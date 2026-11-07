"""Compare fixed retrieval reports without relaxing scientific provenance.

Only the explicitly enumerated statistical float leaves permit an absolute
round-off tolerance. This module is pure standard library and performs no I/O.
It does not replace the source, artifact or model checks done by the auditor.
"""
from __future__ import annotations

import math


ABS_TOL = 1e-12
REL_TOL = 0.0
POLICIES = ('sim10', 'recent10', 'random10', 'mask5', 'mask0')
VARIANTS = ('beam', 'raw', 'fused')
METRICS = tuple(f'{metric}@{k}' for k in (5, 10, 20) for metric in ('recall', 'ndcg'))
COMPARISONS = tuple(
    f'sim10_minus_{policy}:{variant}'
    for policy in POLICIES[1:] for variant in ('raw', 'fused')
) + tuple(f'{policy}:fusion_minus_beam' for policy in POLICIES)
FLOAT_FIELDS = frozenset(('left_mean', 'right_mean', 'delta', 'relative_gain'))
IGNORED_RUNTIME_FIELDS = ('checked_at', 'analysis_cpu_affinity')


def _path(path):
    return '/' + '/'.join(str(part) for part in path)


def _require(condition, message, path=()):
    if not condition:
        raise ValueError(f'{message}: {_path(path)}')


def _mapping_keys(value, keys, path):
    _require(type(value) is dict and set(value) == set(keys), 'Unexpected report schema', path)


def _validate_structure(report):
    _require(type(report) is dict, 'Report must be a dictionary')
    for field in ('point_estimates', 'comparisons') + IGNORED_RUNTIME_FIELDS:
        _require(field in report, 'Missing required report field', (field,))
    _require(type(report['checked_at']) is str and bool(report['checked_at']),
             'Invalid runtime timestamp', ('checked_at',))
    affinity = report['analysis_cpu_affinity']
    _require(type(affinity) is list and bool(affinity)
             and all(type(cpu) is int and cpu >= 0 for cpu in affinity)
             and len(set(affinity)) == len(affinity),
             'Invalid runtime CPU affinity', ('analysis_cpu_affinity',))
    points = report['point_estimates']
    _mapping_keys(points, POLICIES, ('point_estimates',))
    for policy, variants in points.items():
        prefix = ('point_estimates', policy)
        _mapping_keys(variants, VARIANTS, prefix)
        for variant, metrics in variants.items():
            _mapping_keys(metrics, METRICS, prefix + (variant,))
    contrasts = report['comparisons']
    _mapping_keys(contrasts, COMPARISONS, ('comparisons',))
    for comparison, record in contrasts.items():
        prefix = ('comparisons', comparison)
        _mapping_keys(record, ('left', 'right', 'direction', 'metrics'), prefix)
        _mapping_keys(record['metrics'], METRICS, prefix + ('metrics',))
        for metric, values in record['metrics'].items():
            current = prefix + ('metrics', metric)
            _mapping_keys(values, FLOAT_FIELDS | {'paired_bootstrap_ci', 'confidence',
                                                   'nonzero_user_deltas', 'role'}, current)
            ci = values['paired_bootstrap_ci']
            _require(type(ci) is list and len(ci) == 2,
                     'Expected two confidence interval endpoints', current + ('paired_bootstrap_ci',))


def _is_statistical_float(path):
    if (len(path) == 4 and path[0] == 'point_estimates' and path[1] in POLICIES
            and path[2] in VARIANTS and path[3] in METRICS):
        return True
    if (len(path) in (5, 6) and path[0] == 'comparisons' and path[1] in COMPARISONS
            and path[2] == 'metrics' and path[3] in METRICS):
        return ((len(path) == 5 and path[4] in FLOAT_FIELDS)
                or (len(path) == 6 and path[4] == 'paired_bootstrap_ci'
                    and type(path[5]) is int and path[5] in (0, 1)))
    return False


def compare_paired_reports(recorded, fresh):
    """Return a round-off audit, or raise ValueError on any disallowed change.

    Inputs are neither mutated nor serialized. All finite float tolerances are
    absolute (1e-12, no relative tolerance). None is allowed only for matching
    undefined relative gains among the statistical leaves.
    """
    _validate_structure(recorded)
    _validate_structure(fresh)
    differences = []

    def visit(left, right, path):
        _require(type(left) is type(right), 'Report value type changed', path)
        if type(left) is float:
            _require(math.isfinite(left) and math.isfinite(right), 'Nonfinite report value', path)
        if _is_statistical_float(path):
            if left is None and path[-1] == 'relative_gain':
                return
            _require(type(left) is float, 'Expected a statistical float', path)
            difference = abs(left - right)
            _require(math.isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                     'Statistical difference exceeds tolerance', path)
            if left != right:
                differences.append({'path': _path(path), 'recorded': left, 'fresh': right,
                                    'absolute_difference': difference})
            return
        if len(path) == 1 and path[0] in IGNORED_RUNTIME_FIELDS:
            return  # Both reports already passed the runtime value validation.
        if type(left) is dict:
            _require(set(left) == set(right), 'Report dictionary keys changed', path)
            _require(all(type(key) is str for key in left), 'Non-string report key', path)
            for key in sorted(left):
                visit(left[key], right[key], path + (key,))
        elif type(left) is list:
            _require(len(left) == len(right), 'Report list length changed', path)
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, path + (index,))
        else:
            _require(type(left) in (str, int, float, bool, type(None)), 'Unsupported report value', path)
            _require(left == right, 'Exact report field changed', path)

    visit(recorded, fresh, ())
    return {'status': 'passed', 'tolerance': {'abs_tol': ABS_TOL, 'rel_tol': REL_TOL},
            'ignored_runtime_fields': list(IGNORED_RUNTIME_FIELDS),
            'exact_equal_excluding_runtime_fields': not differences,
            'tolerated_differences': differences,
            'max_abs_difference': max((v['absolute_difference'] for v in differences), default=0.0)}
