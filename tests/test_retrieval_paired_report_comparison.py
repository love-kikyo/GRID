import copy
import math

import pytest

from scripts.compare_retrieval_paired_reports import (
    ABS_TOL, COMPARISONS, METRICS, POLICIES, VARIANTS, compare_paired_reports,
)


def fixture():
    metrics = {metric: {'left_mean': .3, 'right_mean': .2, 'delta': .1,
                        'relative_gain': .5, 'paired_bootstrap_ci': [.09, .11],
                        'confidence': .95, 'nonzero_user_deltas': 17,
                        'role': 'secondary_exploratory'} for metric in METRICS}
    report = {
        'status': 'passed', 'phase': 'smoke', 'seed': 42, 'num_examples': 65,
        'checked_at': '2026-11-08T00:00:00+00:00', 'analysis_cpu_affinity': [0, 1, 28, 33],
        'policies': list(POLICIES), 'statistics': {'bootstrap_samples': 5000, 'confidence': .95},
        'source_git_commit': 'commit', 'manifest_sha256': 'manifest',
        'artifacts': [{'path': '/immutable/sim10.h5', 'sha256': 'hash'}],
        'all_point_estimates_match_acceptance': True,
        'primary': {'comparison': 'sim10_minus_recent10:fused', 'metric': 'ndcg@10'},
        'point_estimates': {p: {v: {m: .25 for m in METRICS} for v in VARIANTS} for p in POLICIES},
        'comparisons': {c: {'left': ['sim10', 'fused'], 'right': ['recent10', 'fused'],
                            'direction': 'left_minus_right', 'metrics': copy.deepcopy(metrics)}
                        for c in COMPARISONS},
    }
    report['comparisons']['sim10_minus_mask0:raw']['metrics']['ndcg@5']['paired_bootstrap_ci'][0] = .09279855887629447
    return report


def change(report, path, value):
    current = report
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


CI_PATH = ('comparisons', 'sim10_minus_mask0:raw', 'metrics', 'ndcg@5', 'paired_bootstrap_ci', 0)
POINT_PATH = ('point_estimates', 'sim10', 'fused', 'ndcg@10')
DELTA_PATH = ('comparisons', 'sim10_minus_recent10:fused', 'metrics', 'ndcg@10', 'delta')


def test_exact_report_and_runtime_metadata_are_read_only():
    original = fixture()
    current = copy.deepcopy(original)
    current['checked_at'] = '2026-11-08T01:00:00+00:00'
    current['analysis_cpu_affinity'] = [2]
    before = copy.deepcopy((original, current))
    result = compare_paired_reports(original, current)
    assert result['status'] == 'passed'
    assert result['tolerance'] == {'abs_tol': 1e-12, 'rel_tol': 0.0}
    assert result['tolerated_differences'] == [] and result['max_abs_difference'] == 0.
    assert result['exact_equal_excluding_runtime_fields'] is True
    assert (original, current) == before


def test_real_one_ulp_endpoint_difference_is_recorded():
    original, current = fixture(), fixture()
    change(current, CI_PATH, .09279855887629448)
    result = compare_paired_reports(original, current)
    assert not result['exact_equal_excluding_runtime_fields']
    assert result['max_abs_difference'] == math.ulp(.09279855887629447)
    assert result['tolerated_differences'] == [{
        'path': '/' + '/'.join(map(str, CI_PATH)), 'recorded': .09279855887629447,
        'fresh': .09279855887629448, 'absolute_difference': 1.3877787807814457e-17}]


@pytest.mark.parametrize('path', [POINT_PATH, DELTA_PATH])
def test_other_explicit_statistics_leaves_allow_only_small_roundoff(path):
    original, current = fixture(), fixture()
    change(original, path, 0.)
    change(current, path, ABS_TOL)
    assert compare_paired_reports(original, current)['max_abs_difference'] == ABS_TOL
    change(current, path, math.nextafter(ABS_TOL, math.inf))
    with pytest.raises(ValueError, match='exceeds tolerance'):
        compare_paired_reports(original, current)


def test_no_relative_tolerance_even_for_large_relative_gain():
    path = DELTA_PATH[:-1] + ('relative_gain',)
    original, current = fixture(), fixture()
    change(original, path, 1e6)
    change(current, path, 1e6 + 1e-8)
    with pytest.raises(ValueError, match='exceeds tolerance'):
        compare_paired_reports(original, current)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize('path', [CI_PATH, POINT_PATH, ('statistics', 'confidence')])
def test_nonfinite_values_rejected_even_when_identical(path, value):
    original, current = fixture(), fixture()
    change(original, path, value)
    change(current, path, value)
    with pytest.raises(ValueError, match='Nonfinite'):
        compare_paired_reports(original, current)


@pytest.mark.parametrize('path,value', [
    (POINT_PATH, 0), (DELTA_PATH, True), (('num_examples',), 65.),
    (('all_point_estimates_match_acceptance',), 1),
    (('statistics', 'confidence'), math.nextafter(.95, 1.)),
    (('source_git_commit',), 'different'), (('manifest_sha256',), 'changed'),
    (('artifacts', 0, 'sha256'), 'changed'), (('artifacts', 0, 'path'), '/other'),
    (('seed',), 43), (('num_examples',), 64),
    (DELTA_PATH[:-1] + ('nonzero_user_deltas',), 18),
    (DELTA_PATH[:-1] + ('confidence',), math.nextafter(.95, 1.)),
    (DELTA_PATH[:-1] + ('role',), 'primary'),
    (('primary', 'comparison'), 'sim10_minus_mask0:fused'),
])
def test_identity_config_counts_types_and_roles_remain_exact(path, value):
    original, current = fixture(), fixture()
    change(current, path, value)
    with pytest.raises(ValueError):
        compare_paired_reports(original, current)


@pytest.mark.parametrize('path', [POINT_PATH, DELTA_PATH])
def test_statistic_float_is_required_even_when_both_wrong(path):
    original, current = fixture(), fixture()
    change(original, path, 0)
    change(current, path, 0)
    with pytest.raises(ValueError, match='statistical float'):
        compare_paired_reports(original, current)


def test_relative_gain_none_is_allowed_only_when_both_undefined():
    path = DELTA_PATH[:-1] + ('relative_gain',)
    original, current = fixture(), fixture()
    change(original, path, None)
    change(current, path, None)
    assert compare_paired_reports(original, current)['status'] == 'passed'
    change(current, path, 0.)
    with pytest.raises(ValueError, match='type changed'):
        compare_paired_reports(original, current)
    change(original, DELTA_PATH, None)
    change(current, DELTA_PATH, None)
    change(current, path, None)
    with pytest.raises(ValueError, match='statistical float'):
        compare_paired_reports(original, current)


@pytest.mark.parametrize('action', ['extra_top', 'missing_top', 'missing_runtime', 'extra_metric',
                                   'missing_policy', 'unknown_variant', 'unknown_comparison',
                                   'extra_endpoint', 'short_endpoint', 'list_type'])
def test_schema_changes_fail(action):
    original, current = fixture(), fixture()
    if action == 'extra_top':
        current['extra'] = 1
    elif action == 'missing_top':
        del current['seed']
    elif action == 'missing_runtime':
        del current['checked_at']
    elif action == 'extra_metric':
        current['point_estimates']['sim10']['fused']['ndcg@100'] = .25
    elif action == 'missing_policy':
        del current['point_estimates']['mask0']
    elif action == 'unknown_variant':
        current['point_estimates']['sim10']['click'] = current['point_estimates']['sim10'].pop('raw')
    elif action == 'unknown_comparison':
        current['comparisons']['invented'] = current['comparisons'].pop(COMPARISONS[0])
    elif action == 'extra_endpoint':
        current['comparisons'][COMPARISONS[0]]['metrics']['ndcg@5']['paired_bootstrap_ci'].append(.2)
    elif action == 'short_endpoint':
        current['comparisons'][COMPARISONS[0]]['metrics']['ndcg@5']['paired_bootstrap_ci'].pop()
    else:
        current['policies'] = tuple(current['policies'])
    with pytest.raises(ValueError):
        compare_paired_reports(original, current)


@pytest.mark.parametrize('key,value', [('checked_at', None), ('checked_at', ''),
    ('analysis_cpu_affinity', []), ('analysis_cpu_affinity', [True]),
    ('analysis_cpu_affinity', [1.]), ('analysis_cpu_affinity', [-1]),
    ('analysis_cpu_affinity', [1, 1]), ('analysis_cpu_affinity', '0')])
def test_ignored_runtime_values_still_require_valid_types(key, value):
    original, current = fixture(), fixture()
    original[key] = current[key] = value
    with pytest.raises(ValueError, match='Invalid runtime'):
        compare_paired_reports(original, current)


def test_lookalike_float_key_outside_whitelist_remains_exact():
    original, current = fixture(), fixture()
    original['config'] = {'delta': .1}
    current['config'] = {'delta': math.nextafter(.1, 1.)}
    with pytest.raises(ValueError, match='Exact report field changed'):
        compare_paired_reports(original, current)
