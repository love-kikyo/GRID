"""Stream paired user-level intervals for the frozen five-policy content test."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import h5py
import numpy as np

from scripts.analyze_paired_evaluation import (
    metric_contributions, paired_bootstrap_ci_from_counts, ranked_candidates,
)
from scripts.audit_inference_sensitivity import require, sha


POLICIES = ['sim10', 'recent10', 'random10', 'mask5', 'mask0']
STATISTICS = dict(bootstrap_samples=5000, bootstrap_batch_size=50, bootstrap_seed=2027,
                  confidence=.95, ks=[5, 10, 20],
                  primary_comparison='sim10_minus_recent10:fused', primary_metric='ndcg@10')
CHUNK_SIZE = 4096
IDENTITY = ('user_id', 'example_hash', 'target_sid')
FIELDS = IDENTITY + ('beam_sid', 'beam_score', 'raw_rerank_logit')
VARIANTS = {'beam': 'beam', 'raw': 'click', 'fused': 'fusion@1@0.5'}
METRICS = [f'{metric}@{k}' for k in STATISTICS['ks'] for metric in ('recall', 'ndcg')]


def comparisons():
    result = {}
    for policy in POLICIES[1:]:
        for variant in ('raw', 'fused'):
            result[f'sim10_minus_{policy}:{variant}'] = (('sim10', variant), (policy, variant))
    for policy in POLICIES:
        result[f'{policy}:fusion_minus_beam'] = ((policy, 'fused'), ('sim10', 'beam'))
    return result


def _stat(path):
    value = Path(path).stat()
    return (value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _bind_inputs(directory, m, phase):
    from scripts.audit_retrieval_content import _manifest_check, _run_check
    require(m.get('statistics') == STATISTICS, 'Frozen statistics plan changed')
    require(not any('incomplete' in p.name.lower() for p in directory.iterdir()),
            'Incomplete export present')
    count = _manifest_check(m, phase)
    receipt_path, run_path = directory / 'acceptance.json', directory / 'run.json'
    paths = [receipt_path, run_path, Path(m['reference_test']['path'])] + [directory / f'{p}.h5' for p in POLICIES]
    snapshots = {str(p): _stat(p) for p in paths}
    receipt = json.loads(receipt_path.read_text())
    run = json.loads(run_path.read_text())
    _run_check(run, m, phase, count)
    require(receipt.get('status') == 'passed' and receipt.get('phase') == phase
            and type(receipt.get('seed')) is int and receipt['seed'] == m['seed']
            and receipt.get('num_examples') == count and receipt.get('unique_users') == count,
            'Acceptance identity mismatch')
    require(receipt.get('manifest_sha256') == m['_manifest_sha256']
            and receipt.get('policies') == POLICIES, 'Acceptance manifest/policies mismatch')
    for key in ('source_git_commit', 'checkpoint', 'reference_test', 'frozen_beta'):
        require(receipt.get(key) == m[key], f'Acceptance metadata mismatch: {key}')
    require(set(receipt.get('settings', {})) == set(POLICIES), 'Acceptance policy metrics mismatch')
    require(receipt.get('run') == {'path': str(run_path.resolve()), 'sha256': sha(run_path)},
            'Acceptance run binding mismatch')
    artifacts = receipt.get('artifacts', [])
    require(len(artifacts) == len(POLICIES), 'Acceptance artifact count mismatch')
    records = {r['path']: r['sha256'] for r in artifacts}
    expected_paths = {str((directory / f'{p}.h5').resolve()) for p in POLICIES}
    require(set(records) == expected_paths, 'Acceptance artifact paths mismatch')
    for path, expected in records.items():
        require(sha(path) == expected, 'Acceptance artifact hash mismatch')
    require(all(_stat(path) == value for path, value in snapshots.items()),
            'Input changed during binding')
    return count, receipt, {'path': str(receipt_path.resolve()), 'sha256': sha(receipt_path)}, snapshots


def _contributions(payload):
    result = {}
    for variant, name in VARIANTS.items():
        ranked = ranked_candidates(payload, name)
        result[variant] = {}
        for k in STATISTICS['ks']:
            for metric, value in metric_contributions(payload['target_sid'], ranked, k).items():
                result[variant][f'{metric}@{k}'] = value
    return result


def analyze(output_dir: Path, manifest: dict, phase: str) -> dict:
    """Read-only analysis; all reported intervals are unadjusted user bootstrap CIs."""
    directory = Path(output_dir)
    count, receipt, receipt_record, snapshots = _bind_inputs(directory, manifest, phase)
    pairs = comparisons()
    sums = {p: {v: {k: 0. for k in METRICS} for v in VARIANTS} for p in POLICIES}
    distributions = {name: {metric: Counter() for metric in METRICS} for name in pairs}
    with ExitStack() as stack:
        files = {p: stack.enter_context(h5py.File(directory / f'{p}.h5', 'r')) for p in POLICIES}
        reference = stack.enter_context(h5py.File(manifest['reference_test']['path'], 'r'))
        require(reference['user_id'].shape == (manifest['expected_test_samples'],),
                'Reference sample count mismatch')
        for f in files.values():
            require(f['user_id'].shape == (count,) and f.attrs.get('num_samples') == count,
                    'Paired sample count mismatch')
        for start in range(0, count, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, count)
            identity = {k: reference[k][start:stop] for k in IDENTITY}
            arrays = {p: {k: files[p][k][start:stop] for k in FIELDS} for p in POLICIES}
            values = {}
            for policy, payload in arrays.items():
                for key, value in identity.items():
                    require(np.array_equal(payload[key], value), f'Paired identity mismatch: {policy}/{key}')
                for key in ('beam_sid', 'beam_score'):
                    require(np.array_equal(payload[key], arrays['sim10'][key]),
                            f'Paired candidates/scores mismatch: {policy}/{key}')
                for key in ('beam_score', 'raw_rerank_logit'):
                    require(bool(np.isfinite(payload[key]).all()), f'Nonfinite paired score: {policy}/{key}')
                values[policy] = _contributions(payload)
                for variant, metrics in values[policy].items():
                    for metric, value in metrics.items():
                        sums[policy][variant][metric] += float(value.sum(dtype=np.float64))
            for name, (left, right) in pairs.items():
                for metric in METRICS:
                    delta = values[left[0]][left[1]][metric] - values[right[0]][right[1]][metric]
                    unique, frequencies = np.unique(delta, return_counts=True)
                    distributions[name][metric].update({float(v): int(n) for v, n in zip(unique, frequencies)})
    means = {p: {v: {k: s / count for k, s in metrics.items()} for v, metrics in variants.items()}
             for p, variants in sums.items()}
    for policy in POLICIES:
        for variant in VARIANTS:
            for metric, value in means[policy][variant].items():
                require(abs(value - receipt['settings'][policy]['metrics'][variant][metric]) < 2e-12,
                        f'Acceptance point estimate mismatch: {policy}/{variant}/{metric}')
    results = {}
    for name, (left, right) in pairs.items():
        results[name] = {'left': list(left), 'right': list(right), 'direction': 'left_minus_right', 'metrics': {}}
        for metric in METRICS:
            histogram = dict(sorted(distributions[name][metric].items()))
            require(sum(histogram.values()) == count, 'Paired distribution count mismatch')
            mean = sum(value * n for value, n in histogram.items()) / count
            left_mean, right_mean = means[left[0]][left[1]][metric], means[right[0]][right[1]][metric]
            require(abs(mean - (left_mean - right_mean)) < 2e-12, 'Paired delta mean mismatch')
            interval = paired_bootstrap_ci_from_counts(
                histogram, samples=5000, batch_size=50, confidence=.95,
                rng=np.random.default_rng(2027))
            results[name]['metrics'][metric] = {
                'left_mean': left_mean, 'right_mean': right_mean, 'delta': mean,
                'relative_gain': mean / right_mean if right_mean else None,
                'paired_bootstrap_ci': list(interval), 'confidence': .95,
                'nonzero_user_deltas': count - histogram.get(0., 0),
                'role': ('primary' if name == STATISTICS['primary_comparison']
                         and metric == STATISTICS['primary_metric'] else 'secondary_exploratory'),
            }
    require(all(_stat(path) == value for path, value in snapshots.items()),
            'Input changed during analysis')
    require(sha(directory / 'acceptance.json') == receipt_record['sha256'], 'Acceptance changed during analysis')
    return {
        'status': 'passed', 'phase': phase, 'seed': manifest['seed'], 'checked_at': datetime.now(timezone.utc).isoformat(),
        'num_examples': count, 'policies': POLICIES, 'statistics': STATISTICS,
        'sample_unit': 'paired_user', 'bootstrap_method': 'paired_nonparametric_multinomial_counts',
        'rng_policy': 'Independent reset to seed2027 per contrast/metric; sorted unique delta values',
        'primary': {'comparison': STATISTICS['primary_comparison'], 'metric': STATISTICS['primary_metric']},
        'comparisons': results, 'point_estimates': means,
        'acceptance': receipt_record, 'artifacts': receipt['artifacts'], 'run': receipt['run'],
        'manifest_sha256': manifest['_manifest_sha256'], 'source_git_commit': manifest['source_git_commit'],
        'checkpoint': manifest['checkpoint'], 'reference_test': manifest['reference_test'],
        'frozen_beta': manifest['frozen_beta'], 'analysis_cpu_affinity': sorted(os.sched_getaffinity(0)),
        'no_parameter_selection': True, 'all_point_estimates_match_acceptance': True,
        'limitations': [
            'Intervals are conditional on this checkpoint, user sample and fixed inference policies; not training-seed variability',
            'Secondary intervals are unadjusted and exploratory; no family-wise significance claims or equivalence claim',
            'Main contrast was fixed before the full pass, motivated by the earlier overlapping test prefix; not a new untouched holdout',
            'Recent/random selection retains candidate-similarity ordering; only the chosen history set is varied',
            'Smoke statistics validate the analysis pipeline only and must not be used as an effectiveness result',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    args = parser.parse_args()
    destination = args.output_dir / 'paired.json'
    require(not destination.exists(), 'Paired report already exists')
    manifest = json.loads(args.manifest.read_text())
    manifest['_manifest_sha256'] = sha(args.manifest)
    report = analyze(args.output_dir, manifest, args.phase)
    with destination.open('x') as f:
        f.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'phase': args.phase,
                      'num_examples': report['num_examples'], 'comparisons': len(report['comparisons'])}))


if __name__ == '__main__':
    main()
