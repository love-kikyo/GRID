"""Read-only acceptance for fixed-slot/context seed42 inference diagnostics."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.audit_inference_sensitivity import (
    FIELDS as BASE_FIELDS, _metric_sums, _packed, _ranks, require, sha,
)


POLICIES = ['sim10', 'mask0', 'mask5', 'recent10', 'random10', 'short0', 'short5']
EXTRA_FIELDS = {'context_itemkey', 'history_valid_count', 'candidate_position', 'rerank_input_length'}
FIELDS = BASE_FIELDS | EXTRA_FIELDS
MAX_ITEMKEY = 4096 * sum(4096**i for i in range(4))
SMOKE_FLAGS = ('native_sim10_exact', 'native_active_masks_exact', 'shared_candidates_and_scores',
               'gt_independence_all_policies', 'cache_order_independence',
               'actual_rotary_positions_observed', 'fixed_slots_candidate_positions_equal',
               'paired_short_mask_content_equal', 'parameters_unchanged_and_gradients_absent')


def _manifest_check(m, phase):
    require(phase in {'smoke', 'diagnostic'}, 'Invalid diagnostic phase')
    require(type(m.get('seed')) is int and m['seed'] == 42, 'Only seed42 is authorized')
    require(m.get('policies') == POLICIES and m.get('C') == 20
            and m.get('physical_slots') == 10, 'Diagnostic policies or dimensions changed')
    require(m.get('beta') == .5 and m.get('beam_weight') == 1.
            and m.get('runtime_beta') == .1, 'Frozen fusion weights changed')
    require(m.get('protocol') == 'history-last-user-v1', 'Unexpected data protocol')
    require(bool(m.get('source_git_commit')) and m.get('batch_size') == 32
            and m.get('amp') == 'bf16' and m.get('random_seed') == 42
            and m.get('training_performed', False) is False, 'Invalid frozen inference metadata')
    full_count = m.get('expected_test_samples')
    count = m.get('smoke_samples' if phase == 'smoke' else 'diagnostic_samples')
    require(type(full_count) is int and type(count) is int and 0 < count <= full_count,
            'Invalid diagnostic sample count')
    for record in m.get('execution_sources', []) + m.get('input_metadata', []):
        require(sha(Path(record['path'])) == record['sha256'], 'Source/input hash mismatch')
    record = m['frozen_beta']
    require(sha(Path(record['path'])) == record['sha256'], 'Selection hash mismatch')
    selection = json.loads(Path(record['path']).read_text())
    require(selection.get('input_split') == 'val'
            and selection.get('selection_scope') == 'validation_only', 'Not validation-only selection')
    require(selection['selected']['beam_weight'] == 1.
            and selection['selected']['rerank_weight'] == .5, 'Selection weights mismatch')
    require(selection['input_checkpoint_sha256'] == m['checkpoint']['sha256'],
            'Selection checkpoint mismatch')
    return count


def _run_check(run, m, phase, count):
    require(run.get('status') == 'complete' and run.get('phase') == phase,
            'Run status/phase mismatch')
    require(type(run.get('seed')) is int and run['seed'] == m['seed'], 'Run seed mismatch')
    require(run.get('processed') == count and run.get('expected_samples') == m['expected_test_samples'],
            'Run sample count mismatch')
    require('_manifest_sha256' in m and run.get('manifest_sha256') == m['_manifest_sha256'],
            'Run manifest binding mismatch')
    for key in ('source_git_commit', 'checkpoint', 'policies', 'beta', 'runtime_beta', 'batch_size', 'amp'):
        require(run.get(key) == m[key], f'Run metadata mismatch: {key}')
    require(run.get('training_performed') is False, 'Run must not perform training')
    gate = run.get('smoke_gate', {})
    require(gate.get('status') == 'passed' and all(gate.get(k) is True for k in SMOKE_FLAGS),
            'Missing complete real smoke acceptance')
    require(gate.get('samples') == min(m['batch_size'], m['smoke_samples'])
            and gate.get('policies') == POLICIES and gate.get('random_seed') == m['random_seed']
            and gate.get('random_namespace') == 'grid-retrieval-diagnostic-v1-history-positions',
            'Smoke gate metadata mismatch')
    hardware, end = run.get('hardware', {}), run.get('hardware_end', {})
    require('4090' in hardware.get('name', '') and bool(hardware.get('uuid'))
            and end.get('uuid') == hardware['uuid'], 'GPU identity mismatch')
    budget = run.get('cpu_budget', {})
    cpus = set(budget.get('allowed_cpus', []))
    require(0 < len(cpus) <= 4 and budget.get('allocated_cpus') == 4
            and budget.get('library_threads') == 1, 'Invalid CPU budget')
    snapshots = run.get('cpu_process_tree_samples', [])
    require(len(snapshots) >= 2, 'Missing CPU start/end snapshots')
    for snapshot in snapshots:
        require(set(snapshot.get('allowed_cpus', [])) == cpus and bool(snapshot.get('processes')),
                'Invalid CPU snapshot')
    _prior_smoke_check(run, m, phase)


def _prior_smoke_check(run, m, phase):
    record = run.get('prior_smoke_acceptance')
    if phase == 'smoke':
        require(record is None, 'Smoke cannot reuse a prior acceptance')
        return
    require(isinstance(record, dict) and bool(record.get('path')) and bool(record.get('sha256')),
            'Diagnostic is missing prior smoke acceptance')
    path = Path(record['path'])
    require(sha(path) == record['sha256'], 'Prior smoke acceptance hash mismatch')
    prior = json.loads(path.read_text())
    require(prior.get('status') == 'passed' and prior.get('phase') == 'smoke'
            and type(prior.get('seed')) is int and prior['seed'] == 42
            and prior.get('num_examples') == m['smoke_samples']
            and prior.get('manifest_sha256') == m['_manifest_sha256'], 'Prior smoke identity mismatch')
    for key in ('source_git_commit', 'checkpoint', 'reference_test', 'frozen_beta'):
        require(prior.get(key) == m[key], f'Prior smoke metadata mismatch: {key}')
    prior_run = prior.get('run', {})
    require(bool(prior_run.get('path')) and bool(prior_run.get('sha256')),
            'Missing prior smoke run binding')
    run_path = path.parent / 'run.json'
    require(Path(prior_run['path']).resolve() == run_path.resolve()
            and sha(run_path) == prior_run['sha256'], 'Prior smoke run hash/path mismatch')
    previous = json.loads(run_path.read_text())
    _run_check(previous, m, 'smoke', m['smoke_samples'])
    require(previous['smoke_gate'] == run['smoke_gate'], 'Prior smoke gate differs from diagnostic gate')


def _shape_check(f, n):
    require(set(f.keys()) == FIELDS, 'Unexpected diagnostic HDF5 fields')
    shapes = {'user_id': (n,), 'example_hash': (n, 2), 'target_sid': (n, 4),
              'beam_sid': (n, 20, 4), 'beam_score': (n, 20), 'raw_rerank_logit': (n, 20),
              'rerank_sid': (n, 20, 4), 'rerank_score': (n, 20),
              'context_itemkey': (n, 20, 10), 'history_valid_count': (n,),
              'candidate_position': (n,), 'rerank_input_length': (n,)}
    for key, shape in shapes.items():
        require(f[key].shape == shape, f'Shape mismatch: {key}')
    require(f.attrs.get('num_samples') == n, 'HDF5 sample count mismatch')
    dtypes = {'user_id': 'int64', 'example_hash': 'uint64', 'beam_score': 'float32',
              'raw_rerank_logit': 'float32', 'rerank_score': 'float32',
              'context_itemkey': 'int64', 'history_valid_count': 'int32',
              'candidate_position': 'int32', 'rerank_input_length': 'int16'}
    for key, dtype in dtypes.items():
        require(f[key].dtype == np.dtype(dtype), f'Dtype mismatch: {key}')


def _contexts(a, center, policy):
    history = a['history_valid_count']
    require(np.array_equal(history, center['history_valid_count']) and bool((history >= 0).all()),
            f'History counts changed: {policy}')
    context = a['context_itemkey']
    require(bool(((context >= -1) & (context <= MAX_ITEMKEY)).all()),
            f'Invalid context itemkey: {policy}')
    active = 0 if policy in {'mask0', 'short0'} else (5 if policy in {'mask5', 'short5'} else 10)
    expected = np.minimum(history, active)[:, None]
    require(bool(((context != -1).sum(-1) == expected).all()), f'Context count mismatch: {policy}')
    short = 10 if policy == 'short0' else (5 if policy == 'short5' else 0)
    require(bool((a['candidate_position'] >= 0).all())
            and np.array_equal(a['candidate_position'], center['candidate_position'] - short),
            f'Candidate position mismatch: {policy}')
    require(bool((a['rerank_input_length'] == 13 - short).all()), f'Input length mismatch: {policy}')
    if policy in {'mask0', 'mask5', 'short0', 'short5'}:
        masked = center['context_itemkey'].copy()
        masked[:, :, active:] = -1
        require(np.array_equal(context, masked), f'Controlled contexts changed: {policy}')
    if policy in {'recent10', 'random10'}:
        bags = np.sort(context, axis=-1)
        require(bool((bags == bags[:, :1, :]).all()), f'User selection differs by candidate: {policy}')


def _delta(left, right):
    return {variant: {metric: value - right[variant][metric] for metric, value in values.items()}
            for variant, values in left.items()}


def audit(output_dir: Path, manifest: dict, phase: str) -> dict:
    """Validate all exported rows and return a report without writing anything."""
    output_dir = Path(output_dir)
    require(not any('incomplete' in p.name.lower() for p in output_dir.iterdir()),
            'Incomplete export present')
    count = _manifest_check(manifest, phase)
    run_path = output_dir / 'run.json'
    run = json.loads(run_path.read_text())
    _run_check(run, manifest, phase, count)
    totals = {p: {v: {} for v in ('beam', 'raw', 'fused')} for p in POLICIES}
    coverage = {p: 0 for p in POLICIES}
    std_sums = {p: 0. for p in POLICIES}
    same_context = {p: 0 for p in ('recent10', 'random10')}
    users, history_counts = set(), []
    with ExitStack() as stack:
        reference = stack.enter_context(h5py.File(manifest['reference_test']['path'], 'r'))
        require(reference['user_id'].shape[0] == manifest['expected_test_samples'], 'Reference size mismatch')
        files = {p: stack.enter_context(h5py.File(output_dir / f'{p}.h5', 'r')) for p in POLICIES}
        for f in files.values():
            _shape_check(f, count)
        for start in range(0, count, 1024):
            stop = min(start + 1024, count)
            identity = {k: reference[k][start:stop] for k in ('user_id', 'example_hash', 'target_sid')}
            old_size = len(users)
            users.update(map(int, identity['user_id']))
            require(len(users) - old_size == stop - start, 'Repeated evaluation user')
            center = {k: files['sim10'][k][start:stop] for k in FIELDS}
            history_counts.extend(center['history_valid_count'].tolist())
            for policy in POLICIES:
                a = center if policy == 'sim10' else {k: files[policy][k][start:stop] for k in FIELDS}
                for key, value in identity.items():
                    require(np.array_equal(a[key], value), f'Reference {key} mismatch: {policy}')
                for key in ('beam_sid', 'beam_score'):
                    require(np.array_equal(a[key], center[key]), f'Shared candidates/scores changed: {policy}')
                for key in ('beam_score', 'raw_rerank_logit', 'rerank_score'):
                    require(bool(np.isfinite(a[key]).all()), f'Nonfinite {key}: {policy}')
                for key in ('beam_score', 'rerank_score'):
                    require(bool((a[key][:, :-1] >= a[key][:, 1:]).all()), f'Unsorted {key}: {policy}')
                beam, rerank, target = (_packed(a[k]) for k in ('beam_sid', 'rerank_sid', 'target_sid'))
                ordered = np.sort(beam, axis=1)
                require(bool((ordered[:, 1:] != ordered[:, :-1]).all()), f'Duplicate candidates: {policy}')
                require(np.array_equal(ordered, np.sort(rerank, axis=1)), f'Rerank is not a permutation: {policy}')
                _contexts(a, center, policy)
                matches = beam == target[:, None]
                coverage[policy] += int(matches.any(axis=1).sum())
                fused = a['beam_score'] + np.float32(.5) * a['raw_rerank_logit']
                require(bool(np.isfinite(fused).all()), f'Nonfinite fused score: {policy}')
                for variant, scores in (('beam', None), ('raw', a['raw_rerank_logit']), ('fused', fused)):
                    for metric, value in _metric_sums(_ranks(matches, scores), (5, 10, 20)).items():
                        totals[policy][variant][metric] = totals[policy][variant].get(metric, 0) + value
                std_sums[policy] += float(a['raw_rerank_logit'].astype(np.float64).std(axis=1).sum())
                if policy in same_context:
                    same_context[policy] += int((np.sort(a['context_itemkey'], axis=-1)
                                                 == np.sort(center['context_itemkey'], axis=-1)).all(-1).sum())
    settings = {}
    for policy in POLICIES:
        metrics = {v: {k: value / count for k, value in values.items()} for v, values in totals[policy].items()}
        settings[policy] = {'C': 20, 'metrics': metrics, 'coverage_hits': coverage[policy],
                            'coverage_at_C': coverage[policy] / count,
                            'mean_within_user_raw_logit_std': std_sums[policy] / count,
                            'fusion_minus_beam': {k: metrics['fused'][k] - metrics['beam'][k]
                                                  for k in metrics['fused']}}
    return {
        'status': 'passed', 'phase': phase, 'seed': 42, 'checked_at': datetime.now(timezone.utc).isoformat(),
        'num_examples': count, 'unique_users': len(users), 'settings': settings,
        'minus_sim10': {p: _delta(settings[p]['metrics'], settings['sim10']['metrics'])
                        for p in POLICIES if p != 'sim10'},
        'fixed_minus_short': {f'mask{k}_minus_short{k}': _delta(settings[f'mask{k}']['metrics'],
                                                               settings[f'short{k}']['metrics']) for k in (0, 5)},
        'context_multiset_equal_to_sim10_fraction_per_user_candidate':
            {p: value / (count * 20) for p, value in same_context.items()},
        'history_fewer_than_10_fraction': float(np.mean(np.array(history_counts) < 10)),
        'all_policies_share_candidates_and_scores': True,
        'fixed_slot_candidate_positions_verified': True, 'matched_short_control_contents_verified': True,
        'independent_sort_and_outranker_ranks_verified': True,
        'fusion': {'beam_weight': 1., 'rerank_weight': .5, 'precision': 'float32', 'ties': 'stable original beam order'},
        'sample_scope': f'First {count} rows of the frozen test in original order; exploratory only',
        'artifacts': [{'path': str((output_dir / f'{p}.h5').resolve()), 'sha256': sha(output_dir / f'{p}.h5')}
                      for p in POLICIES],
        'run': {'path': str(run_path.resolve()), 'sha256': sha(run_path)},
        'hardware': run['hardware'], 'hardware_end': run['hardware_end'],
        'cpu_budget': run['cpu_budget'],
        'manifest_sha256': manifest['_manifest_sha256'], 'checkpoint': manifest['checkpoint'],
        'source_git_commit': manifest['source_git_commit'],
        'prior_smoke_acceptance': run.get('prior_smoke_acceptance'),
        'reference_test': manifest['reference_test'], 'frozen_beta': manifest['frozen_beta'],
        'limitations': ['One fixed seed and predeclared test prefix; not a full-test primary result',
                        'No parameter selection, retraining, confidence intervals or significance claims',
                        'History membership, GT independence and hook provenance additionally rely on runner gates',
                        'Same-size recent/random subsets retain candidate-similarity ordering; selection, not ordering, is varied',
                        'Masked/short comparisons use the same content; they remain inference interventions, not training ablations',
                        'Reference hash inherited from manifest; prefix identities and targets are reread',
                        'Runtime beta0.1 ordering is separate from reported FP32 beta0.5 fusion'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'diagnostic'), required=True)
    args = parser.parse_args()
    receipt = args.output_dir / 'acceptance.json'
    require(not receipt.exists(), 'Acceptance already exists')
    manifest = json.loads(args.manifest.read_text())
    manifest['_manifest_sha256'] = sha(args.manifest)
    report = audit(args.output_dir, manifest, args.phase)
    with receipt.open('x') as f:
        f.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'phase': args.phase, 'num_examples': report['num_examples']}))


if __name__ == '__main__':
    main()
