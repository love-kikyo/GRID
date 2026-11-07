"""Read-only acceptance of the five frozen seed42 retrieval-content policies."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.audit_inference_sensitivity import _metric_sums, _packed, _ranks, require, sha
from scripts.audit_retrieval_diagnostic import (
    FIELDS, MAX_ITEMKEY, SMOKE_FLAGS as DIAGNOSTIC_FLAGS, _contexts, _delta, _shape_check,
)


POLICIES = ['sim10', 'recent10', 'random10', 'mask5', 'mask0']
SMOKE_FLAGS = DIAGNOSTIC_FLAGS + ('subset_matches_frozen_diagnostic',)


def _manifest_check(m, phase):
    require(phase in {'smoke', 'full'}, 'Invalid content phase')
    require(type(m.get('seed')) is int and m['seed'] == 42, 'Only seed42 is authorized')
    require(m.get('policies') == POLICIES and m.get('C') == 20
            and m.get('physical_slots') == 10, 'Content policies or dimensions changed')
    require(m.get('beta') == .5 and m.get('beam_weight') == 1.
            and m.get('runtime_beta') == .1, 'Frozen fusion weights changed')
    require(m.get('protocol') == 'history-last-user-v1', 'Unexpected data protocol')
    require(bool(m.get('source_git_commit')) and m.get('batch_size') == 32
            and m.get('amp') == 'bf16' and m.get('random_seed') == 42
            and m.get('training_performed', False) is False, 'Invalid frozen inference metadata')
    expected, full, smoke = (m.get(k) for k in ('expected_test_samples', 'full_samples', 'smoke_samples'))
    require(all(type(v) is int for v in (expected, full, smoke))
            and 0 < smoke <= full == expected, 'Invalid full/smoke sample count')
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
    return smoke if phase == 'smoke' else full


def _run_check(run, m, phase, count):
    require(run.get('status') == 'complete' and run.get('phase') == phase, 'Run status/phase mismatch')
    require(type(run.get('seed')) is int and run['seed'] == m['seed'], 'Run seed mismatch')
    require(run.get('processed') == count and run.get('expected_samples') == m['expected_test_samples'],
            'Run sample count mismatch')
    require('_manifest_sha256' in m and run.get('manifest_sha256') == m['_manifest_sha256'],
            'Run manifest binding mismatch')
    for key in ('source_git_commit', 'checkpoint', 'policies', 'beta', 'runtime_beta', 'batch_size', 'amp'):
        require(run.get(key) == m[key], f'Run metadata mismatch: {key}')
    require(run.get('training_performed') is False, 'Run must not perform training')
    require(run.get('checkpoint_loading') == {'missing': 0, 'unexpected': 0},
            'Checkpoint loading was not exact')
    gate = run.get('smoke_gate', {})
    require(gate.get('status') == 'passed' and all(gate.get(k) is True for k in SMOKE_FLAGS),
            'Missing complete real smoke acceptance')
    require(gate.get('samples') == min(m['batch_size'], m['smoke_samples'])
            and gate.get('policies') == POLICIES and gate.get('random_seed') == 42
            and gate.get('random_namespace') == 'grid-retrieval-diagnostic-v1-history-positions',
            'Smoke gate metadata mismatch')
    positions, lengths = gate.get('candidate_positions', {}), gate.get('input_lengths', {})
    require(set(positions) == set(POLICIES) and set(lengths) == set(POLICIES),
            'Smoke gate layout policies mismatch')
    center = positions['sim10']
    require(isinstance(center, list) and bool(center)
            and all(type(p) is int and p >= 0 for p in center)
            and all(positions[p] == center and lengths[p] == [13] for p in POLICIES),
            'Smoke gate layout mismatch')
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
            'Full export is missing prior smoke acceptance')
    path = Path(record['path'])
    require(path.name == 'acceptance.json' and sha(path) == record['sha256'],
            'Prior smoke acceptance path/hash mismatch')
    prior = json.loads(path.read_text())
    require(prior.get('phase') == 'smoke', 'Prior smoke phase mismatch')
    # Recheck the small smoke itself, not just a status flag or a stale receipt.
    # This recursion terminates because a smoke run must have no prior receipt.
    fresh = audit(path.parent, m, 'smoke')
    require({k: v for k, v in prior.items() if k != 'checked_at'}
            == {k: v for k, v in fresh.items() if k != 'checked_at'},
            'Prior smoke receipt differs from current artifacts')
    previous = json.loads((path.parent / 'run.json').read_text())
    require(previous['smoke_gate'] == run['smoke_gate'], 'Prior smoke gate differs from full gate')
    paired_record = record.get('paired_analysis', {})
    paired_path = path.parent / 'paired.json'
    require(paired_record.get('path') == str(paired_path.resolve())
            and paired_record.get('sha256') == sha(paired_path), 'Prior smoke paired path/hash mismatch')
    paired = json.loads(paired_path.read_text())
    require(paired.get('status') == 'passed' and paired.get('phase') == 'smoke'
            and type(paired.get('seed')) is int and paired['seed'] == 42
            and paired.get('num_examples') == m['smoke_samples'] and paired.get('policies') == POLICIES,
            'Prior smoke paired identity mismatch')
    for key in ('manifest_sha256', 'source_git_commit', 'checkpoint', 'reference_test', 'frozen_beta', 'run', 'artifacts'):
        require(paired.get(key) == prior[key], f'Prior smoke paired binding mismatch: {key}')
    require(paired.get('statistics') == m.get('statistics')
            and paired.get('acceptance') == {'path': str(path.resolve()), 'sha256': record['sha256']}
            and paired.get('no_parameter_selection') is True
            and paired.get('all_point_estimates_match_acceptance') is True,
            'Prior smoke paired statistics/acceptance mismatch')


def audit(output_dir: Path, manifest: dict, phase: str) -> dict:
    """Read every exported row and return a report without modifying artifacts."""
    output_dir = Path(output_dir)
    require(not any('incomplete' in p.name.lower() for p in output_dir.iterdir()),
            'Incomplete export present')
    require({p.name for p in output_dir.glob('*.h5')} == {f'{p}.h5' for p in POLICIES},
            'Unexpected HDF5 policy files')
    count = _manifest_check(manifest, phase)
    run_path = output_dir / 'run.json'
    run = json.loads(run_path.read_text())
    _run_check(run, manifest, phase, count)
    totals = {p: {v: {} for v in ('beam', 'raw', 'fused')} for p in POLICIES}
    coverage, std_sums = ({p: 0 for p in POLICIES}, {p: 0. for p in POLICIES})
    same_context = {p: 0 for p in ('recent10', 'random10')}
    users, short_histories = set(), 0
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
            short_histories += int((center['history_valid_count'] < 10).sum())
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
        'policies': POLICIES, 'num_examples': count, 'unique_users': len(users), 'settings': settings,
        'minus_sim10': {p: _delta(settings[p]['metrics'], settings['sim10']['metrics'])
                        for p in POLICIES if p != 'sim10'},
        'context_multiset_equal_to_sim10_fraction_per_user_candidate':
            {p: value / (count * 20) for p, value in same_context.items()},
        'history_fewer_than_10_fraction': short_histories / count,
        'all_policies_share_candidates_and_scores': True,
        'fixed_slot_candidate_positions_verified': True, 'controlled_mask_contents_verified': True,
        'independent_sort_and_outranker_ranks_verified': True,
        'fusion': {'beam_weight': 1., 'rerank_weight': .5, 'precision': 'float32', 'ties': 'stable original beam order'},
        'sample_scope': ('Complete frozen test in original order' if phase == 'full'
                         else f'First {count} frozen test rows; smoke checks only, not effect evidence'),
        'artifacts': [{'path': str((output_dir / f'{p}.h5').resolve()), 'sha256': sha(output_dir / f'{p}.h5')}
                      for p in POLICIES],
        'run': {'path': str(run_path.resolve()), 'sha256': sha(run_path)},
        'hardware': run['hardware'], 'hardware_end': run['hardware_end'], 'cpu_budget': run['cpu_budget'],
        'manifest_sha256': manifest['_manifest_sha256'], 'checkpoint': manifest['checkpoint'],
        'source_git_commit': manifest['source_git_commit'], 'prior_smoke_acceptance': run.get('prior_smoke_acceptance'),
        'reference_test': manifest['reference_test'], 'frozen_beta': manifest['frozen_beta'],
        'limitations': ['One fixed training seed; no parameter selection or retraining',
                        'Paired confidence intervals are reported separately, not inferred from point estimates',
                        'Recent/random selections retain candidate-similarity ordering; selection, not ordering, is varied',
                        'Mask0 retains the original SID-history KV cache; masks are inference interventions, not training ablations',
                        'History membership, GT independence and actual hook provenance additionally rely on runner gates',
                        'Reference hash inherited from manifest; exported identities and targets are reread',
                        'Runtime beta0.1 ordering is separate from reported FP32 beta0.5 fusion'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
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
