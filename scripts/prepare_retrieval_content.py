"""Freeze an authorized seed42/43/44 five-policy full-test comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from scripts.prepare_inference_sensitivity import ASSET_ROOT, ROOT, SOURCES, record, sha, write_new
from scripts.prepare_retrieval_diagnostic import NEW_SOURCES as DIAGNOSTIC_SOURCES

POLICIES = ['sim10', 'recent10', 'random10', 'mask5', 'mask0']
STATISTICS = dict(bootstrap_samples=5000, bootstrap_batch_size=50, bootstrap_seed=2027,
                  confidence=.95, ks=[5, 10, 20],
                  primary_comparison='sim10_minus_recent10:fused', primary_metric='ndcg@10')
PRIOR_MANIFEST = ASSET_ROOT / 'logs/training_launch/retrieval_diagnostic_seed42_2026-11-07_v1/manifest.json'
PRIOR_MANIFEST_SHA = '160a1a3efa4f9a7c37c2e3effc16c5282eaf71495c25c4249ba15836b98a1e93'
PRIOR_ACCEPTANCE = ASSET_ROOT / 'logs/analysis/retrieval_diagnostic_seed42_2026-11-07_v1/diagnostic/acceptance.json'
PRIOR_ACCEPTANCE_SHA = 'f8571d7b279e3adf357ebebf63c88ec0567182e451ce63e953c662f19dd7d66e'
NEW_SOURCES = [
    'scripts/prepare_retrieval_content.py', 'scripts/run_retrieval_content.py',
    'scripts/retrieval_content_core.py', 'scripts/audit_retrieval_content.py',
    'scripts/analyze_retrieval_content.py', 'scripts/analyze_paired_evaluation.py',
    'tests/test_retrieval_content_manifest.py', 'tests/test_retrieval_content_core.py',
    'tests/test_retrieval_content_runner.py', 'tests/test_retrieval_content_audit.py',
    'tests/test_retrieval_content_paired.py', 'run_retrieval_content_cpu4.sh',
]
MULTISEED_SOURCES = list(dict.fromkeys(SOURCES + DIAGNOSTIC_SOURCES + NEW_SOURCES + [
    'scripts/compare_retrieval_paired_reports.py',
    'tests/test_retrieval_paired_report_comparison.py',
    'run_retrieval_content_multiseed_cpu4.sh',
    'tests/test_retrieval_content_multiseed_launcher.py',
]))
PRIOR_SCANS = {
    43: dict(run_id='132786', best_step='100000',
             manifest_sha256='cec064bd6727bdff742044f5da2c0d0896775ac6af88057be908f151bdd1039b',
             acceptance_sha256='0a4de9f041e08e6379b61a09c6ef85360d2a0f1bf133063b1f9f5429a0f3198a'),
    44: dict(run_id='134298', best_step='080000',
             manifest_sha256='c25225a522a81a5dc8e817477cb76e6957e16649c8781a644ba18e6b4ea75c73',
             acceptance_sha256='22392eab2b3357cdfebc0251cf276db3ff02485d73f5f148a70b3c974aa4f4f8'),
}
INHERITED_FIELDS = ('run_dir', 'data_root', 'training_preflight', 'checkpoint',
                    'reference_test', 'frozen_beta')
SELECTION = ('Recent/random select valid history positions without replacement, retaining repeated items; '
             'then native candidate-similarity ordering')


def scope_fields(seed=42):
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValueError('Content comparison exceeds authorized scope: training seed')
    return dict(schema_version=1, experiment=f'retrieval_content_seed{seed}_full_v1',
                seed=seed, C=20, physical_slots=10, policies=list(POLICIES),
                protocol='history-last-user-v1', asset_root=str(ASSET_ROOT), execution_root=str(ROOT),
                dirty_at_submission=False, beta=.5, beam_weight=1., runtime_beta=.1,
                batch_size=32, num_workers=1, amp='bf16', sequence_length=32,
                hist_itemkey_length=100, expected_test_samples=2061101, smoke_samples=65,
                full_samples=2061101, random_seed=42, statistics=dict(STATISTICS))


def _seed_inputs(seed):
    """Read only small, previously accepted metadata; never scan large assets."""
    source = PRIOR_SCANS[seed]
    prior_path = ASSET_ROOT / f'logs/training_launch/sensitivity_seed{seed}_2026-11-07_v1/manifest.json'
    acceptance_path = ASSET_ROOT / f'logs/analysis/inference_sensitivity_seed{seed}_2026-11-07_v1/full_retrieval/acceptance.json'
    prior_record = record(prior_path, source['manifest_sha256'])
    acceptance_record = record(acceptance_path, source['acceptance_sha256'])
    old, accepted = (json.loads(p.read_text()) for p in (prior_path, acceptance_path))
    if (type(old.get('seed')) is not int or old['seed'] != seed
            or old.get('protocol') != 'history-last-user-v1'
            or old.get('expected_test_samples') != 2061101
            or accepted.get('status') != 'passed' or accepted.get('phase') != 'full'
            or accepted.get('group') != 'retrieval' or type(accepted.get('seed')) is not int
            or accepted['seed'] != seed or accepted.get('num_examples') != 2061101
            or accepted.get('unique_users') != 2061101
            or accepted.get('manifest_sha256') != source['manifest_sha256']):
        raise ValueError('Prior scan seed/protocol/acceptance mismatch')
    run_id = source['run_id']
    run_dir = Path(old['run_dir'])
    analysis = ASSET_ROOT / 'logs/analysis/paired_evaluation'
    expected_paths = {
        'checkpoint': run_dir / 'checkpoints' / f"checkpoint_000_{source['best_step']}.ckpt",
        'reference_test': analysis / f'listwise_seed{seed}_{run_id}_test_full_v1.h5',
        'frozen_beta': analysis / f'listwise_seed{seed}_{run_id}_fusion_val.json',
    }
    if (run_dir.name != f'click-listwise-seed{seed}-{run_id}'
            or not run_dir.is_relative_to(ASSET_ROOT / 'logs/train/runs')
            or old['data_root'] != str(ASSET_ROOT / 'data/taobao_history_last_user_v1')
            or old['training_preflight'] != str(ASSET_ROOT / f'logs/training_preflight/listwise_seed{seed}_{run_id}.json')):
        raise ValueError('Prior run/data/preflight is not the approved training seed')
    for field, expected_path in expected_paths.items():
        if old[field]['path'] != str(expected_path) or accepted.get(field) != old[field]:
            raise ValueError(f'Prior scan identity mismatch: {field}')
    for entry in old['input_metadata'] + old['execution_sources']:
        record(entry['path'], entry['sha256'])
    selection = json.loads(Path(old['frozen_beta']['path']).read_text())
    if (selection.get('input_split') != 'val' or selection.get('selection_scope') != 'validation_only'
            or selection.get('input_checkpoint_sha256') != old['checkpoint']['sha256']
            or selection.get('selected', {}).get('beam_weight') != 1.
            or selection.get('selected', {}).get('rerank_weight') != .5):
        raise ValueError('Frozen validation selection/checkpoint mismatch')
    record(old['frozen_beta']['path'], old['frozen_beta']['sha256'])
    # Inference dependencies inherited from the accepted sweep must remain
    # byte-identical in this worktree; source paths are rebound, not copied.
    old_root = Path(old['execution_root'])
    for entry in old['execution_sources']:
        relative = Path(entry['path']).relative_to(old_root)
        record(ROOT / relative, entry['sha256'])
    return old, list(old['input_metadata']) + [prior_record, acceptance_record]


def _check_multiseed_bindings(m):
    old, metadata = _seed_inputs(m['seed'])
    if any(m.get(k) != old[k] for k in INHERITED_FIELDS) or m.get('input_metadata') != metadata:
        raise ValueError('Content inputs do not match the accepted training seed')
    if m.get('selection') != SELECTION:
        raise ValueError('Content selection rule changed')
    expected_paths = [str((ROOT / p).resolve()) for p in MULTISEED_SOURCES]
    sources = m.get('execution_sources', [])
    if (len(sources) != len(expected_paths)
            or [entry['path'] for entry in sources] != expected_paths):
        raise ValueError('Content execution source list changed')


def prepare(path, seed=42):
    scope = scope_fields(seed)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit sources before freezing the full-test comparison')
    if path.exists():
        raise FileExistsError(f'Refusing existing manifest: {path}')
    if seed in PRIOR_SCANS:
        old, metadata = _seed_inputs(seed)
        m = {k: old[k] for k in INHERITED_FIELDS}
        m.update(scope)
        m.update({
            'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'sampling': 'All 2061101 original test records in frozen order; smoke uses only the first 65',
            'selection': SELECTION,
            'layout_control': 'All five policies retain ten physical context slots and the native candidate position; actual positions and masks observed',
            'candidate_policy': 'One fresh natural beam20 per batch, shared candidates and generation scores; recompute sim10 center, never mix old predictions',
            'hardware_requirement': 'One RTX4090 and four CPUs per job; seed42/43/44 together at most three GPUs and twelve CPUs; one worker and library threads one',
            'scope': f'Five fixed-slot full-test policies, training seed{seed}; no retraining, beta tuning, checkpoint selection or model-source edits',
            'statistics_interpretation': 'Primary: sim10 minus recent10 fused NDCG@10. Other contrasts/metrics exploratory; user-paired intervals do not measure training-seed uncertainty or establish practical importance',
            'input_metadata': metadata,
            'execution_sources': [record(ROOT / p) for p in MULTISEED_SOURCES],
        })
        _check_multiseed_bindings(m)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_new(path, m)
        print(json.dumps({'manifest': str(path), 'sha256': sha(path), 'source_git_commit': m['source_git_commit']}))
        return
    old_record = record(PRIOR_MANIFEST, PRIOR_MANIFEST_SHA)
    acceptance_record = record(PRIOR_ACCEPTANCE, PRIOR_ACCEPTANCE_SHA)
    old = json.loads(PRIOR_MANIFEST.read_text())
    prior = json.loads(PRIOR_ACCEPTANCE.read_text())
    if (prior['status'] != 'passed' or prior['phase'] != 'diagnostic' or prior['seed'] != 42
            or prior['num_examples'] != 16384 or prior['manifest_sha256'] != PRIOR_MANIFEST_SHA):
        raise ValueError('Prior diagnostic identity mismatch')
    for entry in old['input_metadata'] + old['execution_sources']:
        record(entry['path'], entry['sha256'])
    m = {k: old[k] for k in ('run_dir', 'data_root', 'training_preflight', 'checkpoint',
                             'reference_test', 'frozen_beta', 'selection')}
    m.update(scope)
    m.update({
        'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sampling': 'All 2061101 original test records in frozen order; smoke uses only the first 65',
        'layout_control': 'All five policies retain ten physical context slots and the native candidate position; actual positions and masks observed',
        'candidate_policy': 'One fresh natural beam20 per batch, shared candidates and generation scores; recompute sim10 center, never mix old predictions',
        'hardware_requirement': 'One RTX4090 and four CPUs for entire job; one worker; all library threads one',
        'scope': 'Five fixed-slot full-test policies, seed42 only; no retraining, beta tuning, checkpoint selection or model-source edits',
        'statistics_interpretation': 'Primary: sim10 minus recent10 fused NDCG@10. Other contrasts/metrics exploratory; user-paired intervals do not measure training-seed uncertainty or establish practical importance',
        'input_metadata': list(old['input_metadata']) + [old_record, acceptance_record],
        'execution_sources': list(old['execution_sources']) + [record(ROOT / p) for p in NEW_SOURCES],
    })
    paths = [x['path'] for x in m['execution_sources']]
    if len(paths) != len(set(paths)):
        raise ValueError('Repeated source binding')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps({'manifest': str(path), 'sha256': sha(path), 'source_git_commit': m['source_git_commit']}))


def load_manifest(path):
    if sha(path) != os.environ.get('CONTENT_MANIFEST_SHA256'):
        raise ValueError('Content manifest hash mismatch')
    m = json.loads(Path(path).read_text())
    if any(m.get(k) != v for k, v in scope_fields(m.get('seed')).items()):
        raise ValueError('Content comparison exceeds authorized scope')
    for key in ('seed', 'C', 'physical_slots', 'batch_size', 'num_workers', 'expected_test_samples',
                'smoke_samples', 'full_samples', 'random_seed'):
        if type(m[key]) is not int:
            raise ValueError(f'Expected integer scope field: {key}')
    if m['source_git_commit'] != os.environ.get('SOURCE_GIT_COMMIT'):
        raise ValueError('Execution commit binding mismatch')
    for entry in m['execution_sources'] + m['input_metadata']:
        record(entry['path'], entry['sha256'])
    if m['seed'] in PRIOR_SCANS:
        _check_multiseed_bindings(m)
    m['_manifest_sha256'] = sha(path)
    return m


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    args = p.parse_args()
    prepare(args.manifest.resolve(), args.seed)
