"""New one-pass sampled training authority, separate from all diagnostic weights.

The resource probe remains immutable. Its accepted source/input identities are
inherited exactly; only new execution files and the formal configuration are
added. An exact fresh-process integration receipt gates first-pass training.
"""
import argparse
import copy
import json
import os
from pathlib import Path

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new
from src.baselines.liger_data import split_files
from src.baselines.liger_sampled_catalog import sampling_binding


CONFIG = ROOT / 'configs/baselines/liger_sampled_train_seed42_v1.json'
PRIOR = ROOT / 'logs/training_launch/liger_sampled_scaling_2026-11-12_v3/manifest.json'
PRIOR_SHA = 'e0e8fb4737c80c8ae3ba9a15ca9aa638e6a91358970ee62454040da7521bbfa1'
RECEIPT = PRIOR.parent / 'scaling_acceptance.json'
RECEIPT_SHA = 'f706c8189a2dc6548ca51dedbecb458f22c46b9836c20b8c5335cb86e2a05bb7'
GATE_PHASE = 'sampled_training_integration_gate'
FORMAL_PHASE = 'sampled_first_pass_formal'
SOURCES = [
    'scripts/liger_sampled_training_manifest.py', 'scripts/train_liger_sampled_cpu4.py',
    'scripts/probe_liger_sampled_training.py', 'tests/test_liger_sampled_training_manifest.py',
    'tests/test_liger_sampled_training_entrypoints.py', 'tests/test_liger_sampled_training_gate.py',
    'run_liger_sampled_train_cpu4.sh', 'run_liger_sampled_training_gate_cpu4.sh',
]
INHERITED_KEYS = ('upstream_root', 'upstream_commit', 'upstream_files', 'assets', 'catalog',
    'index_artifacts', 'asset_rows', 'data_root', 'train_files', 'seed', 'standardization', 'counts')


def check_config(c):
    assert c['protocol'] == 'liger-fixed-sid-scl-prefix-sampled-one-pass-v1'
    assert c['seed'] == 42 and c['automatic_successor'] is False
    assert c['candidate_config'] == 'configs/baselines/liger_history_last_v1.json'
    assert c['initialization'] == 'fresh seed42 random weights; never load old formal or diagnostic weights'
    assert c['optimizer'] == dict(name='AdamW', learning_rate=.0003, weight_decay=.035, clip_norm=1.)
    assert c['dense_sampling'] == dict(num_negatives=32768, seed=42)
    t = c['training']
    for key in ('batch_size', 'shuffle_buffer', 'prefix_shuffle_buffer', 'history_items',
                'parents_per_epoch', 'prefixes_per_epoch', 'batches_per_epoch', 'final_batch_size',
                'max_epochs', 'stop_after_epoch', 'max_attempts', 'max_optimizer_steps',
                'warmup_steps', 'checkpoint_every_batches', 'chunk_size'):
        assert type(t[key]) is int
    assert (t['batch_size'], t['shuffle_buffer'], t['prefix_shuffle_buffer'], t['history_items']) == (2048, 8192, 8192, 7)
    assert t['supervision'] == 'all_nonempty_history_prefixes_in_last8'
    assert t['loss_weighting'] == 'equal_prefix_examples_both_sid_and_dense' and t['drop_last'] is False
    assert (t['parents_per_epoch'], t['prefixes_per_epoch'], t['batches_per_epoch'], t['final_batch_size']) == (6573674, 45996113, 22460, 81)
    assert t['batches_per_epoch'] == (t['prefixes_per_epoch'] + t['batch_size'] - 1) // t['batch_size']
    assert t['final_batch_size'] == t['prefixes_per_epoch'] - (t['batches_per_epoch'] - 1) * t['batch_size']
    assert (t['max_epochs'], t['stop_after_epoch'], t['max_attempts'], t['max_optimizer_steps'], t['warmup_steps']) == (1, 1, 22460, 22460, 1123)
    assert (t['checkpoint_every_batches'], t['chunk_size']) == (1000, 65536)
    assert t['amp'] == 'FP16 input/SID path with GradScaler; dense loss FP32, TF32 disabled'
    assert t['schedule'] == 'next successful update LR: 5 percent linear warmup then cosine over this one-pass budget; AMP skips do not advance'
    assert t['continuation'] == 'not authorized; any later budget or schedule requires a new explicit phase'
    assert c['standardization'] == dict(fit_scope='frozen_train_seen', reuse_verified_statistics=True)
    assert c['feature_cache'] == dict(backend='cached_seen', storage='process_local_cpu_ram', verify_all_rows=True, cache_trainable_projection=False)
    assert c['validation'] == dict(split='evaluation', records=347416, batch_size=32,
        frequency='once after the complete training pass', candidate_mode='full_cold',
        selection=['hybrid Recall@5 descending', 'hybrid NDCG@5 descending', 'epoch ascending'],
        test_access=False, shard_rows=3200)
    assert c['smoke'] == dict(train_steps=3, save_after_step=1, validate_after_step=2,
                            validation_records=65, deadline_seconds=2400)
    assert c['process_environment'] == {'NUMPY_MADVISE_HUGEPAGE': '0'}
    assert c['resources'] == dict(gpus=1, gpu_model='RTX 4090', cpu_ceiling=4, library_threads=1,
        memory_gb=24, formal_time_limit_hours=12, gate_time_limit_minutes=45)


def require_environment():
    assert os.environ.get('NUMPY_MADVISE_HUGEPAGE') == '0', 'Set NumPy allocation policy before Python starts'


def read_bound(entry):
    assert sha(entry['path']) == entry['sha256']
    return json.loads(Path(entry['path']).read_text())


def predecessor():
    assert sha(PRIOR) == PRIOR_SHA and sha(RECEIPT) == RECEIPT_SHA
    old, receipt = json.loads(PRIOR.read_text()), json.loads(RECEIPT.read_text())
    assert receipt['status'] == 'passed_resource_probe' and receipt['job_id'] == '159877'
    assert receipt['manifest_sha256'] == PRIOR_SHA and receipt['source_git_commit'] == old['source_git_commit']
    assert receipt['exact_cache_equivalence_both_negative_counts'] is True
    assert len(receipt['capacity']) == 8 and all(value == 'passed' for value in receipt['capacity'].values())
    assert receipt['capacity']['cached32k2048'] == 'passed'
    assert all(receipt[key] is False for key in ('formal_training_ready', 'formal_budget_approved',
        'automatic_successor', 'validation_test_records_read', 'formal_checkpoint_produced'))
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(old[group])
    evidence = {str(PRIOR): PRIOR_SHA, str(RECEIPT): RECEIPT_SHA}
    for entry in receipt['reports']:
        assert sha(entry['path']) == entry['sha256']
        evidence[entry['path']] = entry['sha256']
    return old, evidence


def common_inputs(old, evidence):
    root = Path(old['data_root'])
    data_manifest = root / 'protocol_manifest_train.json'
    data = json.loads(data_manifest.read_text())
    validation_files = list(split_files(data, root, 'evaluation'))
    assert list(split_files(data, root, 'train')) == old['train_files']
    metadata = dict(old['input_metadata'], **evidence,
                    **{str(CONFIG): sha(CONFIG), str(data_manifest): sha(data_manifest)})
    return metadata, validation_files


def expected_identities(evidence):
    """Read fixed-input/cache digests, never diagnostic model weight tensors."""
    caches, initial = set(), set()
    for name in ('cached8k1024', 'cached32k1024', 'cached8k2048',
                 'cached32k2048', 'cached8k4096', 'cached32k4096'):
        path = PRIOR.parent / (name + '.json')
        assert evidence[str(path)] == sha(path)
        report = json.loads(path.read_text())
        assert report['status'] == 'passed'
        assert report['cache_verification']['status'] == 'passed'
        assert report['cache_verification']['feature_values_bitwise_equal'] is True
        assert report['cache']['cache_sha256'] == report['cache_verification']['cache_sha256']
        caches.add(report['cache']['cache_sha256'])
        initial.add(report['initial_state_sha256'])
    assert len(caches) == len(initial) == 1
    return dict(expected_feature_cache_sha256=caches.pop(), expected_initial_state_sha256=initial.pop())


def execution_counts(c):
    """The predecessor's inherited counts retain its batch1024 provenance."""
    t = c['training']
    return dict(parents_per_pass=t['parents_per_epoch'], prefix_examples_per_pass=t['prefixes_per_epoch'],
        batch_size=t['batch_size'], prefix_batches_per_pass=t['batches_per_epoch'],
        final_batch_size=t['final_batch_size'], validation_records=c['validation']['records'])


def checkpoint_binding(m):
    """Stable over launch IDs, but diagnostic and formal weights never mix."""
    from scripts.liger_training_manifest import checkpoint_binding as base_binding
    value = base_binding(m)
    value.update(purpose='sampled_formal_one_pass' if m['formal_training_ready'] else 'sampled_integration_gate',
        dense_objective=m['dense_objective'], process_environment=m['process_environment'],
        standardization_sha256=m['standardization']['sha256'],
        supervision='all_nonempty_history_prefixes_in_last8_equal_weight',
        feature_backend='cached_seen_fixed_input_features_not_projection')
    return value


def verify_common(m, old, evidence):
    assert all(m[key] == old[key] for key in INHERITED_KEYS)
    added = {str(ROOT / p) for p in SOURCES}
    assert not added & set(old['execution_sources'])
    assert set(m['execution_sources']) == set(old['execution_sources']) | added
    assert all(m['execution_sources'][path] == digest for path, digest in old['execution_sources'].items())
    metadata, validation = common_inputs(old, evidence)
    assert m['input_metadata'] == metadata and m['validation_files'] == validation
    c = json.loads(CONFIG.read_text())
    check_config(c)
    assert m['config'] == str(CONFIG) and m['config_sha256'] == sha(CONFIG)
    assert m['dense_objective'] == sampling_binding(c['dense_sampling'])
    assert m['process_environment'] == c['process_environment']
    assert m['execution_counts'] == execution_counts(c)
    assert all(m[key] == value for key, value in expected_identities(evidence).items())
    assert m['candidate_universe_finalized'] is True
    assert m['automatic_successor'] is m['test_records_read'] is False
    assert m['initialization'] == c['initialization']
    assert m['stage'] == dict(start_epoch=0, stop_after_epoch=1)
    assert m['dirty_at_submission'] is False
    assert not any(key in m for key in ('resume_checkpoint', 'test_files', 'source_migration'))
    assert Path(m['run_directory']).parent == ROOT / 'logs/train/runs'
    if m['phase'] == GATE_PHASE:
        assert m['formal_training_ready'] is False and 'smoke_acceptance' not in m
    else:
        assert m['phase'] == FORMAL_PHASE and m['formal_training_ready'] is True
        assert 'smoke_acceptance' in m


def verify_gate(entry, m):
    """Require the new exact integration test, not an older capacity receipt."""
    receipt = read_bound(entry)
    assert receipt['status'] == 'passed' and receipt['phase'] == GATE_PHASE
    assert receipt['fresh_process_exact_resume'] is True
    assert receipt['validation_preserved_full_learning_state'] is True
    assert receipt['initial_state_sha256'] == m['expected_initial_state_sha256']
    assert receipt['full_cold_validation_records'] == 65
    assert receipt['config_sha256'] == m['config_sha256']
    assert receipt['execution_sources'] == m['execution_sources']
    assert receipt['dense_objective'] == m['dense_objective']
    assert all(receipt[key] is False for key in ('formal_checkpoint_produced', 'test_records_read', 'automatic_successor'))
    gate = read_bound(receipt['gate_manifest'])
    assert gate['phase'] == GATE_PHASE and gate['formal_training_ready'] is False
    assert gate['automatic_successor'] is gate['test_records_read'] is gate['dirty_at_submission'] is False
    assert gate['initialization'] == m['initialization'] and gate['stage'] == m['stage']
    assert not any(key in gate for key in ('resume_checkpoint', 'test_files', 'smoke_acceptance'))
    assert receipt['manifest_sha256'] == receipt['gate_manifest']['sha256']
    assert receipt['source_git_commit'] == gate['source_git_commit']
    for key in (*INHERITED_KEYS, 'config', 'config_sha256', 'execution_sources', 'input_metadata',
                'validation_files', 'dense_objective', 'process_environment',
                'expected_feature_cache_sha256', 'expected_initial_state_sha256', 'execution_counts'):
        assert gate[key] == m[key], f'Integration gate identity differs: {key}'
    assert isinstance(receipt['job_id'], str) and receipt['job_id']
    assert receipt['reports']
    for artifact in receipt['reports']:
        assert sha(artifact['path']) == artifact['sha256']
    return receipt


def check_assets(m):
    assert sha(m['catalog']['path']) == m['catalog']['sha256']
    assert sha(m['standardization']['path']) == m['standardization']['sha256']
    for entry in m['index_artifacts'].values():
        assert sha(entry['path']) == entry['sha256']
    for entry in m['assets'].values():
        stat = Path(entry['path']).stat()
        assert stat.st_size == entry['size_bytes'] and stat.st_mtime_ns == entry['mtime_ns']


def prepare(path, *, gate_receipt=None):
    path = Path(path).resolve()
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    require_environment()
    old, evidence = predecessor()
    c = json.loads(CONFIG.read_text())
    check_config(c)
    metadata, validation = common_inputs(old, evidence)
    m = copy.deepcopy({key: old[key] for key in INHERITED_KEYS})
    m.update(execution_sources=dict(old['execution_sources'], **{str(ROOT / p): sha(ROOT / p) for p in SOURCES}),
        input_metadata=metadata, validation_files=validation, config=str(CONFIG), config_sha256=sha(CONFIG),
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        phase=FORMAL_PHASE if gate_receipt else GATE_PHASE, formal_training_ready=bool(gate_receipt),
        candidate_universe_finalized=True, automatic_successor=False, test_records_read=False,
        initialization=c['initialization'], dense_objective=sampling_binding(c['dense_sampling']),
        execution_counts=execution_counts(c),
        **expected_identities(evidence),
        process_environment=dict(c['process_environment']), stage=dict(start_epoch=0, stop_after_epoch=1),
        run_directory=str(ROOT / 'logs/train/runs' / path.parent.name))
    if gate_receipt:
        target = Path(gate_receipt).resolve()
        m['smoke_acceptance'] = dict(path=str(target), sha256=sha(target))
        verify_gate(m['smoke_acceptance'], m)
    verify_common(m, old, evidence)
    check_assets(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'], phase=m['phase'])))
    return m


def binding(path):
    path = Path(path).resolve()
    assert sha(path) == os.environ['LIGER_SAMPLED_TRAIN_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT']
    require_environment()
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[group])
    old, evidence = predecessor()
    verify_common(m, old, evidence)
    check_assets(m)
    if m['formal_training_ready']:
        verify_gate(m['smoke_acceptance'], m)
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare-gate', 'prepare-formal'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--gate-receipt', type=Path)
    args = parser.parse_args()
    if (args.mode == 'prepare-formal') != (args.gate_receipt is not None):
        parser.error('Only prepare-formal requires --gate-receipt')
    prepare(args.manifest, gate_receipt=args.gate_receipt)
