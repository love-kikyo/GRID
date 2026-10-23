"""Source-bound, train-only efficiency probe; no formal training authority."""
import json
import os
from pathlib import Path

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new

CONFIG = ROOT / 'configs/baselines/liger_efficiency_probe_v4.json'
PRIOR = ROOT / 'logs/training_launch/liger_prefix_gate_2026-10-22_v3/manifest.json'
PRIOR_SHA = '88edc6ea14e1453054964be65f99a45ce669b44b8f79caa65522ed0870d1afe4'
RECEIPT = PRIOR.parent / 'prefix_acceptance.json'
RECEIPT_SHA = '9dcdba9ca30f0d6882a83d2d01a5ebc47631f366eba592b3d00e178a6f93dd1a'
SOURCES = ['scripts/liger_efficiency_manifest.py', 'scripts/probe_liger_efficiency.py',
    'scripts/accept_liger_efficiency.py', 'src/baselines/liger_feature_cache.py',
    'src/baselines/liger_efficiency_timing.py', 'tests/test_liger_feature_cache.py',
    'tests/test_liger_efficiency_timing.py', 'tests/test_liger_efficiency_probe.py',
    'tests/test_liger_efficiency_manifest.py', 'run_liger_efficiency_cpu4.sh']


def check_config(c):
    assert c['protocol'] == 'liger-fixed-sid-scl-prefix-standardized-efficiency-v4'
    assert c['phase'] == 'fixed_seen_feature_cache_and_batch_capacity_only' and c['seed'] == 42
    assert not any(c[k] for k in ('formal_training_ready', 'automatic_successor',
        'standardization_fit', 'cache_trainable_projection', 'validation_test_access'))
    assert c['optimizer'] == dict(name='AdamW', learning_rate=.0003, weight_decay=.035, clip_norm=1.)
    t = c['training']
    assert (t['batch_size'], t['shuffle_buffer'], t['prefix_shuffle_buffer'], t['history_items']) == (1024, 8192, 8192, 7)
    assert (t['max_attempts'], t['max_optimizer_steps'], t['warmup_steps'], t['chunk_size']) == (4, 19260, 963, 65536)
    assert c['cases'] == [dict(name=name, backend=backend, batch_size=size) for name, backend, size in (
        ('lazy1024', 'lazy', 1024), ('cached1024', 'cached_seen', 1024),
        ('cached2048', 'cached_seen', 2048), ('cached4096', 'cached_seen', 4096))]
    assert (c['fixture_prefixes'], c['warmup_attempts'], c['measured_attempts'], c['deadline_seconds']) == (16384, 1, 3, 2400)
    assert c['validation'] == dict(batch_size=32, access=False)
    assert c['resources'] == dict(gpus=1, gpu_model='RTX 4090', cpu_ceiling=4, library_threads=1, memory_gb=24)


def prepare(path):
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    assert sha(PRIOR) == PRIOR_SHA and sha(RECEIPT) == RECEIPT_SHA
    old, receipt = [json.loads(p.read_text()) for p in (PRIOR, RECEIPT)]
    assert receipt['status'] == 'passed' and receipt['fresh_process_exact_resume']
    assert receipt['manifest_sha256'] == PRIOR_SHA and receipt['source_git_commit'] == old['source_git_commit']
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(old[group])
    for artifact in receipt['reports']:
        assert sha(artifact['path']) == artifact['sha256']
    c = json.loads(CONFIG.read_text())
    check_config(c)
    m = {key: old[key] for key in ('upstream_root', 'upstream_commit', 'upstream_files',
        'execution_sources', 'input_metadata', 'assets', 'catalog', 'index_artifacts',
        'asset_rows', 'data_root', 'train_files', 'seed', 'standardization', 'counts')}
    m['execution_sources'].update({str(ROOT / p): sha(ROOT / p) for p in SOURCES})
    m['input_metadata'].update({str(p): sha(p) for p in (PRIOR, RECEIPT, CONFIG)})
    m.update(config=str(CONFIG), config_sha256=sha(CONFIG),
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        phase=c['phase'], formal_training_ready=False, automatic_successor=False,
        initialization='fresh seed42 for each case; no checkpoint load',
        validation_test_records_read=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])))


def binding(path):
    assert sha(path) == os.environ['LIGER_EFFICIENCY_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and not m['dirty_at_submission']
    assert m['phase'] == 'fixed_seen_feature_cache_and_batch_capacity_only'
    assert not any(m[k] for k in ('formal_training_ready', 'automatic_successor', 'validation_test_records_read'))
    assert not any(k in m for k in ('resume_checkpoint', 'validation_files', 'test_files'))
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[group])
    assert sha(m['config']) == m['config_sha256']
    assert sha(m['standardization']['path']) == m['standardization']['sha256']
    c = json.loads(Path(m['config']).read_text())
    check_config(c)
    from src.baselines.liger_data import split_files
    train = json.loads((Path(m['data_root']) / 'protocol_manifest_train.json').read_text())
    assert list(split_files(train, Path(m['data_root']), 'train')) == m['train_files']
    assert sha(m['catalog']['path']) == m['catalog']['sha256']
    for entry in m['assets'].values():
        stat = Path(entry['path']).stat()
        assert stat.st_size == entry['size_bytes'] and stat.st_mtime_ns == entry['mtime_ns']
    return m, c
