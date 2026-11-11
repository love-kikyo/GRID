"""Create-only sampled resource probe binding; old run evidence is immutable.

Exactly two inherited source files migrated in the accepted CPU implementation.
Verify both old and new git blobs, rather than silently replacing stale source
hashes. No old training checkpoint or held-out data split is inherited.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new

CONFIG = ROOT / 'configs/baselines/liger_sampled_probe_v1.json'
PRIOR = ROOT / 'logs/training_launch/liger_efficiency_2026-10-23_v4/manifest.json'
PRIOR_SHA = 'e7da7da9a54cfd99b1a76e4b855ebbc3098099e60b7f06700a12744ae4595648'
RECEIPT = PRIOR.parent / 'efficiency_acceptance.json'
RECEIPT_SHA = 'c74ad3e93542bf9277318ff157f704dac130871b029dca1cd226b22f07c25bc6'
PRIOR_REVISION = '570d333'
SAMPLED_REVISION = 'a6423ed'
REVISIONS = ('src/baselines/liger_prefix_training.py', 'tests/test_liger_prefix_training.py')
SOURCES = ['scripts/liger_sampled_probe_manifest.py', 'scripts/probe_liger_sampled.py',
    'scripts/accept_liger_sampled.py', 'src/baselines/liger_sampled_catalog.py',
    'tests/test_liger_sampled_catalog.py', 'tests/test_liger_sampled_probe_manifest.py',
    'tests/test_liger_sampled_probe.py', 'run_liger_sampled_probe_cpu4.sh']


def check_config(c):
    assert c['protocol'] == 'liger-fixed-sid-scl-prefix-sampled-resource-v1'
    assert c['phase'] == 'sampled_dense_resource_and_resume_probe_only' and c['seed'] == 42
    assert all(c[k] is False for k in ('formal_training_ready', 'automatic_successor',
        'standardization_fit', 'cache_trainable_projection', 'cache_full_seen_features',
        'validation_test_access'))
    assert c['optimizer'] == dict(name='AdamW', learning_rate=.0003, weight_decay=.035, clip_norm=1.)
    t = c['training']
    assert (t['batch_size'], t['shuffle_buffer'], t['prefix_shuffle_buffer'], t['history_items']) == (1024, 8192, 8192, 7)
    assert (t['max_attempts'], t['max_optimizer_steps'], t['warmup_steps'], t['chunk_size']) == (12, 19260, 963, 65536)
    assert t['schedule_scope'] == 'short resource and resume fixture only; not a formal training budget'
    assert c['cases'] == [dict(name=f'n{count}', backend='lazy', batch_size=1024,
                             num_negatives=count, seed=42) for count in (8192, 32768)]
    assert (c['fixture_prefixes'], c['warmup_attempts'], c['measured_attempts'],
            c['save_after_attempt'], c['deadline_seconds']) == (12288, 2, 10, 2, 2400)
    assert c['validation'] == dict(batch_size=32, access=False)
    assert c['resources'] == dict(gpus=1, gpu_model='RTX 4090', cpu_ceiling=4, library_threads=1, memory_gb=24)


def dense_objectives(c):
    from src.baselines.liger_sampled_catalog import sampling_binding
    return {case['name']: sampling_binding({key: case[key] for key in ('num_negatives', 'seed')})
            for case in c['cases']}


def objective_for_case(m, c, name):
    """Return a new canonical identity; train_step still takes only K and seed."""
    check_config(c)
    expected = dense_objectives(c)
    assert m['dense_objectives'] == expected
    if name not in expected:
        raise ValueError('Unknown sampled resource-probe case')
    return dict(expected[name])


def git_blob_sha(revision, relative):
    payload = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{revision}:{relative}'])
    return hashlib.sha256(payload).hexdigest()


def migrated_sources(old_sources):
    """Require exact accepted transition and reject any unrelated source drift."""
    allowed = {str(ROOT / relative): relative for relative in REVISIONS}
    assert set(allowed) <= set(old_sources)
    current, changes = {}, {}
    for path, old_sha in old_sources.items():
        actual = sha(path)
        if path in allowed:
            relative = allowed[path]
            assert old_sha == git_blob_sha(PRIOR_REVISION, relative), f'Unexpected prior source: {path}'
            accepted = git_blob_sha(SAMPLED_REVISION, relative)
            assert actual == accepted, f'Accepted sampled source changed: {path}'
            changes[path] = dict(prior_sha256=old_sha, current_sha256=actual,
                prior_revision=PRIOR_REVISION, sampled_revision=SAMPLED_REVISION)
        else:
            assert actual == old_sha, f'Unapproved inherited source change: {path}'
        current[path] = actual
    return current, changes


def check_inputs(m):
    """No data reread: validate train shard identities and frozen small assets."""
    assert not any(key in m for key in ('resume_checkpoint', 'validation_files', 'test_files'))
    from src.baselines.liger_data import split_files
    root = Path(m['data_root'])
    train = json.loads((root / 'protocol_manifest_train.json').read_text())
    assert list(split_files(train, root, 'train')) == m['train_files']
    assert sha(m['catalog']['path']) == m['catalog']['sha256']
    assert sha(m['standardization']['path']) == m['standardization']['sha256']
    for entry in m['index_artifacts'].values():
        assert sha(entry['path']) == entry['sha256']
    for entry in m['assets'].values():
        stat = Path(entry['path']).stat()
        assert stat.st_size == entry['size_bytes'] and stat.st_mtime_ns == entry['mtime_ns']


def prepare(path):
    path = Path(path).resolve()
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    assert sha(PRIOR) == PRIOR_SHA and sha(RECEIPT) == RECEIPT_SHA
    old, receipt = [json.loads(p.read_text()) for p in (PRIOR, RECEIPT)]
    assert receipt['status'] == 'passed_efficiency_probe' and receipt['exact_lazy_cached1024_equivalence']
    assert receipt['manifest_sha256'] == PRIOR_SHA and receipt['source_git_commit'] == old['source_git_commit']
    sources, migration = migrated_sources(old['execution_sources'])
    for group in ('input_metadata', 'upstream_files'):
        verify_hashes(old[group])
    for artifact in receipt['reports']:
        assert sha(artifact['path']) == artifact['sha256']
    c = json.loads(CONFIG.read_text())
    check_config(c)
    m = copy.deepcopy({key: old[key] for key in ('upstream_root', 'upstream_commit', 'upstream_files',
        'input_metadata', 'assets', 'catalog', 'index_artifacts', 'asset_rows', 'data_root',
        'train_files', 'seed', 'standardization', 'counts')})
    m['execution_sources'] = dict(sources, **{str(ROOT / p): sha(ROOT / p) for p in SOURCES})
    m['input_metadata'].update({str(p): sha(p) for p in (PRIOR, RECEIPT, CONFIG)})
    m.update(config=str(CONFIG), config_sha256=sha(CONFIG),
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        phase=c['phase'], formal_training_ready=False, automatic_successor=False,
        initialization='fresh seed42 per case; only same-case diagnostic step2 used for resume check',
        validation_test_records_read=False, dense_objectives=dense_objectives(c),
        source_migration=migration)
    check_inputs(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])))
    return m, c


def binding(path):
    path = Path(path).resolve()
    assert sha(path) == os.environ['LIGER_SAMPLED_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and m['dirty_at_submission'] is False
    assert m['phase'] == 'sampled_dense_resource_and_resume_probe_only'
    assert all(m[k] is False for k in ('formal_training_ready', 'automatic_successor', 'validation_test_records_read'))
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[group])
    assert sha(m['config']) == m['config_sha256']
    c = json.loads(Path(m['config']).read_text())
    check_config(c)
    assert m['dense_objectives'] == dense_objectives(c)
    old = json.loads(PRIOR.read_text())
    sources, migration = migrated_sources(old['execution_sources'])
    assert m['source_migration'] == migration
    assert set(m['execution_sources']) == set(sources) | {str(ROOT / p) for p in SOURCES}
    assert all(m['execution_sources'][p] == value for p, value in sources.items())
    check_inputs(m)
    return m, c
