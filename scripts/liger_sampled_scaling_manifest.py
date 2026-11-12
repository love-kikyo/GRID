"""Bound a new train-only read/cache/capacity probe to immutable v1 evidence.

No inherited executable source may change. Cache storage contains only frozen
standardized train-seen inputs, not learned projections. All cases initialize
freshly; this probe neither resumes nor produces formal training checkpoints.
"""
import copy
import json
import os
from pathlib import Path

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new
from scripts.liger_sampled_probe_manifest import check_inputs


CONFIG = ROOT / 'configs/baselines/liger_sampled_scaling_v2.json'
PRIOR = ROOT / 'logs/training_launch/liger_sampled_resource_2026-11-12_v1/manifest.json'
PRIOR_SHA = '49083f28baae7b246e0f63cb500dd168688f64509b82f9e044aa627606014ab6'
RECEIPT = PRIOR.parent / 'sampled_probe_acceptance.json'
RECEIPT_SHA = '73e3467223abb35456b4981c460c38d2ae7cdf71e82240c58d81bd9969700628'
SOURCES = ['scripts/liger_sampled_scaling_manifest.py', 'scripts/probe_liger_sampled_scaling.py',
    'scripts/accept_liger_sampled_scaling.py', 'src/baselines/liger_access_timing.py',
    'tests/test_liger_access_timing.py', 'tests/test_liger_sampled_scaling.py',
    'tests/test_liger_sampled_scaling_manifest.py', 'run_liger_sampled_scaling_cpu4.sh']
INHERITED_KEYS = ('upstream_root', 'upstream_commit', 'upstream_files', 'assets', 'catalog',
    'index_artifacts', 'asset_rows', 'data_root', 'train_files', 'seed', 'standardization', 'counts')


def expected_cases():
    return [dict(name=name, backend=backend, batch_size=size, num_negatives=count, seed=42)
        for name, backend, size, count in (
            ('lazy8k1024', 'lazy', 1024, 8192),
            ('cached8k1024', 'cached_seen', 1024, 8192),
            ('cached32k1024', 'cached_seen', 1024, 32768),
            ('lazy32k1024', 'lazy', 1024, 32768),
            ('cached8k2048', 'cached_seen', 2048, 8192),
            ('cached32k2048', 'cached_seen', 2048, 32768),
            ('cached8k4096', 'cached_seen', 4096, 8192),
            ('cached32k4096', 'cached_seen', 4096, 32768))]


def check_config(c):
    assert c['protocol'] == 'liger-fixed-sid-scl-prefix-sampled-scaling-v2'
    assert c['phase'] == 'sampled_reading_cache_and_batch_capacity_only' and c['seed'] == 42
    assert all(c[k] is False for k in ('formal_training_ready', 'automatic_successor',
        'standardization_fit', 'cache_trainable_projection', 'validation_test_access', 'resume_probe'))
    assert c['cache_full_seen_features'] is True and c['verify_cached_features'] is True
    assert c['cache_storage'] == 'process_local_cpu_ram'
    assert c['optimizer'] == dict(name='AdamW', learning_rate=.0003, weight_decay=.035, clip_norm=1.)
    t = c['training']
    assert (t['batch_size'], t['shuffle_buffer'], t['prefix_shuffle_buffer'], t['history_items']) == (1024, 8192, 8192, 7)
    assert (t['max_attempts'], t['max_optimizer_steps'], t['warmup_steps'], t['chunk_size']) == (20, 19260, 963, 65536)
    assert t['schedule_scope'] == 'short reading and capacity fixture only; not a formal training budget'
    assert c['cases'] == expected_cases()
    assert (c['fixture_prefixes'], c['warmup_attempts'], c['measured_attempts'], c['deadline_seconds']) == (81920, 5, 15, 2400)
    assert c['validation'] == dict(batch_size=32, access=False)
    assert c['resources'] == dict(gpus=1, gpu_model='RTX 4090', cpu_ceiling=4, library_threads=1, memory_gb=24)


def dense_objectives(c):
    from src.baselines.liger_sampled_catalog import sampling_binding
    return {case['name']: sampling_binding({key: case[key] for key in ('num_negatives', 'seed')})
            for case in c['cases']}


def objective_for_case(m, c, name):
    check_config(c)
    expected = dense_objectives(c)
    assert m['dense_objectives'] == expected
    if name not in expected:
        raise ValueError('Unknown sampled scaling case')
    return dict(expected[name])


def predecessor():
    """Check accepted v1's small evidence, without rereading its checkpoints."""
    assert sha(PRIOR) == PRIOR_SHA and sha(RECEIPT) == RECEIPT_SHA
    old, receipt = [json.loads(path.read_text()) for path in (PRIOR, RECEIPT)]
    assert receipt['status'] == 'passed_resource_probe'
    assert receipt['manifest_sha256'] == PRIOR_SHA
    assert receipt['source_git_commit'] == old['source_git_commit']
    assert receipt['capacity'] == {'n8192': 'passed', 'n32768': 'passed'}
    assert all(receipt[key] is False for key in ('formal_training_ready', 'formal_budget_approved',
        'automatic_successor', 'validation_test_records_read', 'formal_checkpoint_produced'))
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(old[group])
    evidence = {str(PRIOR): PRIOR_SHA, str(RECEIPT): RECEIPT_SHA}
    for artifact in receipt['reports']:
        assert sha(artifact['path']) == artifact['sha256']
        evidence[artifact['path']] = artifact['sha256']
        if Path(artifact['path']).name in ('n8192_acceptance.json', 'n32768_acceptance.json'):
            accepted = json.loads(Path(artifact['path']).read_text())
            assert accepted['status'] == 'passed' and accepted['fresh_process_exact_resume'] is True
            assert accepted['manifest_sha256'] == PRIOR_SHA and accepted['job_id'] == receipt['job_id']
            for report in accepted['reports']:
                if Path(report['path']).suffix == '.json':
                    assert sha(report['path']) == report['sha256']
                    evidence[report['path']] = report['sha256']
    return old, evidence


def verify_inheritance(m, old, evidence):
    """No migrations: every old source/input/asset identity must be retained."""
    assert all(m[key] == old[key] for key in INHERITED_KEYS)
    added = {str(ROOT / relative) for relative in SOURCES}
    assert not (added & set(old['execution_sources']))
    assert set(m['execution_sources']) == set(old['execution_sources']) | added
    assert all(m['execution_sources'][path] == digest for path, digest in old['execution_sources'].items())
    expected_inputs = dict(old['input_metadata'], **evidence, **{str(CONFIG): sha(CONFIG)})
    assert m['input_metadata'] == expected_inputs
    assert m['dense_objectives'] == dense_objectives(json.loads(CONFIG.read_text()))
    assert not any(key in m for key in ('resume_checkpoint', 'validation_files', 'test_files', 'source_migration'))


def prepare(path):
    path = Path(path).resolve()
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    old, evidence = predecessor()
    c = json.loads(CONFIG.read_text())
    check_config(c)
    m = copy.deepcopy({key: old[key] for key in INHERITED_KEYS})
    m['execution_sources'] = dict(old['execution_sources'], **{str(ROOT / p): sha(ROOT / p) for p in SOURCES})
    m['input_metadata'] = dict(old['input_metadata'], **evidence, **{str(CONFIG): sha(CONFIG)})
    m.update(config=str(CONFIG), config_sha256=sha(CONFIG),
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        phase=c['phase'], formal_training_ready=False, automatic_successor=False,
        initialization='fresh seed42 per case; no checkpoint load or save',
        validation_test_records_read=False, dense_objectives=dense_objectives(c))
    verify_inheritance(m, old, evidence)
    check_inputs(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])))
    return m, c


def binding(path):
    path = Path(path).resolve()
    assert sha(path) == os.environ['LIGER_SCALING_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and m['dirty_at_submission'] is False
    assert m['phase'] == 'sampled_reading_cache_and_batch_capacity_only'
    assert all(m[k] is False for k in ('formal_training_ready', 'automatic_successor', 'validation_test_records_read'))
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[group])
    assert m['config'] == str(CONFIG) and sha(CONFIG) == m['config_sha256']
    c = json.loads(CONFIG.read_text())
    check_config(c)
    old, evidence = predecessor()
    verify_inheritance(m, old, evidence)
    check_inputs(m)
    return m, c
