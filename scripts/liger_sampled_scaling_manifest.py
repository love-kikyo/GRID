"""Bound a new train-only read/cache/capacity probe to immutable v1 evidence.

Inherited source changes require explicit old/new SHA256 pairs; original source
bytes are checked against the fixed accepted git revision, never rewritten in
old manifests. Cache storage contains only frozen standardized train-seen
inputs. All cases initialize freshly; no formal checkpoint is resumed or saved.
"""
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new
from scripts.liger_sampled_probe_manifest import check_inputs


CONFIG = ROOT / 'configs/baselines/liger_sampled_scaling_v3.json'
PRIOR = ROOT / 'logs/training_launch/liger_sampled_resource_2026-11-12_v1/manifest.json'
PRIOR_SHA = '49083f28baae7b246e0f63cb500dd168688f64509b82f9e044aa627606014ab6'
RECEIPT = PRIOR.parent / 'sampled_probe_acceptance.json'
RECEIPT_SHA = '73e3467223abb35456b4981c460c38d2ae7cdf71e82240c58d81bd9969700628'
PREDECESSOR_REVISION = '13ed6d3ad7d3bf8140215d043cf7fb09b499c438'
# Reviewed repairs only; an additional byte change requires a new explicit pair.
SOURCE_MIGRATIONS = {
    'src/baselines/liger_feature_cache.py': {
        'prior_sha256': 'f8f969a4c585293b1b72fd6b9e97df3a075dc5ec28312becbfed8c5a7b38fe89',
        'current_sha256': 'b255548c0510059c0f31afea46cc884542f4367ee242f1dde4dc673b84578788'},
    'src/baselines/liger_prefix_training.py': {
        'prior_sha256': '9041300f9093b6b6515b16d11364a43669fec5f8c9dcce62fadc634ebc225643',
        'current_sha256': '137e6323982e1357e4f30ff067b236d7a8d1982f926be72452b0f3466eedbe16'},
    'tests/test_liger_feature_cache.py': {
        'prior_sha256': '071c2e88cfea800d021b65a9c001e28e946360675f2506db486e3bb44b7190d7',
        'current_sha256': 'a40d77409f82bc6b1597a92bd565c15b7bed4d2609d11cae29f8bdf8b89edf10'},
    'tests/test_liger_prefix_training.py': {
        'prior_sha256': '55232fc82183985390fd407449f2fdad55420759bf6e2d7448389456b817795c',
        'current_sha256': 'ea350a762750e147572a7e6d19b29f4d2255b161fc5f09dfdf96e2c8ab8f38f7'},
    'tests/test_liger_sampled_probe_manifest.py': {
        'prior_sha256': 'bc22b8439887f25e507ee81c1abfda99e3d1dad18e1e97dc93279bdb98c8d67b',
        'current_sha256': 'bbfe52fb4b9c9e62d1d595f78601425b5dc9c250c3ce0463a7735a2526ab634d'},
}
SOURCES = ['scripts/liger_sampled_scaling_manifest.py', 'scripts/probe_liger_sampled_scaling.py',
    'scripts/accept_liger_sampled_scaling.py', 'src/baselines/liger_access_timing.py',
    'tests/test_liger_access_timing.py', 'tests/test_liger_sampled_scaling.py',
    'tests/test_liger_sampled_scaling_manifest.py', 'run_liger_sampled_scaling_cpu4.sh',
    'src/baselines/liger_row_lookup.py', 'tests/test_liger_row_lookup.py']
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
    assert c['protocol'] == 'liger-fixed-sid-scl-prefix-sampled-scaling-v3'
    assert c['phase'] == 'sampled_reading_cache_and_batch_capacity_only' and c['seed'] == 42
    assert all(c[k] is False for k in ('formal_training_ready', 'automatic_successor',
        'standardization_fit', 'cache_trainable_projection', 'validation_test_access', 'resume_probe'))
    assert c['cache_full_seen_features'] is True and c['verify_cached_features'] is True
    assert c['cache_storage'] == 'process_local_cpu_ram'
    assert c['process_environment'] == {'NUMPY_MADVISE_HUGEPAGE': '0'}
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


def require_process_environment(c):
    """Only this process's NumPy allocation policy; never change kernel THP."""
    assert c['process_environment'] == {'NUMPY_MADVISE_HUGEPAGE': '0'}
    assert os.environ.get('NUMPY_MADVISE_HUGEPAGE') == '0', 'Set NumPy memory policy before importing NumPy'


@lru_cache(maxsize=256)
def historical_source_sha(relative):
    payload = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{PREDECESSOR_REVISION}:{relative}'])
    return hashlib.sha256(payload).hexdigest()


def source_transition(old_sources):
    """Check historical bytes plus exact current old/new migration pairs."""
    allowed = {str(ROOT / relative): relative for relative in SOURCE_MIGRATIONS}
    assert set(allowed) <= set(old_sources), 'Migration file absent from accepted predecessor'
    current, migrations = {}, {}
    for path, old_sha in old_sources.items():
        relative = Path(path).relative_to(ROOT).as_posix()
        assert historical_source_sha(relative) == old_sha, f'Historical git source mismatch: {relative}'
        actual = sha(path)
        if path in allowed:
            approved = SOURCE_MIGRATIONS[relative]
            assert approved['prior_sha256'] == old_sha, f'Unexpected migration predecessor: {relative}'
            expected = approved['current_sha256']
            assert isinstance(expected, str) and len(expected) == 64, 'Migration repair hash not finalized'
            assert actual == expected, f'Unapproved repair bytes: {relative}'
            migrations[path] = dict(prior_revision=PREDECESSOR_REVISION,
                prior_sha256=old_sha, current_sha256=actual)
        else:
            assert actual == old_sha, f'Unapproved inherited source change: {relative}'
        current[path] = actual
    return current, migrations


def predecessor():
    """Check accepted v1's small evidence, without rereading its checkpoints."""
    assert sha(PRIOR) == PRIOR_SHA and sha(RECEIPT) == RECEIPT_SHA
    old, receipt = [json.loads(path.read_text()) for path in (PRIOR, RECEIPT)]
    assert receipt['status'] == 'passed_resource_probe'
    assert receipt['manifest_sha256'] == PRIOR_SHA
    assert receipt['source_git_commit'] == old['source_git_commit'] == PREDECESSOR_REVISION
    assert receipt['capacity'] == {'n8192': 'passed', 'n32768': 'passed'}
    assert all(receipt[key] is False for key in ('formal_training_ready', 'formal_budget_approved',
        'automatic_successor', 'validation_test_records_read', 'formal_checkpoint_produced'))
    source_transition(old['execution_sources'])
    for group in ('input_metadata', 'upstream_files'):
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
    """Permit only reviewed source repairs; preserve every data/asset identity."""
    assert all(m[key] == old[key] for key in INHERITED_KEYS)
    added = {str(ROOT / relative) for relative in SOURCES}
    assert not (added & set(old['execution_sources']))
    assert set(m['execution_sources']) == set(old['execution_sources']) | added
    current, migrations = source_transition(old['execution_sources'])
    assert all(m['execution_sources'][path] == digest for path, digest in current.items())
    assert m['source_migration'] == migrations
    expected_inputs = dict(old['input_metadata'], **evidence, **{str(CONFIG): sha(CONFIG)})
    assert m['input_metadata'] == expected_inputs
    assert m['dense_objectives'] == dense_objectives(json.loads(CONFIG.read_text()))
    assert m['process_environment'] == {'NUMPY_MADVISE_HUGEPAGE': '0'}
    assert not any(key in m for key in ('resume_checkpoint', 'validation_files', 'test_files'))


def prepare(path):
    path = Path(path).resolve()
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    c = json.loads(CONFIG.read_text())
    check_config(c)
    require_process_environment(c)
    old, evidence = predecessor()
    current, migrations = source_transition(old['execution_sources'])
    m = copy.deepcopy({key: old[key] for key in INHERITED_KEYS})
    m['execution_sources'] = dict(current, **{str(ROOT / p): sha(ROOT / p) for p in SOURCES})
    m['input_metadata'] = dict(old['input_metadata'], **evidence, **{str(CONFIG): sha(CONFIG)})
    m.update(config=str(CONFIG), config_sha256=sha(CONFIG),
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        phase=c['phase'], formal_training_ready=False, automatic_successor=False,
        initialization='fresh seed42 per case; no checkpoint load or save',
        validation_test_records_read=False, dense_objectives=dense_objectives(c),
        source_migration=migrations, process_environment=dict(c['process_environment']))
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
    require_process_environment(c)
    old, evidence = predecessor()
    verify_inheritance(m, old, evidence)
    check_inputs(m)
    return m, c
