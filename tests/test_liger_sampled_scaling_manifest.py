import copy
import json
from pathlib import Path

import pytest

from scripts import liger_sampled_scaling_manifest as manifest


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads(manifest.CONFIG.read_text())


def test_contract_separates_equivalence_and_capacity_without_training_authority():
    c = config()
    manifest.check_config(c)
    assert c['training']['max_attempts'] == c['warmup_attempts'] + c['measured_attempts'] == 20
    assert c['fixture_prefixes'] == 20 * max(case['batch_size'] for case in c['cases'])
    assert [case['batch_size'] for case in c['cases'][:4]] == [1024] * 4
    assert c['cases'][0]['backend'] == c['cases'][3]['backend'] == 'lazy'
    assert c['cases'][1]['backend'] == c['cases'][2]['backend'] == 'cached_seen'
    identities = manifest.dense_objectives(c)
    for case in c['cases']:
        identity = manifest.objective_for_case({'dense_objectives': identities}, c, case['name'])
        assert identity == identities[case['name']] and identity is not identities[case['name']]
        assert identity['num_negatives'] == case['num_negatives'] and identity['seed'] == 42


@pytest.mark.parametrize('keys,value', [
    (('formal_training_ready',), True), (('automatic_successor',), True),
    (('validation_test_access',), True), (('resume_probe',), True),
    (('standardization_fit',), True), (('cache_trainable_projection',), True),
    (('cache_full_seen_features',), False), (('verify_cached_features',), False),
    (('cache_storage',), 'persistent_shared_disk'), (('deadline_seconds',), 86400),
    (('process_environment', 'NUMPY_MADVISE_HUGEPAGE'), '1'),
    (('fixture_prefixes',), 99999), (('training', 'max_attempts'), 19260),
    (('training', 'batch_size'), 4096), (('training', 'warmup_steps'), 5),
    (('training', 'history_items'), 99), (('training', 'chunk_size'), 8192),
    (('resources', 'cpu_ceiling'), 24), (('resources', 'gpus'), 3),
    (('resources', 'memory_gb'), 96), (('validation', 'access'), True),
    (('warmup_attempts',), 0), (('measured_attempts',), 5),
    (('cases', 0, 'num_negatives'), 1024), (('cases', 1, 'seed'), 43),
    (('cases', 5, 'batch_size'), 8192), (('seed',), 43),
])
def test_contract_drift_is_rejected(keys, value):
    c = config()
    target = c
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(AssertionError):
        manifest.check_config(c)


@pytest.mark.parametrize('field,value', [('num_negatives', 1), ('seed', 43),
    ('temperature', .2), ('sampling_version', 'unversioned')])
def test_per_case_sampling_identity_cannot_drift(field, value):
    c = config()
    m = {'dense_objectives': manifest.dense_objectives(c)}
    m['dense_objectives']['cached8k2048'][field] = value
    with pytest.raises(AssertionError):
        manifest.objective_for_case(m, c, 'cached8k2048')


def test_unknown_case_and_reordered_capacity_plan_are_rejected():
    c = config()
    with pytest.raises(ValueError):
        manifest.objective_for_case({'dense_objectives': manifest.dense_objectives(c)}, c, 'full')
    c['cases'][0], c['cases'][-1] = c['cases'][-1], c['cases'][0]
    with pytest.raises(AssertionError):
        manifest.check_config(c)


def test_accepted_predecessor_is_immutable_and_repairs_are_explicit():
    old, evidence = manifest.predecessor()
    assert evidence[str(manifest.PRIOR)] == manifest.PRIOR_SHA
    assert evidence[str(manifest.RECEIPT)] == manifest.RECEIPT_SHA
    assert len(old['execution_sources']) == 83
    assert len(evidence) == 9
    current, migration = manifest.source_transition(old['execution_sources'])
    assert len(current) == 83 and len(migration) == 5
    assert set(migration) == {str(ROOT / relative) for relative in manifest.SOURCE_MIGRATIONS}


def inherited_fixture(monkeypatch):
    old = {key: {'value': key} for key in manifest.INHERITED_KEYS}
    old.update(execution_sources={'existing.py': 'unchanged'}, input_metadata={'input.json': 'fixed'})
    evidence = {'old-manifest.json': 'bound'}
    m = copy.deepcopy(old)
    m['execution_sources'].update({str(ROOT / relative): 'new' for relative in manifest.SOURCES})
    m['input_metadata'].update(evidence)
    m['input_metadata'][str(manifest.CONFIG)] = manifest.sha(manifest.CONFIG)
    m['dense_objectives'] = manifest.dense_objectives(config())
    m['source_migration'] = {}
    m['process_environment'] = {'NUMPY_MADVISE_HUGEPAGE': '0'}
    monkeypatch.setattr(manifest, 'source_transition', lambda sources: (dict(sources), {}))
    return m, old, evidence


def test_inherited_source_and_input_bindings_remain_exact_around_migration(monkeypatch):
    manifest.verify_inheritance(*inherited_fixture(monkeypatch))


@pytest.mark.parametrize('change', ['source_change', 'source_drop', 'source_extra',
    'metadata_change', 'metadata_drop', 'asset', 'objective', 'resume', 'test', 'migration', 'environment'])
def test_no_silent_source_asset_or_scope_migration(monkeypatch, change):
    m, old, evidence = inherited_fixture(monkeypatch)
    if change == 'source_change': m['execution_sources']['existing.py'] = 'drift'
    elif change == 'source_drop': del m['execution_sources']['existing.py']
    elif change == 'source_extra': m['execution_sources']['extra.py'] = 'unbound'
    elif change == 'metadata_change': m['input_metadata']['input.json'] = 'drift'
    elif change == 'metadata_drop': del m['input_metadata']['input.json']
    elif change == 'asset': m['assets']['value'] = 'changed'
    elif change == 'objective': m['dense_objectives']['cached8k2048']['seed'] = 99
    elif change == 'resume': m['resume_checkpoint'] = {'path': 'old.pt'}
    elif change == 'test': m['test_files'] = []
    elif change == 'migration': m['source_migration'] = {'unapproved.py': {}}
    else: m['process_environment'] = {'NUMPY_MADVISE_HUGEPAGE': '1'}
    with pytest.raises(AssertionError):
        manifest.verify_inheritance(m, old, evidence)


def test_launcher_runs_one_cpu4_gpu_with_equivalence_before_large_cases():
    source = (ROOT / 'run_liger_sampled_scaling_cpu4.sh').read_text()
    for token in ('#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=4', '#SBATCH --gres=gpu:1',
                  '#SBATCH --mem=24G', '#SBATCH --time=00:45:00', '#SBATCH --partition=rtx4090',
                  '--cpu-bind=verbose,threads', 'USE_TF=0', 'set -euo pipefail',
                  '#SBATCH --job-name=GRID-LIGER-ScalingV3', 'export NUMPY_MADVISE_HUGEPAGE=0'):
        assert token in source
    c = config()
    ordered = ['fixture "$1"'] + [f'case "$1" --case {case["name"]}' for case in c['cases'][:4]]
    ordered += ['equivalence "$1"'] + [f'case "$1" --case {case["name"]}' for case in c['cases'][4:]]
    ordered += ['accept "$1"']
    positions = [source.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert 'sbatch ' not in source and ' resume ' not in source
    assert 'train_liger_cpu4' not in source
    assert 'tests/test_liger_row_lookup.py' in source
    assert source.index('export NUMPY_MADVISE_HUGEPAGE=0') < source.index('run_bound -m pytest')


def test_new_execution_and_test_files_have_explicit_source_bindings():
    assert set(manifest.SOURCES) == {
        'scripts/liger_sampled_scaling_manifest.py', 'scripts/probe_liger_sampled_scaling.py',
        'scripts/accept_liger_sampled_scaling.py', 'src/baselines/liger_access_timing.py',
        'tests/test_liger_access_timing.py', 'tests/test_liger_sampled_scaling.py',
        'tests/test_liger_sampled_scaling_manifest.py', 'run_liger_sampled_scaling_cpu4.sh',
        'src/baselines/liger_row_lookup.py', 'tests/test_liger_row_lookup.py'}


def test_v3_preserves_v2_cases_and_budget_and_only_adds_process_policy():
    previous = json.loads((ROOT / 'configs/baselines/liger_sampled_scaling_v2.json').read_text())
    current = config()
    assert current.pop('protocol') == 'liger-fixed-sid-scl-prefix-sampled-scaling-v3'
    assert previous.pop('protocol') == 'liger-fixed-sid-scl-prefix-sampled-scaling-v2'
    assert current.pop('process_environment') == {'NUMPY_MADVISE_HUGEPAGE': '0'}
    assert current == previous


@pytest.mark.parametrize('value', [None, '1', 'false', ''])
def test_process_memory_policy_must_be_set_before_numpy_import(monkeypatch, value):
    if value is None:
        monkeypatch.delenv('NUMPY_MADVISE_HUGEPAGE', raising=False)
    else:
        monkeypatch.setenv('NUMPY_MADVISE_HUGEPAGE', value)
    with pytest.raises(AssertionError):
        manifest.require_process_environment(config())
    monkeypatch.setenv('NUMPY_MADVISE_HUGEPAGE', '0')
    manifest.require_process_environment(config())


def transition_fixture(monkeypatch):
    changed = str(ROOT / 'changed.py')
    unchanged = str(ROOT / 'unchanged.py')
    old = {changed: 'a' * 64, unchanged: 'b' * 64}
    current = {changed: 'c' * 64, unchanged: 'b' * 64}
    policy = {'changed.py': {'prior_sha256': 'a' * 64, 'current_sha256': 'c' * 64}}
    monkeypatch.setattr(manifest, 'SOURCE_MIGRATIONS', policy)
    monkeypatch.setattr(manifest, 'sha', current.__getitem__)
    monkeypatch.setattr(manifest, 'historical_source_sha', lambda relative: old[str(ROOT / relative)])
    return old, current, policy


def test_source_migration_validates_old_git_and_exact_new_bytes(monkeypatch):
    old, current, _ = transition_fixture(monkeypatch)
    observed, migration = manifest.source_transition(old)
    assert observed == current
    assert migration == {str(ROOT / 'changed.py'): dict(prior_revision=manifest.PREDECESSOR_REVISION,
        prior_sha256='a' * 64, current_sha256='c' * 64)}


@pytest.mark.parametrize('change', ['historical', 'old_policy', 'new_bytes', 'unrelated', 'pending', 'missing'])
def test_unknown_migrations_or_drift_fail_closed(monkeypatch, change):
    old, current, policy = transition_fixture(monkeypatch)
    if change == 'historical':
        monkeypatch.setattr(manifest, 'historical_source_sha', lambda relative: 'f' * 64)
    elif change == 'old_policy': policy['changed.py']['prior_sha256'] = 'd' * 64
    elif change == 'new_bytes': current[str(ROOT / 'changed.py')] = 'd' * 64
    elif change == 'unrelated': current[str(ROOT / 'unchanged.py')] = 'd' * 64
    elif change == 'pending': policy['changed.py']['current_sha256'] = None
    else: del old[str(ROOT / 'changed.py')]
    with pytest.raises(AssertionError):
        manifest.source_transition(old)
