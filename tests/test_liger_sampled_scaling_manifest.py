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


def test_accepted_predecessor_is_immutable_and_every_old_source_is_unchanged():
    old, evidence = manifest.predecessor()
    assert evidence[str(manifest.PRIOR)] == manifest.PRIOR_SHA
    assert evidence[str(manifest.RECEIPT)] == manifest.RECEIPT_SHA
    assert len(old['execution_sources']) == 83
    assert len(evidence) == 9


def inherited_fixture():
    old = {key: {'value': key} for key in manifest.INHERITED_KEYS}
    old.update(execution_sources={'existing.py': 'unchanged'}, input_metadata={'input.json': 'fixed'})
    evidence = {'old-manifest.json': 'bound'}
    m = copy.deepcopy(old)
    m['execution_sources'].update({str(ROOT / relative): 'new' for relative in manifest.SOURCES})
    m['input_metadata'].update(evidence)
    m['input_metadata'][str(manifest.CONFIG)] = manifest.sha(manifest.CONFIG)
    m['dense_objectives'] = manifest.dense_objectives(config())
    return m, old, evidence


def test_inherited_source_and_input_bindings_are_exact_with_no_migration_allowlist():
    manifest.verify_inheritance(*inherited_fixture())


@pytest.mark.parametrize('change', ['source_change', 'source_drop', 'source_extra',
    'metadata_change', 'metadata_drop', 'asset', 'objective', 'resume', 'test', 'migration'])
def test_no_silent_source_asset_or_scope_migration(change):
    m, old, evidence = inherited_fixture()
    if change == 'source_change': m['execution_sources']['existing.py'] = 'drift'
    elif change == 'source_drop': del m['execution_sources']['existing.py']
    elif change == 'source_extra': m['execution_sources']['extra.py'] = 'unbound'
    elif change == 'metadata_change': m['input_metadata']['input.json'] = 'drift'
    elif change == 'metadata_drop': del m['input_metadata']['input.json']
    elif change == 'asset': m['assets']['value'] = 'changed'
    elif change == 'objective': m['dense_objectives']['cached8k2048']['seed'] = 99
    elif change == 'resume': m['resume_checkpoint'] = {'path': 'old.pt'}
    elif change == 'test': m['test_files'] = []
    else: m['source_migration'] = {}
    with pytest.raises(AssertionError):
        manifest.verify_inheritance(m, old, evidence)


def test_launcher_runs_one_cpu4_gpu_with_equivalence_before_large_cases():
    source = (ROOT / 'run_liger_sampled_scaling_cpu4.sh').read_text()
    for token in ('#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=4', '#SBATCH --gres=gpu:1',
                  '#SBATCH --mem=24G', '#SBATCH --time=00:45:00', '#SBATCH --partition=rtx4090',
                  '--cpu-bind=verbose,threads', 'USE_TF=0', 'set -euo pipefail'):
        assert token in source
    c = config()
    ordered = ['fixture "$1"'] + [f'case "$1" --case {case["name"]}' for case in c['cases'][:4]]
    ordered += ['equivalence "$1"'] + [f'case "$1" --case {case["name"]}' for case in c['cases'][4:]]
    ordered += ['accept "$1"']
    positions = [source.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert 'sbatch ' not in source and ' resume ' not in source
    assert 'train_liger_cpu4' not in source


def test_new_execution_and_test_files_have_explicit_source_bindings():
    assert set(manifest.SOURCES) == {
        'scripts/liger_sampled_scaling_manifest.py', 'scripts/probe_liger_sampled_scaling.py',
        'scripts/accept_liger_sampled_scaling.py', 'src/baselines/liger_access_timing.py',
        'tests/test_liger_access_timing.py', 'tests/test_liger_sampled_scaling.py',
        'tests/test_liger_sampled_scaling_manifest.py', 'run_liger_sampled_scaling_cpu4.sh'}
