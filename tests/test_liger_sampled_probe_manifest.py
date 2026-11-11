import copy
import json
from pathlib import Path

import pytest

from scripts import liger_sampled_probe_manifest as manifest


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads(manifest.CONFIG.read_text())


def test_probe_is_bounded_train_only_and_sampling_is_explicit():
    c = config()
    manifest.check_config(c)
    assert c['training']['max_attempts'] == c['warmup_attempts'] + c['measured_attempts'] == 12
    assert c['fixture_prefixes'] == c['training']['max_attempts'] * 1024
    assert c['save_after_attempt'] == c['warmup_attempts']
    m = {'dense_objectives': manifest.dense_objectives(c)}
    for case in c['cases']:
        bound = manifest.objective_for_case(m, c, case['name'])
        assert bound['num_negatives'] == case['num_negatives'] and bound['seed'] == 42
        assert bound['temperature'] == .07
        assert bound is not m['dense_objectives'][case['name']]


@pytest.mark.parametrize('keys,value', [
    (('formal_training_ready',), True), (('automatic_successor',), True),
    (('standardization_fit',), True), (('cache_trainable_projection',), True),
    (('cache_full_seen_features',), True), (('validation_test_access',), True),
    (('deadline_seconds',), 86400), (('fixture_prefixes',), 99999),
    (('training', 'max_attempts'), 19260), (('training', 'batch_size'), 2048),
    (('training', 'chunk_size'), 8192), (('training', 'history_items'), 999),
    (('training', 'warmup_steps'), 2), (('resources', 'cpu_ceiling'), 24),
    (('resources', 'gpus'), 3), (('resources', 'memory_gb'), 96),
    (('validation', 'access'), True), (('warmup_attempts',), 0),
    (('save_after_attempt',), 1), (('measured_attempts',), 2),
    (('cases', 0, 'num_negatives'), 1024), (('cases', 1, 'seed'), 43),
    (('cases', 0, 'backend'), 'cached_seen'), (('seed',), 43),
])
def test_contract_changes_are_rejected(keys, value):
    c = copy.deepcopy(config())
    target = c
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(AssertionError):
        manifest.check_config(c)


@pytest.mark.parametrize('field,value', [('num_negatives', 1), ('seed', 43),
    ('temperature', .2), ('sampling_version', 'unversioned')])
def test_canonical_objective_cannot_drift(field, value):
    c = config()
    m = {'dense_objectives': manifest.dense_objectives(c)}
    m['dense_objectives']['n8192'][field] = value
    with pytest.raises(AssertionError):
        manifest.objective_for_case(m, c, 'n8192')


def test_unknown_case_is_rejected():
    c = config()
    with pytest.raises(ValueError):
        manifest.objective_for_case({'dense_objectives': manifest.dense_objectives(c)}, c, 'full')


def test_frozen_predecessor_and_exact_two_file_transition():
    assert manifest.sha(manifest.PRIOR) == manifest.PRIOR_SHA
    assert manifest.sha(manifest.RECEIPT) == manifest.RECEIPT_SHA
    old = json.loads(manifest.PRIOR.read_text())
    sources, transitions = manifest.migrated_sources(old['execution_sources'])
    assert set(sources) == set(old['execution_sources'])
    assert set(transitions) == {str(ROOT / path) for path in manifest.REVISIONS}
    assert all(row['prior_sha256'] != row['current_sha256'] for row in transitions.values())


@pytest.mark.parametrize('failure', ['unrelated', 'prior', 'accepted', 'missing'])
def test_source_transition_rejects_arbitrary_drift(monkeypatch, failure):
    names = [str(ROOT / relative) for relative in manifest.REVISIONS]
    other = str(ROOT / 'src/baselines/liger_runtime.py')
    old = {name: 'old' for name in names}
    old[other] = 'unchanged'
    monkeypatch.setattr(manifest, 'sha', lambda path: 'new' if path in names else 'unchanged')
    monkeypatch.setattr(manifest, 'git_blob_sha', lambda revision, relative:
                        'old' if revision == manifest.PRIOR_REVISION else 'new')
    if failure == 'unrelated':
        old[other] = 'drift'
    elif failure == 'prior':
        old[names[0]] = 'unknown-old'
    elif failure == 'accepted':
        monkeypatch.setattr(manifest, 'git_blob_sha', lambda revision, relative: 'old')
    else:
        del old[names[0]]
    with pytest.raises(AssertionError):
        manifest.migrated_sources(old)


def test_launcher_has_only_one_four_cpu_gpu_job_and_fresh_process_cases():
    source = (ROOT / 'run_liger_sampled_probe_cpu4.sh').read_text()
    assert '#SBATCH --ntasks=1' in source and '#SBATCH --cpus-per-task=4' in source
    assert '#SBATCH --gres=gpu:1' in source and '#SBATCH --mem=24G' in source
    assert '#SBATCH --partition=rtx4090' in source and '#SBATCH --nodelist=' not in source
    assert '#SBATCH --time=00:45:00' in source and '--cpu-bind=verbose,threads' in source
    assert 'USE_TF=0' in source and 'set -euo pipefail' in source
    ordered = ['fixture "$1"', 'case "$1" --case n8192', 'resume "$1" --case n8192',
               'scripts.accept_liger_sampled case "$1" --case n8192',
               'case "$1" --case n32768', 'resume "$1" --case n32768',
               'scripts.accept_liger_sampled case "$1" --case n32768', 'accept "$1"']
    positions = [source.index(part) for part in ordered]
    assert positions == sorted(positions)
    assert 'sbatch ' not in source and 'train_liger_cpu4' not in source


def test_new_execution_files_have_bindings():
    for relative in ('scripts/probe_liger_sampled.py', 'scripts/accept_liger_sampled.py',
        'src/baselines/liger_sampled_catalog.py', 'run_liger_sampled_probe_cpu4.sh',
        'tests/test_liger_sampled_probe_manifest.py', 'tests/test_liger_sampled_probe.py'):
        assert relative in manifest.SOURCES
