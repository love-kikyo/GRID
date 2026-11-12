import copy
import json
from pathlib import Path

import pytest

from scripts import liger_sampled_training_manifest as manifest


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads(manifest.CONFIG.read_text())


def test_one_pass_counts_and_schedule_are_not_inherited_from_old_three_epoch_fixture():
    c = config()
    manifest.check_config(c)
    t = c['training']
    assert t['max_attempts'] == t['max_optimizer_steps'] == t['batches_per_epoch'] == 22460
    assert t['prefixes_per_epoch'] == (t['batches_per_epoch'] - 1) * t['batch_size'] + 81
    assert t['parents_per_epoch'] == 6573674 and t['warmup_steps'] == 1123
    assert t['warmup_steps'] / t['max_optimizer_steps'] == .05
    assert t['stop_after_epoch'] == t['max_epochs'] == 1
    assert c['dense_sampling'] == {'num_negatives': 32768, 'seed': 42}
    assert not c['validation']['test_access'] and not c['automatic_successor']
    assert manifest.execution_counts(c)['prefix_batches_per_pass'] == 22460


@pytest.mark.parametrize('keys,value', [
    (('seed',), 43), (('automatic_successor',), True),
    (('initialization',), 'resume old teacher'), (('dense_sampling', 'num_negatives'), 8192),
    (('dense_sampling', 'seed'), 43), (('training', 'batch_size'), 4096),
    (('training', 'max_epochs'), 3), (('training', 'stop_after_epoch'), 2),
    (('training', 'max_epochs'), True), (('training', 'max_optimizer_steps'), 19260),
    (('training', 'max_attempts'), 22461), (('training', 'warmup_steps'), 963),
    (('training', 'batches_per_epoch'), 44919), (('training', 'final_batch_size'), 1105),
    (('training', 'drop_last'), True), (('training', 'prefixes_per_epoch'), 6573674),
    (('training', 'checkpoint_every_batches'), 0), (('training', 'history_items'), 10),
    (('training', 'prefix_shuffle_buffer'), 1), (('training', 'supervision'), 'final_gt_only'),
    (('feature_cache', 'verify_all_rows'), False), (('feature_cache', 'cache_trainable_projection'), True),
    (('feature_cache', 'storage'), 'shared_disk'), (('standardization', 'fit_scope'), 'all_splits'),
    (('validation', 'records'), 65), (('validation', 'candidate_mode'), 'sampled_cold'),
    (('validation', 'test_access'), True), (('validation', 'selection'), ['test Recall@5']),
    (('smoke', 'validation_records'), 64), (('smoke', 'train_steps'), 1),
    (('smoke', 'deadline_seconds'), 86400), (('resources', 'cpu_ceiling'), 24),
    (('resources', 'gpus'), 3), (('resources', 'memory_gb'), 96),
    (('resources', 'formal_time_limit_hours'), 36), (('process_environment', 'NUMPY_MADVISE_HUGEPAGE'), '1'),
])
def test_unapproved_training_resource_or_evaluation_changes_rejected(keys, value):
    c = config()
    target = c
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(AssertionError): manifest.check_config(c)


@pytest.mark.parametrize('value', [None, '', '1', 'false'])
def test_numpy_policy_must_be_explicitly_set_before_process_start(monkeypatch, value):
    if value is None: monkeypatch.delenv('NUMPY_MADVISE_HUGEPAGE', raising=False)
    else: monkeypatch.setenv('NUMPY_MADVISE_HUGEPAGE', value)
    with pytest.raises(AssertionError): manifest.require_environment()
    monkeypatch.setenv('NUMPY_MADVISE_HUGEPAGE', '0')
    manifest.require_environment()


def test_actual_accepted_resource_evidence_and_input_identities_are_unchanged():
    old, evidence = manifest.predecessor()
    assert len(old['execution_sources']) == 93
    assert evidence[str(manifest.PRIOR)] == manifest.PRIOR_SHA
    assert evidence[str(manifest.RECEIPT)] == manifest.RECEIPT_SHA
    identities = manifest.expected_identities(evidence)
    assert set(identities) == {'expected_feature_cache_sha256', 'expected_initial_state_sha256'}
    assert all(len(value) == 64 for value in identities.values())
    metadata, validation = manifest.common_inputs(old, evidence)
    assert metadata[str(manifest.CONFIG)] == manifest.sha(manifest.CONFIG)
    assert len(validation) > 0
    assert all(Path(entry['path']).parent.name == 'evaluation' for entry in validation)
    assert old['counts']['prefix_examples_per_pass'] == config()['training']['prefixes_per_epoch']
    assert old['counts']['batch_size'] == 1024  # Provenance, never the new training budget.


def common_fixture(monkeypatch):
    c = config()
    old = {key: {'value': key} for key in manifest.INHERITED_KEYS}
    old.update(execution_sources={'original.py': 'unchanged'}, input_metadata={'old.json': 'old'})
    metadata, validation = {'old.json': 'old', 'new.json': 'new'}, [{'path': 'evaluation/data', 'size': 3}]
    identities = dict(expected_feature_cache_sha256='c' * 64, expected_initial_state_sha256='a' * 64)
    monkeypatch.setattr(manifest, 'common_inputs', lambda old, evidence: (metadata, validation))
    monkeypatch.setattr(manifest, 'expected_identities', lambda evidence: identities)
    m = copy.deepcopy({key: old[key] for key in manifest.INHERITED_KEYS})
    m.update(execution_sources=dict(old['execution_sources'], **{str(ROOT / p): 'new' for p in manifest.SOURCES}),
        input_metadata=copy.deepcopy(metadata), validation_files=copy.deepcopy(validation),
        config=str(manifest.CONFIG), config_sha256=manifest.sha(manifest.CONFIG),
        dense_objective=manifest.sampling_binding(c['dense_sampling']), process_environment=c['process_environment'],
        execution_counts=manifest.execution_counts(c),
        candidate_universe_finalized=True, automatic_successor=False, test_records_read=False,
        initialization=c['initialization'], stage=dict(start_epoch=0, stop_after_epoch=1),
        dirty_at_submission=False, phase=manifest.GATE_PHASE, formal_training_ready=False,
        run_directory=str(ROOT / 'logs/train/runs/example'), **identities)
    return m, old, {}


def test_common_manifest_adds_only_new_sources_and_validation_identity(monkeypatch):
    manifest.verify_common(*common_fixture(monkeypatch))


@pytest.mark.parametrize('change', ['source', 'source_drop', 'source_extra', 'input', 'validation',
    'asset', 'objective', 'cache', 'initial', 'counts', 'purpose', 'resume', 'test', 'migration', 'scope', 'directory'])
def test_common_manifest_rejects_drift_or_wrong_authority(monkeypatch, change):
    m, old, evidence = common_fixture(monkeypatch)
    if change == 'source': m['execution_sources']['original.py'] = 'changed'
    elif change == 'source_drop': del m['execution_sources']['original.py']
    elif change == 'source_extra': m['execution_sources']['extra.py'] = 'unbound'
    elif change == 'input': m['input_metadata']['old.json'] = 'changed'
    elif change == 'validation': m['validation_files'] = []
    elif change == 'asset': m['assets']['value'] = 'different'
    elif change == 'objective': m['dense_objective']['seed'] = 43
    elif change == 'cache': m['expected_feature_cache_sha256'] = 'changed'
    elif change == 'initial': m['expected_initial_state_sha256'] = 'changed'
    elif change == 'counts': m['execution_counts']['batch_size'] = 1024
    elif change == 'purpose': m['formal_training_ready'] = True
    elif change == 'resume': m['resume_checkpoint'] = {'path': 'diagnostic.pt'}
    elif change == 'test': m['test_files'] = []
    elif change == 'migration': m['source_migration'] = {}
    elif change == 'scope': m['automatic_successor'] = True
    else: m['run_directory'] = str(ROOT / 'logs/analysis/old')
    with pytest.raises(AssertionError): manifest.verify_common(m, old, evidence)


def gate_fixture(monkeypatch):
    m, _, _ = common_fixture(monkeypatch)
    gate = copy.deepcopy(m)
    gate['source_git_commit'] = 'gate-revision'
    m.update(phase=manifest.FORMAL_PHASE, formal_training_ready=True,
             smoke_acceptance={'path': 'receipt.json', 'sha256': 'receipt'})
    receipt = dict(status='passed', phase=manifest.GATE_PHASE, fresh_process_exact_resume=True,
        validation_preserved_full_learning_state=True, initial_state_sha256=m['expected_initial_state_sha256'],
        full_cold_validation_records=65, config_sha256=m['config_sha256'],
        execution_sources=copy.deepcopy(m['execution_sources']), dense_objective=copy.deepcopy(m['dense_objective']),
        formal_checkpoint_produced=False, test_records_read=False, automatic_successor=False,
        gate_manifest=dict(path='gate.json', sha256='gate-hash'), manifest_sha256='gate-hash',
        source_git_commit='gate-revision', job_id='123', reports=[dict(path='part.json', sha256='part-hash')])
    monkeypatch.setattr(manifest, 'read_bound', lambda entry: gate if entry['path'] == 'gate.json' else receipt)
    monkeypatch.setattr(manifest, 'sha', lambda path: 'part-hash')
    return m, gate, receipt


def test_formal_authority_requires_matching_exact_new_gate(monkeypatch):
    m, _, receipt = gate_fixture(monkeypatch)
    assert manifest.verify_gate(m['smoke_acceptance'], m) is receipt


@pytest.mark.parametrize('change', ['status', 'phase', 'resume', 'records', 'source', 'config', 'objective',
    'test', 'formal_weights', 'successor', 'manifest_hash', 'gate_source', 'gate_input', 'gate_phase',
    'gate_resume', 'gate_scope', 'gate_stage', 'report_hash', 'report_missing', 'validation_state', 'initial'])
def test_old_or_incomplete_gate_cannot_authorize_formal_run(monkeypatch, change):
    m, gate, r = gate_fixture(monkeypatch)
    if change == 'status': r['status'] = 'passed_resource_probe'
    elif change == 'phase': r['phase'] = 'training_resume_smoke'
    elif change == 'resume': r['fresh_process_exact_resume'] = False
    elif change == 'records': r['full_cold_validation_records'] = 0
    elif change == 'source': r['execution_sources']['original.py'] = 'changed'
    elif change == 'config': r['config_sha256'] = 'different'
    elif change == 'objective': r['dense_objective']['num_negatives'] = 8192
    elif change == 'test': r['test_records_read'] = True
    elif change == 'formal_weights': r['formal_checkpoint_produced'] = True
    elif change == 'successor': r['automatic_successor'] = True
    elif change == 'manifest_hash': r['manifest_sha256'] = 'wrong'
    elif change == 'gate_source': gate['source_git_commit'] = 'different'
    elif change == 'gate_input': gate['input_metadata']['old.json'] = 'different'
    elif change == 'gate_phase': gate['formal_training_ready'] = True
    elif change == 'gate_resume': gate['resume_checkpoint'] = {'path': 'old.pt'}
    elif change == 'gate_scope': gate['automatic_successor'] = True
    elif change == 'gate_stage': gate['stage']['stop_after_epoch'] = 3
    elif change == 'report_hash': r['reports'][0]['sha256'] = 'wrong'
    elif change == 'validation_state': r['validation_preserved_full_learning_state'] = False
    elif change == 'initial': r['initial_state_sha256'] = 'wrong'
    else: r['reports'] = []
    with pytest.raises(AssertionError): manifest.verify_gate(m['smoke_acceptance'], m)


def test_checkpoint_binding_is_stable_across_launches_but_never_gate_to_formal(monkeypatch):
    m, _, _ = common_fixture(monkeypatch)
    # Supply the base binding's concrete mapping-shaped metadata.
    m['assets'] = {'features': {'sha256': 'asset'}}
    m['catalog'] = {'sha256': 'catalog'}
    m['standardization'] = {'sha256': 'stats'}
    original = manifest.checkpoint_binding(m)
    m.update(job_id='new-job', source_git_commit='new-doc-only-commit', run_directory='new-directory')
    assert manifest.checkpoint_binding(m) == original
    m['formal_training_ready'] = True
    formal = manifest.checkpoint_binding(m)
    assert formal['purpose'] != original['purpose']
    assert formal['dense_objective'] == original['dense_objective']


@pytest.mark.parametrize('filename,expected_time,module,sequence', [
    ('run_liger_sampled_train_cpu4.sh', '12:00:00', 'scripts.train_liger_sampled_cpu4', ['run', 'accept']),
    ('run_liger_sampled_training_gate_cpu4.sh', '00:45:00', 'scripts.probe_liger_sampled_training', ['continuous', 'resumed', 'accept']),
])
def test_launchers_bound_resources_and_have_no_automatic_successor(filename, expected_time, module, sequence):
    source = (ROOT / filename).read_text()
    for token in ('#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=4', '#SBATCH --gres=gpu:1',
                  '#SBATCH --mem=24G', '#SBATCH --partition=rtx4090', '--cpu-bind=verbose,threads',
                  'set -euo pipefail', 'USE_TF=0', 'export NUMPY_MADVISE_HUGEPAGE=0',
                  'LIGER_SAMPLED_TRAIN_MANIFEST_SHA256', 'tests/test_liger_row_lookup.py',
                  f'#SBATCH --time={expected_time}'):
        assert token in source
    locations = [source.index(f'{module} {mode} "$1"') for mode in sequence]
    assert locations == sorted(locations)
    assert source.index('export NUMPY_MADVISE_HUGEPAGE=0') < source.index('run_bound -m pytest')
    assert 'sbatch ' not in source and '#SBATCH --nodelist=' not in source
    assert 'checkpoint' not in source.lower()  # Only manifest-specified fresh first pass.


def test_every_new_execution_file_has_source_binding():
    assert len(manifest.SOURCES) == len(set(manifest.SOURCES)) == 8
    assert 'scripts/train_liger_sampled_cpu4.py' in manifest.SOURCES
    assert 'scripts/probe_liger_sampled_training.py' in manifest.SOURCES
    assert 'tests/test_liger_sampled_training_manifest.py' in manifest.SOURCES
