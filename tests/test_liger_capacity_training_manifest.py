"""Formal authority is one fresh pass plus validation, never probe continuation."""
import copy
import json
from pathlib import Path

import pytest

from scripts import liger_capacity_training_manifest as contract


@pytest.fixture
def bound(tmp_path, monkeypatch):
    config_bytes = contract.CONFIG.read_bytes()
    legacy_bytes = (Path(__file__).resolve().parents[1] /
        'configs/baselines/liger_sampled_train_seed42_v1.json').read_bytes()
    files = {}
    for name in ('old.py', 'routes.py', 'new.py', 'upstream.py', 'meta.json', 'asset.npy',
                 'catalog.npy', 'keys.npy', 'rows.npy', 'receipt.json'):
        files[name] = tmp_path / name
        files[name].write_text(name)
    config, legacy, stats = [tmp_path / name for name in ('formal.json', 'legacy.json', 'stats.json')]
    config.write_bytes(config_bytes)
    legacy.write_bytes(legacy_bytes)
    stats.write_text(json.dumps(dict(stats_sha256='s' * 64, seen_rows_sha256='r' * 64)))
    item = lambda p: dict(path=str(p), sha256=contract.sha(p))
    sources = {str(files[name]): contract.sha(files[name]) for name in ('old.py', 'routes.py')}
    old = dict(phase='capacity384_resource_only', formal_training_ready=False,
        automatic_successor=False, test_records_read=False, dirty_at_submission=False,
        initialization='old probe', source_git_commit='a' * 40,
        execution_sources={str(files['old.py']): contract.sha(files['old.py'])},
        input_metadata={str(files['meta.json']): contract.sha(files['meta.json'])},
        upstream_files={str(files['upstream.py']): contract.sha(files['upstream.py'])},
        upstream_root=str(tmp_path), upstream_commit='c' * 40,
        config=str(legacy), config_sha256=contract.sha(legacy), capacity_config=item(config),
        standardization=item(stats), assets={'frozen_sid': item(files['asset.npy'])},
        catalog=item(files['catalog.npy']), index_artifacts={'sid_keys': item(files['keys.npy']), 'asset_rows': item(files['rows.npy'])},
        dense_objective={'num_negatives': 32768, 'seed': 42}, seed=42,
        expected_feature_cache_sha256='d' * 64, validation_files=['val'], train_files=['train'])
    prior = tmp_path / 'prior.json'
    prior.write_text(json.dumps(old))
    receipt = item(files['receipt.json'])
    evidence = {str(prior): contract.sha(prior), receipt['path']: receipt['sha256']}
    monkeypatch.setattr(contract, 'ROOT', tmp_path)
    monkeypatch.setattr(contract, 'CONFIG', config)
    monkeypatch.setattr(contract, 'PRIOR', prior)
    monkeypatch.setattr(contract, 'PRIOR_SHA', contract.sha(prior))
    monkeypatch.setattr(contract, 'SOURCES', ['new.py'])
    monkeypatch.setattr(contract, 'predecessor', lambda: (copy.deepcopy(old),
        {'training': receipt, 'validation': receipt, 'routes': receipt},
        copy.deepcopy(evidence), 'e' * 64, copy.deepcopy(sources)))
    monkeypatch.setattr(contract, 'git', lambda root, *args: '' if args[0] == 'status' else 'b' * 40)
    monkeypatch.setenv('NUMPY_MADVISE_HUGEPAGE', '0')
    path = tmp_path / 'launch/gate/manifest.json'
    m = contract.prepare(path)
    return path, m, files


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=contract.sha(path))


def make_gate_receipt(path, m):
    directory = Path(m['run_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_entries = []
    for step in (1, 3):
        target = directory / f'step0{step}.pt'
        target.write_text(f'toy checkpoint {step}')
        checkpoint_entries.append(dict(path=str(target), sha256=contract.sha(target),
            full_state_sha256=str(step) * 64, attempts=step, updates=step, next_batch=step))
    one, final = checkpoint_entries
    steps = [dict(attempt=i, updates=i, rows=2048, total_loss=30-i,
                  prefix_identity_sha256=str(i)*64, candidate_pool_sha256=str(i)*64)
             for i in (1, 2, 3)]
    shared = dict(status='passed', phase=contract.GATE_PHASE,
        job_id='12345', source_git_commit=m['source_git_commit'], manifest_sha256=contract.sha(path),
        initial_state_sha256=m['expected_initial_state_sha256'],
        checkpoint_binding=contract.checkpoint_binding(m), gpus=[{'uuid':'same-card'}],
        final_progress=dict(updates=3, attempts=3, prefix_count=6144, skipped_updates=0),
        state_after_three=final['full_state_sha256'], formal_training_ready=False, formal_checkpoint_produced=False,
        test_records_read=False, automatic_successor=False)
    a = dict(shared, resumed=False, steps=steps, step_one_checkpoint=one, validation_checkpoint=final)
    b = dict(shared, resumed=True, steps=steps[1:], loaded_checkpoint=one, restore_exact=True)
    train = write_json(directory/'continuous.json', a)
    resumed = write_json(directory/'resumed.json', b)
    artifact = directory/'validation/part-00000.npz'
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_bytes(b'toy serialized predictions')
    artifact_entry = dict(path=str(artifact), sha256=contract.sha(artifact), rows=65)
    vr = dict(status='passed', records=65, batches=3, cold_rows=23915712, checkpoint=final,
        gt_independence_verified=True, model_weights_unchanged=True, test_records_read=False,
        model_sha256='9'*64, metrics={'hybrid': {'Recall@5': 0.}}, artifacts=[artifact_entry])
    validation = write_json(directory/'validation/report.json', vr)
    va = dict(status='passed', records=65, checkpoint=final, report=validation, metrics=vr['metrics'])
    accepted = write_json(directory/'validation/acceptance.json', va)
    wrapper = dict(shared, checkpoint=final, validation_report=validation,
        validation_acceptance=accepted, proof=dict(model_weights_unchanged=True,
            before_model_sha256='9'*64, after_model_sha256='9'*64, load_scope='model_weights_only'),
        loaded=dict(checkpoint=final, model_state_sha256='9'*64, load_scope='model_weights_only'))
    wrapper_ref = write_json(directory/'validation_gate.json', wrapper)
    refs = dict(train_report=train, resumed_report=resumed, validation_gate_report=wrapper_ref,
                validation_report=validation, validation_acceptance=accepted)
    receipt = dict(status='passed_capacity384_training_gate', phase=contract.GATE_PHASE,
        exact_fresh_process_replay=True, two_gpu_validation_records=65,
        validation_model_weights_unchanged=True, formal_training_ready=False,
        formal_checkpoint_produced=False, test_records_read=False, automatic_successor=False,
        gate_manifest=dict(path=str(path), sha256=contract.sha(path)),
        manifest_sha256=contract.sha(path), source_git_commit=m['source_git_commit'],
        initial_state_sha256=m['expected_initial_state_sha256'],
        formal_config_sha256=m['formal_config']['sha256'], execution_sources=m['execution_sources'],
        job_id='12345', reports=list(refs.values())+[artifact_entry, one, final], **refs)
    target = directory/'gate_acceptance.json'
    write_json(target, receipt)
    return target, receipt


def test_gate_preserves_runtime_bindings_and_does_not_load_probe_weights(bound):
    path, m, _ = bound
    c = contract.verify(m)
    assert c['model']['d_model'] == 384 and c['training']['physical_batch_size'] == 1024
    assert m['phase'] == contract.GATE_PHASE and m['gate_acceptance'] is None
    assert m['formal_training_ready'] is False and m['expected_initial_state_sha256'] == 'e'*64
    assert 'resume_checkpoint' not in m and 'initial_checkpoint' not in m
    assert m['config'] != m['formal_config']['path']
    assert contract.checkpoint_binding(m)['feature_cache_sha256'] == 'd'*64
    with pytest.raises(AssertionError): contract.prepare(path)


def test_formal_requires_exact_new_handoff_and_changes_only_purpose(bound):
    path, gate, _ = bound
    target, _ = make_gate_receipt(path, gate)
    formal_path = path.parent.parent/'formal/manifest.json'
    m = contract.prepare(formal_path, phase='formal', gate_receipt=target)
    assert contract.verify(m)['training']['max_epochs'] == 1
    assert m['phase'] == contract.FORMAL_PHASE and m['formal_training_ready'] is True
    a, b = contract.checkpoint_binding(gate), contract.checkpoint_binding(m)
    assert a.pop('purpose') == contract.GATE_PHASE and b.pop('purpose') == contract.FORMAL_PHASE
    assert a == b
    assert m['run_directory'] != gate['run_directory']


@pytest.mark.parametrize('phase,receipt', [('formal', None), ('gate', 'old-checkpoint'), ('resume', None)])
def test_no_scope_inference_from_optional_arguments(bound, phase, receipt):
    path, _, _ = bound
    with pytest.raises(AssertionError): contract.prepare(path.with_name('other.json'), phase, receipt)


@pytest.mark.parametrize('key,value', [
    ('phase', 'old-epoch3'), ('formal_training_ready', True), ('formal_training_ready', 0),
    ('automatic_successor', True), ('test_records_read', True), ('dirty_at_submission', True),
    ('initialization', 'load probe'), ('source_git_commit', 'x'*40), ('seed', 43),
    ('expected_initial_state_sha256', '1'*64), ('run_directory', '/tmp/outside'),
    ('gate_acceptance', {}), ('config', '/tmp/replacement.json'), ('formal_config', {}),
    ('execution_sources', {}), ('input_metadata', {}), ('resource_acceptances', {}),
])
def test_reject_manifest_scope_data_or_source_drift(bound, key, value):
    _, m, _ = bound
    m[key] = value
    with pytest.raises((AssertionError, ValueError, FileNotFoundError)): contract.verify(m)


@pytest.mark.parametrize('key', ['resume_checkpoint', 'test_files', 'initial_checkpoint', 'extra_budget'])
def test_reject_unknown_manifest_fields(bound, key):
    _, m, _ = bound
    m[key] = 'unauthorized'
    with pytest.raises(AssertionError): contract.verify(m)


@pytest.mark.parametrize('key', ['new.py', 'old.py', 'routes.py', 'meta.json', 'upstream.py'])
def test_reject_bound_file_bytes_drift(bound, key):
    _, m, files = bound
    files[key].write_text('changed')
    with pytest.raises((AssertionError, ValueError)): contract.verify(m)


@pytest.mark.parametrize('section,key,value', [
    ('training', 'physical_batch_size', 512), ('training', 'accumulation_steps', 4),
    ('training', 'max_epochs', 3), ('training', 'drop_last', True),
    ('training', 'max_attempts', True), ('training', 'warmup_steps', 963),
    ('validation', 'candidate_mode', 'sampled'), ('validation', 'records', 65),
    ('validation', 'test_access', True), ('resources', 'cpu_ceiling', 8),
    ('resources', 'memory_gb', 24.0), ('model', 'd_model', 128),
])
def test_config_values_and_types_are_frozen(bound, section, key, value):
    _, m, _ = bound
    c = contract.verify(m)
    c[section][key] = value
    with pytest.raises(AssertionError): contract.check_config(c)


def test_config_file_cannot_self_authorize_by_rehash(bound):
    _, m, _ = bound
    c = json.loads(contract.CONFIG.read_text())
    c['training']['max_epochs'] = 2
    contract.CONFIG.write_text(json.dumps(c))
    m['formal_config']['sha256'] = contract.sha(contract.CONFIG)
    with pytest.raises(AssertionError): contract.verify(m)


def test_environment_requires_both_source_and_manifest_binding(bound, monkeypatch):
    path, m, _ = bound
    monkeypatch.setenv('SOURCE_GIT_COMMIT', m['source_git_commit'])
    monkeypatch.setenv('LIGER_CAPACITY_TRAIN_MANIFEST_SHA256', contract.sha(path))
    assert contract.binding(path)[0] == m
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'a'*40)
    with pytest.raises(AssertionError): contract.binding(path)


@pytest.mark.parametrize('key,value', [
    ('status', 'passed_resource_training'), ('phase', 'old'),
    ('exact_fresh_process_replay', False), ('two_gpu_validation_records', 64),
    ('validation_model_weights_unchanged', False), ('formal_training_ready', True),
    ('test_records_read', True), ('automatic_successor', True), ('formal_checkpoint_produced', True),
    ('formal_config_sha256', 'bad'), ('initial_state_sha256', 'bad'), ('execution_sources', {}),
])
def test_formal_rejects_weak_or_wrong_gate_receipt(bound, key, value):
    path, m, _ = bound
    target, receipt = make_gate_receipt(path, m)
    receipt[key] = value
    write_json(target, receipt)
    with pytest.raises(AssertionError):
        contract.prepare(path.parent.parent/'formal/manifest.json', phase='formal', gate_receipt=target)


@pytest.mark.parametrize('key', ['train_report', 'resumed_report', 'validation_report',
                               'validation_acceptance', 'validation_gate_report'])
def test_gate_report_bytes_cannot_change_after_acceptance(bound, key):
    path, m, _ = bound
    target, receipt = make_gate_receipt(path, m)
    Path(receipt[key]['path']).write_text('{}')
    with pytest.raises(AssertionError):
        contract.prepare(path.parent.parent/'formal/manifest.json', phase='formal', gate_receipt=target)


def test_full_configuration_math_and_resources_are_explicit():
    c = contract.check_config(json.loads(contract.CONFIG.read_text()))
    assert c['training']['prefixes_per_epoch'] == 45996113
    assert c['training']['parents_per_epoch'] == 6573674
    assert c['training']['batches_per_epoch'] == 22460 and c['training']['final_batch_size'] == 81
    assert c['resources']['training_gpus'] == 1 and c['resources']['validation_gpus'] == 2
    assert c['resources']['cpu_ceiling'] == 4 and c['validation']['cold_rows'] == 23915712
