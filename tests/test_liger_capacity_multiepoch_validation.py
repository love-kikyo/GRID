"""CPU-only tests of full-state epoch handoff and validation orchestration."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import validate_liger_capacity_multiepoch as val
from scripts.probe_liger_cpu4 import sha
from src.baselines.liger_early_stopping import initial_policy, advance_policy
from src.baselines.liger_prefix_training import initial_progress
from src.baselines.liger_state import tree_digest


def config():
    return json.loads(Path('configs/baselines/liger_capacity384_multiepoch_seed42_v1.json').read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return val.entry(path)


def saved_state(tmp_path, c, epoch=1, policy=None, diagnostic=True):
    t = c['training']
    policy = copy.deepcopy(initial_policy() if policy is None else policy)
    p = initial_progress(max_optimizer_steps=t['max_optimizer_steps'], max_attempts=t['max_attempts'])
    n = epoch * t['batches_per_epoch']
    p.update(epoch=epoch-1, next_batch=t['batches_per_epoch'], samples_in_epoch=t['prefixes_per_epoch'],
        attempts=n, batches_consumed=n, updates=n, prefix_count=epoch*t['prefixes_per_epoch'],
        final_count=epoch*t['parents_per_epoch'], early_stopping=policy,
        best=copy.deepcopy(policy['best']), validations=copy.deepcopy(policy['validations']))
    binding = dict(purpose='capacity384_multiepoch_gate' if diagnostic else 'capacity384_multiepoch_formal')
    state = dict(format_version=1, binding=binding, model={'weight': torch.tensor([1., 2.])},
        optimizer={'state': {0: dict(step=torch.tensor(float(n)), exp_avg=torch.tensor([.3, .4]),
                                    exp_avg_sq=torch.tensor([.7, .8]))},
                   'param_groups': [dict(params=[0], lr=.0002)]},
        scheduler=dict(total=t['max_optimizer_steps'], warmup=t['warmup_steps'],
                       base_lr=c['optimizer']['learning_rate'], updates=n),
        scaler=dict(scale=65536., growth_factor=2., backoff_factor=.5, growth_interval=2000, _growth_tracker=3),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=[torch.tensor([1, 2, 3], dtype=torch.uint8)]), progress=p, model_training=True)
    path = tmp_path/f'epoch{epoch:02d}-before.pt'
    torch.save(state, path)
    cp = dict(path=str(path), sha256=sha(path), full_state_sha256=tree_digest(state),
              **{key:p[key] for key in ('epoch', 'next_batch', 'updates', 'attempts')})
    return state, cp


def evidence(tmp_path, checkpoint, c, epoch=1, recall=.1, ndcg=.08, diagnostic=True):
    metrics = {mode: {'Recall@5':recall, 'NDCG@5':ndcg,
                     'Recall@10':recall, 'NDCG@10':ndcg,
                     'Recall@20':recall, 'NDCG@20':ndcg} for mode in ('hybrid', 'beam_only')}
    records = c['smoke']['validation_records'] if diagnostic else c['validation']['records']
    report = dict(status='passed', records=records, checkpoint=checkpoint, metrics=metrics,
        cold_rows=23915712, model_weights_unchanged=True, gt_independence_verified=True,
        test_records_read=False, raw_generation_coverage=dict(hits={'20':1}, recall={'20':1/records}))
    report_ref = write_json(tmp_path/f'v{epoch:02d}/report.json', report)
    accepted = dict(status='passed', records=records, checkpoint=checkpoint, report=report_ref,
        metrics=metrics, full_cold_rows=23915712, model_weights_unchanged=True, test_records_read=False)
    accepted_ref = write_json(tmp_path/f'v{epoch:02d}/acceptance.json', accepted)
    return dict(epoch=epoch, checkpoint=checkpoint, report=report_ref,
                acceptance=accepted_ref, metrics=metrics)


@pytest.fixture
def gate(tmp_path):
    c = config()
    c['training'].update(batches_per_epoch=3, prefixes_per_epoch=6144, parents_per_epoch=850)
    state, cp = saved_state(tmp_path, c)
    v = evidence(tmp_path, cp, c)
    return SimpleNamespace(c=c, state=state, cp=cp, validation=v, path=tmp_path/'after.pt', root=tmp_path)


def test_cpu_handoff_changes_only_progress_and_preserves_process_rng(gate):
    h = gate
    rng = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())
    cp, p, proof = val.finalize_checkpoint(h.cp, h.state['binding'], h.c, h.validation, h.path, diagnostic=True)
    saved = torch.load(h.path, weights_only=False)
    assert sha(h.cp['path']) == h.cp['sha256']
    assert tree_digest(saved) == cp['full_state_sha256'] and sha(h.path) == cp['sha256']
    for key in set(saved)-{'progress'}:
        assert tree_digest(saved[key]) == tree_digest(h.state[key])
    assert saved['progress'] == p
    assert p['epoch'] == 1 and p['next_batch'] == p['samples_in_epoch'] == 0
    assert p['attempts'] == p['updates'] == 3 and p['prefix_count'] == 6144
    assert p['early_stopping']['completed_epochs'] == 1 and p['early_stopping']['bad_epochs'] == 0
    assert p['best'] == p['early_stopping']['best'] and p['validations'] == [h.validation]
    assert proof['non_progress_state_unchanged']
    assert proof['before_non_progress_sha256'] == proof['after_non_progress_sha256']
    assert proof['preserved_cuda_rng_devices'] == 1 and proof['changed_fields'] == ['progress']
    assert rng[0] == random.getstate()
    np.testing.assert_array_equal(rng[1][1], np.random.get_state()[1])
    torch.testing.assert_close(rng[2], torch.get_rng_state(), rtol=0, atol=0)


def test_formal_census_full_validation_and_immutable_schedule(tmp_path):
    c = config()
    s, cp = saved_state(tmp_path, c, diagnostic=False)
    v = evidence(tmp_path, cp, c, diagnostic=False)
    next_cp, p, proof = val.finalize_checkpoint(cp, s['binding'], c, v, tmp_path/'full.pt')
    saved = torch.load(next_cp['path'], weights_only=False)
    assert p['attempts'] == 22460 and p['prefix_count'] == 45996113
    assert saved['scheduler']['total'] == 269520 and saved['scheduler']['warmup'] == 1123
    assert proof['diagnostic'] is False


def test_gate_census_cannot_be_used_by_formal_handoff(gate):
    with pytest.raises(AssertionError):
        val.finalize_checkpoint(gate.cp, gate.state['binding'], gate.c, gate.validation, gate.path)


def test_repeated_handoff_cannot_overwrite_existing_checkpoint(gate):
    h = gate
    val.finalize_checkpoint(h.cp, h.state['binding'], h.c, h.validation, h.path, diagnostic=True)
    before = h.path.read_bytes()
    with pytest.raises(FileExistsError):
        val.finalize_checkpoint(h.cp, h.state['binding'], h.c, h.validation, h.path, diagnostic=True)
    assert h.path.read_bytes() == before


def test_source_checkpoint_must_never_be_overwritten(gate):
    h = gate
    with pytest.raises(AssertionError):
        val.finalize_checkpoint(h.cp, h.state['binding'], h.c, h.validation, h.cp['path'], diagnostic=True)
    assert sha(h.cp['path']) == h.cp['sha256']


@pytest.mark.parametrize('tamper', ['bytes', 'digest', 'binding', 'training_mode', 'rng_devices',
    'scheduler_total', 'scheduler_update', 'optimizer_step', 'optimizer_nonfinite', 'model_nonfinite',
    'prefixes', 'parents', 'pending_epoch', 'entry_cursor', 'policy_counter', 'policy_mirror'])
def test_invalid_training_state_cannot_be_promoted(gate, tamper):
    h = gate
    s, cp = h.state, h.cp
    if tamper == 'binding': s['binding']['protocol'] = 'changed'
    elif tamper == 'training_mode': s['model_training'] = False
    elif tamper == 'rng_devices': s['rng']['cuda'].append(s['rng']['cuda'][0])
    elif tamper == 'scheduler_total': s['scheduler']['total'] -= 1
    elif tamper == 'scheduler_update': s['scheduler']['updates'] -= 1
    elif tamper == 'optimizer_step': s['optimizer']['state'][0]['step'] -= 1
    elif tamper == 'optimizer_nonfinite': s['optimizer']['state'][0]['exp_avg'][0] = float('nan')
    elif tamper == 'model_nonfinite': s['model']['weight'][0] = float('inf')
    elif tamper == 'prefixes': s['progress']['prefix_count'] -= 1
    elif tamper == 'parents': s['progress']['final_count'] -= 1
    elif tamper == 'pending_epoch': s['progress']['epoch'] += 1
    elif tamper == 'policy_counter': s['progress']['early_stopping']['bad_epochs'] += 1
    elif tamper == 'policy_mirror': s['progress']['best'] = {'wrong':True}
    torch.save(s, cp['path'])
    cp.update(sha256=sha(cp['path']), full_state_sha256=tree_digest(s))
    h.validation = evidence(h.root, cp, h.c)
    if tamper == 'bytes':
        with open(cp['path'], 'ab') as f: f.write(b'corrupt')
    elif tamper == 'digest': cp['full_state_sha256'] = 'wrong'
    elif tamper == 'entry_cursor': cp['attempts'] -= 1
    expected_binding = {'purpose':'capacity384_multiepoch_gate'}
    with pytest.raises((AssertionError, ValueError)):
        val.finalize_checkpoint(cp, expected_binding, h.c, h.validation, h.path, diagnostic=True)
    assert not h.path.exists()


@pytest.mark.parametrize('tamper', ['report_bytes', 'acceptance_bytes', 'records', 'checkpoint',
    'metric', 'status', 'cold', 'weights', 'gt_independence', 'test_access', 'epoch'])
def test_unaccepted_or_tampered_validation_cannot_advance_policy(gate, tamper):
    h = gate
    r = json.loads(Path(h.validation['report']['path']).read_text())
    if tamper == 'records': r['records'] -= 1
    elif tamper == 'checkpoint': r['checkpoint']['updates'] += 1
    elif tamper == 'metric': r['metrics']['hybrid']['Recall@5'] += .1
    elif tamper == 'status': r['status'] = 'failed'
    elif tamper == 'cold': r['cold_rows'] -= 1
    elif tamper == 'weights': r['model_weights_unchanged'] = False
    elif tamper == 'gt_independence': r['gt_independence_verified'] = False
    elif tamper == 'test_access': r['test_records_read'] = True
    h.validation['report'] = write_json(Path(h.validation['report']['path']), r)
    a = json.loads(Path(h.validation['acceptance']['path']).read_text())
    a['report'] = h.validation['report']
    h.validation['acceptance'] = write_json(Path(h.validation['acceptance']['path']), a)
    if tamper in ('report_bytes', 'acceptance_bytes'):
        key = tamper.split('_')[0]
        with open(h.validation[key]['path'], 'a') as f: f.write(' ')
    elif tamper == 'epoch': h.validation['epoch'] = 2
    with pytest.raises((AssertionError, ValueError)):
        val.finalize_checkpoint(h.cp, h.state['binding'], h.c, h.validation, h.path, diagnostic=True)
    assert not h.path.exists()


def test_sixth_epoch_handoff_keeps_patience_and_can_stop(gate):
    h = gate
    policy = initial_policy()
    for epoch in range(1, 6):
        s, cp = saved_state(h.root, h.c, epoch, policy)
        v = evidence(h.root, cp, h.c, epoch, recall=.2 if epoch == 1 else .1)
        policy = advance_policy(policy, v, h.c)
    s, cp = saved_state(h.root, h.c, 6, policy)
    v = evidence(h.root, cp, h.c, 6, recall=.1)
    _, p, _ = val.finalize_checkpoint(cp, s['binding'], h.c, v, h.path, diagnostic=True)
    assert p['early_stopping']['bad_epochs'] == 5
    assert p['early_stopping']['stopped'] and p['early_stopping']['stop_reason'] == 'patience'
    assert p['best']['epoch'] == 1 and len(p['validations']) == 6


@pytest.fixture
def formal(tmp_path, monkeypatch):
    import scripts
    from scripts.train_liger_capacity_multiepoch import prior_epoch_state
    c = config()
    s, cp = saved_state(tmp_path, c, diagnostic=False)
    directory = tmp_path/'run'/'epoch01'
    cp_path = directory/'segments'/'segment000'/'checkpoints'/'before-validation.pt'
    cp_path.parent.mkdir(parents=True)
    cp_path.write_bytes(Path(cp['path']).read_bytes())
    cp['path'] = str(cp_path)
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    m = dict(phase='capacity384_multiepoch_formal', formal_training_ready=True,
        run_directory=str(directory.parent), source_git_commit='source', expected_initial_state_sha256='initial')
    ctx = SimpleNamespace(binding=s['binding'])
    train = dict(status='passed', epoch=1, manifest_sha256=sha(manifest), source_git_commit='source',
        checkpoint=cp, checkpoint_binding=s['binding'], final_progress=s['progress'])
    trainref = write_json(directory/'training_result.json', train)
    receipt = dict(status='passed_training', awaiting_validation=True, source_git_commit='source',
        manifest_sha256=sha(manifest), test_records_read=False, automatic_successor=False,
        completed_training_passes=1, completed_validation_passes=0, report=trainref,
        checkpoint=cp, checkpoint_binding=s['binding'], final_progress=s['progress'],
        counts={key:s['progress'][key] for key in ('attempts', 'updates', 'skipped_updates', 'prefix_count', 'final_count', 'data_chain')})
    write_json(directory/'training_acceptance.json', receipt)
    contract = SimpleNamespace(FORMAL_PHASE=m['phase'], binding=lambda _: (m,c),
                               checkpoint_binding=lambda _: copy.deepcopy(s['binding']))
    trainmodule = SimpleNamespace(build_context=lambda *args, **kwargs: ctx)
    monkeypatch.setitem(sys.modules, 'scripts.liger_capacity_multiepoch_manifest', contract)
    monkeypatch.setattr(scripts, 'liger_capacity_multiepoch_manifest', contract, raising=False)
    monkeypatch.setitem(sys.modules, 'scripts.train_liger_capacity_multiepoch', trainmodule)
    monkeypatch.setattr(scripts, 'train_liger_capacity_multiepoch', trainmodule, raising=False)
    monkeypatch.setenv('SLURM_JOB_ID', 'val-job')
    def load(actual, actual_cp, *, expected_progress):
        assert actual is ctx and actual_cp == cp and expected_progress == s['progress']
        return dict(progress=copy.deepcopy(s['progress']), load_scope='model_weights_only')
    def evaluate(actual, output, records, actual_cp):
        assert actual is ctx and records == 347416 and actual_cp == cp
        v = evidence(tmp_path, cp, c, diagnostic=False)
        r = json.loads(Path(v['report']['path']).read_text())
        a = json.loads(Path(v['acceptance']['path']).read_text())
        rref = write_json(output/'report.json', r)
        a['report'] = rref
        write_json(output/'acceptance.json', a)
        return r, a, dict(model_weights_unchanged=True)
    monkeypatch.setattr(val, 'load_model_checkpoint', load)
    monkeypatch.setattr(val, 'evaluate_context', evaluate)
    return SimpleNamespace(c=c, m=m, state=s, cp=cp, directory=directory,
                           manifest=manifest, receipt=receipt, contract=contract,
                           prior_epoch_state=prior_epoch_state)


def test_formal_runner_publishes_resumable_state_and_no_automatic_next_job(formal):
    h = formal
    val.run(h.manifest, 1)
    accepted = json.loads((h.directory/'acceptance.json').read_text())
    result = json.loads((h.directory/'result.json').read_text())
    assert accepted['status'] == result['status'] == 'passed'
    assert accepted['checkpoint'] == accepted['next_checkpoint'] == result['next_checkpoint']
    assert accepted['next_progress']['epoch'] == 1
    assert accepted['next_progress']['updates'] == 22460
    assert accepted['policy']['completed_epochs'] == 1 and not accepted['stopped']
    assert accepted['requires_manual_next_job'] and not accepted['automatic_successor']
    assert not accepted['test_records_read']
    assert sha(accepted['report']['path']) == accepted['report']['sha256']


def test_validation_handoff_is_consumed_by_real_next_epoch_reader(formal):
    h = formal
    val.run(h.manifest, 1)
    accepted = json.loads((h.directory/'acceptance.json').read_text())
    p, cp, receipt_ref = h.prior_epoch_state(h.manifest, h.m, h.c, 2)
    assert p == accepted['next_progress'] and cp == accepted['next_checkpoint']
    assert receipt_ref == val.entry(h.directory/'acceptance.json')
    state = torch.load(cp['path'], weights_only=False)
    assert state['progress'] == p and state['scheduler'] == h.state['scheduler']


@pytest.mark.parametrize('field', ['status', 'manifest', 'source', 'awaiting', 'test', 'auto',
                                  'completed_training', 'completed_validation', 'counts', 'report_bytes'])
def test_training_receipt_rejects_drift_before_validation(formal, field):
    h = formal
    r = h.receipt
    if field == 'status': r['status'] = 'failed'
    elif field == 'manifest': r['manifest_sha256'] = 'wrong'
    elif field == 'source': r['source_git_commit'] = 'wrong'
    elif field == 'awaiting': r['awaiting_validation'] = False
    elif field == 'test': r['test_records_read'] = True
    elif field == 'auto': r['automatic_successor'] = True
    elif field == 'completed_training': r['completed_training_passes'] = 2
    elif field == 'completed_validation': r['completed_validation_passes'] = 1
    elif field == 'counts': r['counts']['updates'] -= 1
    else:
        with open(r['report']['path'], 'a') as f: f.write(' ')
    write_json(h.directory/'training_acceptance.json', r)
    with pytest.raises((AssertionError, ValueError)):
        val.run(h.manifest, 1)
    assert not (h.directory/'acceptance.json').exists()


def test_validation_failure_does_not_promote_training_checkpoint(formal, monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError('GPU validation failed')
    monkeypatch.setattr(val, 'evaluate_context', fail)
    with pytest.raises(RuntimeError): val.run(formal.manifest, 1)
    assert json.loads((formal.directory/'result.json').read_text())['status'] == 'failed'
    assert not (formal.directory/'acceptance.json').exists()
    assert not (formal.directory/'after-validation.pt').exists()


def test_completed_epoch_validation_cannot_be_repeated(formal):
    val.run(formal.manifest, 1)
    with pytest.raises(AssertionError): val.run(formal.manifest, 1)
