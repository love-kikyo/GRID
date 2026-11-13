"""Formal 384 training boundary: accounting, immutable handoff and CPU audit."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import train_liger_capacity_cpu4 as runner
from src.baselines.liger_prefix_training import initial_progress
from src.baselines.liger_sampled_catalog import sampling_binding
from src.baselines.liger_state import BudgetSchedule, tree_digest


def config():
    return dict(training=dict(batch_size=2, physical_batch_size=1, accumulation_steps=2,
        prefixes_per_epoch=5, parents_per_epoch=3, batches_per_epoch=3, final_batch_size=1,
        max_optimizer_steps=3, max_attempts=3, warmup_steps=1, checkpoint_every_batches=2),
        optimizer=dict(learning_rate=.0003, weight_decay=.035),
        dense_sampling=dict(num_negatives=2, seed=42))


def synthetic_steps(*, skipped=False):
    c = config()
    count = updates = finals = 0
    chain = '0'*64
    result = []
    for attempt, rows in enumerate((2, 2, 1), 1):
        skip = skipped and attempt == 2
        identity = hashlib.sha256(str(attempt).encode()).hexdigest()
        chain = hashlib.sha256((chain+identity).encode()).hexdigest()
        lr = runner.next_learning_rate(updates, c)
        updates += not skip
        count += rows
        finals += 1
        result.append(dict(epoch=1, attempt=attempt, batch=attempt, rows=rows,
            microbatch=1, microbatches=rows, final_target_count=1,
            candidate_pool_sha256=hashlib.sha256(f'pool-{attempt}'.encode()).hexdigest(),
            candidate_classes=rows+2, prefix_identity_sha256=identity,
            dense_objective=sampling_binding(c['dense_sampling']), sampling_attempt=attempt-1,
            gradient_tensors=152, skipped=skip, sid_loss=2., dense_loss=3., total_loss=5.,
            learning_rate=lr, scale_before=32., scale_after=16. if skip else 32.,
            grad_norm=None if skip else .5, updates=updates, cumulative_prefixes=count,
            cumulative_final_targets=finals, data_chain=chain,
            input_seconds=.1, training_seconds=.2, seconds=.3,
            peak_allocated_bytes=10, peak_reserved_bytes=20))
    return result


def write_steps(path, steps):
    path.write_text(''.join(json.dumps(s)+'\n' for s in steps))


def complete_progress(*, skipped=False):
    steps = synthetic_steps(skipped=skipped)
    p = initial_progress(max_optimizer_steps=3, max_attempts=3)
    p.update(next_batch=3, samples_in_epoch=5, batches_consumed=3, attempts=3,
             prefix_count=5, final_count=3, updates=3-int(skipped), skipped_updates=int(skipped),
             data_chain=steps[-1]['data_chain'])
    return p


@pytest.mark.parametrize('skipped', [False, True])
def test_audit_counts_tail_microbatches_amp_and_schedule(tmp_path, skipped):
    path = tmp_path/'steps.jsonl'
    write_steps(path, synthetic_steps(skipped=skipped))
    actual = runner.audit_step_log(path, config(), seen_rows=10)
    assert actual['attempts'] == 3 and actual['prefix_count'] == 5 and actual['final_count'] == 3
    assert actual['updates'] == 3-skipped and actual['skipped_updates'] == skipped
    assert actual['data_chain'] == synthetic_steps()[-1]['data_chain']
    assert actual['input_seconds'] == pytest.approx(.3)
    assert actual['training_seconds'] == pytest.approx(.6)
    assert actual['total_step_seconds'] == pytest.approx(.9)
    assert actual['checkpoint_cursors'][2]['updates'] == 2-skipped
    assert actual['checkpoint_cursors'][2]['prefix_count'] == 4


@pytest.mark.parametrize('change', ['tail', 'missing', 'extra', 'parent_count', 'lr', 'sampling',
    'sampling_attempt', 'identity', 'pool_hash', 'pool_few', 'pool_many', 'gradient', 'loss',
    'skip', 'scale', 'updates', 'cumulative', 'seconds', 'memory', 'nan', 'micro', 'micro_count',
    'chain', 'bool_count', 'bool_loss', 'negative_time'])
def test_audit_rejects_corrupt_log_evidence(tmp_path, change):
    steps = synthetic_steps()
    s = steps[-1]
    if change == 'tail': s['rows'] = 2
    elif change == 'missing': steps.pop()
    elif change == 'extra': steps.append(copy.deepcopy(s))
    elif change == 'parent_count': s['final_target_count'] = 0
    elif change == 'lr': s['learning_rate'] = .0003
    elif change == 'sampling': s['dense_objective']['num_negatives'] = 9
    elif change == 'sampling_attempt': s['sampling_attempt'] = 0
    elif change == 'identity': s['prefix_identity_sha256'] = 'bad'
    elif change == 'pool_hash': s['candidate_pool_sha256'] = 'bad'
    elif change == 'pool_few': s['candidate_classes'] = 2
    elif change == 'pool_many': s['candidate_classes'] = 4
    elif change == 'gradient': s['gradient_tensors'] = 150
    elif change == 'loss': s['total_loss'] = 6.
    elif change == 'skip': s['skipped'] = True
    elif change == 'scale': s['scale_after'] = 1.
    elif change == 'updates': s['updates'] = 2
    elif change == 'cumulative': s['cumulative_prefixes'] = 4
    elif change == 'seconds': s['seconds'] = .2
    elif change == 'memory': s['peak_reserved_bytes'] = 5
    elif change == 'nan': s['training_seconds'] = float('nan')
    elif change == 'micro': s['microbatch'] = 2
    elif change == 'micro_count': s['microbatches'] = 2
    elif change == 'chain': s['data_chain'] = '0'*64
    elif change == 'bool_count': s['rows'] = True
    elif change == 'bool_loss': s['sid_loss'] = True
    else: s['input_seconds'] = -.1
    path = tmp_path/'steps.jsonl'
    write_steps(path, steps)
    with pytest.raises(AssertionError): runner.audit_step_log(path, config(), seen_rows=10)


def test_complete_training_does_not_advance_epoch_or_select_model():
    p = complete_progress()
    before = copy.deepcopy(p)
    runner.check_training_complete(p, config())
    assert p == before and p['epoch'] == 0 and p['next_batch'] == 3
    assert p['best'] is None and p['validations'] == []


@pytest.mark.parametrize('key,value', [('epoch', 1), ('next_batch', 2), ('prefix_count', 4),
    ('final_count', 2), ('samples_in_epoch', 4), ('best', {'epoch': 1}), ('validations', [{}]),
    ('updates', 0)])
def test_complete_training_rejects_partial_or_falsely_validated_state(key, value):
    p = complete_progress()
    p[key] = value
    with pytest.raises((AssertionError, ValueError)): runner.check_training_complete(p, config())


def checkpoint_fixture(tmp_path):
    c, p = config(), complete_progress()
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([.1])) for _ in range(152)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.035)
    scheduler = BudgetSchedule(optimizer, total=3, warmup=1, base_lr=.0003)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.sum() for parameter in model).backward()
        optimizer.step(); scheduler.step()
    state = dict(format_version=1, binding={'purpose': 'unit'}, model=model.state_dict(),
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler={'scale': 32.},
        rng={'cuda': [torch.zeros(4, dtype=torch.uint8)]}, progress=p, model_training=True)
    path = tmp_path/'checkpoint.pt'
    torch.save(state, path)
    entry = dict(path=str(path), sha256=runner.sha(path), full_state_sha256=tree_digest(state),
                 epoch=0, next_batch=3, updates=3, attempts=3)
    return c, state, entry


def test_checkpoint_audit_checks_full_state_immutable_identity_and_optimizer(tmp_path):
    c, state, entry = checkpoint_fixture(tmp_path)
    p = runner.audit_checkpoint(entry, entry['path'], state['progress'], c, state['binding'])
    assert p == complete_progress()


@pytest.mark.parametrize('change', ['purpose', 'nonfinite_model', 'nonfinite_optimizer', 'state_hash',
    'schedule', 'lr', 'wd', 'optimizer_count', 'optimizer_step', 'cuda_count', 'scaler', 'mode',
    'cursor', 'bytes', 'path', 'entry_cursor'])
def test_checkpoint_audit_rejects_inconsistent_state(tmp_path, change):
    c, state, entry = checkpoint_fixture(tmp_path)
    expected_p = copy.deepcopy(state['progress'])
    binding, path = {'purpose': 'unit'}, Path(entry['path'])
    if change == 'purpose': state['binding']['purpose'] = 'diagnostic'
    elif change == 'nonfinite_model': state['model']['0'].fill_(float('nan'))
    elif change == 'nonfinite_optimizer': state['optimizer']['state'][0]['exp_avg'].fill_(float('inf'))
    elif change == 'state_hash': entry['full_state_sha256'] = '0'*64
    elif change == 'schedule': state['scheduler']['total'] = 4
    elif change == 'lr': state['optimizer']['param_groups'][0]['lr'] = 1.
    elif change == 'wd': state['optimizer']['param_groups'][0]['weight_decay'] = 1.
    elif change == 'optimizer_count': state['optimizer']['state'].pop(0)
    elif change == 'optimizer_step': state['optimizer']['state'][0]['step'].fill_(2)
    elif change == 'cuda_count': state['rng']['cuda'].append(torch.zeros(4))
    elif change == 'scaler': state['scaler']['scale'] = float('nan')
    elif change == 'mode': state['model_training'] = False
    elif change == 'cursor': state['progress']['final_count'] = 2
    elif change == 'path': entry['path'] = str(tmp_path/'wrong.pt')
    elif change == 'entry_cursor': entry['updates'] = 2
    torch.save(state, path)
    if change != 'state_hash': entry['full_state_sha256'] = tree_digest(state)
    entry['sha256'] = '0'*64 if change == 'bytes' else runner.sha(path)
    with pytest.raises((AssertionError, ValueError)):
        runner.audit_checkpoint(entry, path, expected_p, c, binding)


@pytest.fixture
def run_fixture(monkeypatch, tmp_path):
    c = config()
    manifest_path = tmp_path/'manifest.json'
    manifest_path.write_text('{}')
    asset = tmp_path/'asset.bin'; asset.write_bytes(b'fixed unit asset')
    m = dict(phase=runner.contract.FORMAL_PHASE, formal_training_ready=True,
        run_directory=str(tmp_path/'run'), source_git_commit='a'*40,
        initialization='fresh', expected_initial_state_sha256='b'*64,
        expected_feature_cache_sha256='c'*64, assets={'content': {'path':str(asset),'sha256':runner.sha(asset)}})
    monkeypatch.setenv('SLURM_JOB_ID', '12345')
    monkeypatch.setattr(runner.contract, 'binding', lambda path: (m, c))
    monkeypatch.setattr(runner.contract, 'checkpoint_binding', lambda m: {'purpose': 'unit'})
    ctx = SimpleNamespace(initial_sha='b'*64, binding={'purpose': 'unit'},
        parameter_report=dict(total_params=41868160, trainable_params=41868160,
            backbone_params=34222464, parameter_tensors=152, tied_sid_input_output=True),
        resources=dict(allocated_cpus=4,user_cpu_ceiling=4,library_threads=1,allowed_cpus=[2]),
        gpus=[dict(name='4090',uuid='unit',total_bytes=10000)], versions={'numpy':'unit'},
        numpy_policy={'unit':True}, feature_cache_report=dict(
            build={'cache_sha256':'c'*64,'cache_bytes':5909890560},
            verification={'cache_sha256':'c'*64,'cache_bytes':5909890560,'status':'passed','feature_values_bitwise_equal':True}),
        setup_seconds=.5, seen=range(10),cold=range(10),cpu_checks=[])
    def cpu_check():
        value = dict(allowed_cpus=[2],processes=[dict(threads_checked=1)])
        ctx.cpu_checks.append(value)
        return value
    ctx.cpu_check=cpu_check;cpu_check()
    ctx.batch_calls=[]
    def batches(**kwargs):
        ctx.batch_calls.append(kwargs)
        assert kwargs == dict(training=True,epoch=0,start_batch=0)
        yield from enumerate(([None]*2,[None]*2,[None]))
    ctx.batches=batches
    monkeypatch.setattr(runner,'build_context',lambda manifest:ctx)
    steps=synthetic_steps()
    def logical_step(context,records,p,microbatch):
        step=copy.deepcopy(steps[p['attempts']])
        assert microbatch==1 and len(records)==step['rows']
        p.update(attempts=step['attempt'],batches_consumed=step['attempt'],next_batch=step['attempt'],
            updates=step['updates'],prefix_count=step['cumulative_prefixes'],
            samples_in_epoch=step['cumulative_prefixes'],final_count=step['cumulative_final_targets'],
            data_chain=step['data_chain'])
        return step
    monkeypatch.setattr(runner,'logical_step',logical_step)
    saved=[]
    def save(ctx,p,path):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(p))
        entry=dict(path=str(path),sha256=runner.sha(path),full_state_sha256='d'*64,
            epoch=p['epoch'],next_batch=p['next_batch'],updates=p['updates'],attempts=p['attempts'])
        saved.append((entry,copy.deepcopy(p)))
        return entry
    monkeypatch.setattr(runner,'save_bound_checkpoint',save)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:10)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:20)
    monkeypatch.setattr(runner,'accept_cpu',lambda:(ctx.resources,cpu_check))
    import scripts.accept_liger_sampled_scaling as old_accept
    monkeypatch.setattr(old_accept,'validate_numpy_policy',lambda *args:None)
    return SimpleNamespace(path=manifest_path,m=m,c=c,ctx=ctx,saved=saved)


def test_formal_run_reads_only_one_training_pass_and_hands_off_without_validation(run_fixture):
    f=run_fixture
    runner.run(f.path)
    directory=Path(f.m['run_directory'])
    r=json.loads((directory/'training_result.json').read_text())
    assert r['status']=='passed' and r['awaiting_validation']
    assert r['completed_training_passes']==1 and r['completed_validation_passes']==0
    assert r['final_progress']==complete_progress()
    assert [Path(e['path']).name for e,_ in f.saved]==['pass01-batch000002.pt','pass01-before-validation.pt']
    assert not (directory/'training_acceptance.json').exists()
    assert not any('validation' in p.name for p in directory.iterdir())
    assert f.ctx.batch_calls==[dict(training=True,epoch=0,start_batch=0)]
    assert r['initial_progress']['attempts']==0 and r['initial_progress']['updates']==0
    runner.audit_step_log(directory/'steps.jsonl',f.c,seen_rows=10)
    with pytest.raises(FileExistsError):runner.run(f.path)


def test_failed_training_preserves_failed_report_and_no_acceptance(run_fixture,monkeypatch):
    f=run_fixture
    monkeypatch.setattr(runner,'logical_step',lambda *args:(_ for _ in ()).throw(RuntimeError('unit failure')))
    with pytest.raises(RuntimeError,match='unit failure'):runner.run(f.path)
    directory=Path(f.m['run_directory'])
    r=json.loads((directory/'training_result.json').read_text())
    assert r['status']=='failed' and 'unit failure' in r['error']
    assert not (directory/'training_acceptance.json').exists() and not f.saved


def test_gate_manifest_cannot_run_formal_training(run_fixture):
    f=run_fixture;f.m['formal_training_ready']=False
    with pytest.raises(AssertionError):runner.run(f.path)
    assert not Path(f.m['run_directory']).exists()


def test_training_acceptance_independently_audits_checkpoints_and_publishes_handoff(run_fixture,monkeypatch):
    f=run_fixture;runner.run(f.path)
    audited=[]
    def audit(entry,path,progress,c,binding):
        audited.append(Path(path).name)
        assert binding=={'purpose':'unit'}
        return json.loads(Path(path).read_text())
    monkeypatch.setattr(runner,'audit_checkpoint',audit)
    receipt=runner.accept(f.path)
    assert receipt['status']=='passed_training' and receipt['awaiting_validation']
    assert receipt['checkpoint']==f.saved[-1][0]
    assert receipt['counts']['prefix_count']==5 and receipt['counts']['final_count']==3
    assert receipt['completed_validation_passes']==0
    assert audited==['pass01-batch000002.pt','pass01-before-validation.pt']
    assert receipt['asset_verification']['files']=={v['path']:v['sha256'] for v in f.m['assets'].values()}
    with pytest.raises(FileExistsError):runner.accept(f.path)


@pytest.mark.parametrize('change',['init','job','source','purpose','validation','automatic','params',
    'cpu','gpu','cache','asset','checks','missing_ckpt','selected_ckpt','final_count','status','memory'])
def test_acceptance_rejects_report_or_handoff_drift(run_fixture,monkeypatch,change):
    f=run_fixture;runner.run(f.path)
    path=Path(f.m['run_directory'])/'training_result.json';r=json.loads(path.read_text())
    if change=='init':r['initial_state_sha256']='0'*64
    elif change=='job':r['job_id']='other'
    elif change=='source':r['source_git_commit']='0'*40
    elif change=='purpose':r['checkpoint_binding']['purpose']='diagnostic'
    elif change=='validation':r['completed_validation_passes']=1
    elif change=='automatic':r['automatic_successor']=True
    elif change=='params':r['parameters']['total_params']=10070912
    elif change=='cpu':r['resources']['allocated_cpus']=8
    elif change=='gpu':r['gpus'].append(copy.deepcopy(r['gpus'][0]))
    elif change=='cache':r['feature_cache']['verification']['cache_sha256']='0'*64
    elif change=='asset':r['asset_verification']['files']={}
    elif change=='checks':r['cpu_checks']=[]
    elif change=='missing_ckpt':r['checkpoints'].pop(0)
    elif change=='selected_ckpt':r['checkpoint']=r['checkpoints'][0]
    elif change=='final_count':r['final_progress']['final_count']=2
    elif change=='status':r['status']='failed'
    else:r['gpus'][0]['total_bytes']=1
    path.write_text(json.dumps(r))
    monkeypatch.setattr(runner,'audit_checkpoint',lambda entry,path,progress,c,binding:json.loads(Path(path).read_text()))
    with pytest.raises((AssertionError,ValueError)):runner.accept(f.path)
    assert not path.with_name('training_acceptance.json').exists()


@pytest.mark.parametrize('devices',[1,2])
def test_context_reuses_exact_runtime_and_only_rebinds_purpose(tmp_path,monkeypatch,devices):
    import scripts.liger_capacity_runtime as resource_runtime
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/baselines/liger_capacity384_train_seed42_v1.json').read_text())
    formal=tmp_path/'formal.json';formal.write_text(json.dumps(c))
    capacity=tmp_path/'capacity.json';capacity.write_text(json.dumps({'model':c['model']}))
    m=dict(formal_config={'path':str(formal)},capacity_config={'path':str(capacity)},
        expected_initial_state_sha256='a'*64,standardization={'sha256':'b'*64})
    expected=dict(purpose='formal-unit',standardization_sha256='b'*64,
        standardization_stats_sha256='c'*64,standardization_seen_rows_sha256='d'*64)
    ctx=SimpleNamespace(initial_sha='a'*64,parameter_report=dict(total_params=41868160,
        backbone_params=34222464,parameter_tensors=152),config=copy.deepcopy(c),
        standardization={'stats_sha256':'c'*64,'seen_rows_sha256':'d'*64},
        binding={'purpose':'old-diagnostic'},tf=SimpleNamespace(__version__='unit'))
    calls=[]
    def create(manifest,config,*,microbatch,devices):
        calls.append((manifest,config,microbatch,devices));return ctx
    monkeypatch.setattr(resource_runtime,'build_context',create)
    monkeypatch.setattr(runner.contract,'checkpoint_binding',lambda manifest:expected)
    assert runner.build_context(m,devices=devices) is ctx
    assert calls==[(m,{'model':c['model']},1024,devices)]
    assert ctx.binding==expected and ctx.formal_config==c
    ctx.initial_sha='0'*64
    with pytest.raises(AssertionError,match='initialization'):runner.build_context(m,devices=devices)


def test_frozen_assets_verified_bytewise_not_only_by_metadata(tmp_path):
    path=tmp_path/'asset';path.write_bytes(b'correct')
    m={'assets':{'item':{'path':str(path),'sha256':runner.sha(path)}}}
    proof=runner.verify_asset_bytes(m)
    assert proof['files']=={str(path):runner.sha(path)} and proof['seconds']>0
    path.write_bytes(b'changed')
    with pytest.raises(AssertionError,match='Frozen asset bytes changed'):runner.verify_asset_bytes(m)
