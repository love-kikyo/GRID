"""Global learning schedule, epoch accounting and immutable interruption logs."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import train_liger_capacity_multiepoch as r
from src.baselines.liger_early_stopping import advance_policy
from src.baselines.liger_sampled_catalog import sampling_binding
from src.baselines.liger_state import BudgetSchedule, tree_digest


def config():
    return dict(training=dict(batch_size=2, physical_batch_size=1, accumulation_steps=2,
        prefixes_per_epoch=5, parents_per_epoch=3, batches_per_epoch=3, final_batch_size=1,
        max_optimizer_steps=36, max_attempts=36, warmup_steps=1, checkpoint_every_batches=2,
        min_epochs=6, max_epochs=12), optimizer=dict(learning_rate=.0003, weight_decay=.035),
        dense_sampling=dict(num_negatives=2, seed=42), validation=dict(records=7),
        early_stopping=dict(monitor='hybrid Recall@5', mode='max', min_delta=0.,
                            patience=5, min_epochs=6, max_epochs=12))


def validation(epoch=1, recall=.1):
    ref = dict(path=f'/mock/epoch{epoch}.json', sha256='a'*64)
    return dict(epoch=epoch, checkpoint=ref, report=ref, acceptance=ref,
                metrics=dict(hybrid={'Recall@5':recall, 'NDCG@5':.05}))


def finish(p, c):
    p = copy.deepcopy(p)
    policy = advance_policy(p['early_stopping'], validation(p['epoch']+1), c)
    p.update(epoch=p['epoch']+1, next_batch=0, samples_in_epoch=0,
             early_stopping=policy, best=policy['best'], validations=policy['validations'])
    return p


def steps(initial=None, *, skip_batch=None, start=1, stop=3):
    c = config(); p = copy.deepcopy(initial or r.new_progress(c))
    out = []
    for batch in range(start, stop+1):
        rows = min(2, 5-p['samples_in_epoch'])
        identity = hashlib.sha256(f'{p["epoch"]}:{batch}'.encode()).hexdigest()
        lr = r.next_learning_rate(p['updates'], c)
        skip = batch == skip_batch
        p['attempts'] += 1; p['batches_consumed'] += 1; p['next_batch'] += 1
        p['updates'] += not skip; p['skipped_updates'] += skip
        p['prefix_count'] += rows; p['samples_in_epoch'] += rows; p['final_count'] += 1
        p['data_chain'] = hashlib.sha256((p['data_chain']+identity).encode()).hexdigest()
        out.append(dict(epoch=p['epoch']+1, attempt=p['attempts'], batch=batch, rows=rows,
            microbatch=1, microbatches=rows, final_target_count=1,
            candidate_pool_sha256=hashlib.sha256(f'pool-{p["attempts"]}'.encode()).hexdigest(),
            candidate_classes=rows+2, prefix_identity_sha256=identity,
            dense_objective=sampling_binding(c['dense_sampling']), sampling_attempt=p['attempts']-1,
            gradient_tensors=152, skipped=skip, sid_loss=2., dense_loss=3., total_loss=5.,
            learning_rate=lr, scale_before=32., scale_after=16. if skip else 32.,
            grad_norm=None if skip else .5, updates=p['updates'], cumulative_prefixes=p['prefix_count'],
            cumulative_final_targets=p['final_count'], data_chain=p['data_chain'],
            input_seconds=.1, training_seconds=.2, seconds=.3,
            peak_allocated_bytes=10, peak_reserved_bytes=20))
    return out, p


def write_steps(path, rows):
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))


@pytest.mark.parametrize('skip', [None, 2, 3])
def test_two_epochs_preserve_schedule_skip_and_tail_accounting(tmp_path, skip):
    c = config(); p = r.new_progress(c)
    for epoch in (1, 2):
        records, end = steps(p, skip_batch=skip)
        f = tmp_path/f'{epoch}.jsonl'; write_steps(f, records)
        audit = r.audit_step_log(f, c, p, complete=True, seen_rows=10)
        assert audit['progress'] == end
        assert end['attempts'] == epoch*3 and end['prefix_count'] == epoch*5
        assert end['final_count'] == epoch*3
        assert audit['steps'] == 3 and audit['checkpoint_cursors'][2]['next_batch'] == 2
        assert records[0]['sampling_attempt'] == (epoch-1)*3
        assert records[0]['learning_rate'] == r.next_learning_rate(p['updates'], c)
        r.check_training_complete(end, c, epoch)
        p = finish(end, c)
        r.check_progress_state(p, c)
    assert p['epoch'] == 2 and p['early_stopping']['bad_epochs'] == 1


@pytest.mark.parametrize('key', ['attempt', 'batch', 'epoch', 'rows', 'microbatch', 'microbatches',
    'gradient_tensors', 'updates', 'cumulative_prefixes', 'cumulative_final_targets',
    'sampling_attempt', 'final_target_count', 'candidate_classes'])
def test_log_integer_corruption_rejected(tmp_path, key):
    rows, _ = steps(); rows[-1][key] += 1
    f=tmp_path/'steps.jsonl'; write_steps(f, rows)
    with pytest.raises((AssertionError, ValueError)):
        r.audit_step_log(f, config(), r.new_progress(config()), complete=True, seen_rows=10)


@pytest.mark.parametrize('change', ['lr', 'identity', 'pool', 'chain', 'loss', 'nan', 'skip',
    'scale', 'time', 'memory', 'missing', 'extra', 'partial', 'sampling'])
def test_log_evidence_corruption_rejected(tmp_path, change):
    rows, _ = steps(); last=rows[-1]
    if change=='lr': last['learning_rate']=.7
    elif change=='identity': last['prefix_identity_sha256']='bad'
    elif change=='pool': last['candidate_pool_sha256']='bad'
    elif change=='chain': last['data_chain']='0'*64
    elif change=='loss': last['total_loss']=6.
    elif change=='nan': last['sid_loss']=float('nan')
    elif change=='skip': last['skipped']=True
    elif change=='scale': last['scale_after']=1.
    elif change=='time': last['seconds']=.2
    elif change=='memory': last['peak_reserved_bytes']=1
    elif change=='missing': rows.pop()
    elif change=='extra': rows.append(copy.deepcopy(last))
    elif change=='sampling': last['dense_objective']['num_negatives']=3
    f=tmp_path/'steps.jsonl'; write_steps(f, rows)
    if change=='partial': f.write_text(f.read_text().rstrip('\n'))
    with pytest.raises((AssertionError, ValueError)):
        r.audit_step_log(f, config(), r.new_progress(config()), complete=True, seen_rows=10)


def test_interrupted_tail_ignored_only_with_explicit_checkpoint_cursor(tmp_path):
    c=config(); rows, end=steps(); f=tmp_path/'steps.jsonl'
    write_steps(f, rows); f.write_text(f.read_text()+'{partial killed write')
    partial=r.audit_step_log(f,c,r.new_progress(c),stop_batch=2,seen_rows=10)
    assert partial['progress']['next_batch']==2 and partial['steps']==2
    resumed_rows, expected=steps(partial['progress'],start=3)
    g=tmp_path/'resumed.jsonl'; write_steps(g,resumed_rows)
    actual=r.audit_step_log(g,c,partial['progress'],complete=True,seen_rows=10)
    assert actual['progress']==expected==end
    assert f.read_text().endswith('{partial killed write')
    with pytest.raises((AssertionError, ValueError)):
        r.audit_step_log(f,c,r.new_progress(c),complete=True,seen_rows=10)


def test_empty_final_segment_can_republish_completed_cursor(tmp_path):
    _, end=steps(); f=tmp_path/'empty'; f.write_text('')
    actual=r.audit_step_log(f,config(),end,complete=True,seen_rows=10)
    assert actual['steps']==0 and actual['progress']==end


@pytest.mark.parametrize('change', ['epoch','policy_count','bad_count','best','history','budget'])
def test_policy_progress_and_budget_cannot_drift(change):
    c=config(); _,p=steps(); p=finish(p,c)
    if change=='epoch': p['epoch']=2
    elif change=='policy_count': p['early_stopping']['completed_epochs']=2
    elif change=='bad_count': p['early_stopping']['bad_epochs']=4
    elif change=='best': p['best']=None
    elif change=='history': p['validations']=[]
    else: p['max_optimizer_steps']=37
    with pytest.raises((AssertionError, ValueError)): r.check_progress_state(p,c)


def checkpoint_fixture(tmp_path, epoch=1):
    c=config(); p=r.new_progress(c)
    for i in range(epoch):
        _,p=steps(p)
        if i+1<epoch:p=finish(p,c)
    model=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([.1])) for _ in range(152)])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.035)
    scheduler=BudgetSchedule(optimizer,total=36,warmup=1,base_lr=.0003)
    for _ in range(p['updates']):
        optimizer.zero_grad(set_to_none=True);sum(x.sum() for x in model).backward()
        optimizer.step();scheduler.step()
    state=dict(format_version=1,binding={'purpose':'unit'},model=model.state_dict(),
        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler={'scale':32.},
        rng={'cuda':[torch.zeros(4,dtype=torch.uint8)]},progress=p,model_training=True)
    file=tmp_path/'checkpoint.pt';torch.save(state,file)
    ref=dict(path=str(file),sha256=r.sha(file),full_state_sha256=tree_digest(state),
        **{k:p[k] for k in ('epoch','next_batch','updates','attempts')})
    return c,p,state,ref


@pytest.mark.parametrize('epoch',[1,2])
def test_checkpoint_audit_restored_global_optimizer_and_policy(tmp_path,epoch):
    c,p,state,ref=checkpoint_fixture(tmp_path,epoch)
    assert r.audit_checkpoint(ref,ref['path'],p,c,state['binding'])==p


@pytest.mark.parametrize('change',['optimizer_step','moment_nan','lr','schedule_total','rng_devices',
    'scaler','mode','policy','binding','model_nan','state_digest'])
def test_checkpoint_corruption_rejected(tmp_path,change):
    c,p,state,ref=checkpoint_fixture(tmp_path,2)
    if change=='optimizer_step':state['optimizer']['state'][0]['step']+=1
    elif change=='moment_nan':state['optimizer']['state'][0]['exp_avg'][0]=float('nan')
    elif change=='lr':state['optimizer']['param_groups'][0]['lr']=1.
    elif change=='schedule_total':state['scheduler']['total']=3
    elif change=='rng_devices':state['rng']['cuda']*=2
    elif change=='scaler':state['scaler']['scale']=0.
    elif change=='mode':state['model_training']=False
    elif change=='policy':state['progress']['early_stopping']['bad_epochs']=5
    elif change=='binding':state['binding']={'purpose':'old_one_pass'}
    elif change=='model_nan':state['model']['0'][0]=float('nan')
    torch.save(state,ref['path']);ref['sha256']=r.sha(ref['path'])
    ref['full_state_sha256']=tree_digest(state) if change!='state_digest' else '0'*64
    with pytest.raises((AssertionError,ValueError)):
        r.audit_checkpoint(ref,ref['path'],p,c,{'purpose':'unit'})


def test_checkpoint_path_must_belong_to_same_epoch_segment(tmp_path):
    d=tmp_path/'epoch01'; p=d/'segments/segment000/checkpoints/batch000002.pt'
    assert r.segment_from_checkpoint({'path':str(p)},d)==p.parent.parent
    with pytest.raises(AssertionError):r.segment_from_checkpoint({'path':str(p)},tmp_path/'epoch02')
    with pytest.raises(AssertionError):r.segment_from_checkpoint({'path':str(d/'elsewhere.pt')},d)


def test_initial_epoch_does_not_load_old_weights_or_validation():
    c=config(); p,cp,receipt=r.prior_epoch_state(None,{},c,1)
    assert p==r.new_progress(c) and cp is receipt is None
    assert p['updates']==p['epoch']==0 and p['early_stopping']['completed_epochs']==0


def test_epoch_bounds_and_path(tmp_path):
    m={'run_directory':str(tmp_path)}
    assert r.epoch_directory(m,config(),12)==tmp_path/'epoch12'
    for epoch in [0,13,True,1.5]:
        with pytest.raises(AssertionError):r.epoch_directory(m,config(),epoch)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return r.entry(path)


def test_segment_chain_commits_only_restored_prefix(tmp_path, monkeypatch):
    c=config(); manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    m=dict(run_directory=str(tmp_path/'run'), source_git_commit='commit',expected_initial_state_sha256='b'*64)
    binding={'purpose':'unit'}
    monkeypatch.setattr(r,'_contract',lambda:SimpleNamespace(checkpoint_binding=lambda _:binding))
    directory=r.epoch_directory(m,c,1); s0=directory/'segments/segment000';s1=directory/'segments/segment001'
    s0.mkdir(parents=True);s1.mkdir()
    p=r.new_progress(c); rows,end=steps()
    write_steps(s0/'steps.jsonl',rows)
    partial=r.audit_step_log(s0/'steps.jsonl',c,p,stop_batch=2)['progress']
    cp=s0/'checkpoints/batch000002.pt';cp.parent.mkdir();cp.write_bytes(b'fixture')
    ref=dict(path=str(cp),sha256=r.sha(cp),full_state_sha256='c'*64,
        **{k:partial[k] for k in ('epoch','next_batch','updates','attempts')})
    write_json(cp.with_suffix('.pt.json'),ref)
    common=dict(status='ready',epoch=1,manifest_sha256=r.sha(manifest),source_git_commit='commit',
                checkpoint_binding=binding,initial_state_sha256='b'*64,previous_acceptance=None)
    write_json(s0/'preflight.json',dict(common,initial_progress=p,loaded_checkpoint=None,resume_segment=None))
    write_json(s1/'preflight.json',dict(common,initial_progress=partial,loaded_checkpoint=ref,resume_segment='segment000'))
    resumed,_=steps(partial,start=3);write_steps(s1/'steps.jsonl',resumed)
    # An abandoned tail may even contain a truncated kill-time write. It is
    # preserved and hash-bound, but not interpreted as completed learning.
    with (s0/'steps.jsonl').open('a') as f:f.write('{unfinished')
    result=r.audit_segments(manifest,m,c,1,s1,checkpoint_checks=False)
    assert result['progress']==end and result['steps']==3
    assert len(result['segments'])==2 and len(result['checkpoints'])==1
    assert result['segments'][0]['ignored_tail_after_checkpoint']
    assert (s0/'steps.jsonl').read_text().endswith('{unfinished')
    pre=json.loads((s1/'preflight.json').read_text());pre['initial_progress']['updates']-=1
    write_json(s1/'preflight.json',pre)
    with pytest.raises((AssertionError,ValueError)):
        r.audit_segments(manifest,m,c,1,s1,checkpoint_checks=False)


def prior_fixture(tmp_path,monkeypatch):
    c=config();manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    m=dict(run_directory=str(tmp_path/'run'),source_git_commit='commit')
    binding={'purpose':'unit'}
    monkeypatch.setattr(r,'_contract',lambda:SimpleNamespace(checkpoint_binding=lambda _:binding))
    previous=r.epoch_directory(m,c,1); previous.mkdir(parents=True)
    oldcp=dict(path='/previous/train.pt',sha256='c'*64,full_state_sha256='d'*64,
               epoch=0,next_batch=3,attempts=3,updates=3)
    metrics={'hybrid':{'Recall@5':.1,'NDCG@5':.05}}
    vr=write_json(previous/'validation/report.json',dict(status='passed',records=7,checkpoint=oldcp,metrics=metrics))
    va=write_json(previous/'validation/acceptance.json',dict(status='passed',records=7,checkpoint=oldcp,metrics=metrics))
    _,p=steps()
    v=dict(epoch=1,checkpoint=oldcp,report=vr,acceptance=va,metrics=metrics)
    policy=advance_policy(p['early_stopping'],v,c)
    p.update(epoch=1,next_batch=0,samples_in_epoch=0,early_stopping=policy,best=policy['best'],validations=policy['validations'])
    cp=previous/'after-validation.pt';cp.write_bytes(b'bound complete state fixture')
    ref=dict(path=str(cp),sha256=r.sha(cp),full_state_sha256='e'*64,
             **{k:p[k] for k in ('epoch','next_batch','updates','attempts')})
    report=write_json(previous/'result.json',dict(status='passed',epoch=1,manifest_sha256=r.sha(manifest),
        source_git_commit='commit',next_checkpoint=ref,next_progress=p))
    receipt=dict(status='passed',epoch=1,completed_passes=1,manifest_sha256=r.sha(manifest),
        source_git_commit='commit',checkpoint_binding=binding,test_records_read=False,automatic_successor=False,
        stopped=False,checkpoint=ref,next_progress=p,policy=policy,report=report)
    write_json(previous/'acceptance.json',receipt)
    return c,m,manifest,previous,receipt,p


def test_next_epoch_requires_bound_full_validation_and_policy(tmp_path,monkeypatch):
    c,m,manifest,previous,receipt,p=prior_fixture(tmp_path,monkeypatch)
    actual,cp,ref=r.prior_epoch_state(manifest,m,c,2)
    assert actual==p and cp==receipt['checkpoint'] and ref==r.entry(previous/'acceptance.json')


@pytest.mark.parametrize('change',['stop','policy','source','manifest','checkpoint','report','validation','epoch'])
def test_next_epoch_rejects_unaccepted_or_changed_handoff(tmp_path,monkeypatch,change):
    c,m,manifest,previous,receipt,p=prior_fixture(tmp_path,monkeypatch)
    if change=='stop':receipt['stopped']=True
    elif change=='policy':receipt['policy']['bad_epochs']=7
    elif change=='source':receipt['source_git_commit']='other'
    elif change=='manifest':receipt['manifest_sha256']='0'*64
    elif change=='checkpoint':Path(receipt['checkpoint']['path']).write_bytes(b'changed')
    elif change=='report':Path(receipt['report']['path']).write_text('{}')
    elif change=='validation':Path(p['validations'][0]['report']['path']).write_text('{}')
    else:receipt['completed_passes']=2
    write_json(previous/'acceptance.json',receipt)
    with pytest.raises((AssertionError,ValueError)):
        r.prior_epoch_state(manifest,m,c,2)


def test_run_and_independent_accept_mocked_device(tmp_path,monkeypatch):
    c=config(); manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    m=dict(run_directory=str(tmp_path/'run'),phase='formal',formal_training_ready=True,
        source_git_commit='commit',expected_initial_state_sha256='b'*64,initialization='fresh',
        expected_feature_cache_sha256='c'*64,assets={})
    binding={'purpose':'unit'}
    monkeypatch.setattr(r,'_contract',lambda:SimpleNamespace(FORMAL_PHASE='formal',
        binding=lambda _: (m,c),checkpoint_binding=lambda _:binding))
    monkeypatch.setenv('SLURM_JOB_ID','fixture')
    checks=[dict(allowed_cpus=[2],processes=[dict(threads_checked=1)])]
    resources=dict(allocated_cpus=4,user_cpu_ceiling=4,library_threads=1,allowed_cpus=[2])
    cache=dict(build=dict(cache_sha256='c'*64,cache_bytes=5909890560),
        verification=dict(status='passed',feature_values_bitwise_equal=True,cache_sha256='c'*64,cache_bytes=5909890560))
    def batches(**kwargs):
        for i,rows in enumerate((2,2,1)):yield i,list(range(rows))
    ctx=SimpleNamespace(initial_sha='b'*64,binding=binding,
        parameter_report=dict(total_params=41868160,trainable_params=41868160,backbone_params=34222464,
                              parameter_tensors=152,tied_sid_input_output=True),
        resources=resources,gpus=[dict(name='4090',total_bytes=100)],versions={'numpy':'fixture'},
        numpy_policy={},feature_cache_report=cache,setup_seconds=.1,cpu_checks=checks,
        cpu_check=lambda:checks[0],batches=batches,seen=range(11542755),cold=range(23915712))
    monkeypatch.setattr(r,'build_context',lambda *a,**kw:ctx)
    monkeypatch.setattr(r,'verify_asset_bytes',lambda _:dict(files={},seconds=.1))
    for method in ('synchronize','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,method,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:10)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:20)
    def logical(ctx,records,p,micro):
        rows,end=steps(p,start=p['next_batch']+1,stop=p['next_batch']+1)
        p.clear();p.update(end);return rows[0]
    monkeypatch.setattr(r,'logical_step',logical)
    saved={}
    def save(ctx,p,path):
        path.write_bytes(b'fixture state '+str(p['next_batch']).encode())
        ref=dict(path=str(path),sha256=r.sha(path),full_state_sha256='d'*64,
                 **{k:p[k] for k in ('epoch','next_batch','updates','attempts')})
        saved[str(path)]=copy.deepcopy(p);return ref
    monkeypatch.setattr(r,'save_bound_checkpoint',save)
    monkeypatch.setattr(r,'audit_checkpoint',lambda ref,path,p,c,b: p if saved[str(path)]==p else pytest.fail('badcheckpoint'))
    monkeypatch.setattr(r,'accept_cpu',lambda:(resources,lambda:checks[0]))
    import scripts.accept_liger_sampled_scaling as scaling
    monkeypatch.setattr(scaling,'validate_numpy_policy',lambda *args:None)
    result=r.run(manifest,1)
    receipt=r.accept(manifest,1)
    assert result['status']==receipt['status'].removesuffix('_training')=='passed'
    assert receipt['counts']['attempts']==3 and receipt['counts']['prefix_count']==5
    assert receipt['completed_training_passes']==1 and receipt['completed_validation_passes']==0
    assert receipt['final_progress']['early_stopping']['completed_epochs']==0
    assert len(saved)==2 and len(receipt['segments'])==1
    with pytest.raises(FileExistsError):r.run(manifest,1)
