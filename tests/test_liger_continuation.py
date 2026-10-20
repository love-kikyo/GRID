"""One-epoch authorization, immutable protocol, and full-state migration."""
import copy
import json
import math
import hashlib
from types import SimpleNamespace

import pytest
import torch

from scripts import liger_training_manifest as manifest
from scripts import train_liger_cpu4 as entry
from scripts.probe_liger_cpu4 import sha, write_new
from src.baselines.liger_state import BudgetSchedule, capture, save_checkpoint, tree_digest
from src.baselines.liger_training import initial_progress


@pytest.mark.parametrize('start,stop,valid', [(0,1,True),(1,2,True),(2,3,True),
    (0,2,False),(1,3,False),(3,4,False),(-1,0,False),(1,1,False)])
def test_stage_is_one_epoch_within_original_budget(start, stop, valid):
    m = dict(phase='first_epoch_formal_training' if start == 0 else 'epoch_continuation',
             stage=dict(start_epoch=start, stop_after_epoch=stop))
    config = {'training': {'max_epochs': 3}}
    if valid: assert manifest.stage_bounds(m, config) == (start, stop)
    else:
        with pytest.raises(AssertionError): manifest.stage_bounds(m, config)


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, 'ROOT', tmp_path)
    config = dict(training=dict(max_epochs=3, max_optimizer_steps=6, records_per_epoch=3,
        batches_per_epoch=2, batch_size=2), validation=dict(records=3))
    cfg = tmp_path / 'config.json'
    write_new(cfg, config)
    old = dict(formal_training_ready=True, source_git_commit='old', config=str(cfg), config_sha256=sha(cfg),
        input_metadata={}, upstream_root='up', upstream_commit='u', upstream_files={}, assets={},
        catalog={'sha256':'catalog'}, index_artifacts={}, asset_rows=10, seed=42, data_root='data',
        train_files=['train'], validation_files=['val'], smoke_acceptance={'sha256':'smoke'},
        execution_sources={str(tmp_path/'scripts/train_liger_cpu4.py'):'old',
                           str(tmp_path/'src/baselines/liger_training.py'):'unchanged'})
    old_path = tmp_path/'old-manifest.json'
    write_new(old_path, old)
    checkpoint_path = tmp_path/'logs/train/runs/prior/checkpoint.json'
    checkpoint_path.parent.mkdir(parents=True)
    write_new(checkpoint_path, {})
    cp = dict(path=str(checkpoint_path), sha256=sha(checkpoint_path), epoch=1, next_batch=0, updates=2)
    p = initial_progress()
    p.update(epoch=1, batches_consumed=2, updates=2, validations=[dict(report=cp,acceptance=cp,checkpoint=cp)],
             best={'epoch':1})
    r = dict(status='passed', manifest_sha256=sha(old_path), source_git_commit='old', job_id='old-job',
        test_records_read=False, stopped_for_author_acceptance=True, automatic_successor=False,
        final_progress=p, best=p['best'], checkpoints=[cp])
    report_path = tmp_path/'result.json'
    write_new(report_path, r)
    receipt_path = tmp_path/'acceptance.json'
    write_new(receipt_path, dict(status='passed',report=dict(path=str(report_path),sha256=sha(report_path)),
        job_id='old-job',no_test_access=True,completed_epochs=1,updates=2,skipped_updates=0,best=p['best'],
        original_max_epochs=3,full_training_records=3,full_validation_records=3))
    m = copy.deepcopy(old)
    m.update(phase='epoch_continuation',stage=dict(start_epoch=1,stop_after_epoch=2),resume_checkpoint=cp)
    changed = str(tmp_path/'scripts/train_liger_cpu4.py')
    m['execution_sources'][changed] = 'new'
    m['continuation'] = dict(manifest=dict(path=str(old_path),sha256=sha(old_path)),
        acceptance=dict(path=str(receipt_path),sha256=sha(receipt_path)),changed_orchestration_sources=[changed])
    return m, old, r


def test_predecessor_is_accepted_and_only_orchestration_can_change(continuation):
    m, old, r = continuation
    assert manifest.verify_continuation(m) == (old, r)


@pytest.mark.parametrize('mutation', ['core','config','train','seed','drop_source','checkpoint',
                                    'receipt','stop','cursor'])
def test_continuation_rejects_protocol_or_lineage_changes(continuation, mutation):
    m, _, _ = continuation
    if mutation == 'core':
        m['execution_sources'][str(manifest.ROOT/'src/baselines/liger_training.py')] = 'changed'
    elif mutation == 'config': m['config_sha256'] = 'changed'
    elif mutation == 'train': m['train_files'] = ['test']
    elif mutation == 'seed': m['seed'] = 43
    elif mutation == 'drop_source': m['execution_sources'].pop(str(manifest.ROOT/'src/baselines/liger_training.py'))
    elif mutation == 'checkpoint': m['resume_checkpoint'] = dict(m['resume_checkpoint'],sha256='changed')
    elif mutation == 'receipt': m['continuation']['acceptance']['sha256'] = 'changed'
    elif mutation == 'stop': m['stage']['stop_after_epoch'] = 3
    elif mutation == 'cursor': m['stage'] = dict(start_epoch=2,stop_after_epoch=3)
    with pytest.raises(AssertionError): manifest.verify_continuation(m)


def test_restore_preserves_every_state_value_and_migrates_only_new_saves(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.035)
    schedule = BudgetSchedule(optimizer,total=19260,warmup=963)
    scaler = torch.amp.GradScaler('cuda',enabled=False)
    model(torch.ones(1,2)).sum().backward()
    optimizer.step(); optimizer.zero_grad(); schedule.step()
    p = initial_progress(); p.update(updates=1)
    old_binding = {'purpose':'formal','execution_sources':{'entry':'old'}}
    ctx = SimpleNamespace(components=(model,optimizer,schedule,scaler), binding={'purpose':'formal','execution_sources':{'entry':'new'}})
    state = capture(*ctx.components,p,old_binding)
    path = tmp_path/'accepted.pt'
    save_checkpoint(path,state)
    original_hash = sha(path)
    m = dict(phase='epoch_continuation',resume_checkpoint=dict(path=str(path),sha256=original_hash))
    monkeypatch.setattr(entry,'verify_continuation',lambda _: ({},dict(final_progress=p)))
    monkeypatch.setattr(entry,'checkpoint_binding',lambda _: old_binding)
    # Ensure restoring, not merely observing an unchanged initialized model.
    with torch.no_grad(): model.weight.add_(5)
    torch.rand(4)
    restored, audit = entry.restore_formal(ctx,m)
    assert audit['full_state_exact'] and audit['restored_updates'] == 1
    assert audit['next_learning_rate'] == .0003*2/963
    assert audit['full_state_sha256'] == tree_digest(capture(*ctx.components,restored,old_binding))
    assert sha(path) == original_hash
    successor = capture(*ctx.components,restored,ctx.binding)
    saved = torch.load(path,map_location='cpu',weights_only=False)
    assert successor['binding'] != saved['binding']
    successor['binding'] = saved['binding']
    assert tree_digest(successor) == tree_digest(saved)


def test_new_launcher_runs_continuation_regressions():
    assert 'tests/test_liger_continuation.py' in (manifest.ROOT/'run_liger_train_cpu4.sh').read_text()


@pytest.mark.parametrize('score', [.1, .2, .3])
def test_independent_acceptance_of_second_epoch_with_prior_best(tmp_path, monkeypatch, score):
    """Exercise final acceptance too, including 6420-row log and carried best."""
    from src.baselines import liger_runtime, liger_validation
    path = tmp_path/'manifest.json'
    write_new(path,{})
    budget = dict(max_epochs=3, max_optimizer_steps=19260, batch_size=1024,
                  batches_per_epoch=6420, records_per_epoch=6573674)
    ctx = SimpleNamespace(config={'training':budget},binding={'purpose':'unit'},seen=None,index=None)
    monkeypatch.setattr(liger_runtime,'Runtime',lambda *args,**kwargs:ctx)
    monkeypatch.setenv('SLURM_JOB_ID','unit')
    before = initial_progress()
    old_v = dict(epoch=1,checkpoint={'path':'old-best'},metrics={'hybrid':{'Recall@5':.2,'NDCG@5':.1}})
    before.update(epoch=1,batches_consumed=6420,updates=6420,validations=[old_v],
        best=dict(epoch=1,checkpoint=old_v['checkpoint'],**old_v['metrics']['hybrid']))
    source = {'progress':before}
    m = dict(formal_training_ready=True,phase='epoch_continuation',source_git_commit='unit',
        run_directory=str(tmp_path),stage=dict(start_epoch=1,stop_after_epoch=2),resume_checkpoint={'path':'source'})
    monkeypatch.setattr(entry,'binding',lambda _:m)
    monkeypatch.setattr(entry,'verify_continuation',lambda _:({},dict(final_progress=before)))
    cp_path = tmp_path/'checkpoint.json'
    write_new(cp_path,{})
    cp = dict(path=str(cp_path),sha256=sha(cp_path))
    vp = tmp_path/'validation.json'
    model = {'w':torch.tensor([1.])}
    write_new(vp,dict(model_sha256=tree_digest(model)))
    metrics = {'hybrid':{'Recall@5':score,'NDCG@5':.1}}
    receipt = dict(metrics=metrics)
    ap = tmp_path/'val-acceptance.json'
    write_new(ap,receipt)
    monkeypatch.setattr(liger_validation,'audit',lambda *args,**kwargs:receipt)
    v = dict(epoch=2,checkpoint=cp,metrics=metrics,
             report=dict(path=str(vp),sha256=sha(vp)),acceptance=dict(path=str(ap),sha256=sha(ap)))
    chain = before['data_chain']
    rows = []
    for i in range(1,6421):
        identity = str(i).zfill(64)
        chain = hashlib.sha256((chain+identity).encode()).hexdigest()
        lr = .0003*.5*(1+math.cos(math.pi*(6420+i-963)/(19260-963)))
        rows.append(json.dumps(dict(batch=i,epoch=2,rows=1024 if i < 6420 else 618,
            gradient_tensors=152,learning_rate=lr,seconds=1,skipped=False,updates=6420+i,
            batch_identity_sha256=identity,data_chain=chain)))
    (tmp_path/'steps.jsonl').write_text('\n'.join(rows)+'\n')
    best = dict(epoch=2,checkpoint=cp,**metrics['hybrid']) if score > .2 else before['best']
    after = copy.deepcopy(before)
    after.update(epoch=2,batches_consumed=12840,updates=12840,validations=[old_v,v],best=best,data_chain=chain)
    state = dict(binding=ctx.binding,progress=after,scheduler=dict(total=19260,warmup=963,updates=12840),
        optimizer={'state':{i:dict(step=12840) for i in range(152)}},scaler={'scale':524288},
        rng={'cuda':[torch.tensor([1])]},model=model)
    monkeypatch.setattr(torch,'load',lambda file,**kwargs:source if file == 'source' else state)
    r = dict(status='passed',job_id='unit',manifest_sha256=sha(path),source_git_commit='unit',
        stopped_for_author_acceptance=True,test_records_read=False,automatic_successor=False,
        resources=dict(allocated_cpus=4,user_cpu_ceiling=4,allowed_cpus=[0,1,2,3]),
        cpu_checks=[dict(allowed_cpus=[0,1,2,3],processes=['unit'])],initial_progress=before,
        final_progress=after,resume_verification=dict(full_state_exact=True,full_state_sha256=tree_digest(source)),
        checkpoints=[cp],validations=[v],best=best)
    write_new(tmp_path/'result.json',r)
    entry.accept(path)
    accepted = json.loads((tmp_path/'acceptance.json').read_text())
    assert accepted['completed_epochs'] == 2 and accepted['updates'] == 12840
    assert accepted['best'] == best and accepted['original_max_epochs'] == 3
