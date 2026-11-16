"""Small, CPU-only fixtures for the external controller's read-only gate."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import liger_autorun_evidence as gate
from src.baselines.liger_early_stopping import initial_policy, advance_policy


def config():
    return json.loads(Path('configs/baselines/liger_capacity384_multiepoch_seed42_v1.json').read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return gate.reference(path)


def cp(path='/tmp/formal.pt', epoch=0):
    return dict(path=str(path), sha256='a'*64, full_state_sha256='b'*64,
                epoch=epoch, next_batch=22460, attempts=(epoch+1)*22460,
                updates=(epoch+1)*22460-6)


def validation(epoch=1, recall=.2):
    metrics = {mode: {f'{metric}@{k}': recall for metric in ('Recall', 'NDCG') for k in (5, 10, 20)}
               for mode in ('hybrid', 'beam_only')}
    return dict(epoch=epoch, checkpoint=cp(epoch=epoch-1),
                report=dict(path='/tmp/report.json', sha256='c'*64),
                acceptance=dict(path='/tmp/acceptance.json', sha256='d'*64), metrics=metrics)


def transition(epoch=1, improving=False):
    c, policy = config(), initial_policy()
    for number in range(1, epoch):
        policy = advance_policy(policy, validation(number, .01*number if improving else .2), c)
    before = dict(epoch=epoch-1, next_batch=22460, samples_in_epoch=45996113,
        batches_consumed=epoch*22460, attempts=epoch*22460, prefix_count=epoch*45996113,
        final_count=epoch*6573674, updates=epoch*22460-6, skipped_updates=6,
        max_optimizer_steps=269520, max_attempts=269520, data_chain='f'*64,
        best=policy['best'], validations=policy['validations'], early_stopping=policy)
    value = validation(epoch, .01*epoch if improving else .2)
    checkpoint = value['checkpoint']
    after_policy = advance_policy(policy, value, c)
    after = copy.deepcopy(before)
    after.update(epoch=epoch, next_batch=0, samples_in_epoch=0, early_stopping=after_policy,
                 best=after_policy['best'], validations=after_policy['validations'])
    proof = dict(non_progress_state_unchanged=True, before_non_progress_sha256='e'*64,
        after_non_progress_sha256='e'*64, changed_fields=['progress'], preserved_cuda_rng_devices=1,
        diagnostic=False, original_checkpoint=checkpoint)
    return c, before, after, after_policy, value, proof, checkpoint


@pytest.mark.parametrize('epoch,improving,stopped,reason', [
    (1, False, False, None), (5, False, False, None), (6, False, True, 'patience'),
    (6, True, False, None), (11, True, False, None), (12, True, True, 'max_epochs')])
def test_policy_transition_min6_patience5_max12(epoch, improving, stopped, reason):
    c, before, after, policy, value, proof, checkpoint = transition(epoch, improving)
    observed = gate.check_transition(before, after, policy, value, proof, checkpoint, c, epoch)
    assert observed['stopped'] is stopped and observed['stop_reason'] == reason


@pytest.mark.parametrize('change', ['max_epochs', 'min_epochs', 'patience', 'wrong_monitor',
    'bad_counter', 'stop_flag', 'best', 'updates', 'attempts', 'prefixes', 'parents',
    'tail', 'before_epoch', 'next_cursor', 'validation_epoch', 'checkpoint',
    'model_changed', 'preserved_hash', 'rng_devices', 'extra_changed_field', 'diagnostic'])
def test_transition_rejects_tampering(change):
    c, before, after, policy, value, proof, checkpoint = transition()
    if change == 'max_epochs': c['training']['max_epochs'] = c['early_stopping']['max_epochs'] = 13
    elif change == 'min_epochs': c['training']['min_epochs'] = c['early_stopping']['min_epochs'] = 1
    elif change == 'patience': c['early_stopping']['patience'] = 2
    elif change == 'wrong_monitor': c['early_stopping']['monitor'] = 'test Recall@5'
    elif change == 'bad_counter': policy['bad_epochs'] += 1
    elif change == 'stop_flag': policy['stopped'] = True
    elif change == 'best': policy['best']['Recall@5'] = .9
    elif change == 'updates': after['updates'] -= 1
    elif change == 'attempts': before['attempts'] -= 1
    elif change == 'prefixes': before['prefix_count'] -= 1
    elif change == 'parents': before['final_count'] -= 1
    elif change == 'tail': before['samples_in_epoch'] -= 81
    elif change == 'before_epoch': before['epoch'] = 1
    elif change == 'next_cursor': after['next_batch'] = 1
    elif change == 'validation_epoch': value['epoch'] = 2
    elif change == 'checkpoint': value['checkpoint'] = dict(checkpoint, sha256='f'*64)
    elif change == 'model_changed': proof['non_progress_state_unchanged'] = False
    elif change == 'preserved_hash': proof['after_non_progress_sha256'] = 'd'*64
    elif change == 'rng_devices': proof['preserved_cuda_rng_devices'] = 2
    elif change == 'extra_changed_field': proof['changed_fields'].append('optimizer')
    elif change == 'diagnostic': proof['diagnostic'] = True
    with pytest.raises(AssertionError):
        gate.check_transition(before, after, policy, value, proof, checkpoint, c, 1)


def test_reference_rejects_drift_path_and_malformed_hash(tmp_path):
    path = tmp_path/'report.json'
    ref = write(path, {'ok': True})
    assert gate.read_bound(ref, path) == {'ok': True}
    for damaged in (dict(ref, sha256='invalid'), dict(ref, extra=True)):
        with pytest.raises(AssertionError): gate.read_bound(damaged)
    with pytest.raises(AssertionError): gate.read_bound(ref, tmp_path/'different.json')
    path.write_text('{}')
    with pytest.raises(AssertionError): gate.read_bound(ref)


@pytest.fixture
def predictions(tmp_path):
    directory = tmp_path/'validation'
    directory.mkdir()
    n, targets = 5, np.asarray([0, 5, 19, 50, 1], dtype=np.int64)
    rows = np.tile(np.arange(20, dtype=np.int64), (n, 1))
    scores = -np.tile(np.arange(20, dtype=np.float64), (n, 1))
    arrays = dict(user_ids=np.arange(n, dtype=np.int64), target_rows=targets,
        record_sha256=np.asarray([hashlib.sha256(str(i).encode()).hexdigest() for i in range(n)], dtype='S64'),
        raw_beam_rows=rows, valid_beam_rows=rows, hybrid_rows=rows, beam_only_rows=rows,
        hybrid_scores=scores, beam_only_scores=scores)
    artifacts = []
    for i, start in enumerate((0, 4)):
        path = directory/f'part-{i:05d}.npz'
        with path.open('wb') as handle: np.savez(handle, **{k: v[start:start+4] for k, v in arrays.items()})
        artifacts.append(dict(gate.reference(path), rows=min(4, n-start)))
    ranks = [1, 6, 20, 0, 2]
    metrics = {mode: {f'{metric}@{k}':sum((1 if metric=='Recall' else 1/math.log2(rank+1))
                for rank in ranks if 0 < rank <= k)/n for metric in ('Recall', 'NDCG') for k in (5,10,20)}
               for mode in ('hybrid', 'beam_only')}
    hits = {str(k):sum(0<rank<=k for rank in ranks) for k in (5,10,20)}
    times = {k:0. for k in ('input_seconds','packing_seconds','generation_seconds','ranking_seconds',
                          'serialization_metrics_seconds','gt_independence_seconds')}
    batch_log = directory/'batches.jsonl'
    batch_log.write_text(''.join(json.dumps(dict(batch=i,rows=count,scanned_rows=23915712,**times))+'\n'
                                for i,count in enumerate((2,2,1))))
    report = dict(records=n,batches=3,cold_rows=23915712,artifacts=artifacts,
        record_identity_sha256=hashlib.sha256(arrays['record_sha256'].tobytes()).hexdigest(),
        valid_generated_slots=n*20,metrics=metrics,
        raw_generation_coverage=dict(hits=hits,recall={k:v/n for k,v in hits.items()}),
        batch_log=gate.reference(batch_log),timing=times)
    return directory,report,arrays


def check_predictions(directory, report):
    return gate.audit_prediction_artifacts(report,directory,expected_records=5,batch_size=2,
                                           cold_rows=23915712,shard_rows=4,asset_rows=100)


def test_prediction_audit_recomputes_identity_metrics_and_tail(predictions):
    directory, report, _ = predictions
    result = check_predictions(directory,report)
    assert result == dict(records=5,artifacts=2,batches=3,final_batch=1,cold_rows=23915712,
                         metrics_recomputed=True,identities_recomputed=True,candidate_mapping_rechecked=False)


@pytest.mark.parametrize('change', ['records','batches','cold','identity','valid_beams','metric',
    'coverage','artifact_count','artifact_rows','artifact_bytes','duplicate_users','input_identity',
    'invalid_target','repeated_rank','bad_score','missing_hybrid','raw_coverage','batch_tail','batch_cold','timing'])
def test_prediction_audit_rejects_corruption(predictions,change):
    directory,r,arrays = predictions
    if change == 'records': r['records'] -= 1
    elif change == 'batches': r['batches'] -= 1
    elif change == 'cold': r['cold_rows'] -= 1
    elif change == 'identity': r['record_identity_sha256'] = 'a'*64
    elif change == 'valid_beams': r['valid_generated_slots'] -= 1
    elif change == 'metric': r['metrics']['hybrid']['NDCG@10'] += .01
    elif change == 'coverage': r['raw_generation_coverage']['hits']['20'] -= 1
    elif change == 'artifact_count': r['artifacts'].pop()
    elif change == 'artifact_rows': r['artifacts'][0]['rows'] -= 1
    elif change == 'artifact_bytes':
        with Path(r['artifacts'][0]['path']).open('ab') as handle: handle.write(b'x')
    elif change in ('batch_tail','batch_cold','timing'):
        path = Path(r['batch_log']['path'])
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        if change == 'batch_tail': lines[-1]['rows'] = 2
        elif change == 'batch_cold': lines[-1]['scanned_rows'] -= 1
        else: lines[-1]['input_seconds'] = float('nan')
        path.write_text(''.join(json.dumps(x)+'\n' for x in lines))
        r['batch_log'] = gate.reference(path)
    else:
        for k,v in arrays.items(): arrays[k] = v.copy()
        if change == 'duplicate_users': arrays['user_ids'][1] = arrays['user_ids'][0]
        elif change == 'input_identity': arrays['record_sha256'][0] = b'a'*64
        elif change == 'invalid_target': arrays['target_rows'][0] = -1
        elif change == 'repeated_rank': arrays['hybrid_rows'][0,1] = arrays['hybrid_rows'][0,0]
        elif change == 'bad_score': arrays['hybrid_scores'][0,0] = float('nan')
        elif change == 'missing_hybrid': arrays['hybrid_rows'][0,0] = -1; arrays['hybrid_scores'][0,0] = -np.inf
        elif change == 'raw_coverage': arrays['raw_beam_rows'][0,0] = 51
        path = directory/'part-00000.npz'
        with path.open('wb') as handle: np.savez(handle, **{k:v[:4] for k,v in arrays.items()})
        r['artifacts'][0] = dict(gate.reference(path),rows=4)
    with pytest.raises((AssertionError,KeyError,ValueError)):
        check_predictions(directory,r)


@pytest.fixture
def completed(tmp_path,monkeypatch):
    c,before,after,policy,value,proof,checkpoint = transition()
    manifest_path=tmp_path/'manifest.json';write(manifest_path,{'frozen':True})
    directory=tmp_path/'run/epoch01';segment=directory/'segments/segment000'
    m=dict(phase='capacity384_multiepoch_formal',formal_training_ready=True,
           run_directory=str(directory.parent),source_git_commit='source',expected_initial_state_sha256='initial')
    binding={'purpose':'capacity384_multiepoch_formal'}
    cp_path=segment/'checkpoints/before-validation.pt';cp_path.parent.mkdir(parents=True)
    cp_path.write_bytes(b'prevalidated model and optimizer')
    checkpoint.update(path=str(cp_path),sha256=gate.sha(cp_path))
    common=dict(epoch=1,manifest_sha256=gate.sha(manifest_path),source_git_commit='source',
                test_records_read=False,automatic_successor=False)
    pre=write(segment/'preflight.json',{'ready':True})
    log=segment/'steps.jsonl';log.write_text('audited full steps\n')
    segments=[dict(segment=str(segment),preflight=pre,steps=gate.reference(log))]
    train_report=dict(common,status='passed',job_id='101',awaiting_validation=True,
        completed_training_passes=1,completed_validation_passes=0,checkpoint_binding=binding,
        initial_state_sha256='initial',checkpoint=checkpoint,final_progress=before)
    train=dict(train_report,status='passed_training',report=write(directory/'training_result.json',train_report),
        measured=dict(progress=before,steps=22460,segments=segments),segments=segments,
        counts={k:before[k] for k in ('attempts','updates','skipped_updates','prefix_count','final_count','data_chain')})
    train_ref=write(directory/'training_acceptance.json',train)
    preservation=dict(model_weights_unchanged=True,before_model_sha256='f'*64,
                      after_model_sha256='f'*64,load_scope='model_weights_only')
    r=dict(status='passed',records=347416,checkpoint=checkpoint,metrics=value['metrics'],
        cold_rows=23915712,model_weights_unchanged=True,gt_independence_verified=True,
        test_records_read=False,raw_generation_coverage={},model_sha256='f'*64,
        resources=dict(job_id='102',allocated_cpus=4,library_threads=1),
        gpus=[dict(name='RTX 4090'),dict(name='RTX 4090')],shard_sizes=[11957856,11957856])
    vref=write(directory/'validation/report.json',r)
    va=dict(r,report=vref,full_cold_rows=23915712)
    value.update(report=vref,acceptance=write(directory/'validation/acceptance.json',va),
                 state_preservation=preservation)
    policy=advance_policy(before['early_stopping'],value,c)
    after.update(early_stopping=policy,best=policy['best'],validations=policy['validations'])
    post_path=directory/'after-validation.pt';post_path.write_bytes(b'complete single GPU state with updated progress')
    post=dict(checkpoint,path=str(post_path),sha256=gate.sha(post_path),epoch=1,next_batch=0)
    result=dict(common,status='passed',phase=m['phase'],job_id='102',requires_manual_next_job=True,
        training_acceptance=train_ref,completed_passes=1,validation=value,validations=policy['validations'],
        best=policy['best'],policy=policy,stopped=False,stop_reason=None,checkpoint=post,next_checkpoint=post,
        next_progress=after,training_progress=before,handoff_proof=proof,
        load_proof=dict(checkpoint=checkpoint,progress=before,model_state_sha256='f'*64,load_scope='model_weights_only'),
        validation_model_weights_unchanged=True,raw_generation_coverage={})
    accepted=dict(result,report=write(directory/'result.json',result),checkpoint_binding=binding,
        full_validation_records=347416,full_training_prefixes=45996113,full_training_parents=6573674)
    write(directory/'acceptance.json',accepted)
    monkeypatch.setattr(gate,'audit_prediction_artifacts',lambda *a,**kw:dict(metrics_recomputed=True))
    return manifest_path,m,c,binding,directory


def test_full_bound_receipt_chain(completed):
    path,m,c,binding,directory=completed
    r=gate.verify_bound_epoch(path,m,c,binding,1,'101','102')
    assert r['status']=='accepted' and r['next_epoch']==2 and not r['stopped']
    assert r['acceptance']==gate.reference(directory/'acceptance.json')
    assert r['external_controller_authority_required'] and not r['checkpoint_tensors_reloaded']


@pytest.mark.parametrize('epoch,train,val',[(0,'101','102'),(13,'101','102'),(True,'101','102'),
    (1,'101','101'),(1,'0','102'),(1,'101_1','102'),(1,'999','102'),(1,'101','999')])
def test_epoch_and_job_identity_must_match(completed,epoch,train,val):
    path,m,c,binding,_=completed
    with pytest.raises((AssertionError,FileNotFoundError)):
        gate.verify_bound_epoch(path,m,c,binding,epoch,train,val)


@pytest.mark.parametrize('field,value',[('stopped',True),('requires_manual_next_job',False),
    ('automatic_successor',True),('test_records_read',True),('full_validation_records',65),
    ('full_training_prefixes',6144),('full_training_parents',850),('validation_model_weights_unchanged',False),
    ('job_id','999'),('source_git_commit','other'),('manifest_sha256','f'*64),('status','failed')])
def test_top_receipt_cannot_silently_bypass_formal_evidence(completed,field,value):
    path,m,c,binding,directory=completed
    p=directory/'acceptance.json';r=json.loads(p.read_text());r[field]=value;write(p,r)
    with pytest.raises(AssertionError):gate.verify_bound_epoch(path,m,c,binding,1,'101','102')


@pytest.mark.parametrize('filename',['after-validation.pt','segments/segment000/checkpoints/before-validation.pt',
    'training_result.json','result.json','validation/report.json','validation/acceptance.json',
    'segments/segment000/preflight.json','segments/segment000/steps.jsonl'])
def test_referenced_bytes_are_immutable(completed,filename):
    path,m,c,binding,directory=completed
    with (directory/filename).open('ab') as handle:handle.write(b' ')
    with pytest.raises(AssertionError):gate.verify_bound_epoch(path,m,c,binding,1,'101','102')


def test_no_hidden_checkpoint_tensor_loading_or_submission():
    text=Path(gate.__file__).read_text()
    for forbidden in ('torch.load(', 'sbatch', 'subprocess.', '.write_text(', '.write_bytes('):
        assert forbidden not in text


def test_optimized_python_cannot_disable_acceptance_checks():
    result=subprocess.run([sys.executable,'-O','-c','import scripts.liger_autorun_evidence'],
                          capture_output=True,text=True)
    assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr
