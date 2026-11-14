"""Bounded real-GPU replay, validation handoff, and cross-epoch resume gate.

The three-batch diagnostic epoch is explicitly NOT a full training pass. Its
model/schedule/update path is the new formal path; its weights never seed the
fresh formal run. Historical sources and checkpoints remain immutable.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

from scripts import liger_capacity_multiepoch_manifest as contract
from scripts.probe_liger_cpu4 import sha, write_new
from scripts.probe_liger_sampled_training import gate_clock, check_deadline, clean_steps


def entry(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def diagnostic_config(c):
    c = copy.deepcopy(c)
    c['training'].update(prefixes_per_epoch=6144, parents_per_epoch=850,
                         batches_per_epoch=3, final_batch_size=2048)
    c['validation']['records'] = 65
    return c


def gate(path):
    m, c = contract.binding(path)
    assert m['phase'] == contract.GATE_PHASE and not m['formal_training_ready']
    return m, c, Path(m['run_directory'])


def base_report(path, m, clock):
    return dict(status='failed', phase=m['phase'], manifest_sha256=sha(path),
        source_git_commit=m['source_git_commit'], job_id=os.environ['SLURM_JOB_ID'],
        formal_training_ready=False, formal_checkpoint_produced=False,
        test_records_read=False, automatic_successor=False, clock=clock)


def run_training(path, mode):
    import torch
    from scripts.train_liger_capacity_multiepoch import build_context, new_progress, progress_budget
    from scripts.train_liger_capacity_cpu4 import verify_asset_bytes
    from scripts.liger_capacity_runtime import logical_step
    from scripts.train_liger_sampled_cpu4 import save_bound_checkpoint, restore_bound_checkpoint
    from src.baselines.liger_prefix_training import check_progress
    from src.baselines.liger_state import capture, tree_digest
    m, c, directory = gate(path)
    first = mode == 'continuous'
    cross = mode.startswith('cross_')
    if first: directory.mkdir(parents=True, exist_ok=False)
    clock, clock_entry = gate_clock(path, m, not first)
    target = directory / (mode + '.json')
    assert not target.exists()
    report = dict(base_report(path, m, clock_entry), steps=[], mode=mode)
    begin = time.monotonic()
    try:
        if first: report['asset_byte_verification'] = verify_asset_bytes(m)
        ctx = build_context(m, devices=1)
        p = new_progress(c)
        report.update(initial_state_sha256=ctx.initial_sha, checkpoint_binding=ctx.binding,
            gpus=ctx.gpus, resources=ctx.resources, parameters=ctx.parameter_report,
            cache=ctx.feature_cache_report)
        if mode == 'resumed':
            previous = contract.read_bound(entry(directory/'continuous.json'))
            checkpoint = previous['step_one_checkpoint']
        elif cross:
            previous = contract.read_bound(entry(directory/'validation_gate.json'))
            checkpoint = previous['next_checkpoint']
        else:
            checkpoint = None
        if checkpoint:
            p, proof = restore_bound_checkpoint(ctx, checkpoint)
            report.update(loaded_checkpoint=checkpoint, restore_proof=proof)
        dc = diagnostic_config(c)
        check_progress(p, progress_budget(dc))
        report['initial_progress'] = copy.deepcopy(p)
        stream = ctx.batches(training=True, epoch=p['epoch'], start_batch=p['next_batch'])
        target_attempts = 5 if cross else 3
        try:
            while p['attempts'] < target_attempts:
                check_deadline(clock)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                index, records = next(stream)
                input_end = time.monotonic()
                assert index == p['next_batch'] and len(records) == 2048
                step = logical_step(ctx, records, p, 1024)
                torch.cuda.synchronize()
                ended = time.monotonic()
                step.update(input_seconds=input_end-started, training_seconds=ended-input_end,
                    seconds=ended-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                assert not step['skipped'] and step['gradient_tensors'] == 152
                report['steps'].append(step)
                check_progress(p, progress_budget(dc))
                ctx.cpu_check()
                print('[liger-multiepoch-gate] '+json.dumps(step), flush=True)
                if first and p['attempts'] == 1:
                    report['step_one_checkpoint'] = save_bound_checkpoint(ctx, p, directory/'step01.pt')
        finally:
            stream.close()
        report['final_state_sha256'] = tree_digest(capture(*ctx.components, p, ctx.binding))
        if first:
            assert p['final_count'] == 850
            report['validation_checkpoint'] = save_bound_checkpoint(ctx, p, directory/'step03.pt')
        report.update(status='passed', final_progress=copy.deepcopy(p), cpu_checks=ctx.cpu_checks)
        contract.binding(path)
        check_deadline(clock)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-begin, checked_at=datetime.now().astimezone().isoformat())
        write_new(target, report)


def verify_pair(a, b, *, cross=False):
    assert a['status'] == b['status'] == 'passed'
    expected = [4, 5] if cross else [1, 2, 3]
    assert [s['attempt'] for s in a['steps']] == expected
    assert [s['attempt'] for s in b['steps']] == ([4, 5] if cross else [2, 3])
    assert clean_steps(a['steps'] if cross else a['steps'][1:]) == clean_steps(b['steps'])
    for key in ('final_progress', 'final_state_sha256', 'initial_state_sha256',
                'checkpoint_binding', 'gpus', 'clock', 'manifest_sha256', 'job_id', 'source_git_commit'):
        assert a[key] == b[key], key
    assert b['restore_proof']['full_state_exact'] is True
    expected_checkpoint = a['loaded_checkpoint'] if cross else a['step_one_checkpoint']
    assert b['loaded_checkpoint'] == expected_checkpoint
    assert b['restore_proof']['full_state_sha256'] == expected_checkpoint['full_state_sha256']
    if cross:
        assert a['restore_proof']['full_state_exact'] is True
        assert a['initial_progress'] == b['initial_progress']
        assert a['initial_progress']['epoch'] == 1 and a['initial_progress']['next_batch'] == 0
        assert a['final_progress']['epoch'] == 1 and a['final_progress']['next_batch'] == 2
        assert a['final_progress']['prefix_count'] == 10240
    else:
        assert a['initial_progress']['updates'] == a['initial_progress']['attempts'] == 0
        assert b['initial_progress']['updates'] == b['initial_progress']['attempts'] == 1
        assert a['final_progress']['prefix_count'] == 6144
    for r in (a, b):
        for s in r['steps']:
            assert s['rows'] == 2048 and s['microbatch'] == 1024 and s['microbatches'] == 2
            assert s['gradient_tensors'] == 152 and s['skipped'] is False
            assert 32768 < s['candidate_classes'] <= 34816
            assert abs(s['total_loss']-s['sid_loss']-s['dense_loss']) < 1e-10


def run_validation(path):
    from scripts.train_liger_capacity_multiepoch import build_context
    from scripts.validate_liger_capacity_cpu4 import load_model_checkpoint, evaluate_context
    from scripts.validate_liger_capacity_multiepoch import finalize_checkpoint
    m, c, directory = gate(path)
    a, b = [contract.read_bound(entry(directory/name)) for name in ('continuous.json', 'resumed.json')]
    verify_pair(a, b)
    clock, clock_entry = gate_clock(path, m, True)
    r = base_report(path, m, clock_entry)
    try:
        ctx = build_context(m, devices=2)
        loaded = load_model_checkpoint(ctx, a['validation_checkpoint'], expected_progress=a['final_progress'])
        report, accepted, proof = evaluate_context(ctx, directory/'validation', 65,
            a['validation_checkpoint'], validation_limit=65)
        v = dict(epoch=1, checkpoint=a['validation_checkpoint'], report=accepted['report'],
            acceptance=entry(directory/'validation/acceptance.json'), metrics=report['metrics'],
            state_preservation=proof)
        next_checkpoint, next_progress, transition_proof = finalize_checkpoint(
            a['validation_checkpoint'], ctx.binding, diagnostic_config(c), v,
            directory/'after-validation.pt', diagnostic=True)
        r.update(status='passed', loaded=loaded, validation_entry=v, proof=proof,
            transition_proof=transition_proof, next_checkpoint=next_checkpoint,
            next_progress=next_progress, resources=ctx.resources, gpus=ctx.gpus, cpu_checks=ctx.cpu_checks)
        check_deadline(clock)
        contract.binding(path)
    except Exception:
        r['error'] = traceback.format_exc()
        raise
    finally:
        r['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(directory/'validation_gate.json', r)


def verify_gate_reports(m, c, directory):
    directory = Path(directory)
    names = ('continuous.json', 'resumed.json', 'validation_gate.json',
             'cross_continuous.json', 'cross_resumed.json', 'policy_test.json')
    a, b, v, x, y, policy = [json.loads((directory/name).read_text()) for name in names]
    verify_pair(a, b)
    verify_pair(x, y, cross=True)
    binding = contract.checkpoint_binding(m)
    for r in (a, b, v, x, y):
        assert r['status'] == 'passed' and r['phase'] == contract.GATE_PHASE
        assert r['source_git_commit'] == m['source_git_commit'] and r['job_id'] == a['job_id']
        assert r['manifest_sha256'] == a['manifest_sha256']
        assert all(r[k] is False for k in ('formal_training_ready', 'formal_checkpoint_produced',
                                         'test_records_read', 'automatic_successor'))
        assert r['resources']['allocated_cpus'] == 4 and r['resources']['library_threads'] == 1
        assert all(q['allowed_cpus'] == r['resources']['allowed_cpus'] for q in r['cpu_checks'])
    for r in (a, b, x, y):
        assert r['checkpoint_binding'] == binding
        assert r['initial_state_sha256'] == m['expected_initial_state_sha256']
        assert r['cache']['build']['cache_sha256'] == m['expected_feature_cache_sha256']
        assert r['cache']['verification']['cache_sha256'] == m['expected_feature_cache_sha256']
    assert a['asset_byte_verification']['files'] == {v['path']:v['sha256'] for v in m['assets'].values()}
    assert v['loaded']['checkpoint'] == a['validation_checkpoint']
    assert v['proof']['model_weights_unchanged'] is True
    assert v['proof']['before_model_sha256'] == v['proof']['after_model_sha256']
    transition = v['transition_proof']
    assert transition['non_progress_state_unchanged'] is True
    assert transition['before_non_progress_sha256'] == transition['after_non_progress_sha256']
    assert transition['changed_fields'] == ['progress'] and transition['preserved_cuda_rng_devices'] == 1
    assert transition['diagnostic'] is True and transition['original_checkpoint'] == a['validation_checkpoint']
    assert v['next_checkpoint'] == x['loaded_checkpoint'] == y['loaded_checkpoint']
    assert v['next_progress'] == x['initial_progress']
    assert v['next_progress']['early_stopping']['completed_epochs'] == 1
    assert v['next_progress']['early_stopping']['stopped'] is False
    assert v['next_progress']['early_stopping']['bad_epochs'] == 0
    assert a['final_progress']['attempts'] == a['final_progress']['updates'] == 3
    assert x['final_progress']['attempts'] == x['final_progress']['updates'] == 5
    assert x['steps'][0]['prefix_identity_sha256'] != a['steps'][0]['prefix_identity_sha256']
    vr = contract.read_bound(v['validation_entry']['report'])
    va = contract.read_bound(v['validation_entry']['acceptance'])
    assert vr['records'] == va['records'] == 65 and vr['cold_rows'] == 23915712
    assert vr['metrics'] == va['metrics'] == v['validation_entry']['metrics']
    assert vr['checkpoint'] == va['checkpoint'] == a['validation_checkpoint']
    assert vr['gt_independence_verified'] is True and len(v['gpus']) == 2
    assert va['report'] == v['validation_entry']['report']
    assert va['full_cold_rows'] == 23915712 and va['model_weights_unchanged'] is True
    assert v['proof']['load_scope'] == 'model_weights_only'
    assert vr['model_sha256'] == v['proof']['before_model_sha256']
    assert policy == policy_evidence(c)
    refs = [entry(directory/name) for name in names] + [v['validation_entry']['report'],
        v['validation_entry']['acceptance'], a['step_one_checkpoint'], a['validation_checkpoint'],
        v['next_checkpoint']] + vr['artifacts']
    for e in refs: assert sha(e['path']) == e['sha256']
    return refs


def policy_evidence(c):
    from src.baselines.liger_early_stopping import initial_policy, advance_policy
    def simulate(scores):
        state = initial_policy()
        for epoch, (r, n) in enumerate(scores, 1):
            v = dict(epoch=epoch, metrics={'hybrid': {'Recall@5': r, 'NDCG@5': n}},
                checkpoint={'path': f'diagnostic-{epoch}', 'sha256': '0'*64},
                report={'path': f'report-{epoch}', 'sha256': '0'*64},
                acceptance={'path': f'acceptance-{epoch}', 'sha256': '0'*64})
            state = advance_policy(json.loads(json.dumps(state)), v, c)
            if state['stopped']: return state
        return state
    plateau = simulate([(.1, .05+i*.001) for i in range(12)])
    improving = simulate([(.1+i*.001, .05) for i in range(12)])
    assert plateau['completed_epochs'] == 6 and plateau['bad_epochs'] == 5 and plateau['stopped']
    assert improving['completed_epochs'] == 12 and improving['bad_epochs'] == 0 and improving['stopped']
    return dict(status='passed', scope='synthetic policy only; no training epochs implied',
        min_epochs=6, max_epochs=12, patience=5, plateau_stops_at=6,
        improving_stops_at=12, tie_ndcg_does_not_reset_patience=True,
        serialized_resume_preserves_policy=True)


def accept(path):
    m, c, directory = gate(path)
    clock, _ = gate_clock(path, m, True)
    write_new(directory/'policy_test.json', policy_evidence(c))
    refs = verify_gate_reports(m, c, directory)
    check_deadline(clock)
    write_new(directory/'gate_acceptance.json', dict(status='passed_capacity384_multiepoch_gate',
        phase=m['phase'], gate_manifest=entry(path), manifest_sha256=sha(path),
        source_git_commit=m['source_git_commit'], job_id=os.environ['SLURM_JOB_ID'],
        formal_config_sha256=m['formal_config']['sha256'], execution_sources=m['execution_sources'],
        initial_state_sha256=m['expected_initial_state_sha256'], exact_fresh_process_replay=True,
        cross_epoch_exact_replay=True, two_gpu_validation_records=65,
        validation_model_weights_unchanged=True, early_stopping_policy_verified=True,
        formal_training_ready=False, formal_checkpoint_produced=False,
        test_records_read=False, automatic_successor=False, reports=refs,
        checked_at=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['continuous', 'resumed', 'validation', 'cross_continuous', 'cross_resumed', 'accept'])
    p.add_argument('manifest', type=Path)
    args = p.parse_args()
    path = args.manifest.resolve()
    if args.mode == 'validation': run_validation(path)
    elif args.mode == 'accept': accept(path)
    else: run_training(path, args.mode)
