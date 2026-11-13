"""Three fresh 384-wide updates, exact replay, then independent two-GPU val.

All stages use the formal entrypoint helpers but a distinct diagnostic purpose.
No diagnostic checkpoint can initialize the separately authorized formal run.
"""
import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
import traceback

from scripts.probe_liger_cpu4 import sha, write_new
from scripts.probe_liger_capacity import check_replay
from scripts.probe_liger_sampled_training import gate_clock, check_deadline
from scripts import liger_capacity_training_manifest as contract


def entry(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def gate(path):
    m, c = contract.binding(path)
    assert m['phase'] == contract.GATE_PHASE and m['formal_training_ready'] is False
    return m, c, Path(m['run_directory'])


def check_train_reports(path, m, a, b):
    check_replay(a, b)
    assert [x['attempt'] for x in a['steps']] == [1, 2, 3]
    assert [x['attempt'] for x in b['steps']] == [2, 3]
    for r in (a, b):
        assert r['manifest_sha256'] == sha(path) and r['phase'] == contract.GATE_PHASE
        assert r['job_id'] == os.environ['SLURM_JOB_ID']
        assert r['source_git_commit'] == m['source_git_commit']
        assert r['initial_state_sha256'] == m['expected_initial_state_sha256']
        assert r['checkpoint_binding'] == contract.checkpoint_binding(m)
        assert not any(r[k] for k in ('formal_training_ready', 'formal_checkpoint_produced',
                                      'test_records_read', 'automatic_successor'))
        assert r['resources']['allocated_cpus'] == 4 and r['resources']['library_threads'] == 1
        assert all(x['allowed_cpus'] == r['resources']['allowed_cpus'] for x in r['cpu_checks'])
        assert r['final_progress']['attempts'] == r['final_progress']['updates'] == 3
        assert r['final_progress']['prefix_count'] == 6144
        assert r['final_progress']['final_count'] == sum(x['final_target_count'] for x in a['steps'])
        assert r['cache']['verification']['status'] == 'passed'
        assert r['cache']['build']['cache_sha256'] == m['expected_feature_cache_sha256']
        assert r['cache']['verification']['cache_sha256'] == m['expected_feature_cache_sha256']
        assert sha(r['clock']['path']) == r['clock']['sha256']
        check_deadline(json.loads(Path(r['clock']['path']).read_text()))
        for step in r['steps']:
            assert step['microbatch'] == 1024 and step['microbatches'] == 2 and step['rows'] == 2048
            assert step['gradient_tensors'] == 152 and step['skipped'] is False
            assert 32768 < step['candidate_classes'] <= 34816
            assert abs(step['total_loss'] - step['sid_loss'] - step['dense_loss']) < 1e-10
            assert 0 < step['peak_allocated_bytes'] <= step['peak_reserved_bytes'] <= r['gpus'][0]['total_bytes']
    assert a['final_progress'] == b['final_progress']
    assets = a['asset_byte_verification']
    assert assets['files'] == {value['path']:value['sha256'] for value in m['assets'].values()}
    assert math.isfinite(assets['seconds']) and assets['seconds'] > 0
    assert a['validation_checkpoint']['full_state_sha256'] == a['state_after_three']
    for item in (a['step_one_checkpoint'], a['validation_checkpoint']):
        assert sha(item['path']) == item['sha256']


def run_training(path, resumed=False):
    import torch
    from scripts.train_liger_capacity_cpu4 import build_context, verify_asset_bytes
    from scripts.liger_capacity_runtime import new_progress, logical_step
    from scripts.train_liger_sampled_cpu4 import save_bound_checkpoint, restore_bound_checkpoint
    from src.baselines.liger_state import capture, tree_digest
    m, c, directory = gate(path)
    if not resumed: directory.mkdir(parents=True, exist_ok=False)
    target = directory / ('resumed.json' if resumed else 'continuous.json')
    assert not target.exists()
    clock, clock_entry = gate_clock(path, m, resumed)
    report = dict(status='failed', phase=m['phase'], manifest_sha256=sha(path),
        source_git_commit=m['source_git_commit'], job_id=os.environ['SLURM_JOB_ID'], resumed=resumed,
        formal_training_ready=False, formal_checkpoint_produced=False, test_records_read=False,
        automatic_successor=False, clock=clock_entry, steps=[])
    started = time.monotonic()
    try:
        if not resumed:
            report['asset_byte_verification'] = verify_asset_bytes(m)
        ctx = build_context(m, devices=1)
        p = new_progress()
        report.update(initial_state_sha256=ctx.initial_sha, checkpoint_binding=ctx.binding,
            gpus=ctx.gpus, resources=ctx.resources, cache=ctx.feature_cache_report,
            parameters=ctx.parameter_report)
        if resumed:
            a = json.loads((directory / 'continuous.json').read_text())
            assert a['status'] == 'passed' and a['manifest_sha256'] == sha(path)
            p, proof = restore_bound_checkpoint(ctx, a['step_one_checkpoint'])
            report.update(loaded_checkpoint=a['step_one_checkpoint'], restore_exact=proof['full_state_exact'])
        stream = ctx.batches(training=True, start_batch=p['next_batch'])
        try:
            while p['attempts'] < c['smoke']['train_steps']:
                check_deadline(clock)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin = time.monotonic()
                index, records = next(stream)
                input_end = time.monotonic()
                assert index == p['next_batch'] and len(records) == 2048
                step = logical_step(ctx, records, p, 1024)
                torch.cuda.synchronize()
                end = time.monotonic()
                step.update(input_seconds=input_end-begin, training_seconds=end-input_end,
                    seconds=end-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                report['steps'].append(step)
                ctx.cpu_check()
                print('[liger-capacity-training-gate] ' + json.dumps(step), flush=True)
                assert not step['skipped']
                if p['attempts'] == 1:
                    report['step_one_checkpoint'] = save_bound_checkpoint(ctx, p, directory / 'step01.pt')
        finally:
            stream.close()
        report['state_after_three'] = tree_digest(capture(*ctx.components, p, ctx.binding))
        if not resumed:
            report['validation_checkpoint'] = save_bound_checkpoint(ctx, p, directory / 'step03.pt')
        report.update(status='passed', final_progress=copy.deepcopy(p), cpu_checks=ctx.cpu_checks)
        contract.binding(path)
        check_deadline(clock)
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, checked_at=datetime.now().astimezone().isoformat())
        write_new(target, report)


def accept_training(path):
    m, c, directory = gate(path)
    a, b = [json.loads((directory / name).read_text()) for name in ('continuous.json', 'resumed.json')]
    check_train_reports(path, m, a, b)
    write_new(directory / 'training_gate_phase_acceptance.json', dict(status='passed',
        manifest_sha256=sha(path), job_id=os.environ['SLURM_JOB_ID'],
        train_report=entry(directory/'continuous.json'), resumed_report=entry(directory/'resumed.json'),
        checkpoint=a['validation_checkpoint'], progress=a['final_progress']))


def run_validation(path):
    from scripts.train_liger_capacity_cpu4 import build_context
    from scripts.validate_liger_capacity_cpu4 import load_model_checkpoint, evaluate_context
    m, c, directory = gate(path)
    receipt = json.loads((directory / 'training_gate_phase_acceptance.json').read_text())
    assert receipt['status'] == 'passed' and receipt['manifest_sha256'] == sha(path)
    assert receipt['job_id'] == os.environ['SLURM_JOB_ID']
    for key in ('train_report', 'resumed_report'):
        assert sha(receipt[key]['path']) == receipt[key]['sha256']
    clock, clock_entry = gate_clock(path, m, True)
    target = directory / 'validation_gate.json'
    assert not target.exists()
    report = dict(status='failed', phase=m['phase'], job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path), clock=clock_entry,
        formal_training_ready=False, formal_checkpoint_produced=False, test_records_read=False,
        automatic_successor=False)
    try:
        ctx = build_context(m, devices=2)
        loaded = load_model_checkpoint(ctx, receipt['checkpoint'], expected_progress=receipt['progress'])
        evaluated, audited, proof = evaluate_context(ctx, directory / 'validation', 65,
            receipt['checkpoint'], validation_limit=65)
        assert evaluated['metrics'] == audited['metrics'] and proof['model_weights_unchanged']
        report.update(status='passed', loaded=loaded, checkpoint=receipt['checkpoint'], proof=proof,
            validation_report=entry(directory/'validation/report.json'),
            validation_acceptance=entry(directory/'validation/acceptance.json'),
            gpus=ctx.gpus, resources=ctx.resources, cpu_checks=ctx.cpu_checks)
        contract.binding(path)
        check_deadline(clock)
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(target, report)


def accept(path):
    m, c, directory = gate(path)
    a, b = [json.loads((directory/name).read_text()) for name in ('continuous.json', 'resumed.json')]
    check_train_reports(path, m, a, b)
    v = json.loads((directory/'validation_gate.json').read_text())
    assert v['status'] == 'passed' and v['job_id'] == os.environ['SLURM_JOB_ID']
    assert v['manifest_sha256'] == sha(path) and v['checkpoint'] == a['validation_checkpoint']
    assert v['proof']['model_weights_unchanged']
    assert v['proof']['before_model_sha256'] == v['proof']['after_model_sha256']
    for name in ('validation_report', 'validation_acceptance'):
        assert sha(v[name]['path']) == v[name]['sha256']
    report, receipt = [json.loads(Path(v[key]['path']).read_text())
                       for key in ('validation_report', 'validation_acceptance')]
    assert report['records'] == receipt['records'] == 65
    assert report['cold_rows'] == 23915712 and report['batches'] == 3
    assert report['checkpoint'] == receipt['checkpoint'] == a['validation_checkpoint']
    assert receipt['report'] == v['validation_report'] and report['metrics'] == receipt['metrics']
    for artifact in report['artifacts']: assert sha(artifact['path']) == artifact['sha256']
    references = dict(train_report=entry(directory/'continuous.json'),
        resumed_report=entry(directory/'resumed.json'), validation_gate_report=entry(directory/'validation_gate.json'),
        validation_report=v['validation_report'], validation_acceptance=v['validation_acceptance'])
    write_new(directory/'gate_acceptance.json', dict(status='passed_capacity384_training_gate',
        phase=m['phase'], job_id=os.environ['SLURM_JOB_ID'], gate_manifest=entry(path),
        manifest_sha256=sha(path), source_git_commit=m['source_git_commit'],
        formal_config_sha256=m['formal_config']['sha256'], execution_sources=m['execution_sources'],
        initial_state_sha256=m['expected_initial_state_sha256'], exact_fresh_process_replay=True,
        two_gpu_validation_records=65, validation_model_weights_unchanged=True,
        formal_training_ready=False, formal_checkpoint_produced=False,
        test_records_read=False, automatic_successor=False,
        reports=list(references.values()) + report['artifacts'] + [a['step_one_checkpoint'], a['validation_checkpoint']],
        **references, checked_at=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['continuous', 'resumed', 'accept_train', 'validation', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    path = args.manifest.resolve()
    if args.mode in ('continuous', 'resumed'): run_training(path, args.mode == 'resumed')
    else: {'accept_train': accept_training, 'validation': run_validation, 'accept': accept}[args.mode](path)
