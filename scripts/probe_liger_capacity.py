"""Bounded real-data capacity cases and exact fresh-process replay; no full run."""
import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from scripts.liger_capacity_manifest import binding
from scripts.probe_liger_cpu4 import sha, write_new


def clean_step(step):
    return {k: v for k, v in step.items() if k not in (
        'seconds', 'input_seconds', 'training_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')}


def check_replay(a, b):
    assert a['status'] == b['status'] == 'passed'
    assert b['resumed'] is True and a['resumed'] is False
    assert [clean_step(x) for x in a['steps'][1:3]] == [clean_step(x) for x in b['steps']]
    for key in ('state_after_three', 'initial_state_sha256', 'checkpoint_binding', 'gpus'):
        assert a[key] == b[key], key
    assert b['loaded_checkpoint'] == a['step_one_checkpoint']
    assert b['restore_exact'] is True


def base_report(path, m, micro, resumed):
    return dict(status='failed', job_id=os.environ.get('SLURM_JOB_ID'),
        phase=m['phase'], source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
        microbatch=micro, logical_batch_size=2048, resumed=resumed, steps=[],
        formal_training_ready=False, formal_checkpoint_produced=False,
        automatic_successor=False, test_records_read=False)


def run(path, micro, resumed=False):
    import torch
    from scripts.liger_capacity_runtime import build_context, new_progress, logical_step
    from scripts.train_liger_sampled_cpu4 import save_bound_checkpoint, restore_bound_checkpoint
    from src.baselines.liger_state import capture, tree_digest
    from src.baselines.liger_efficiency_timing import process_memory_snapshot
    m, c = binding(path)
    assert micro in c['training']['physical_batch_sizes']
    directory = path.parent / f'micro{micro}'
    directory.mkdir(exist_ok=True)
    report_path = directory / ('resumed.json' if resumed else 'continuous.json')
    assert not report_path.exists()
    report = base_report(path, m, micro, resumed)
    started, deadline = time.monotonic(), time.monotonic()+c['deadline_seconds']
    stage = 'setup'
    try:
        previous = json.loads((directory/'continuous.json').read_text()) if resumed else None
        if resumed and previous['status'] == 'cuda_oom':
            report.update(status='skipped_after_cuda_oom')
            return
        ctx = build_context(m, c, microbatch=micro)
        report.update(initial_state_sha256=ctx.initial_sha, parameters=ctx.parameter_report,
            checkpoint_binding=ctx.binding, gpus=ctx.gpus, resources=ctx.resources,
            numpy_memory_policy=ctx.numpy_policy, cache=ctx.feature_cache_report,
            setup_seconds=ctx.setup_seconds)
        p = new_progress()
        if resumed:
            p, proof = restore_bound_checkpoint(ctx, previous['step_one_checkpoint'])
            report.update(loaded_checkpoint=previous['step_one_checkpoint'],
                          restore_exact=proof['full_state_exact'])
        end = 3 if resumed else c['training']['max_attempts_per_case']
        stream = ctx.batches(training=True, start_batch=p['next_batch'])
        try:
            while p['attempts'] < end:
                if time.monotonic() > deadline: raise TimeoutError('Capacity case deadline')
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin = time.monotonic()
                index, records = next(stream)
                input_seconds = time.monotonic()-begin
                assert index == p['next_batch'] and len(records) == 2048
                stage = 'training_attempt'
                train_begin = time.monotonic()
                step = logical_step(ctx, records, p, micro)
                torch.cuda.synchronize()
                step.update(input_seconds=input_seconds, training_seconds=time.monotonic()-train_begin,
                    seconds=time.monotonic()-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                report['steps'].append(step)
                print('[liger-capacity] '+json.dumps(step), flush=True)
                stage = 'checkpoint_or_audit'
                ctx.cpu_check()
                if p['attempts'] == 1:
                    report['step_one_checkpoint'] = save_bound_checkpoint(ctx, p, directory/'step01.pt')
                if p['attempts'] == 3:
                    report['state_after_three'] = tree_digest(capture(*ctx.components, p, ctx.binding))
                if time.monotonic() > deadline: raise TimeoutError('Capacity case deadline')
        finally:
            stream.close()
        assert p['skipped_updates'] == 0, 'Short capacity probe requires all updates to succeed'
        if not resumed and micro == 512:
            report['validation_checkpoint'] = save_bound_checkpoint(ctx, p, directory/'step13-diagnostic.pt')
        if not resumed:
            measured = report['steps'][c['training']['warmup_attempts']:]
            report['measured'] = {key: statistics.mean(x[key] for x in measured)
                for key in ('seconds', 'input_seconds', 'training_seconds')}
            report['measured'].update(prefixes_per_second=2048/report['measured']['seconds'],
                projected_pass_hours=22460*report['measured']['seconds']/3600,
                peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in report['steps']),
                peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in report['steps']))
        report.update(status='passed', final_progress=copy.deepcopy(p), cpu_checks=ctx.cpu_checks,
                      memory=process_memory_snapshot())
        binding(path)
    except torch.cuda.OutOfMemoryError:
        report.update(status='cuda_oom' if stage == 'training_attempt' else 'failed',
                      failed_stage=stage, error=traceback.format_exc())
        if report['status'] != 'cuda_oom' or micro == 512: raise
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      checked_at=datetime.now().astimezone().isoformat())
        write_new(report_path, report)


def accept(path):
    m, c = binding(path)
    reports, cases = [], {}
    initial, identities = set(), []
    for micro in (512, 1024):
        a_path, b_path = [path.parent/f'micro{micro}'/name for name in ('continuous.json', 'resumed.json')]
        a, b = [json.loads(p.read_text()) for p in (a_path, b_path)]
        for report in (a, b):
            assert report['manifest_sha256'] == sha(path) and report['job_id'] == os.environ['SLURM_JOB_ID']
            assert report['source_git_commit'] == m['source_git_commit']
            assert not any(report[k] for k in ('formal_training_ready', 'formal_checkpoint_produced',
                                              'automatic_successor', 'test_records_read'))
        if a['status'] == 'cuda_oom':
            assert micro == 1024 and b['status'] == 'skipped_after_cuda_oom'
            cases[str(micro)] = dict(status='cuda_oom')
        else:
            check_replay(a, b)
            assert len(a['steps']) == 13 and a['final_progress']['updates'] == 13
            assert a['final_progress']['prefix_count'] == 26624
            initial.add(a['initial_state_sha256'])
            identities.append([x['prefix_identity_sha256'] for x in a['steps']])
            for step in a['steps']:
                assert not step['skipped'] and step['gradient_tensors'] == 152
                assert step['rows'] == 2048 and step['microbatches'] == 2048//micro
                assert step['candidate_classes'] > 32768 and step['seconds'] > 0
                assert all(math.isfinite(step[k]) and step[k] >= 0 for k in (
                    'seconds', 'input_seconds', 'training_seconds', 'sid_loss', 'dense_loss', 'grad_norm'))
                assert abs(step['total_loss']-step['sid_loss']-step['dense_loss']) < 1e-10
            measured = a['steps'][3:]
            for key in ('seconds', 'input_seconds', 'training_seconds'):
                assert a['measured'][key] == statistics.mean(x[key] for x in measured)
            assert a['measured']['projected_pass_hours'] == 22460*a['measured']['seconds']/3600
            for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                assert a['measured'][key] == max(x[key] for x in a['steps'])
            assert a['parameters']['total_params'] == 41868160
            assert a['cache']['build']['cache_sha256'] == m['expected_feature_cache_sha256']
            assert a['resources']['allocated_cpus'] == 4 and a['resources']['library_threads'] == 1
            assert all(x['allowed_cpus'] == a['resources']['allowed_cpus'] for x in a['cpu_checks'])
            assert all(0 < x['peak_allocated_bytes'] <= x['peak_reserved_bytes'] <= a['gpus'][0]['total_bytes'] for x in a['steps'])
            entry = a['step_one_checkpoint']
            assert sha(entry['path']) == entry['sha256']
            cases[str(micro)] = dict(status='passed', **a['measured'])
        reports.extend(dict(path=str(p), sha256=sha(p)) for p in (a_path, b_path))
    assert len(initial) == 1 and all(x == identities[0] for x in identities)
    checkpoint = json.loads((path.parent/'micro512/continuous.json').read_text())['validation_checkpoint']
    assert sha(checkpoint['path']) == checkpoint['sha256']
    write_new(path.parent/'training_acceptance.json', dict(status='passed_resource_training',
        job_id=os.environ['SLURM_JOB_ID'], manifest_sha256=sha(path), source_git_commit=m['source_git_commit'],
        cases=cases, reports=reports, validation_checkpoint=checkpoint,
        exact_fresh_process_replay=True, test_records_read=False, formal_training_ready=False,
        automatic_successor=False, checked_at=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['continuous', 'resumed', 'accept'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--microbatch', type=int, choices=[512, 1024], default=512)
    args = parser.parse_args()
    path = args.manifest.resolve()
    if args.mode == 'accept': accept(path)
    else: run(path, args.microbatch, args.mode == 'resumed')
