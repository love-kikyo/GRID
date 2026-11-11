"""Bounded sampled-loss GPU resource probe, never a learning-quality study.

Both cases consume the same real training-prefix snapshot with fresh weights.
Each runs in a separate process and is replayed from its own step-two state in
another process. Timings include the complete sampled train step, but exclude
input snapshot construction, initialization, checkpoint I/O and replay costs.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

from scripts.liger_sampled_probe_manifest import binding, objective_for_case, prepare
from scripts.probe_liger_cpu4 import sha, write_new


def base_report(path, manifest):
    return dict(status='failed', phase=manifest['phase'], job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=manifest['source_git_commit'], manifest_sha256=sha(path),
        formal_training_ready=False, automatic_successor=False,
        validation_test_records_read=False, formal_checkpoint_produced=False)


def deadline(receipt):
    if time.time() >= receipt['deadline_unix_seconds']:
        raise TimeoutError('Shared forty-minute sampled-probe deadline expired')


def capacity_oom(stage, attempt, resumed=False):
    """Only an explicit CUDA OOM in a fresh case's update is capacity data."""
    return (not resumed and stage == 'training_attempt' and isinstance(attempt, int)
            and not isinstance(attempt, bool) and 1 <= attempt <= 12)


def fixture(path):
    import torch  # Torch precedes the compute-node-only TensorFlow import.
    from src.baselines.liger_prefix_runtime import PrefixRuntime
    from src.baselines.liger_efficiency_timing import process_memory_snapshot
    from scripts.accept_liger_efficiency import validate_fixture
    m, c = binding(path)
    started = time.time()
    receipt = base_report(path, m)
    receipt.update(started_unix_seconds=started,
                   deadline_unix_seconds=started + c['deadline_seconds'])
    ctx = PrefixRuntime(m, build_model=False)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    ctx.tf = tf
    stream = ctx.batches(training=True)
    examples, batches = [], []
    try:
        while len(examples) < c['fixture_prefixes']:
            deadline(receipt)
            begin = time.monotonic()
            index, batch = next(stream)
            batches.append(dict(index=index, rows=len(batch), seconds=time.monotonic() - begin))
            examples.extend(batch)
    finally:
        stream.close()
    target = path.parent / 'prefix_fixture.json'
    write_new(target, dict(examples=[asdict(e) for e in examples]))
    ctx.cpu_check()
    receipt.update(status='passed', fixture=dict(path=str(target), sha256=sha(target)),
        prefix_examples=len(examples), unique_parents=len({e.parent_user_id for e in examples}),
        final_target_count=sum(e.is_final for e in examples), data_batches=batches,
        resources=ctx.resources, cpu_checks=ctx.cpu_checks, memory=process_memory_snapshot(),
        elapsed_seconds=time.time() - started, checked_at=datetime.now().astimezone().isoformat())
    validate_fixture(receipt, examples, asset_rows=m['asset_rows'],
                     expected_prefixes=c['fixture_prefixes'], batch_size=c['training']['batch_size'])
    deadline(receipt)
    binding(path)
    deadline(receipt)
    write_new(path.parent / 'fixture_receipt.json', receipt)
    print('[liger-sampled] train-only shared prefix snapshot ready', flush=True)


def load_fixture(path, m, *, check_deadline=True):
    from src.baselines.liger_supervision import PrefixExample
    from scripts.accept_liger_efficiency import validate_fixture
    receipt_path = path.parent / 'fixture_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'passed' and receipt['job_id'] == os.environ['SLURM_JOB_ID']
    assert receipt['manifest_sha256'] == sha(path)
    assert receipt['source_git_commit'] == m['source_git_commit'] and receipt['phase'] == m['phase']
    if check_deadline:
        deadline(receipt)
    assert sha(receipt['fixture']['path']) == receipt['fixture']['sha256']
    payload = json.loads(Path(receipt['fixture']['path']).read_text())['examples']
    examples = [PrefixExample(**dict(v, sequence=tuple(v['sequence']))) for v in payload]
    validate_fixture(receipt, examples, asset_rows=m['asset_rows'], expected_prefixes=12288, batch_size=1024)
    return receipt, examples, dict(path=str(receipt_path), sha256=sha(receipt_path))


def read_case_report(path, m, name, *, resumed=False):
    target = path.parent / f"{name}_{'resumed' if resumed else 'continuous'}.json"
    report = json.loads(target.read_text())
    assert report['manifest_sha256'] == sha(path)
    assert report['job_id'] == os.environ['SLURM_JOB_ID']
    assert report['source_git_commit'] == m['source_git_commit'] and report['name'] == name
    assert report['resumed'] is resumed
    return report


def run_case(path, name, *, resumed=False):
    import torch
    from src.baselines.liger_prefix_runtime import PrefixRuntime
    from src.baselines.liger_prefix_training import initial_progress, check_progress, train_step
    from src.baselines.liger_state import capture, restore_checkpoint, save_checkpoint, tree_digest
    from src.baselines.liger_efficiency_timing import TimedCatalog, timed_call, process_memory_snapshot
    m, c = binding(path)
    setting = next(case for case in c['cases'] if case['name'] == name)
    objective = objective_for_case(m, c, name)
    output = path.parent / f"{name}_{'resumed' if resumed else 'continuous'}.json"
    assert not output.exists()
    receipt, examples, fixture_entry = load_fixture(path, m)
    report = dict(base_report(path, m), **setting, resumed=resumed,
                  fixture_receipt=fixture_entry, steps=[], memory=[])
    started = time.monotonic()
    stage, attempt, ctx = 'preconditions', None, None
    try:
        previous = None
        if resumed:
            previous = read_case_report(path, m, name)
            if previous['status'] in ('cuda_oom', 'skipped_after_cuda_oom'):
                report.update(status='skipped_after_cuda_oom', reason='Fresh case has no completed trajectory to replay')
                return
            assert previous['status'] == 'passed'
        elif name == 'n32768':
            gate_path = path.parent / 'n8192_acceptance.json'
            gate = json.loads(gate_path.read_text())
            assert gate['manifest_sha256'] == sha(path) and gate['job_id'] == report['job_id']
            assert gate['source_git_commit'] == m['source_git_commit']
            report['preceding_case_acceptance'] = dict(path=str(gate_path), sha256=sha(gate_path))
            if gate['status'] == 'capacity_oom':
                report.update(status='skipped_after_cuda_oom', reason='Do not attempt more negatives after an 8K CUDA capacity failure')
                return
            assert gate['status'] == 'passed'
        deadline(receipt)
        stage = 'runtime_initialization'
        ctx = PrefixRuntime(m)
        ctx.binding = dict(ctx.binding, dense_objective=objective,
                           probe_fixture_sha256=fixture_entry['sha256'], probe_case=name)
        report.update(gpu=ctx.gpu, resources=ctx.resources, versions=ctx.versions,
            initial_state_sha256=ctx.initial_sha, checkpoint_binding=ctx.binding,
            setup_seconds=time.monotonic() - started,
            train_seen_rows=len(ctx.seen), cold_rows=len(ctx.cold))
        report['memory'].append(dict(stage='after_runtime', **process_memory_snapshot()))
        p = initial_progress(**ctx.training_budget)
        if resumed:
            stage = 'checkpoint_restore'
            entry = previous['resume_checkpoint']
            assert sha(entry['path']) == entry['sha256']
            begin = time.monotonic()
            p = restore_checkpoint(entry['path'], *ctx.components, ctx.binding)
            report['checkpoint_load_seconds'] = time.monotonic() - begin
            report['loaded_checkpoint'] = entry
            assert p['attempts'] == p['updates'] == p['next_batch'] == c['save_after_attempt']
            assert p['dense_objective'] == objective
        budget = dict(ctx.training_budget, batch_size=setting['batch_size'],
                      prefixes_per_epoch=m['counts']['prefix_examples_per_pass'])
        check_progress(p, budget)
        catalog = TimedCatalog(ctx.catalog)
        for index in range(p['attempts'], c['training']['max_attempts']):
            stage, attempt = 'before_training_attempt', index + 1
            deadline(receipt)
            size = setting['batch_size']
            batch = examples[index * size:(index + 1) * size]
            assert len(batch) == size
            catalog.reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            stage = 'training_attempt'
            training, timing = timed_call(train_step, *ctx.components, ctx.method, ctx.official,
                batch, ctx.sids, ctx.content, ctx.seen, catalog, p,
                dense_sampling=dict(num_negatives=setting['num_negatives'], seed=setting['seed']),
                synchronize=torch.cuda.synchronize)
            stage = 'post_update_checks'
            assert not training['skipped'], 'Resource probe requires successful updates, not AMP-skipped attempts'
            fetch = catalog.snapshot()
            assert fetch['calls'] == 1 and fetch['rows'] == training['dense_candidate_classes']
            assert training['sampled_negative_classes'] == setting['num_negatives']
            check_progress(p, budget)
            step = dict(training=training, measured=index >= c['warmup_attempts'],
                training_wall_seconds=timing['wall_seconds'], catalog_fetch=fetch,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
            report['steps'].append(step)
            report['memory'].append(dict(stage=f'update{index + 1}', **process_memory_snapshot()))
            ctx.cpu_check()
            print('[liger-sampled] ' + json.dumps(dict(case=name, resumed=resumed, **step)), flush=True)
            if not resumed and p['attempts'] == c['save_after_attempt']:
                stage = 'diagnostic_checkpoint_write'
                checkpoint = path.parent / f'{name}_step2.pt'
                begin = time.monotonic()
                save_checkpoint(checkpoint, capture(*ctx.components, p, ctx.binding))
                report['resume_checkpoint'] = dict(path=str(checkpoint), sha256=sha(checkpoint))
                report['diagnostic_checkpoint_write_seconds'] = time.monotonic() - begin
        stage = 'state_fingerprint'
        report.update(final_state_sha256=tree_digest(capture(*ctx.components, p, ctx.binding)),
                      final_progress=p, cpu_checks=ctx.cpu_checks,
                      timing_scope='preloaded prefixes; synchronized complete sampled train step; excludes data snapshot, setup, checkpoint and replay')
        deadline(receipt)
        stage = 'final_binding_check'
        binding(path)
        deadline(receipt)
        stage = 'complete'
        report['status'] = 'passed'
    except torch.cuda.OutOfMemoryError:
        is_capacity = capacity_oom(stage, attempt, resumed)
        report.update(status='cuda_oom' if is_capacity else 'failed', error=traceback.format_exc(),
                      failure_stage=stage, failure_attempt=attempt)
        if not is_capacity:
            raise
        report['memory'].append(dict(stage='failure', **process_memory_snapshot()))
        ctx.cpu_check()
        report['cpu_checks'] = ctx.cpu_checks
    except Exception:
        report.update(status='failed', error=traceback.format_exc(),
                      failure_stage=stage, failure_attempt=attempt)
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, last_stage=stage,
                      checked_at=datetime.now().astimezone().isoformat())
        write_new(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'fixture', 'case', 'resume'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--case', choices=['n8192', 'n32768'])
    args = parser.parse_args()
    path = args.manifest.resolve()
    if args.mode == 'prepare':
        prepare(path)
    elif args.mode == 'fixture':
        fixture(path)
    else:
        if args.case is None:
            parser.error('--case is required for case/resume')
        run_case(path, args.case, resumed=args.mode == 'resume')
