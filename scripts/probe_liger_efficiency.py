"""Bounded train-only cache equivalence and batch-capacity measurement.

Every case runs in a separate Slurm step/process with fresh initialization.
Inputs are preloaded from one frozen-prefix snapshot: measured training times
exclude TFRecord reading, cache construction, setup and any validation. No
checkpoint is produced, and no short-run loss is used to select a model.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
import traceback

from scripts.liger_efficiency_manifest import binding, prepare
from scripts.probe_liger_cpu4 import sha, write_new


def base_report(path, m):
    return dict(status='failed', phase=m['phase'], job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
        formal_training_ready=False, automatic_successor=False,
        validation_test_records_read=False, formal_checkpoint_produced=False)


def deadline(receipt):
    if time.time() >= receipt['deadline_unix_seconds']:
        raise TimeoutError('Shared forty-minute efficiency-probe deadline expired')


def capacity_oom(stage, batch_size, attempt):
    """Only an explicit CUDA OOM inside a large-batch update is capacity data."""
    return (stage == 'training_attempt' and batch_size > 1024
            and isinstance(attempt, int) and not isinstance(attempt, bool) and 1 <= attempt <= 4)


def fixture(path):
    import torch
    from src.baselines.liger_prefix_runtime import PrefixRuntime
    from src.baselines.liger_efficiency_timing import process_memory_snapshot
    m, c = binding(path)
    started = time.time()
    receipt = base_report(path, m)
    receipt['started_unix_seconds'] = started
    receipt['deadline_unix_seconds'] = started + c['deadline_seconds']
    ctx = PrefixRuntime(m, build_model=False)
    # Compatible compute node only; Torch precedes CPU-only TensorFlow.
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
            start = time.monotonic()
            index, values = next(stream)
            batches.append(dict(index=index, rows=len(values), seconds=time.monotonic() - start))
            examples.extend(values)
    finally:
        stream.close()
    assert len(examples) == 16384 and len({(e.parent_user_id, e.prefix_target_index) for e in examples}) == len(examples)
    target = path.parent / 'prefix_fixture.json'
    write_new(target, dict(examples=[asdict(e) for e in examples]))
    ctx.cpu_check()
    receipt.update(status='passed', fixture=dict(path=str(target), sha256=sha(target)),
        prefix_examples=len(examples), unique_parents=len({e.parent_user_id for e in examples}),
        final_target_count=sum(e.is_final for e in examples), data_batches=batches,
        resources=ctx.resources, cpu_checks=ctx.cpu_checks, memory=process_memory_snapshot(),
        elapsed_seconds=time.time() - started, checked_at=datetime.now().astimezone().isoformat())
    from scripts.accept_liger_efficiency import validate_fixture
    validate_fixture(receipt, examples, asset_rows=m['asset_rows'])
    deadline(receipt)
    write_new(path.parent / 'fixture_receipt.json', receipt)
    print('[liger-efficiency] train-only prefix snapshot ready', flush=True)


def load_fixture(path, m, *, check_deadline=True):
    from src.baselines.liger_supervision import PrefixExample
    receipt_path = path.parent / 'fixture_receipt.json'
    r = json.loads(receipt_path.read_text())
    assert r['status'] == 'passed' and r['job_id'] == os.environ['SLURM_JOB_ID']
    assert r['manifest_sha256'] == sha(path) and r['source_git_commit'] == m['source_git_commit']
    if check_deadline:
        deadline(r)
    assert sha(r['fixture']['path']) == r['fixture']['sha256']
    values = json.loads(Path(r['fixture']['path']).read_text())['examples']
    examples = [PrefixExample(**dict(v, sequence=tuple(v['sequence']))) for v in values]
    from scripts.accept_liger_efficiency import validate_fixture
    validate_fixture(r, examples, asset_rows=m['asset_rows'])
    return r, examples, dict(path=str(receipt_path), sha256=sha(receipt_path))


def run_case(path, name):
    import numpy as np
    import torch
    from src.baselines.liger_prefix_runtime import PrefixRuntime
    from src.baselines.liger_prefix_training import initial_progress, check_progress, train_step
    from src.baselines.liger_state import capture, tree_digest
    from src.baselines.liger_feature_cache import build_cached_seen, verify_cached_seen
    from src.baselines.liger_efficiency_timing import (
        TimedCatalog, TimedModelForward, timed_call, process_memory_snapshot)
    m, c = binding(path)
    setting = next(case for case in c['cases'] if case['name'] == name)
    output = path.parent / f'{name}.json'
    assert not output.exists()
    fixture_receipt, examples, fixture_entry = load_fixture(path, m)
    report = dict(base_report(path, m), **setting, fixture_receipt=fixture_entry, steps=[], memory=[])
    started = time.monotonic()
    stage, current_attempt, ctx = 'preconditions', None, None
    try:
        if setting['batch_size'] > 1024:
            from scripts.accept_liger_efficiency import check_equivalence_receipt
            report['equivalence_receipt'] = check_equivalence_receipt(path)
        if name == 'cached4096':
            previous = json.loads((path.parent / 'cached2048.json').read_text())
            assert previous['manifest_sha256'] == sha(path) and previous['job_id'] == report['job_id']
            if previous['status'] == 'cuda_oom':
                report.update(status='skipped_after_cuda_oom', reason='Do not attempt a larger batch after capacity failure')
                return
            assert previous['status'] == 'passed'
        deadline(fixture_receipt)
        stage = 'runtime_initialization'
        ctx = PrefixRuntime(m)
        report.update(gpu=ctx.gpu, resources=ctx.resources, versions=ctx.versions,
                      initial_state_sha256=ctx.initial_sha, setup_seconds=time.monotonic() - started)
        report['memory'].append(dict(stage='before_cache', **process_memory_snapshot()))
        if setting['backend'] == 'cached_seen':
            expected = dict(standardization_stats_sha256=ctx.standardization['stats_sha256'],
                seen_rows_sha256=ctx.standardization['seen_rows_sha256'],
                source_sha256=ctx.standardization['content_sha256'])
            cache_start = time.monotonic()
            stage = 'cache_build'
            content, catalog, summary = build_cached_seen(ctx.content, ctx.seen,
                chunk_size=65536, expected_binding=expected)
            report['cache_build_seconds'] = time.monotonic() - cache_start
            verify_start = time.monotonic()
            stage = 'cache_verification'
            report['cache_verification'] = verify_cached_seen(content, catalog,
                expected_binding=expected, chunk_size=65536)
            report['cache_verify_seconds'] = time.monotonic() - verify_start
            # Seen, cold, true row0, and multidimensional row addressing use the
            # same full-asset coordinate system, without reading held-out data.
            stage = 'cache_lookup_check'
            rows = np.concatenate((np.array([0, len(content) - 1]), ctx.seen[:7], ctx.cold[:7]))
            np.testing.assert_array_equal(content[rows.reshape(4, 4)], ctx.content[rows.reshape(4, 4)])
            assert len(content) - len(ctx.seen) == 23915712
            assert catalog.content is content and catalog.rows is ctx.seen
            ctx.content, ctx.catalog = content, catalog
            report['cache'] = summary
            report['mixed_global_row_lookup_exact'] = True
        else:
            report['cache_build_seconds'] = report['cache_verify_seconds'] = 0.
        stage = 'training_setup'
        report['memory'].append(dict(stage='after_cache', **process_memory_snapshot()))
        timed_catalog = TimedCatalog(ctx.catalog)
        timed_forward = TimedModelForward(ctx.official, synchronize=torch.cuda.synchronize)
        p = initial_progress(**ctx.training_budget)
        batch_size = setting['batch_size']
        assert len(examples) >= 4 * batch_size
        for attempt in range(4):
            stage, current_attempt = 'before_training_attempt', attempt + 1
            deadline(fixture_receipt)
            current = examples[attempt * batch_size:(attempt + 1) * batch_size]
            timed_catalog.reset()
            timed_forward.reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            stage = 'training_attempt'
            measurement, timing = timed_call(train_step, *ctx.components, ctx.method, timed_forward,
                current, ctx.sids, ctx.content, ctx.seen, timed_catalog, p,
                chunk_size=65536, synchronize=torch.cuda.synchronize)
            stage = 'post_update_checks'
            assert not measurement['skipped'], 'Efficiency case requires all four successful updates'
            fetch = timed_catalog.snapshot()
            assert fetch['rows'] == 2 * len(ctx.seen) and fetch['calls'] == 2 * math.ceil(len(ctx.seen) / 65536)
            check_progress(p, dict(ctx.training_budget, batch_size=batch_size,
                prefixes_per_epoch=m['counts']['prefix_examples_per_pass']))
            step = dict(training=measurement, measured=attempt >= c['warmup_attempts'],
                training_wall_seconds=timing['wall_seconds'], catalog_fetch=fetch,
                model_forward=timed_forward.snapshot(),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
            report['steps'].append(step)
            report['memory'].append(dict(stage=f'update{attempt + 1}', **process_memory_snapshot()))
            ctx.cpu_check()
            print('[liger-efficiency] ' + json.dumps(dict(case=name, **step)), flush=True)
        # Exclude only execution/source binding, which describes a backend, not
        # learning state. All weights, optimizer, RNG, scaler and cursors remain.
        stage = 'state_fingerprint'
        state = capture(*ctx.components, p, ctx.binding)
        state.pop('binding')
        report.update(learning_state_sha256=tree_digest(state), final_progress=p,
            cpu_checks=ctx.cpu_checks, train_seen_rows=len(ctx.seen), cold_rows=len(ctx.cold),
            checkpoint_binding=ctx.binding,
            timing_scope='preloaded prefix inputs; synchronized training wall; catalog CPU fetch may overlap GPU; no pure-GPU subtraction',
            status='passed')
        deadline(fixture_receipt)
        stage = 'final_binding_check'
        binding(path)
        stage = 'complete'
    except torch.cuda.OutOfMemoryError:
        # A separate process owns each case. Never continue training after OOM.
        is_capacity = capacity_oom(stage, setting['batch_size'], current_attempt)
        report.update(status='cuda_oom' if is_capacity else 'failed', error=traceback.format_exc(),
            failure_stage=stage, failure_attempt=current_attempt)
        if not is_capacity:
            raise
        report['memory'].append(dict(stage='failure', **process_memory_snapshot()))
        ctx.cpu_check()
        report['cpu_checks'] = ctx.cpu_checks
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        report['failure_stage'] = stage
        report['failure_attempt'] = current_attempt
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      last_stage=stage,
                      checked_at=datetime.now().astimezone().isoformat())
        write_new(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'fixture', 'case'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--case', choices=['lazy1024', 'cached1024', 'cached2048', 'cached4096'])
    args = parser.parse_args()
    path = args.manifest.resolve()
    if args.mode == 'prepare':
        prepare(path)
    elif args.mode == 'fixture':
        fixture(path)
    else:
        if args.case is None:
            parser.error('--case is required')
        run_case(path, args.case)
