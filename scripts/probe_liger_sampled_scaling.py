"""Train-only reading diagnostics and sampled-loss batch-capacity probe.

Historical run artifacts remain immutable; this version uses an explicitly
audited source migration. Frozen-input RAM caching must reproduce the complete
learning trajectory at batch1024 before larger batches are allowed. Larger
batches are resource cases, not equivalent optimization trajectories. Each case
owns a fresh process, model and (when requested) immutable CPU feature cache.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

from scripts.liger_sampled_scaling_manifest import binding, objective_for_case, prepare
from scripts.probe_liger_cpu4 import sha, write_new
from scripts.probe_liger_sampled import base_report, deadline


def numpy_memory_policy(np_module=None):
    """Verify a process-local allocation policy, not a global kernel change.

    NumPy reads this environment variable on import. Checking the live getter
    detects a too-late environment change. This mitigation does not establish
    hugepage compaction as the cause of the previous preparation slowdown.
    """
    if np_module is None:
        import numpy as np_module
    if os.environ.get('NUMPY_MADVISE_HUGEPAGE') != '0':
        raise ValueError('NUMPY_MADVISE_HUGEPAGE=0 must be set before Python/NumPy import')
    core = getattr(np_module, '_core', None)
    getter = getattr(getattr(core, '_multiarray_umath', None), '_get_madvise_hugepage', None)
    if not callable(getter) or getter() is not False:
        raise ValueError('Cannot verify that process-local NumPy hugepage advice is disabled')
    return dict(environment_value='0', effective_madvise_hugepage=False,
        numpy_version=np_module.__version__, verification_api='numpy._core._multiarray_umath._get_madvise_hugepage',
        scope='current NumPy process only; no global kernel settings written',
        interpretation='diagnostic mitigation; hugepage compaction is not a proven cause of previous slowdown')


def record_phase(report, phase, *, started, progress=None):
    """Bounded diagnostic heartbeat; no CUDA synchronization or RNG mutation."""
    from src.baselines.liger_access_timing import io_snapshot
    from src.baselines.liger_efficiency_timing import process_memory_snapshot
    allowed = sorted(os.sched_getaffinity(0))
    assert 1 <= len(allowed) <= 4
    event = dict(phase=phase, elapsed_seconds=time.monotonic() - started,
        allowed_cpus=allowed, memory=process_memory_snapshot(), io=io_snapshot())
    if progress is not None:
        event.update(progress)
    report['phase_progress'].append(event)
    print('[liger-scaling-phase] ' + json.dumps(dict(case=report['name'], **event)), flush=True)


def cache_progress_callback(report, receipt, *, started, every_chunks=16):
    """Check the shared deadline every completed chunk; log only bounded groups."""
    if type(every_chunks) is not int or every_chunks <= 0:
        raise ValueError('Progress heartbeat interval must be a positive integer')
    counts = {}
    def progress(*, stage, completed_rows, total_rows):
        deadline(receipt)
        assert stage in ('cache_build', 'cache_verification')
        assert type(completed_rows) is int and type(total_rows) is int
        assert 0 < completed_rows <= total_rows
        counts[stage] = counts.get(stage, 0) + 1
        if counts[stage] % every_chunks == 0 or completed_rows == total_rows:
            record_phase(report, stage + '_progress', started=started,
                progress=dict(completed_rows=completed_rows, total_rows=total_rows,
                              completed_chunks=counts[stage]))
    return progress


def capacity_oom(stage, batch_size, attempt):
    return (stage == 'training_attempt' and batch_size in (2048, 4096)
            and isinstance(attempt, int) and not isinstance(attempt, bool) and 1 <= attempt <= 20)


def fixture(path):
    import torch  # Initialize Torch before compute-node-only TensorFlow.
    from src.baselines.liger_prefix_runtime import PrefixRuntime
    from src.baselines.liger_efficiency_timing import process_memory_snapshot
    from scripts.accept_liger_efficiency import validate_fixture
    m, c = binding(path)
    started = time.time()
    receipt = base_report(path, m)
    receipt.update(started_unix_seconds=started,
                   deadline_unix_seconds=started + c['deadline_seconds'],
                   numpy_memory_policy=numpy_memory_policy())
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
                     expected_prefixes=c['fixture_prefixes'], batch_size=1024)
    binding(path)
    deadline(receipt)
    write_new(path.parent / 'fixture_receipt.json', receipt)
    print('[liger-scaling] shared train-only prefix snapshot ready', flush=True)


def load_fixture(path, m, *, check_deadline=True):
    from src.baselines.liger_supervision import PrefixExample
    from scripts.accept_liger_efficiency import validate_fixture
    receipt_path = path.parent / 'fixture_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'passed' and receipt['job_id'] == os.environ['SLURM_JOB_ID']
    assert receipt['manifest_sha256'] == sha(path)
    assert receipt['source_git_commit'] == m['source_git_commit'] and receipt['phase'] == m['phase']
    assert receipt['numpy_memory_policy']['environment_value'] == '0'
    assert receipt['numpy_memory_policy']['effective_madvise_hugepage'] is False
    if check_deadline:
        deadline(receipt)
    assert sha(receipt['fixture']['path']) == receipt['fixture']['sha256']
    values = json.loads(Path(receipt['fixture']['path']).read_text())['examples']
    examples = [PrefixExample(**dict(v, sequence=tuple(v['sequence']))) for v in values]
    validate_fixture(receipt, examples, asset_rows=m['asset_rows'], expected_prefixes=81920, batch_size=1024)
    return receipt, examples, dict(path=str(receipt_path), sha256=sha(receipt_path))


def check_large_case_gate(path, m, setting):
    """Do not infer permission from a successful training process alone."""
    target = path.parent / 'cache_equivalence.json'
    gate = json.loads(target.read_text())
    assert gate['status'] == 'passed_exact_cache_equivalence'
    assert gate['manifest_sha256'] == sha(path) and gate['source_git_commit'] == m['source_git_commit']
    assert gate['job_id'] == os.environ['SLURM_JOB_ID']
    for entry in gate['reports']:
        assert sha(entry['path']) == entry['sha256']
    skip = False
    if setting['batch_size'] == 4096:
        previous = path.parent / (setting['name'].removesuffix('4096') + '2048.json')
        report = json.loads(previous.read_text())
        assert report['manifest_sha256'] == sha(path) and report['job_id'] == gate['job_id']
        assert report['source_git_commit'] == m['source_git_commit']
        assert report['batch_size'] == 2048 and report['num_negatives'] == setting['num_negatives']
        assert report['status'] in ('passed', 'cuda_oom')
        skip = report['status'] == 'cuda_oom'
    return dict(path=str(target), sha256=sha(target)), skip


def run_case(path, name):
    import torch
    from src.baselines.liger_prefix_runtime import PrefixRuntime
    from src.baselines.liger_prefix_training import initial_progress, check_progress, train_step
    from src.baselines.liger_state import capture, tree_digest
    from src.baselines.liger_full_catalog import RowCatalog
    from src.baselines.liger_feature_cache import build_cached_seen, verify_cached_seen
    from src.baselines.liger_efficiency_timing import TimedCatalog, timed_call, process_memory_snapshot
    from src.baselines.liger_access_timing import (
        AccessTimer, TimedStandardizedContent, CatalogAccessView, summarize_access, io_snapshot, delta_io)
    m, c = binding(path)
    setting = next(case for case in c['cases'] if case['name'] == name)
    objective = objective_for_case(m, c, name)
    output = path.parent / f'{name}.json'
    assert not output.exists()
    receipt, examples, fixture_entry = load_fixture(path, m)
    report = dict(base_report(path, m), **setting, fixture_receipt=fixture_entry, steps=[], memory=[],
                  phase_progress=[], numpy_memory_policy=numpy_memory_policy())
    started = time.monotonic()
    stage, attempt, ctx = 'preconditions', None, None
    try:
        if setting['batch_size'] > 1024:
            entry, skip = check_large_case_gate(path, m, setting)
            report['equivalence_receipt'] = entry
            if skip:
                report.update(status='skipped_after_cuda_oom', reason='Same negative count failed batch2048 capacity')
                return
        deadline(receipt)
        stage = 'runtime_initialization'
        record_phase(report, stage, started=started)
        ctx = PrefixRuntime(m)
        ctx.binding = dict(ctx.binding, dense_objective=objective,
            probe_fixture_sha256=fixture_entry['sha256'], probe_case=name,
            feature_backend=setting['backend'], probe_batch_size=setting['batch_size'])
        raw = AccessTimer(ctx.raw_content)
        standardized = TimedStandardizedContent(raw, ctx.standardization,
            expected_content_sha256=m['assets']['model_embedding_copy']['sha256'])
        content, catalog = standardized, RowCatalog(standardized, ctx.seen)
        report.update(gpu=ctx.gpu, resources=ctx.resources, versions=ctx.versions,
            initial_state_sha256=ctx.initial_sha, checkpoint_binding=ctx.binding,
            setup_seconds=time.monotonic() - started,
            train_seen_rows=len(ctx.seen), cold_rows=len(ctx.cold))
        report['memory'].append(dict(stage='after_runtime', **process_memory_snapshot()))
        record_phase(report, 'after_runtime', started=started)
        deadline(receipt)
        if setting['backend'] == 'cached_seen':
            expected = dict(standardization_stats_sha256=ctx.standardization['stats_sha256'],
                seen_rows_sha256=ctx.standardization['seen_rows_sha256'],
                source_sha256=ctx.standardization['content_sha256'])
            stage = 'cache_build'
            record_phase(report, stage, started=started)
            begin = time.monotonic()
            progress = cache_progress_callback(report, receipt, started=started)
            content, catalog, report['cache'] = build_cached_seen(
                standardized, ctx.seen, expected_binding=expected, chunk_size=65536,
                progress_callback=progress)
            report['cache_build_seconds'] = time.monotonic() - begin
            record_phase(report, 'after_cache_build', started=started)
            deadline(receipt)
            stage = 'cache_verification'
            record_phase(report, stage, started=started)
            begin = time.monotonic()
            report['cache_verification'] = verify_cached_seen(content, catalog,
                expected_binding=expected, chunk_size=65536, progress_callback=progress)
            report['cache_verify_seconds'] = time.monotonic() - begin
            record_phase(report, 'after_cache_verification', started=started)
        else:
            report['cache_build_seconds'] = report['cache_verify_seconds'] = 0.
        deadline(receipt)
        stage = 'training_setup'
        history = AccessTimer(content)
        timed_catalog = TimedCatalog(CatalogAccessView(catalog, history))
        ctx.content, ctx.catalog = history, timed_catalog
        report['memory'].append(dict(stage='after_backend', **process_memory_snapshot()))
        record_phase(report, 'after_backend', started=started)
        p = initial_progress(**ctx.training_budget)
        budget = dict(ctx.training_budget, batch_size=setting['batch_size'],
                      prefixes_per_epoch=m['counts']['prefix_examples_per_pass'])
        for index in range(c['training']['max_attempts']):
            stage, attempt = 'before_training_attempt', index + 1
            deadline(receipt)
            size = setting['batch_size']
            batch = examples[index * size:(index + 1) * size]
            assert len(batch) == size
            for timer in (raw, standardized, history, timed_catalog):
                timer.reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = io_snapshot()
            stage = 'training_attempt'
            training, timing = timed_call(train_step, *ctx.components, ctx.method, ctx.official,
                batch, ctx.sids, ctx.content, ctx.seen, ctx.catalog, p,
                dense_sampling=dict(num_negatives=setting['num_negatives'], seed=setting['seed']),
                synchronize=torch.cuda.synchronize)
            io = delta_io(before, io_snapshot())
            stage = 'post_update_checks'
            assert not training['skipped'], 'Resource probe requires successful optimizer updates'
            fetch = timed_catalog.snapshot()
            assert fetch['calls'] == 1 and fetch['rows'] == training['dense_candidate_classes']
            check_progress(p, budget)
            step = dict(training=training, measured=index >= c['warmup_attempts'],
                training_wall_seconds=timing['wall_seconds'], catalog_fetch=fetch,
                access=summarize_access(raw, standardized, history, fetch), io_delta=io,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
            report['steps'].append(step)
            report['memory'].append(dict(stage=f'update{index + 1}', **process_memory_snapshot()))
            ctx.cpu_check()
            print('[liger-scaling] ' + json.dumps(dict(case=name, **step)), flush=True)
        stage = 'state_fingerprint'
        state = capture(*ctx.components, p, ctx.binding)
        state.pop('binding')  # Backend identity is validated separately, not learning state.
        report.update(learning_state_sha256=tree_digest(state), final_progress=p,
            cpu_checks=ctx.cpu_checks,
            timing_scope='instrumented preloaded real prefixes; synchronized full train step; cache/setup/input/validation excluded')
        record_phase(report, 'training_complete', started=started)
        stage = 'final_binding_check'
        binding(path)
        deadline(receipt)
        stage = 'complete'
        record_phase(report, stage, started=started)
        report['status'] = 'passed'
    except torch.cuda.OutOfMemoryError:
        valid = capacity_oom(stage, setting['batch_size'], attempt)
        report.update(status='cuda_oom' if valid else 'failed', error=traceback.format_exc(),
                      failure_stage=stage, failure_attempt=attempt)
        if not valid:
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
    parser.add_argument('mode', choices=['prepare', 'fixture', 'case'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--case')
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
