"""Same-configuration sampled training/resume/full-cold integration gate.

Three real streaming updates only, using the authorized one-pass schedule.
These diagnostic weights never initialize the separate fresh formal run.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

from scripts.liger_sampled_training_manifest import binding, GATE_PHASE
from scripts.probe_liger_cpu4 import sha, write_new


def check_deadline(clock):
    if time.time() > clock['deadline_unix_seconds']:
        raise TimeoutError('Shared 40-minute sampled training integration deadline exceeded')


def gate_clock(path, manifest, resumed):
    target = path.parent / 'gate_clock.json'
    if not resumed:
        now = time.time()
        write_new(target, dict(job_id=os.environ['SLURM_JOB_ID'], manifest_sha256=sha(path),
                              started_unix_seconds=now, deadline_unix_seconds=now + 2400))
    clock = json.loads(target.read_text())
    assert clock['job_id'] == os.environ['SLURM_JOB_ID'] and clock['manifest_sha256'] == sha(path)
    assert clock['deadline_unix_seconds'] - clock['started_unix_seconds'] == 2400
    check_deadline(clock)
    return clock, dict(path=str(target), sha256=sha(target))


def clean_steps(steps):
    return [{k: v for k, v in step.items() if k not in
             ('seconds', 'input_seconds', 'training_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')}
            for step in steps]


def check_gate_pair(continuous, resumed):
    """Pure evidence checks; predictions and file hashes are checked separately."""
    a, b = continuous, resumed
    assert [s['attempt'] for s in a['steps']] == [1, 2, 3]
    assert [s['attempt'] for s in b['steps']] == [2, 3]
    assert clean_steps(a['steps'][1:]) == clean_steps(b['steps'])
    for key in ('initial_state_sha256', 'final_state_sha256', 'final_progress', 'gpu',
                'versions', 'checkpoint_binding', 'clock', 'numpy_memory_policy'):
        assert a[key] == b[key], f'Fresh-process replay mismatch: {key}'
    assert a['resume_checkpoint'] == b['loaded_checkpoint']
    assert b['resume_proof']['full_state_exact'] is True
    assert b['resume_proof']['full_state_sha256'] == a['resume_checkpoint']['full_state_sha256']
    for report in (a, b):
        proof = report['validation_state_proof']
        assert proof['full_learning_state_unchanged'] is True
        assert proof['before_state_sha256'] == proof['after_state_sha256']
        assert report['final_progress']['updates'] == report['final_progress']['attempts'] == 3
        assert report['final_progress']['prefix_count'] == 6144
        assert report['final_progress']['final_count'] == sum(s['final_target_count'] for s in a['steps'])


def run(path, resumed=False):
    import torch
    from src.baselines.liger_prefix_training import initial_progress, check_progress, train_step
    from src.baselines.liger_state import capture, tree_digest
    from scripts.train_liger_sampled_cpu4 import (
        build_context, save_bound_checkpoint, restore_bound_checkpoint, perform_validation)
    m = binding(path)
    assert m['phase'] == GATE_PHASE and not m['formal_training_ready']
    directory = path.parent / ('resumed' if resumed else 'continuous')
    directory.mkdir(exist_ok=False)
    clock, clock_entry = gate_clock(path, m, resumed)
    report = dict(status='failed', phase=m['phase'], job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path), resumed=resumed,
        formal_checkpoint_produced=False, test_records_read=False, formal_training_ready=False,
        automatic_successor=False, steps=[], clock=clock_entry)
    started = time.monotonic()
    try:
        ctx = build_context(m, deadline=lambda: check_deadline(clock))
        p = initial_progress(**ctx.training_budget)
        budget = dict(ctx.training_budget, batch_size=2048, prefixes_per_epoch=45996113)
        if resumed:
            previous = json.loads((path.parent / 'continuous/report.json').read_text())
            assert previous['status'] == 'passed' and previous['manifest_sha256'] == sha(path)
            entry = previous['resume_checkpoint']
            p, proof = restore_bound_checkpoint(ctx, entry)
            assert p['next_batch'] == p['updates'] == p['attempts'] == 1
            report.update(loaded_checkpoint=entry, resume_proof=proof)
        check_progress(p, budget)
        stream = ctx.batches(training=True, epoch=0, start_batch=p['next_batch'])
        try:
            while p['attempts'] < 3:
                check_deadline(clock)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin = time.monotonic()
                index, records = next(stream)
                input_seconds = time.monotonic() - begin
                assert index == p['next_batch'] and len(records) == 2048
                train_begin = time.monotonic()
                step = train_step(*ctx.components, ctx.method, ctx.official, records,
                    ctx.sids, ctx.content, ctx.seen, ctx.catalog, p,
                    dense_sampling=dict(num_negatives=32768, seed=42))
                torch.cuda.synchronize()
                step.update(input_seconds=input_seconds, training_seconds=time.monotonic() - train_begin,
                    seconds=time.monotonic() - begin,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), data_chain=p['data_chain'])
                assert not step['skipped'] and 0 < step['final_target_count'] < 2048
                report['steps'].append(step)
                check_progress(p, budget)
                ctx.cpu_check()
                print('[liger-sampled-gate] ' + json.dumps(step), flush=True)
                if p['attempts'] == 1:
                    report['resume_checkpoint'] = save_bound_checkpoint(ctx, p, directory / 'step1.pt')
                if p['attempts'] == 2:
                    check_deadline(clock)
                    weight = dict(model_sha256=tree_digest(ctx.model.state_dict()), updates=p['updates'])
                    evaluated, audited, proof = perform_validation(ctx, p, directory / 'validation',
                        65, weight, validation_limit=65)
                    report.update(validation=dict(path=str(directory / 'validation/report.json'),
                        sha256=sha(directory / 'validation/report.json')), validation_state_proof=proof)
                    assert evaluated['metrics'] == audited['metrics']
                    ctx.cpu_check()
            report.update(final_state_sha256=tree_digest(capture(*ctx.components, p, ctx.binding)),
                initial_state_sha256=ctx.initial_sha, final_progress=p, gpu=ctx.gpu,
                resources=ctx.resources, cpu_checks=ctx.cpu_checks, versions=ctx.versions,
                checkpoint_binding=ctx.binding, feature_cache=ctx.feature_cache_report,
                numpy_memory_policy=ctx.numpy_memory_policy,
                elapsed_seconds=time.monotonic() - started)
        finally:
            stream.close()
        binding(path)
        check_deadline(clock)
        report['status'] = 'passed'
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(directory / 'report.json', report)


def accept(path):
    import numpy as np
    from scripts.train_liger_sampled_cpu4 import build_context
    from src.baselines.liger_prefix_training import check_progress
    from src.baselines.liger_sampled_catalog import sampling_binding
    from src.baselines.liger_validation import audit
    m = binding(path)
    assert m['phase'] == GATE_PHASE and not m['formal_training_ready']
    ctx = build_context(m, build_model=False)
    a, b = [json.loads((path.parent / name / 'report.json').read_text())
            for name in ('continuous', 'resumed')]
    artifacts = []
    for name, report in (('continuous', a), ('resumed', b)):
        assert report['status'] == 'passed' and report['job_id'] == os.environ['SLURM_JOB_ID']
        assert report['phase'] == m['phase'] and report['source_git_commit'] == m['source_git_commit']
        assert report['manifest_sha256'] == sha(path) and report['checkpoint_binding'] == ctx.binding
        assert not any(report[k] for k in ('test_records_read', 'formal_checkpoint_produced',
                                           'formal_training_ready', 'automatic_successor'))
        assert report['resumed'] is (name == 'resumed')
        assert report['initial_state_sha256'] == m['expected_initial_state_sha256']
        assert '4090' in report['gpu']['name'] and report['gpu']['uuid']
        resource = report['resources']
        assert resource['allocated_cpus'] == resource['user_cpu_ceiling'] == 4
        assert 1 <= len(resource['allowed_cpus']) <= 4 and resource['library_threads'] == 1
        assert len(report['cpu_checks']) >= len(report['steps']) + 1
        assert all(c['allowed_cpus'] == resource['allowed_cpus'] and c['processes'] for c in report['cpu_checks'])
        cache = report['feature_cache']
        assert cache['build']['cache_sha256'] == cache['verification']['cache_sha256'] == m['expected_feature_cache_sha256']
        assert cache['verification']['status'] == 'passed' and cache['verification']['feature_values_bitwise_equal']
        assert cache['build']['rows'] == 11542755 and cache['build']['cache_bytes'] == 5909890560
        assert report['numpy_memory_policy']['effective_madvise_hugepage'] is False
        for step in report['steps']:
            assert step['rows'] == step['prefix_count'] == 2048
            assert not step['skipped'] and step['gradient_tensors'] == 152
            assert step['dense_objective'] == sampling_binding(dict(num_negatives=32768, seed=42))
            assert step['sampled_negative_classes'] == 32768
            assert step['sampling_attempt'] == step['attempt'] - 1
            assert step['dense_candidate_classes'] == step['forced_positive_classes'] + 32768
            assert 0 < step['final_target_count'] < 2048 and 0 < step['batch_unique_parents'] <= 2048
            assert step['batch_identity_sha256'] == step['prefix_identity_sha256']
            assert abs(step['learning_rate'] - .0003 * step['updates'] / 1123) < 1e-16
            assert step['seconds'] > 0 and step['training_seconds'] > 0 and step['input_seconds'] >= 0
            assert 0 < step['peak_allocated_bytes'] <= step['peak_reserved_bytes'] <= report['gpu']['total_bytes']
        check_progress(report['final_progress'], dict(ctx.training_budget,
            batch_size=2048, prefixes_per_epoch=45996113))
        entry = report['validation']
        assert sha(entry['path']) == entry['sha256']
        val = json.loads(Path(entry['path']).read_text())
        assert val['records'] == 65 and val['batches'] == 3
        recalculated = audit(Path(entry['path']).parent, ctx.seen, ctx.index, 65, 23915712,
                             val['checkpoint'], write_receipt=False)
        val_acceptance = Path(entry['path']).parent / 'acceptance.json'
        assert recalculated == json.loads(val_acceptance.read_text())
        assert sha(report['clock']['path']) == report['clock']['sha256']
        check_deadline(json.loads(Path(report['clock']['path']).read_text()))
        for artifact in (path.parent / name / 'report.json', Path(entry['path']), val_acceptance):
            artifacts.append(dict(path=str(artifact), sha256=sha(artifact)))
    check_gate_pair(a, b)
    entry = a['resume_checkpoint']
    assert sha(entry['path']) == entry['sha256']
    va, vb = [json.loads(Path(r['validation']['path']).read_text()) for r in (a, b)]
    assert va['metrics'] == vb['metrics'] and va['record_identity_sha256'] == vb['record_identity_sha256']
    assert len(va['artifacts']) == len(vb['artifacts'])
    for x, y in zip(va['artifacts'], vb['artifacts']):
        with np.load(x['path'], allow_pickle=False) as p, np.load(y['path'], allow_pickle=False) as q:
            assert p.files == q.files
            for key in p.files:
                np.testing.assert_array_equal(p[key], q[key])
        artifacts.extend([x, y])
    artifacts.extend([entry, a['clock']])
    ctx.cpu_check()
    binding(path)
    write_new(path.parent / 'training_gate_acceptance.json', dict(status='passed', phase=GATE_PHASE,
        job_id=os.environ['SLURM_JOB_ID'], checked_at=datetime.now().astimezone().isoformat(),
        gate_manifest=dict(path=str(path), sha256=sha(path)), manifest_sha256=sha(path),
        source_git_commit=m['source_git_commit'], execution_sources=m['execution_sources'],
        config_sha256=m['config_sha256'], dense_objective=sampling_binding(dict(num_negatives=32768, seed=42)),
        fresh_process_exact_resume=True, full_cold_validation_records=65,
        validation_preserved_full_learning_state=True, final_state_sha256=a['final_state_sha256'],
        initial_state_sha256=a['initial_state_sha256'], gpu=a['gpu'], reports=artifacts,
        formal_checkpoint_produced=False, test_records_read=False, automatic_successor=False))
    print('[liger-sampled-gate] exact fresh-process resume and full-cold transition accepted', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['continuous', 'resumed', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    path = args.manifest.resolve()
    if args.mode == 'accept':
        accept(path)
    else:
        run(path, resumed=args.mode == 'resumed')
