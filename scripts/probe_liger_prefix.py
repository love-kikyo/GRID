"""One necessary gate for the new formal adapter, not another fitting study.

Fresh initialization, real prefix batch1024, exact complete seen/cold catalogs,
shared standardized features, and bitwise fresh-process continuation. Does not
fit a tiny catalog, choose hyperparameters, or authorize a long training budget.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

from scripts.liger_prefix_manifest import binding, prepare
from scripts.probe_liger_cpu4 import sha, write_new


def run(path, resumed=False):
    import torch
    from src.baselines.liger_prefix_runtime import PrefixRuntime
    from src.baselines.liger_prefix_training import initial_progress, check_progress, train_step
    from src.baselines.liger_state import capture, restore_checkpoint, save_checkpoint, tree_digest
    from src.baselines.liger_validation import evaluate
    m = binding(path)
    directory = path.parent / ('resumed' if resumed else 'continuous')
    directory.mkdir(exist_ok=False)
    report = dict(status='failed', phase=m['phase'], job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path), resumed=resumed,
        trained_checkpoint_for_paper=False, test_records_read=False, formal_training_ready=False,
        automatic_successor=False, steps=[])
    started = time.monotonic()
    try:
        ctx = PrefixRuntime(m)
        p = initial_progress(**ctx.training_budget)
        budget = dict(ctx.training_budget, batch_size=ctx.config['training']['batch_size'],
                      prefixes_per_epoch=m['counts']['prefix_examples_per_pass'])
        if resumed:
            previous = json.loads((path.parent / 'continuous/report.json').read_text())
            entry = previous['resume_checkpoint']
            assert previous['status'] == 'passed' and sha(entry['path']) == entry['sha256']
            p = restore_checkpoint(entry['path'], *ctx.components, ctx.binding)
            assert p['next_batch'] == p['updates'] == p['attempts'] == 1
            report['loaded_checkpoint'] = entry
        check_progress(p, budget)
        stream = ctx.batches(training=True, epoch=p['epoch'], start_batch=p['next_batch'])
        try:
            while p['attempts'] < 3:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                index, examples = next(stream)
                assert index == p['next_batch'] and len(examples) == 1024
                step = train_step(*ctx.components, ctx.method, ctx.official, examples, ctx.sids,
                    ctx.content, ctx.seen, ctx.catalog, p, chunk_size=65536)
                torch.cuda.synchronize()
                step['seconds'] = time.monotonic() - start
                step['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                assert not step['skipped']
                assert 0 < step['final_target_count'] < step['rows']
                report['steps'].append(step)
                check_progress(p, budget)
                ctx.cpu_check()
                print('[liger-prefix] ' + json.dumps(step), flush=True)
                if p['attempts'] == 1:
                    checkpoint = directory / 'step1.pt'
                    save_checkpoint(checkpoint, capture(*ctx.components, p, ctx.binding))
                    report['resume_checkpoint'] = dict(path=str(checkpoint), sha256=sha(checkpoint))
                if p['attempts'] == 2:
                    weight = dict(model_sha256=tree_digest(ctx.model.state_dict()), updates=p['updates'])
                    validation = evaluate(ctx.model, ctx.method, ctx.official,
                        ctx.batches(training=False, validation_limit=65), ctx.sids, ctx.content,
                        ctx.index, ctx.seen, ctx.cold, directory / 'validation', 65, weight)
                    report['validation'] = dict(path=str(directory / 'validation/report.json'),
                                                sha256=sha(directory / 'validation/report.json'))
                    p['best'] = dict(epoch=1, **validation['metrics']['hybrid'], **weight)
                    ctx.cpu_check()
            report.update(final_state_sha256=tree_digest(capture(*ctx.components, p, ctx.binding)),
                initial_state_sha256=ctx.initial_sha, final_progress=p, gpu=ctx.gpu,
                resources=ctx.resources, cpu_checks=ctx.cpu_checks, versions=ctx.versions,
                train_seen_rows=len(ctx.seen), cold_rows=len(ctx.cold),
                checkpoint_binding=ctx.binding, elapsed_seconds=time.monotonic() - started)
        finally:
            stream.close()
        binding(path)
        report['status'] = 'passed'
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(directory / 'report.json', report)


def accept(path):
    import numpy as np
    from src.baselines.liger_prefix_runtime import PrefixRuntime
    from src.baselines.liger_prefix_training import check_progress
    from src.baselines.liger_validation import audit
    m = binding(path)
    ctx = PrefixRuntime(m, build_model=False)
    a, b = [json.loads((path.parent / name / 'report.json').read_text()) for name in ('continuous', 'resumed')]
    artifacts = []
    for name, report in (('continuous', a), ('resumed', b)):
        assert report['status'] == 'passed' and report['job_id'] == os.environ['SLURM_JOB_ID']
        assert report['phase'] == m['phase'] and report['source_git_commit'] == m['source_git_commit']
        assert report['manifest_sha256'] == sha(path) and report['checkpoint_binding'] == ctx.binding
        assert not any(report[k] for k in ('test_records_read', 'trained_checkpoint_for_paper', 'formal_training_ready', 'automatic_successor'))
        assert report['train_seen_rows'] == 11542755 and report['cold_rows'] == 23915712
        assert report['resources']['allocated_cpus'] == report['resources']['user_cpu_ceiling'] == 4
        assert len(report['cpu_checks']) == (4 if name == 'resumed' else 5)
        assert all(c['allowed_cpus'] == report['resources']['allowed_cpus'] and c['processes'] for c in report['cpu_checks'])
        for step in report['steps']:
            assert step['rows'] == step['prefix_count'] == 1024
            assert not step['skipped'] and step['gradient_tensors'] == 152
            assert 0 < step['final_target_count'] < 1024 and 0 < step['batch_unique_parents'] <= 1024
            assert step['batch_identity_sha256'] == step['prefix_identity_sha256']
            assert abs(step['learning_rate'] - .0003 * step['updates'] / 963) < 1e-16
        progress = report['final_progress']
        check_progress(progress, dict(ctx.training_budget, batch_size=1024,
            prefixes_per_epoch=m['counts']['prefix_examples_per_pass']))
        assert progress['updates'] == progress['next_batch'] == progress['attempts'] == 3
        assert progress['samples_in_epoch'] == progress['prefix_count'] == 3072
        entry = report['validation']
        assert sha(entry['path']) == entry['sha256']
        val = json.loads(Path(entry['path']).read_text())
        assert val['records'] == 65 and val['batches'] == 3
        audit(Path(entry['path']).parent, ctx.seen, ctx.index, 65, 23915712, val['checkpoint'])
        for artifact in (path.parent / name / 'report.json', Path(entry['path']), Path(entry['path']).parent / 'acceptance.json'):
            artifacts.append(dict(path=str(artifact), sha256=sha(artifact)))
    assert [s['attempt'] for s in a['steps']] == [1, 2, 3]
    assert [s['attempt'] for s in b['steps']] == [2, 3]
    clean = lambda steps: [{k: v for k, v in s.items() if k not in ('seconds', 'peak_allocated_bytes')} for s in steps]
    assert clean(a['steps'][1:]) == clean(b['steps'])
    assert a['initial_state_sha256'] == b['initial_state_sha256']
    assert a['final_state_sha256'] == b['final_state_sha256'] and a['gpu'] == b['gpu']
    assert a['resume_checkpoint'] == b['loaded_checkpoint']
    assert sha(a['resume_checkpoint']['path']) == a['resume_checkpoint']['sha256']
    assert a['final_progress']['final_count'] == sum(s['final_target_count'] for s in a['steps'])
    va, vb = [json.loads(Path(r['validation']['path']).read_text()) for r in (a, b)]
    assert va['metrics'] == vb['metrics'] and va['record_identity_sha256'] == vb['record_identity_sha256']
    assert len(va['artifacts']) == len(vb['artifacts'])
    for x, y in zip(va['artifacts'], vb['artifacts']):
        with np.load(x['path'], allow_pickle=False) as p, np.load(y['path'], allow_pickle=False) as q:
            assert p.files == q.files
            for key in p.files:
                np.testing.assert_array_equal(p[key], q[key])
    ctx.cpu_check()
    # Two warm steps only: expose an indicative cost, never call it a stable benchmark.
    seconds = sum(s['seconds'] for s in a['steps'][1:]) / 2
    write_new(path.parent / 'prefix_acceptance.json', dict(status='passed', job_id=os.environ['SLURM_JOB_ID'],
        checked_at=datetime.now().astimezone().isoformat(), fresh_process_exact_resume=True,
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path), reports=artifacts,
        execution_sources=m['execution_sources'], config_sha256=m['config_sha256'],
        standardization=m['standardization'], counts=m['counts'],
        final_state_sha256=a['final_state_sha256'], gpu=a['gpu'],
        warm_step_seconds_mean=seconds,
        indicative_single_prefix_pass_training_hours=seconds * m['counts']['prefix_batches_per_pass'] / 3600,
        timing_scope='two warm steps; validation/startup/checkpoint/replay excluded; not a stable benchmark',
        no_test_access=True, formal_checkpoint_produced=False, formal_training_ready=False,
        automatic_successor=False, next_action='author confirms new total training budget before formal launch'))
    print('[liger-prefix] exact resume and full-cold validation integration passed; no automatic formal successor', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'continuous', 'resumed', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    path = args.manifest.resolve()
    if args.mode == 'prepare':
        prepare(path)
    elif args.mode == 'accept':
        accept(path)
    else:
        run(path, resumed=args.mode == 'resumed')
