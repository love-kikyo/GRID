"""Real batch1024, complete training/cold catalogs, fresh-process resume gate."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

from scripts.liger_training_manifest import binding, prepare
from scripts.probe_liger_cpu4 import sha, write_new


def run(path, resumed):
    import torch
    from src.baselines.liger_runtime import Runtime
    from src.baselines.liger_state import capture, restore_checkpoint, save_checkpoint, tree_digest
    from src.baselines.liger_training import initial_progress, train_step, check_progress
    from src.baselines.liger_validation import evaluate
    m = binding(path)
    assert m['phase'] == 'training_resume_smoke' and not m['formal_training_ready']
    directory = path.parent / ('resumed' if resumed else 'continuous')
    directory.mkdir(exist_ok=False)
    report = dict(status='failed', job_id=os.environ['SLURM_JOB_ID'], phase=m['phase'],
                  source_git_commit=m['source_git_commit'], manifest_sha256=sha(path),
                  trained_checkpoint_for_paper=False, test_records_read=False, resumed=resumed, steps=[])
    started = time.monotonic()
    try:
        ctx = Runtime(m)
        p = initial_progress()
        if resumed:
            previous = json.loads((path.parent / 'continuous/report.json').read_text())
            entry = previous['resume_checkpoint']
            assert previous['status'] == 'passed' and sha(entry['path']) == entry['sha256']
            p = restore_checkpoint(entry['path'], *ctx.components, ctx.binding)
            assert p['next_batch'] == p['updates'] == 1
            report['loaded_checkpoint'] = entry
        check_progress(p, ctx.config['training'])
        stream = ctx.batches(training=True, epoch=p['epoch'], start_batch=p['next_batch'])
        try:
            while p['next_batch'] < 3:
                torch.cuda.synchronize()
                step_start = time.monotonic()
                i, records = next(stream)
                assert i == p['next_batch'] and len(records) == 1024
                step = train_step(*ctx.components, ctx.method, ctx.official, records, ctx.sids,
                                  ctx.content, ctx.seen, ctx.catalog, p)
                torch.cuda.synchronize()
                step['seconds'] = time.monotonic() - step_start
                assert not step['skipped']
                report['steps'].append(step)
                check_progress(p, ctx.config['training'])
                ctx.cpu_check()
                print('[liger-resume] ' + json.dumps(step), flush=True)
                if p['next_batch'] == 1:
                    checkpoint = directory / 'step1.pt'
                    save_checkpoint(checkpoint, capture(*ctx.components, p, ctx.binding))
                    report['resume_checkpoint'] = dict(path=str(checkpoint), sha256=sha(checkpoint))
                if p['next_batch'] == 2:
                    # Bind validation to exact model contents, not a path that differs between branches.
                    weight = dict(model_sha256=tree_digest(ctx.model.state_dict()), updates=p['updates'])
                    r = evaluate(ctx.model, ctx.method, ctx.official,
                        ctx.batches(training=False, validation_limit=65), ctx.sids, ctx.content,
                        ctx.index, ctx.seen, ctx.cold, directory / 'validation', 65, weight)
                    report['validation'] = dict(path=str(directory / 'validation/report.json'),
                                               sha256=sha(directory / 'validation/report.json'))
                    # Persist selection-shaped state and verify it survives the next update.
                    p['best'] = dict(epoch=1, **r['metrics']['hybrid'], **weight)
                    ctx.cpu_check()
            report.update(final_state_sha256=tree_digest(capture(*ctx.components, p, ctx.binding)),
                initial_state_sha256=ctx.initial_sha, final_progress=p, gpu=ctx.gpu,
                resources=ctx.resources, cpu_checks=ctx.cpu_checks, versions=ctx.versions,
                train_seen_rows=len(ctx.seen), cold_rows=len(ctx.cold),
                elapsed_seconds=time.monotonic() - started)
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
    from src.baselines.liger_runtime import Runtime
    from src.baselines.liger_validation import audit
    m = binding(path)
    assert m['phase'] == 'training_resume_smoke'
    ctx = Runtime(m, build_model=False)
    continuous, resumed = [json.loads((path.parent / name / 'report.json').read_text())
                           for name in ('continuous', 'resumed')]
    reports = []
    for name, r in (('continuous', continuous), ('resumed', resumed)):
        assert r['status'] == 'passed' and r['job_id'] == os.environ['SLURM_JOB_ID']
        assert r['source_git_commit'] == m['source_git_commit'] and r['manifest_sha256'] == sha(path)
        assert not r['test_records_read'] and not r['trained_checkpoint_for_paper']
        assert r['train_seen_rows'] == 11542755 and r['cold_rows'] == 23915712
        assert r['resources']['allocated_cpus'] == r['resources']['user_cpu_ceiling'] == 4
        assert len(r['cpu_checks']) == (4 if name == 'resumed' else 5)
        assert all(c['allowed_cpus'] == r['resources']['allowed_cpus'] and c['processes'] for c in r['cpu_checks'])
        for step in r['steps']:
            assert step['rows'] == 1024 and not step['skipped'] and step['gradient_tensors'] == 152
            expected_lr = .0003 * step['updates'] / 963
            assert abs(step['learning_rate'] - expected_lr) < 1e-16
        assert r['final_progress']['updates'] == r['final_progress']['next_batch'] == 3
        assert r['final_progress']['samples_in_epoch'] == 3072
        entry = r['validation']
        assert sha(entry['path']) == entry['sha256']
        v = json.loads(Path(entry['path']).read_text())
        assert v['records'] == 65 and v['batches'] == 3  # real 32/32/1 tail
        audit(Path(entry['path']).parent, ctx.seen, ctx.index, 65, 23915712, v['checkpoint'])
        reports.append(dict(path=str(path.parent / name / 'report.json'), sha256=sha(path.parent / name / 'report.json')))
        reports.append(entry)
        accept_path = Path(entry['path']).parent / 'acceptance.json'
        reports.append(dict(path=str(accept_path), sha256=sha(accept_path)))
    assert [s['batch'] for s in continuous['steps']] == [1, 2, 3]
    assert [s['batch'] for s in resumed['steps']] == [2, 3]
    clean = lambda steps: [{k: v for k, v in s.items() if k != 'seconds'} for s in steps]
    assert clean(continuous['steps'][1:]) == clean(resumed['steps'])
    assert continuous['initial_state_sha256'] == resumed['initial_state_sha256']
    assert continuous['final_state_sha256'] == resumed['final_state_sha256']
    assert continuous['gpu'] == resumed['gpu']
    assert continuous['resume_checkpoint'] == resumed['loaded_checkpoint']
    assert sha(continuous['resume_checkpoint']['path']) == continuous['resume_checkpoint']['sha256']
    va, vb = [json.loads(Path(r['validation']['path']).read_text()) for r in (continuous, resumed)]
    assert va['metrics'] == vb['metrics'] and va['record_identity_sha256'] == vb['record_identity_sha256']
    assert len(va['artifacts']) == len(vb['artifacts'])
    for a, b in zip(va['artifacts'], vb['artifacts']):
        with np.load(a['path'], allow_pickle=False) as x, np.load(b['path'], allow_pickle=False) as y:
            assert x.files == y.files
            for key in x.files: np.testing.assert_array_equal(x[key], y[key])
    ctx.cpu_check()
    write_new(path.parent / 'resume_acceptance.json', dict(status='passed', job_id=os.environ['SLURM_JOB_ID'],
        checked_at=datetime.now().astimezone().isoformat(), fresh_process_exact_resume=True,
        source_git_commit=m['source_git_commit'], manifest_sha256=sha(path), reports=reports,
        execution_sources=m['execution_sources'], config_sha256=m['config_sha256'],
        final_state_sha256=continuous['final_state_sha256'], gpu=continuous['gpu'],
        no_test_access=True, formal_checkpoint_produced=False))
    print('[liger-resume] exact fresh-process continuation and independent validation audit passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'continuous', 'resumed', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    path = args.manifest.resolve()
    if args.mode == 'prepare': prepare(path)
    elif args.mode == 'accept': accept(path)
    else: run(path, resumed=args.mode == 'resumed')
