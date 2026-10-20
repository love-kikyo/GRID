import inspect
import json
import subprocess

import pytest

from scripts import liger_training_manifest as manifest
from scripts import train_liger_cpu4, probe_liger_resume


@pytest.mark.parametrize('script', ['run_liger_train_cpu4.sh', 'run_liger_resume_smoke_cpu4.sh'])
def test_launchers_limit_resources_and_gate_success(script):
    path = manifest.ROOT / script
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    for value in ('--ntasks=1', '--cpus-per-task=4', '--gres=gpu:1', '--cpu-bind=verbose,threads',
                  'set -euo pipefail', 'SOURCE_GIT_COMMIT', 'LIGER_TRAIN_MANIFEST_SHA256'):
        assert value in text
    assert 'sbatch ' not in text and ' accept ' in text
    if 'smoke' in script:
        assert text.index(' continuous ') < text.index(' resumed ') < text.index(' accept ')


def test_configuration_keeps_approved_budget_and_selection():
    config = json.loads(manifest.CONFIG.read_text())
    candidate = json.loads((manifest.ROOT / config['candidate_config']).read_text())
    manifest.verify_config(config, candidate)
    config['training']['max_epochs'] = 4
    with pytest.raises(AssertionError): manifest.verify_config(config, candidate)


def test_prepare_rejects_dirty_execution_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, 'git', lambda *args: ' M src/changed.py')
    with pytest.raises(AssertionError): manifest.prepare(tmp_path / 'manifest.json')


def test_smoke_is_fresh_process_real_data_and_not_formal_initialization():
    source = inspect.getsource(probe_liger_resume.run)
    assert 'restore_checkpoint' in source and "ctx.batches(training=True" in source
    assert "len(records) == 1024" in source and "validation_limit=65" in source
    assert 'formal_training_ready' in source
    assert all((manifest.ROOT / name).is_file() for name in manifest.SOURCES)


def test_formal_entry_requires_gate_and_stops_after_first_epoch():
    source = inspect.getsource(train_liger_cpu4.run)
    assert "assert m['formal_training_ready']" in source
    assert 'start_epoch, stop_epoch = stage_bounds(m, ctx.config)' in source
    assert 'test_files' not in source and 'test_records_read=False' in source
    assert source.index('audited = audit(') < source.index('if better_validation(')
    assert 'checkpoint_every_batches' in source and "stream.close()" in source


def test_resume_binding_distinguishes_smoke_from_formal():
    m = dict(config_sha256='x', execution_sources={}, input_metadata={}, upstream_commit='u',
             assets={}, catalog={'sha256': 'c'}, index_artifacts={})
    smoke = manifest.checkpoint_binding(dict(m, formal_training_ready=False))
    formal = manifest.checkpoint_binding(dict(m, formal_training_ready=True))
    assert smoke != formal and smoke['purpose'] == 'smoke' and formal['purpose'] == 'formal'


@pytest.mark.parametrize('start_epoch,score', [(0, .2), (1, .1), (1, .3), (2, .2)])
def test_formal_control_flow_preflight_tail_validation_and_stop(tmp_path, monkeypatch, start_epoch, score):
    """Exercise the real orchestrator with tiny stand-ins, not a GPU cost probe."""
    from types import SimpleNamespace
    import torch
    from src.baselines import liger_runtime, liger_training, liger_validation
    from src.baselines.liger_state import BudgetSchedule
    from scripts.probe_liger_cpu4 import sha, write_new
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    m = dict(formal_training_ready=True, phase='first_epoch_formal_training',
             run_directory=str(tmp_path / 'run'), source_git_commit='unit', initialization='seed42 random')
    monkeypatch.setattr(train_liger_cpu4, 'binding', lambda _: m)
    monkeypatch.setenv('SLURM_JOB_ID', 'unit')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters())
    sched = BudgetSchedule(opt, total=6, warmup=2)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    budget = dict(batch_size=2, records_per_epoch=3, batches_per_epoch=2, max_epochs=3,
                  max_optimizer_steps=6, stop_after_epoch=1, checkpoint_every_batches=1, chunk_size=2)
    ctx = SimpleNamespace(components=(model, opt, sched, scaler), model=model, binding={'purpose': 'unit'},
        config=dict(training=budget, validation=dict(records=3)), method=None, official=None,
        sids=None, content=None, seen=[0], cold=[1], index=None, catalog=None,
        initial_sha='unit', resources={}, gpu={}, versions={}, cpu_checks=[],
        cpu_check=lambda: None,
        batches=lambda **kwargs: (batch for batch in [(0, ['a', 'b']), (1, ['c'])]))
    monkeypatch.setattr(liger_runtime, 'Runtime', lambda _: ctx)
    if start_epoch:
        import copy
        from src.baselines.liger_training import initial_progress
        m.update(phase='epoch_continuation', stage=dict(start_epoch=start_epoch, stop_after_epoch=start_epoch+1),
                 resume_checkpoint={'path': 'unit'})
        p = initial_progress()
        prior = [dict(epoch=e, checkpoint={'path': f'old-{e}'}, metrics={'hybrid': {'Recall@5': .2, 'NDCG@5': .1}})
                 for e in range(1, start_epoch+1)]
        p.update(epoch=start_epoch, batches_consumed=start_epoch*2, updates=start_epoch*2,
                 validations=prior, best=dict(epoch=1, checkpoint=prior[0]['checkpoint'], **prior[0]['metrics']['hybrid']))
        for _ in range(start_epoch*2): sched.step()
        monkeypatch.setattr(train_liger_cpu4, 'restore_formal', lambda *_: (copy.deepcopy(p), {}))
        def batches(**kwargs):
            if kwargs['training']: assert kwargs['epoch'] == start_epoch and kwargs['start_batch'] == 0
            return (batch for batch in [(0, ['a', 'b']), (1, ['c'])])
        ctx.batches = batches
    consumed = []
    def step(*args, **kwargs):
        records, p = args[6], args[11]
        consumed.append(len(records))
        sched.step()
        p['updates'] += 1
        p['batches_consumed'] += 1
        p['next_batch'] += 1
        p['samples_in_epoch'] += len(records)
        return dict(batch=p['next_batch'], rows=len(records))
    monkeypatch.setattr(liger_training, 'train_step', step)
    metrics = {'Recall@5': score, 'NDCG@5': .1}
    def evaluate(*args, **kwargs):
        directory, expected, checkpoint = args[9:12]
        directory.mkdir()
        assert expected == 3
        report = dict(metrics={'hybrid': metrics}, checkpoint=checkpoint)
        write_new(directory / 'report.json', report)
        return report
    def audit(directory, *args):
        receipt = dict(report=dict(path=str(directory / 'report.json'), sha256=sha(directory / 'report.json')))
        write_new(directory / 'acceptance.json', receipt)
        return receipt
    monkeypatch.setattr(liger_validation, 'evaluate', evaluate)
    monkeypatch.setattr(liger_validation, 'audit', audit)
    train_liger_cpu4.run(path)
    result = json.loads((tmp_path / 'run/result.json').read_text())
    preflight = json.loads((tmp_path / 'run/preflight.json').read_text())
    assert preflight['status'] == 'ready' and result['status'] == 'passed'
    assert consumed == [2, 1] and result['final_progress']['epoch'] == start_epoch + 1
    assert result['final_progress']['updates'] == (start_epoch + 1)*2
    assert len(result['final_progress']['validations']) == start_epoch + 1
    assert result['stopped_for_author_acceptance'] and not result['automatic_successor']
    assert len(result['checkpoints']) == 4
    assert result['best']['epoch'] == (start_epoch + 1 if score > .2 else 1)
