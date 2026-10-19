import inspect
import json
import subprocess

import pytest

from scripts import probe_liger_full_catalog as probe


def test_prepare_rejects_dirty_source_before_creating_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'git', lambda *args: ' M changed')
    with pytest.raises(AssertionError):
        probe.prepare(tmp_path / 'manifest.json')
    assert not (tmp_path / 'manifest.json').exists()


def test_launcher_has_one_gpu_four_cpus_and_no_automatic_training():
    path = probe.ROOT / 'run_liger_full_catalog_probe_cpu4.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    for expected in ['#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=4', '#SBATCH --gres=gpu:1',
                     '--cpu-bind=verbose,threads', 'LIGER_SCALE_MANIFEST_SHA256', 'GRID_CPU_BUDGET=4']:
        assert expected in text
    assert 'sbatch ' not in text and '-m src.train' not in text
    assert text.index('catalog run') < text.index('catalog accept')


def test_run_reuses_train_fixture_without_held_out_reader_or_checkpoint_write():
    text = inspect.getsource(probe.run)
    assert 'full_catalog_cross_entropy' in text and 'catalog, chunk_size=' in text
    assert 'torch.save' not in text and 'tf_record_iterator' not in text
    assert 'evaluation.model_forward' in text and 'scaler.step(optimizer)' in text
    assert all((probe.ROOT / name).is_file() for name in probe.SOURCES)


def test_failed_probe_cannot_be_accepted(tmp_path, monkeypatch):
    (tmp_path / 'scale_probe.json').write_text(json.dumps({'status': 'failed'}))
    monkeypatch.setattr(probe, 'binding', lambda path: {})
    with pytest.raises(AssertionError):
        probe.accept(tmp_path / 'manifest.json')
    assert not (tmp_path / 'scale_acceptance.json').exists()


@pytest.mark.parametrize('corruption', [None, 'cpu', 'missing_step', 'universe', 'gradient_error'])
def test_acceptance_requires_both_steps_resources_and_scope(tmp_path, monkeypatch, corruption):
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    monkeypatch.setenv('SLURM_JOB_ID', 'unit-test')
    m = dict(source_git_commit='abc', catalog=dict(rows=11), masked_count=7,
             asset_rows=18, chunk_size=4, optimizer_steps=2)
    monkeypatch.setattr(probe, 'binding', lambda path: m)
    snapshot = dict(allowed_cpus=[1, 2, 3, 4], processes=[dict(threads_checked=1)])
    r = dict(status='passed', phase='full_catalog_scale_only', source_git_commit='abc',
        job_id='unit-test', manifest=dict(path=str(path), sha256=probe.sha(path)),
        formal_training_ready=False, candidate_universe_finalized=False, held_out_records_read=False,
        catalog_rows=11, masked_count=7, batch_size=32, fixture_users=list(range(32)),
        chunk_size=4, dense_dtype='float32', resources=dict(allocated_cpus=4, allowed_cpus=[1, 2, 3, 4]),
        cpu_start=snapshot, cpu_end=snapshot, gpu=dict(total_bytes=1000),
        official_projection_checks=[dict(dropout_training=x, backward_restores_rng=True,
            gradient_tensors=2, gradient_max_abs_error=1e-12, loss_abs_error=1e-12) for x in [False, True]],
        steps=[dict(step=i, optimizer_step_verified=True, gradient_tensors=10, sid_loss=1.,
            dense_loss=2., total_loss=3., grad_norm=1., forward_seconds=.1, total_step_seconds=.3,
            peak_allocated_bytes=100, peak_reserved_bytes=200) for i in [1, 2]])
    if corruption == 'cpu': r['resources']['allowed_cpus'] = list(range(24))
    if corruption == 'missing_step': r['steps'].pop()
    if corruption == 'universe': r['candidate_universe_finalized'] = True
    if corruption == 'gradient_error': r['official_projection_checks'][0]['gradient_max_abs_error'] = 1.
    (tmp_path / 'scale_probe.json').write_text(json.dumps(r))
    if corruption:
        with pytest.raises(AssertionError):
            probe.accept(path)
        assert not (tmp_path / 'scale_acceptance.json').exists()
    else:
        probe.accept(path)
        assert json.loads((tmp_path / 'scale_acceptance.json').read_text())['status'] == 'passed'


@pytest.mark.parametrize('corruption', [None, 'nan', 'median', 'missing_backend', 'rng'])
def test_backend_diagnostic_audit_rejects_incomplete_or_invalid_measurements(corruption):
    from scripts.liger_projection_diagnostic import verify_diagnostic
    checks = [dict(precision=p, training=t, passed=True, rng_equal=True, parameter_tensors=17,
                   output_max_abs_error=1e-12, gradient_max_abs_error=1e-12)
              for p in ('float64', 'float32', 'amp_float16') for t in (False, True)]
    timings = [dict(backend=b, chunk_size=c, median_forward_seconds=1., median_total_seconds=3.,
        measured=[dict(forward_seconds=1., total_seconds=3., backward_recompute_seconds=2.,
                       peak_allocated_bytes=100) for _ in range(3)])
        for c in (8192, 65536) for b in ('conv1d', 'linear')]
    report = dict(checks=checks, timings=timings, measured_repeats=3, warmup_repeats=1)
    if corruption == 'nan': checks[0]['gradient_max_abs_error'] = float('nan')
    if corruption == 'median': timings[0]['median_total_seconds'] = .01
    if corruption == 'missing_backend': timings.pop()
    if corruption == 'rng': checks[0]['rng_equal'] = False
    if corruption:
        with pytest.raises(AssertionError):
            verify_diagnostic(report)
    else:
        verify_diagnostic(report)
