"""Safety gates for the distinct real-record batch probe, without TensorFlow."""
import inspect
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import probe_liger_batch_scaling as probe


def test_dirty_prepare_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'git', lambda *args: ' M tracked.py')
    path = tmp_path / 'manifest.json'
    with pytest.raises(AssertionError):
        probe.prepare(path)
    assert not path.exists()


def test_failed_report_cannot_be_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'binding', lambda path: {})
    (tmp_path / 'batch_probe.json').write_text(json.dumps({'status': 'failed'}))
    with pytest.raises(AssertionError):
        probe.accept(tmp_path / 'manifest.json')
    assert not (tmp_path / 'batch_acceptance.json').exists()


def test_launcher_is_single_bounded_job():
    path = probe.ROOT / 'run_liger_batch_scaling_cpu4.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    for setting in ('--ntasks=1', '--cpus-per-task=4', '--gres=gpu:1', '--time=00:45:00',
                    '--nodelist=aias-compute-2', 'LIGER_BATCH_MANIFEST_SHA256', 'SOURCE_GIT_COMMIT'):
        assert setting in text
    assert text.count('srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads') == 3
    assert 'sbatch ' not in text and 'set -euo pipefail' in text
    assert all((probe.ROOT / name).is_file() for name in probe.SOURCES)


def test_timing_and_runtime_boundaries():
    source = inspect.getsource(probe.measure_case)
    assert source.index('new_model(') < source.index('for index in range(')
    assert source.index('started =') < source.index('take_batch(') < source.index('data_done =')
    assert source.index('pack_sequences(') < source.index('data_done =')
    run_source = inspect.getsource(probe.run)
    assert run_source.index('import torch') < run_source.index('import tensorflow')
    assert "tf.config.set_visible_devices([], 'GPU')" in run_source
    assert 'except torch.cuda.OutOfMemoryError' in run_source
    assert "status='skipped_after_oom' if stopped" in run_source
    assert 'fixture_report' not in inspect.getsource(probe)
    assert 'torch.save(' not in inspect.getsource(probe)


def fake_tf(records, opened):
    def iterator(path, options):
        opened.append((path, options))
        return iter(records[path])

    def feature(values):
        return SimpleNamespace(int64_list=SimpleNamespace(value=values), WhichOneof=lambda _: 'int64_list')

    def parse(raw):
        uid, seq = json.loads(raw)
        fields = {'user_id': feature([uid]), 'sequence_data': feature(seq)}
        return SimpleNamespace(features=SimpleNamespace(feature=fields))

    return SimpleNamespace(compat=SimpleNamespace(v1=SimpleNamespace(io=SimpleNamespace(tf_record_iterator=iterator))),
                           train=SimpleNamespace(Example=SimpleNamespace(FromString=parse)))


def test_train_stream_preserves_gt_and_reads_only_manifest_files():
    opened = []
    tf = fake_tf({'train_a': [b'[1, [0, 8]]'], 'train_b': [b'[-2, [5, 6, 7]]']}, opened)
    iterator = probe.real_records(tf, [{'path': 'train_a'}, {'path': 'train_b'}])
    assert list(iterator) == [(1, [0, 8], b'[1, [0, 8]]'), (-2, [5, 6, 7], b'[-2, [5, 6, 7]]')]
    assert opened == [('train_a', 'GZIP'), ('train_b', 'GZIP')]


def test_train_stream_does_not_open_later_shards_early():
    opened = []
    tf = fake_tf({'train_a': [b'[1, [0, 8]]'], 'train_b': []}, opened)
    iterator = probe.real_records(tf, [{'path': 'train_a'}, {'path': 'train_b'}])
    assert next(iterator)[0] == 1
    iterator.close()
    assert opened == [('train_a', 'GZIP')]


@pytest.mark.parametrize('corruption', [None, 'cpu', 'held_out', 'case_file', 'summary', 'updates'])
def test_acceptance_requires_scope_resources_and_matching_case_artifacts(tmp_path, monkeypatch, corruption):
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    monkeypatch.setenv('SLURM_JOB_ID', 'unit-test')
    m = dict(source_git_commit='abc', catalog=dict(rows=11), masked_count=7, asset_rows=18,
             chunk_size=65536, batch_sizes=[2], warmup_steps=1, measured_steps=3,
             train_records=100, max_consumed_records=8)
    monkeypatch.setattr(probe, 'binding', lambda path: m)
    steps = [dict(step=i + 1, warmup=i == 0, optimizer_step_verified=True, gradient_tensors=152,
        data_seconds=1., forward_seconds=2., backward_update_seconds=3., total_seconds=6.,
        sid_loss=1., dense_loss=2., total_loss=3., grad_norm=1.,
        peak_allocated_bytes=10, peak_reserved_bytes=20) for i in range(4)]
    case = dict(batch_size=2, status='completed', steps=steps, initial_state_sha256='b' * 64,
        summary=probe.summarize(steps, 2, 100),
        attempts=[dict(users=[i * 2, i * 2 + 1], records=2, serialized_sha256='a' * 64) for i in range(4)])
    snapshot = dict(allowed_cpus=[1, 2, 3, 4], processes=[dict(threads_checked=1)])
    r = dict(status='passed', phase='real_train_batch_capacity_only', source_git_commit='abc',
        job_id='unit-test', manifest=dict(path=str(path), sha256=probe.sha(path)),
        formal_training_ready=False, candidate_universe_finalized=False, held_out_records_read=False,
        formal_batch_selected=False, tf32=False, includes_train_read_decode_pack=True,
        catalog_rows=11, masked_count=7, chunk_size=65536, dense_dtype='float32', projection_backend='linear',
        cases=[case], gpu=dict(total_bytes=1000),
        sweep_summary=probe.verify_cases([case], [2], 1, 3, 100, 1000),
        resources=dict(allocated_cpus=4, user_cpu_ceiling=4, allowed_cpus=[1, 2, 3, 4]),
        cpu_checks=[snapshot, snapshot])
    if corruption == 'cpu': r['resources']['allocated_cpus'] = 24
    if corruption == 'held_out': r['held_out_records_read'] = True
    if corruption == 'summary': case['summary']['samples_per_second'] *= 2
    if corruption == 'updates': steps[3]['optimizer_step_verified'] = False
    (tmp_path / 'batch_probe.json').write_text(json.dumps(r))
    (tmp_path / 'batch_case_2.json').write_text(json.dumps({} if corruption == 'case_file' else case))
    if corruption:
        with pytest.raises(AssertionError):
            probe.accept(path)
        assert not (tmp_path / 'batch_acceptance.json').exists()
    else:
        probe.accept(path)
        accepted = json.loads((tmp_path / 'batch_acceptance.json').read_text())
        assert accepted['status'] == 'passed' and not accepted['formal_training_ready']
