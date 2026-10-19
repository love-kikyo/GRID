import inspect
import json
import subprocess

import numpy as np
import pytest

from scripts import build_liger_sid_index as builder


def test_dirty_prepare_refuses_to_create_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, 'git', lambda *args: ' M tracked.py')
    path = tmp_path / 'manifest.json'
    with pytest.raises(AssertionError):
        builder.prepare(path)
    assert not path.exists()


def test_launcher_has_one_cpu_no_gpu_no_training_or_successor():
    path = builder.ROOT / 'run_liger_sid_index_cpu1.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    for token in ('#SBATCH --cpus-per-task=1', '#SBATCH --gres=gpu:0', '--time=00:20:00',
                  '--cpu-bind=verbose,threads', 'LIGER_INDEX_MANIFEST_SHA256'):
        assert token in text
    assert 'sbatch ' not in text and 'tensorflow' not in inspect.getsource(builder)
    assert text.index('sid_index run') < text.index('sid_index accept')
    assert all((builder.ROOT / name).is_file() for name in builder.SOURCES)


@pytest.mark.parametrize('corruption', [None, 'report', 'identity', 'scope', 'cpu', 'rows', 'asset'])
def test_end_to_end_index_and_independent_acceptance(tmp_path, monkeypatch, corruption):
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    asset_path = tmp_path / 'source.npy'
    np.save(asset_path, np.array([[1, 2, 3, 4], [0, 0, 0, 0], [4095] * 4], dtype=np.int32))
    m = dict(phase='global_sid_identity_only', source_git_commit='test', asset_rows=3, chunk_size=2,
             asset=dict(path=str(asset_path), sha256=builder.sha(asset_path)))
    monkeypatch.setattr(builder, 'binding', lambda path: m)
    monkeypatch.setattr(builder, 'cpu_guard', lambda: dict(allowed_cpus=[1], processes=[dict(threads=1)]))
    monkeypatch.setenv('SLURM_JOB_ID', 'unit-test')
    builder.run(path)
    report_path = tmp_path / 'index_report.json'
    r = json.loads(report_path.read_text())
    assert r['identity']['rows_checked'] == 3
    if corruption == 'report': r['status'] = 'failed'
    if corruption == 'identity': r['identity']['row_zero_preserved'] = False
    if corruption == 'scope': r['candidate_universe_finalized'] = True
    if corruption == 'cpu': r['cpu_snapshots'][0]['allowed_cpus'] = [1, 2]
    if corruption == 'rows':
        a = r['artifacts']['asset_rows']
        np.save(a['path'], np.array([0, 1, 2], dtype=np.int32))
        a['sha256'] = builder.sha(a['path'])  # A valid permutation, but wrong SID-row pairing.
    if corruption == 'asset':
        np.save(asset_path, np.zeros((3, 4), dtype=np.int32))
    report_path.write_text(json.dumps(r))
    if corruption:
        with pytest.raises(AssertionError):
            builder.accept(path)
        assert not (tmp_path / 'index_acceptance.json').exists()
    else:
        builder.accept(path)
        a = json.loads((tmp_path / 'index_acceptance.json').read_text())
        assert a['status'] == 'passed' and a['report']['sha256'] == builder.sha(report_path)
        assert not a['formal_training_ready']


def test_sid_aliases_fail_before_publishing_index_artifacts(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    asset_path = tmp_path / 'source.npy'
    np.save(asset_path, np.zeros((2, 4), dtype=np.int32))
    m = dict(phase='global_sid_identity_only', source_git_commit='test', asset_rows=2, chunk_size=2,
             asset=dict(path=str(asset_path), sha256=builder.sha(asset_path)))
    monkeypatch.setattr(builder, 'binding', lambda path: m)
    monkeypatch.setattr(builder, 'cpu_guard', lambda: dict(allowed_cpus=[1], processes=[dict(threads=1)]))
    monkeypatch.setenv('SLURM_JOB_ID', 'unit-test')
    with pytest.raises(ValueError, match='aliases'):
        builder.run(path)
    assert json.loads((tmp_path / 'index_report.json').read_text())['status'] == 'failed'
    assert not (tmp_path / 'sid_keys.npy').exists()
