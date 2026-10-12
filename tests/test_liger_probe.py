"""Provenance and launch controls; no scheduler or model execution."""
import json
from pathlib import Path

import pytest

from scripts import probe_liger_cpu4 as probe


def test_binding_rejects_changed_sources(tmp_path):
    path = tmp_path / 'input'
    path.write_text('old')
    binding = {str(path): probe.sha(path)}
    probe.verify_hashes(binding)
    path.write_text('changed')
    with pytest.raises(ValueError, match='fingerprint'):
        probe.verify_hashes(binding)


def test_report_is_create_only(tmp_path):
    path = tmp_path / 'report.json'
    probe.write_new(path, {'status': 'failed'})
    with pytest.raises(FileExistsError):
        probe.write_new(path, {'status': 'passed'})
    assert json.loads(path.read_text())['status'] == 'failed'


def test_prepare_rejects_dirty_checkout_before_any_write(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'git', lambda *args: ' M unsafe.py')
    path = tmp_path / 'manifest.json'
    with pytest.raises(ValueError, match='Commit source'):
        probe.prepare(path, tmp_path)
    assert not path.exists()


def test_upstream_lock_and_execution_sources_are_complete():
    lock = json.loads(probe.LOCK.read_text())
    assert lock['commit'] == 'b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524'
    assert {'src/tiger.py', 'src/evaluation.py', 'LICENSE'} <= set(lock['files'])
    assert all(len(value) == 64 for value in lock['files'].values())
    assert all((probe.ROOT / name).is_file() for name in probe.SOURCES)
    assert 'src/baselines/liger_inputs.py' in probe.SOURCES


def test_launcher_is_bounded_offline_and_not_a_formal_training_entry():
    text = (probe.ROOT / 'run_probe_liger_cpu4.sh').read_text()
    for required in ['#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=4',
                     '#SBATCH --gres=gpu:1', '#SBATCH --time=00:20:00',
                     '--cpu-bind=verbose,threads', 'GRID_CPU_BUDGET=4',
                     'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                     'BASELINE_MANIFEST_SHA256', 'SOURCE_GIT_COMMIT', 'set -euo pipefail']:
        assert required in text
    assert '-m src.train' not in text and 'sbatch ' not in '\n'.join(text.splitlines()[16:])
    assert 'pip install' not in text and 'conda env' not in text
