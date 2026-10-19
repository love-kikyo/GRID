import inspect
import json
import subprocess

import pytest

from scripts import probe_liger_full_cold as probe


def test_prepare_refuses_dirty_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'git', lambda *args: ' M tracked.py')
    path = tmp_path / 'manifest.json'
    with pytest.raises(AssertionError): probe.prepare(path)
    assert not path.exists()


def test_failed_report_has_no_acceptance_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'binding', lambda path: {})
    (tmp_path / 'cold_probe.json').write_text(json.dumps(dict(status='failed')))
    with pytest.raises(AssertionError): probe.accept(tmp_path / 'manifest.json')
    assert not (tmp_path / 'cold_acceptance.json').exists()


def test_launcher_limits_resources_and_has_no_training_successor():
    path = probe.ROOT / 'run_liger_full_cold_cpu4.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    for setting in ('--cpus-per-task=4', '--ntasks=1', '--gres=gpu:1', '--time=00:30:00',
                    '--cpu-bind=verbose,threads', 'SOURCE_GIT_COMMIT', 'LIGER_COLD_MANIFEST_SHA256'):
        assert setting in text
    assert 'sbatch ' not in text and 'set -euo pipefail' in text
    assert all((probe.ROOT / name).is_file() for name in probe.SOURCES)


def test_scope_is_validation_only_random_model_no_checkpoint_no_optimizer():
    source = inspect.getsource(probe.run)
    assert source.index('import torch') < source.index('import tensorflow')
    assert 'optimizer' not in source and 'torch.save' not in source and 'test_files' not in source
    assert "record_stream(tf, m['validation_files']" in source
    assert 'gt_independence_verified' in source and 'cold_rows_scanned' in source
    assert 'full_cold_rank(cache' in source and "model.eval()" in source


def test_frozen_policy_preserves_full_cold_and_approved_budget():
    p = json.loads(probe.CONFIG.read_text())
    assert p['candidate_universe_finalized'] and not p['formal_training_ready']
    assert p['cold_rows'] + p['seen_rows'] == p['asset_rows'] == 35458467
    assert p['inference']['cutoffs'] == [5, 10, 20]
    assert p['inference']['candidate_set'] == 'valid natural beam20 union every cold asset row'
    t = p['training_budget']
    assert t['max_optimizer_steps'] == t['max_epochs'] * t['batches_per_epoch'] == 19260
    assert t['warmup_steps'] == 963 and t['stop_for_acceptance_after_epochs'] == 1
