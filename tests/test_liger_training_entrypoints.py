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
    assert "budget['stop_after_epoch'] == 1" in source
    assert 'test_files' not in source and 'test_records_read=False' in source
    assert source.index('audited = audit(') < source.index('if better_validation(')
    assert 'checkpoint_every_batches' in source and "stream.close()" in source


def test_resume_binding_distinguishes_smoke_from_formal():
    m = dict(config_sha256='x', execution_sources={}, input_metadata={}, upstream_commit='u',
             assets={}, catalog={'sha256': 'c'}, index_artifacts={})
    smoke = manifest.checkpoint_binding(dict(m, formal_training_ready=False))
    formal = manifest.checkpoint_binding(dict(m, formal_training_ready=True))
    assert smoke != formal and smoke['purpose'] == 'smoke' and formal['purpose'] == 'formal'
