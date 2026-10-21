"""CPU-only safety gates; no TensorFlow, GPU execution, or checkpoint loading."""
import copy
import inspect
import json
import re
import subprocess
import sys

import pytest

from scripts import probe_liger_learning as probe
from src.runtime_cpu import THREAD_VARIABLES


def config():
    return json.loads(probe.CONFIG.read_text())


def test_checked_in_config_has_exact_bounded_four_cell_design():
    c = config()
    probe.check_config(c)
    assert c['cases'] == [
        dict(name='raw_final', standardized=False, supervision='final_only'),
        dict(name='standardized_final', standardized=True, supervision='final_only'),
        dict(name='raw_prefix', standardized=False, supervision='all_prefix'),
        dict(name='standardized_prefix', standardized=True, supervision='all_prefix'),
    ]
    assert c['reserved_scope'] == 'train parents excluded from optimization, not excluded from feature statistics'
    assert c['comparison_scope'] == 'equal attempted target-token budget; final-target exposures differ'


@pytest.mark.parametrize('key,value', [
    ('phase', 'formal_training'), ('fit_parents', 256), ('reserved_parents', 0),
    ('batch_size', 64), ('catalog_rows', 8192), ('attempts_per_case', 2048),
    ('evaluation_attempts', [256, 1024]), ('full_catalog_smoke_attempts', 0),
    ('deadline_seconds', 7200), ('history_items', 8), ('warmup_steps', 52),
    ('learning_rate', .001), ('weight_decay', 0), ('seed', 43), ('split_seed', 1),
    ('loss_weighting', 'equal_parent_examples'),
    ('standardization_fit_scope', 'all_items_including_validation_test'),
    ('formal_training_ready', True), ('validation_test_access', True),
    ('automatic_successor', True),
])
def test_config_rejects_relaxed_or_changed_safety_limits(key, value):
    c = config()
    c[key] = value
    with pytest.raises(AssertionError):
        probe.check_config(c)


@pytest.mark.parametrize('mutation', ['missing', 'reordered', 'duplicate', 'supervision', 'normalization'])
def test_config_rejects_changed_four_cell_comparison(mutation):
    c = config()
    if mutation == 'missing': c['cases'].pop()
    if mutation == 'reordered': c['cases'].reverse()
    if mutation == 'duplicate': c['cases'][1] = copy.deepcopy(c['cases'][0])
    if mutation == 'supervision': c['cases'][0]['supervision'] = 'all_prefix'
    if mutation == 'normalization': c['cases'][0]['standardized'] = True
    with pytest.raises(AssertionError):
        probe.check_config(c)


@pytest.mark.parametrize('now,expired', [(99.999, False), (100., True), (100.001, True)])
def test_deadline_rejects_at_or_after_budget_boundary(monkeypatch, now, expired):
    monkeypatch.setattr(probe.time, 'monotonic', lambda: now)
    if expired:
        with pytest.raises(TimeoutError, match='partial cases are not an accepted four-cell comparison'):
            probe.check_deadline(100.)
    else:
        probe.check_deadline(100.)


@pytest.fixture
def bound_manifest(tmp_path, monkeypatch):
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config()))
    m = dict(source_git_commit='a' * 40, dirty_at_submission=False,
             formal_training_ready=False, automatic_successor=False,
             validation_test_records_read=False, config=str(config_path),
             config_sha256=probe.sha(config_path), phase='train_only_learning_diagnostic',
             execution_sources={}, input_metadata={str(config_path): probe.sha(config_path)},
             upstream_files={})
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        source = tmp_path / f'{group}.txt'
        source.write_text(f'bound fixture for {group}\n')
        m[group][str(source)] = probe.sha(source)
    path = tmp_path / 'manifest.json'

    def publish(manifest=None):
        value = m if manifest is None else manifest
        path.write_text(json.dumps(value))
        monkeypatch.setenv('LIGER_LEARNING_MANIFEST_SHA256', probe.sha(path))
        monkeypatch.setenv('SOURCE_GIT_COMMIT', value['source_git_commit'])
        return path

    publish()
    return m, path, publish


def test_binding_accepts_exact_hashes_and_returns_checked_config(bound_manifest):
    m, path, _ = bound_manifest
    checked, c = probe.binding(path)
    assert checked == m and c == config()


@pytest.mark.parametrize('field', ['validation_files', 'resume_checkpoint'])
def test_binding_rejects_forbidden_input_even_when_empty(bound_manifest, field):
    m, path, publish = bound_manifest
    m[field] = [] if field == 'validation_files' else None
    publish()
    with pytest.raises(AssertionError):
        probe.binding(path)


@pytest.mark.parametrize('field', ['dirty_at_submission', 'formal_training_ready',
                                   'automatic_successor', 'validation_test_records_read'])
def test_binding_rejects_unsafe_scope_flags(bound_manifest, field):
    m, path, publish = bound_manifest
    m[field] = True
    publish()
    with pytest.raises(AssertionError):
        probe.binding(path)


@pytest.mark.parametrize('variable', ['LIGER_LEARNING_MANIFEST_SHA256', 'SOURCE_GIT_COMMIT'])
def test_binding_rejects_wrong_external_hash_or_source_commit(bound_manifest, monkeypatch, variable):
    _, path, _ = bound_manifest
    monkeypatch.setenv(variable, '0' * 64)
    with pytest.raises(AssertionError):
        probe.binding(path)


@pytest.mark.parametrize('group', ['execution_sources', 'input_metadata', 'upstream_files'])
def test_binding_rejects_wrong_dependency_fingerprint(bound_manifest, group):
    m, path, publish = bound_manifest
    target = next(iter(m[group]))
    m[group][target] = '0' * 64
    publish()  # Manifest itself is valid; dependency verification must fail.
    with pytest.raises(ValueError, match='fingerprint changed'):
        probe.binding(path)


def test_binding_rejects_config_changed_after_manifest_creation(bound_manifest):
    m, path, _ = bound_manifest
    changed = config()
    changed['attempts_per_case'] = 8192
    from pathlib import Path
    Path(m['config']).write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='fingerprint changed'):
        probe.binding(path)


def test_launcher_cpu_gpu_time_thread_and_serial_execution_contract():
    path = probe.ROOT / 'run_liger_learning_probe_cpu4.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    for setting in ('--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--gres=gpu:1',
                    '--time=00:45:00', '--mem=24G', '--nodelist=aias-compute-2'):
        assert f'#SBATCH {setting}' in text
    assert 'set -euo pipefail' in text and 'export GRID_CPU_BUDGET=4' in text
    for name in THREAD_VARIABLES:
        assert re.search(rf'\b{re.escape(name)}=1(?=\s|$)', text)
    for token in ('OMP_DYNAMIC=FALSE', 'MKL_DYNAMIC=FALSE', 'TOKENIZERS_PARALLELISM=false',
                  'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'WANDB_MODE=disabled',
                  '${SOURCE_GIT_COMMIT:?', '${LIGER_LEARNING_MANIFEST_SHA256:?'):
        assert token in text
    commands = [line.strip() for line in text.replace('\\\n', ' ').splitlines()
                if line.startswith('srun ')]
    assert len(commands) == 3
    assert '-m pytest -q' in commands[0]
    assert '-m scripts.probe_liger_learning run "$1"' in commands[1]
    assert '-m scripts.probe_liger_learning accept "$1"' in commands[2]
    for command in commands:
        assert 'srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1' in command
        assert '&' not in command
    assert not re.search(r'(^|\s)(sbatch|nohup)\b', text)
    assert all((probe.ROOT / source).is_file() for source in probe.SOURCES)


def test_probe_import_is_cpu_only_and_does_not_load_checkpoints():
    subprocess.run([sys.executable, '-c',
        'import sys; import scripts.probe_liger_learning; '
        'assert "tensorflow" not in sys.modules; assert "torch" not in sys.modules'],
        cwd=probe.ROOT, check=True)
    source = inspect.getsource(probe)
    assert 'torch.load(' not in source
    run = inspect.getsource(probe.run)
    assert run.index('import torch') < run.index('import tensorflow')
    assert "tf.config.set_visible_devices([], 'GPU')" in run
    assert "tf.config.threading.set_intra_op_parallelism_threads(1)" in run
    assert "tf.config.threading.set_inter_op_parallelism_threads(1)" in run
