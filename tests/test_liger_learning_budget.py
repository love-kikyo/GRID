"""CPU-only gates for the fresh, single-condition 8192-attempt diagnostic."""
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from scripts import probe_liger_learning_budget as probe
from src.runtime_cpu import THREAD_VARIABLES


def config():
    return json.loads(probe.CONFIG.read_text())


def test_checked_in_config_is_exactly_one_fresh_raw_final_condition():
    c = config()
    probe.check_config(c)
    assert c['case'] == dict(name='raw_final', standardized=False, supervision='final_only')
    assert c['evaluation_attempts'] == [0, 256] + list(range(512, 8193, 512))
    assert c['attempts_per_case'] == 8192 and c['warmup_steps'] == 51
    assert c['fresh_initialization'] is True
    assert c['sid_loss_weight'] == c['dense_loss_weight'] == 1
    assert 'resume_checkpoint' not in c and 'validation_files' not in c


@pytest.mark.parametrize('key,value', [
    ('phase', 'formal_training'), ('fit_parents', 256), ('reserved_parents', 0),
    ('batch_size', 64), ('catalog_rows', 8192), ('attempts_per_case', 16384),
    ('evaluation_attempts', [0, 8192]), ('deadline_seconds', 7200),
    ('history_items', 8), ('warmup_steps', 409), ('learning_rate', .001),
    ('weight_decay', 0), ('seed', 43), ('split_seed', 1),
    ('loss_weighting', 'equal_parent_examples'), ('sid_loss_weight', 0),
    ('dense_loss_weight', 0), ('fresh_initialization', False),
    ('formal_training_ready', True), ('validation_test_access', True),
    ('automatic_successor', True),
])
def test_config_rejects_changed_budget_scope_and_loss(key, value):
    c = config()
    c[key] = value
    with pytest.raises(AssertionError):
        probe.check_config(c)


@pytest.mark.parametrize('key,value', [('name', 'standardized_final'),
                                     ('standardized', True), ('supervision', 'all_prefix')])
def test_config_cannot_change_or_expand_the_approved_condition(key, value):
    c = config()
    c['case'][key] = value
    with pytest.raises(AssertionError):
        probe.check_config(c)


@pytest.mark.parametrize('field', ['cases', 'resume_checkpoint', 'validation_files'])
def test_config_rejects_extra_conditions_resume_and_heldout_fields(field):
    c = config()
    c[field] = None
    with pytest.raises(AssertionError):
        probe.check_config(c)


@pytest.fixture
def bound_manifest(tmp_path, monkeypatch):
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config()))
    m = dict(source_git_commit='a' * 40, dirty_at_submission=False,
             formal_training_ready=False, automatic_successor=False,
             validation_test_records_read=False, fresh_initialization=True,
             config=str(config_path), config_sha256=probe.sha(config_path),
             phase='train_only_learning_budget_diagnostic',
             execution_sources={}, input_metadata={str(config_path): probe.sha(config_path)},
             upstream_files={})
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        source = tmp_path / f'{group}.txt'
        source.write_text(f'bound fixture for {group}\n')
        m[group][str(source)] = probe.sha(source)
    path = tmp_path / 'manifest.json'

    def publish():
        path.write_text(json.dumps(m))
        monkeypatch.setenv('LIGER_BUDGET_MANIFEST_SHA256', probe.sha(path))
        monkeypatch.setenv('SOURCE_GIT_COMMIT', m['source_git_commit'])
        return path

    publish()
    return m, path, publish


def test_binding_accepts_exact_safe_manifest(bound_manifest):
    m, path, _ = bound_manifest
    actual, c = probe.binding(path)
    assert actual == m and c == config()


@pytest.mark.parametrize('field', ['validation_files', 'resume_checkpoint'])
def test_binding_rejects_heldout_or_resume_field_even_when_empty(bound_manifest, field):
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


def test_binding_requires_fresh_initialization(bound_manifest):
    m, path, publish = bound_manifest
    m['fresh_initialization'] = False
    publish()
    with pytest.raises(AssertionError):
        probe.binding(path)


@pytest.mark.parametrize('variable', ['LIGER_BUDGET_MANIFEST_SHA256', 'SOURCE_GIT_COMMIT'])
def test_binding_rejects_wrong_external_binding(bound_manifest, monkeypatch, variable):
    _, path, _ = bound_manifest
    monkeypatch.setenv(variable, '0' * 64)
    with pytest.raises(AssertionError):
        probe.binding(path)


@pytest.mark.parametrize('group', ['execution_sources', 'input_metadata', 'upstream_files'])
def test_binding_rejects_wrong_dependency_hash(bound_manifest, group):
    m, path, publish = bound_manifest
    m[group][next(iter(m[group]))] = '0' * 64
    publish()
    with pytest.raises(ValueError, match='fingerprint changed'):
        probe.binding(path)


def test_binding_rejects_config_file_changed_after_binding(bound_manifest):
    m, path, _ = bound_manifest
    c = config()
    c['attempts_per_case'] = 16384
    Path(m['config']).write_text(json.dumps(c))
    with pytest.raises((ValueError, AssertionError)):
        probe.binding(path)


def test_launcher_enforces_resources_threads_and_serial_run_accept():
    path = probe.ROOT / 'run_liger_learning_budget_cpu4.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    for setting in ('--job-name=GRID-LIGER-BudgetV3', '--nodes=1', '--ntasks=1',
                    '--cpus-per-task=4', '--gres=gpu:1', '--time=00:45:00',
                    '--mem=24G', '--nodelist=aias-compute-2'):
        assert f'#SBATCH {setting}' in text
    assert 'set -euo pipefail' in text and 'export GRID_CPU_BUDGET=4' in text
    for name in THREAD_VARIABLES:
        assert re.search(rf'\b{re.escape(name)}=1(?=\s|$)', text)
    for token in ('OMP_DYNAMIC=FALSE', 'MKL_DYNAMIC=FALSE', 'TOKENIZERS_PARALLELISM=false',
                  'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'WANDB_MODE=disabled',
                  '${SOURCE_GIT_COMMIT:?', '${LIGER_BUDGET_MANIFEST_SHA256:?'):
        assert token in text
    commands = [line.strip() for line in text.replace('\\\n', ' ').splitlines()
                if line.startswith('srun ')]
    assert len(commands) == 3 and '-m pytest -q' in commands[0]
    for test in ('test_liger_learning_budget.py', 'test_liger_learning_probe.py', 'test_liger_supervision.py'):
        assert f'tests/{test}' in commands[0]
    assert '-m scripts.probe_liger_learning_budget run "$1"' in commands[1]
    assert '-m scripts.probe_liger_learning_budget accept "$1"' in commands[2]
    for command in commands:
        assert 'srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1' in command
        assert '&' not in command
    assert not re.search(r'(^|\s)(sbatch|nohup)\b', text)
    assert all((probe.ROOT / source).is_file() for source in probe.SOURCES)


def test_import_has_no_ml_runtime_or_checkpoint_load():
    subprocess.run([sys.executable, '-c',
        'import sys; import scripts.probe_liger_learning_budget; '
        'assert "tensorflow" not in sys.modules; assert "torch" not in sys.modules'],
        cwd=probe.ROOT, check=True)
    assert 'torch.load(' not in inspect.getsource(probe)


def test_beam_format_each_layer_accepts_inclusive_min_max_and_rejects_adjacent_tokens():
    offsets = np.array([1, 4097, 8193, 12289])
    tokens = np.tile(offsets, (1, 20, 1))
    tokens[0, 1] = offsets + 4095  # Inclusive upper endpoint at each layer.
    tokens[0, 2] = offsets - 1
    tokens[0, 3] = offsets + 4096
    result = probe.beam_format(tokens)
    np.testing.assert_allclose(result['codebook_valid_fraction'], [.9] * 4)
    assert result['all_codebooks_valid_fraction'] == .9


def test_beam_format_rejects_token_from_wrong_layer_even_if_global_vocab_token():
    wrong_layers = np.array([4097, 8193, 12289, 1])
    result = probe.beam_format(np.tile(wrong_layers, (2, 20, 1)))
    assert result['codebook_valid_fraction'] == [0.] * 4
    assert result['all_codebooks_valid_fraction'] == 0.


def test_beam_format_all_layers_fraction_is_joint_format_validity_not_item_mapping():
    # The helper has no SID index: codebook-valid tuples need not map to items.
    tokens = np.tile([4096, 8192, 12288, 16384], (1, 20, 1))
    for layer in range(4):
        tokens[0, layer, layer] = 0  # Different beam invalid in each layer.
    result = probe.beam_format(tokens)
    np.testing.assert_allclose(result['codebook_valid_fraction'], [.95] * 4)
    assert result['all_codebooks_valid_fraction'] == .8
    assert set(result) == {'codebook_valid_fraction', 'all_codebooks_valid_fraction'}
