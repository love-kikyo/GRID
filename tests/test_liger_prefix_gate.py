"""CPU-only contract checks for the new gate, without importing TensorFlow."""
import copy
import json
from pathlib import Path

import pytest

from scripts import liger_prefix_manifest as manifest


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / 'configs/baselines/liger_prefix_gate_v3.json').read_text())


def test_config_is_integration_only():
    value = config()
    manifest.check_config(value)
    assert value['training']['max_attempts'] == 3
    assert value['training']['max_optimizer_steps'] > value['training']['max_attempts']
    assert 'NOT an approved formal training budget' in value['training']['schedule_scope']


@pytest.mark.parametrize('keys,value', [
    (('formal_training_ready',), True), (('automatic_successor',), True),
    (('training', 'max_attempts'), 19260), (('training', 'batch_size'), 32),
    (('training', 'supervision'), 'final_only'),
    (('training', 'loss_weighting'), 'sid_only_prefix'),
    (('training', 'prefix_shuffle_buffer'), 1),
    (('validation', 'candidate_mode'), 'beam_only'),
    (('validation', 'test_access'), True),
    (('resources', 'cpu_ceiling'), 24),
])
def test_gate_does_not_authorize_protocol_or_budget_changes(keys, value):
    altered = copy.deepcopy(config())
    target = altered
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(AssertionError):
        manifest.check_config(altered)


def test_audited_inventory_gives_exact_prefix_budget_without_rescan(monkeypatch):
    monkeypatch.setattr(manifest, 'sha', lambda path: 'a' * 64)
    inventory = dict(status='passed', history_items=7, held_out_records_read=False,
        records=6573674, unique_users=6573674, consumed_input_plus_target_occurrences=52569787)
    counts = manifest.inventory_counts(inventory)
    assert counts['prefix_examples_per_pass'] == 45996113
    assert counts['prefix_batches_per_pass'] == 44919
    assert counts['final_batch_size'] == 81
    assert counts['final_targets_per_pass'] == 6573674
    assert not counts['new_tfrecord_scan'] and not counts['formal_budget_approved']
    inventory['consumed_input_plus_target_occurrences'] = 8 * 6573674 + 1
    with pytest.raises(AssertionError):
        manifest.inventory_counts(inventory)


def test_launcher_limits_whole_job_and_has_no_successor():
    launcher = (ROOT / 'run_liger_prefix_gate_cpu4.sh').read_text()
    assert '#SBATCH --cpus-per-task=4' in launcher and '#SBATCH --ntasks=1' in launcher
    assert '#SBATCH --gres=gpu:1' in launcher and '#SBATCH --time=00:45:00' in launcher
    assert 'LIGER_PREFIX_MANIFEST_SHA256' in launcher and 'SOURCE_GIT_COMMIT' in launcher
    assert launcher.count('--cpu-bind=verbose,threads') == 4
    for mode in ('continuous', 'resumed', 'accept'):
        assert f'scripts.probe_liger_prefix {mode}' in launcher
    assert 'sbatch ' not in launcher and 'train_liger_cpu4' not in launcher


def test_gate_uses_new_training_path_without_refitting_or_small_catalog():
    source = (ROOT / 'scripts/probe_liger_prefix.py').read_text()
    assert 'from src.baselines.liger_prefix_runtime import PrefixRuntime' in source
    assert 'from src.baselines.liger_prefix_training import' in source
    assert 'restore_checkpoint' in source and 'np.testing.assert_array_equal' in source
    assert 'fit_standardization' not in source and 'diagnostic_catalog' not in source
    assert 'formal_training_ready=False' in source and 'automatic_successor=False' in source


def test_inventory_schema_matches_existing_receipt():
    # Small immutable JSON only; not an asset scan or a new result computation.
    receipt = json.loads(manifest.INVENTORY_RECEIPT.read_text())
    assert receipt['status'] == 'passed'
    assert receipt['report']['path'] == str(manifest.INVENTORY)
    assert receipt['report']['sha256'] == manifest.sha(manifest.INVENTORY)
