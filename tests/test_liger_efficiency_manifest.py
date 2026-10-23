import copy
import json
from pathlib import Path

import pytest

from scripts import liger_efficiency_manifest as manifest


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads(manifest.CONFIG.read_text())


def test_bounded_config_separates_equivalence_from_capacity_and_long_training():
    c = config()
    manifest.check_config(c)
    assert c['training']['max_attempts'] == c['warmup_attempts'] + c['measured_attempts'] == 4
    assert c['fixture_prefixes'] == 4 * max(case['batch_size'] for case in c['cases'])
    assert c['cases'][0]['batch_size'] == c['cases'][1]['batch_size']


@pytest.mark.parametrize('keys,value', [
    (('formal_training_ready',), True), (('automatic_successor',), True),
    (('cache_trainable_projection',), True), (('standardization_fit',), True),
    (('validation_test_access',), True), (('deadline_seconds',), 86400),
    (('fixture_prefixes',), 99999), (('training', 'max_attempts'), 19260),
    (('training', 'chunk_size'), 8192), (('training', 'history_items'), 999),
    (('resources', 'cpu_ceiling'), 24), (('validation', 'access'), True),
])
def test_contract_changes_are_rejected(keys, value):
    c = copy.deepcopy(config())
    target = c
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(AssertionError):
        manifest.check_config(c)


def test_previous_acceptance_is_frozen_and_exact():
    assert manifest.sha(manifest.PRIOR) == manifest.PRIOR_SHA
    assert manifest.sha(manifest.RECEIPT) == manifest.RECEIPT_SHA
    r = json.loads(manifest.RECEIPT.read_text())
    assert r['fresh_process_exact_resume'] and r['status'] == 'passed'


def test_launcher_uses_separate_processes_and_equivalence_before_capacity():
    source = (ROOT / 'run_liger_efficiency_cpu4.sh').read_text()
    assert '#SBATCH --ntasks=1' in source and '#SBATCH --cpus-per-task=4' in source
    assert '#SBATCH --gres=gpu:1' in source and '#SBATCH --mem=24G' in source
    assert '#SBATCH --time=00:45:00' in source and '--cpu-bind=verbose,threads' in source
    assert 'USE_TF=0' in source and 'set -euo pipefail' in source
    ordered = ['fixture "$1"', '--case lazy1024', '--case cached1024',
               'equivalence "$1"', '--case cached2048', '--case cached4096', 'accept "$1"']
    positions = [source.index(part) for part in ordered]
    assert positions == sorted(positions)
    assert 'sbatch ' not in source


def test_every_new_execution_file_has_a_manifest_binding():
    for relative in ('scripts/probe_liger_efficiency.py', 'scripts/accept_liger_efficiency.py',
        'src/baselines/liger_feature_cache.py', 'src/baselines/liger_efficiency_timing.py',
        'run_liger_efficiency_cpu4.sh', 'tests/test_liger_efficiency_manifest.py'):
        assert relative in manifest.SOURCES
