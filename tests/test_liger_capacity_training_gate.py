"""Small evidence and authority checks for the actual formal-entrypoint gate."""
import ast
import copy
import json
from pathlib import Path
import time

import pytest

from scripts import probe_liger_capacity_training as probe


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(probe.contract, 'checkpoint_binding', lambda m: {'purpose': 'gate'})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    clock = tmp_path / 'clock.json'
    clock.write_text(json.dumps({'deadline_unix_seconds': time.time()+1000}))
    cp1, cp3 = tmp_path / 'step1.pt', tmp_path / 'step3.pt'
    cp1.write_bytes(b'one')
    cp3.write_bytes(b'three')
    m = dict(source_git_commit='c'*40, expected_initial_state_sha256='i'*64,
             expected_feature_cache_sha256='f'*64, assets={'sid':probe.entry(cp1)})
    steps = [dict(attempt=i, updates=i, rows=2048, microbatch=1024, microbatches=2,
        gradient_tensors=152, skipped=False, candidate_classes=34700,
        final_target_count=10, sid_loss=2., dense_loss=3., total_loss=5.,
        peak_allocated_bytes=100, peak_reserved_bytes=120, seconds=1.,
        input_seconds=.4, training_seconds=.6) for i in (1, 2, 3)]
    a = dict(status='passed', resumed=False, manifest_sha256=probe.sha(manifest),
        phase=probe.contract.GATE_PHASE, job_id='123', source_git_commit=m['source_git_commit'],
        initial_state_sha256=m['expected_initial_state_sha256'], checkpoint_binding={'purpose': 'gate'},
        state_after_three='s3', formal_training_ready=False, formal_checkpoint_produced=False,
        test_records_read=False, automatic_successor=False, steps=steps,
        gpus=[{'total_bytes': 200}], resources={'allocated_cpus': 4, 'library_threads': 1, 'allowed_cpus': [2]},
        cpu_checks=[{'allowed_cpus': [2]}], cache={'verification': {'status': 'passed', 'cache_sha256': m['expected_feature_cache_sha256']},
        'build': {'cache_sha256': m['expected_feature_cache_sha256']}},
        asset_byte_verification={'files': {str(cp1):probe.sha(cp1)}, 'seconds': 1.},
        clock=probe.entry(clock), final_progress={'attempts': 3, 'updates': 3, 'prefix_count': 6144, 'final_count': 30},
        step_one_checkpoint=probe.entry(cp1),
        validation_checkpoint=dict(probe.entry(cp3), full_state_sha256='s3'))
    b = copy.deepcopy(a)
    b.update(resumed=True, restore_exact=True, loaded_checkpoint=a['step_one_checkpoint'], steps=copy.deepcopy(steps[1:]))
    return manifest, m, a, b


def test_actual_gate_pair_accepts_only_exact_learning_replay(pair):
    path, m, a, b = pair
    b['steps'][0]['seconds'] = 200.
    probe.check_train_reports(path, m, a, b)


@pytest.mark.parametrize('field,value', [
    ('source_git_commit', 'wrong'), ('initial_state_sha256', 'wrong'),
    ('job_id', '456'), ('formal_training_ready', True), ('formal_checkpoint_produced', True),
    ('test_records_read', True), ('automatic_successor', True),
    ('checkpoint_binding', {'purpose': 'formal'}), ('restore_exact', False),
    ('state_after_three', 'different'), ('resumed', False)])
def test_gate_rejects_wrong_authority_or_restore(pair, field, value):
    path, m, a, b = pair
    b[field] = value
    with pytest.raises(AssertionError): probe.check_train_reports(path, m, a, b)


@pytest.mark.parametrize('field,value', [
    ('microbatch', 512), ('microbatches', 4), ('rows', 81), ('skipped', True),
    ('gradient_tensors', 151), ('candidate_classes', 32768), ('total_loss', 6),
    ('peak_allocated_bytes', 0), ('peak_reserved_bytes', 201)])
def test_gate_rejects_both_reports_agreeing_on_wrong_step(pair, field, value):
    path, m, a, b = pair
    a['steps'][1][field] = b['steps'][0][field] = value
    with pytest.raises(AssertionError): probe.check_train_reports(path, m, a, b)


def test_changed_checkpoint_bytes_rejected(pair):
    path, m, a, b = pair
    Path(a['validation_checkpoint']['path']).write_bytes(b'changed')
    with pytest.raises(AssertionError): probe.check_train_reports(path, m, a, b)


@pytest.mark.parametrize('changed', ['asset_files', 'asset_seconds', 'cache_verification'])
def test_rejects_incomplete_input_verification(pair, changed):
    path, m, a, b = pair
    if changed == 'asset_files': a['asset_byte_verification']['files'] = {}
    elif changed == 'asset_seconds': a['asset_byte_verification']['seconds'] = float('nan')
    else: b['cache']['verification']['cache_sha256'] = 'wrong'
    with pytest.raises(AssertionError): probe.check_train_reports(path, m, a, b)


@pytest.mark.parametrize('kind,gpus,limit,module', [
    ('training_gate', 2, '00:45:00', 'probe_liger_capacity_training'),
    ('formal_train', 1, '12:00:00', 'train_liger_capacity_cpu4'),
    ('formal_val', 2, '06:00:00', 'validate_liger_capacity_cpu4')])
def test_launchers_resource_caps_and_no_auto_submission(kind, gpus, limit, module):
    root = Path(__file__).resolve().parents[1]
    text = (root / f'run_liger_capacity_{kind}_cpu4.sh').read_text()
    assert '#SBATCH --ntasks=1' in text and '#SBATCH --cpus-per-task=4' in text
    assert f'#SBATCH --gres=gpu:{gpus}' in text and '#SBATCH --mem=24G' in text
    assert f'#SBATCH --time={limit}' in text
    assert '--cpu-bind=verbose,threads' in text and '--kill-on-bad-exit=1' in text
    assert 'set -euo pipefail' in text and 'NUMPY_MADVISE_HUGEPAGE=0' in text
    assert f'-m scripts.{module}' in text and 'sbatch ' not in text
    if kind == 'training_gate':
        lines = [line for line in text.splitlines() if '-m scripts.' in line]
        assert [line.split()[1] for line in lines] == ['1', '1', '1', '2', '2']
        assert [line.split()[-2] for line in lines] == ['continuous', 'resumed', 'accept_train', 'validation', 'accept']


def test_gate_cli_does_not_expose_dataset_budget_or_formal_initialization_override():
    tree = ast.parse(Path(probe.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument']
    assert [node.args[0].value for node in calls] == ['mode', 'manifest']
    choices = next(value.value for value in calls[0].keywords if value.arg == 'choices')
    assert ast.literal_eval(choices) == ['continuous', 'resumed', 'accept_train', 'validation', 'accept']
