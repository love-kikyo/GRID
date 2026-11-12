"""CPU checks of fresh-process gate decisions without real data or GPU."""
import copy
import json
from pathlib import Path

import pytest

from scripts import probe_liger_sampled_training as gate


def pair():
    steps = [dict(attempt=n, rows=2048, updates=n, skipped=False, final_target_count=300,
                  training_seconds=1., input_seconds=.5, seconds=1.5,
                  peak_allocated_bytes=20, peak_reserved_bytes=30, learning_rate=n * 1e-7)
             for n in range(1, 4)]
    checkpoint = dict(path='step1.pt', sha256='a' * 64, full_state_sha256='b' * 64)
    a = dict(steps=steps, initial_state_sha256='i' * 64, final_state_sha256='f' * 64,
             final_progress=dict(updates=3, attempts=3, prefix_count=6144, final_count=900),
             gpu={'name': '4090'}, versions={'numpy': '2.0.2'}, checkpoint_binding={'kind': 'gate'},
             clock={'sha256': 'c' * 64}, numpy_memory_policy={'effective_madvise_hugepage': False},
             resume_checkpoint=checkpoint,
             validation_state_proof=dict(full_learning_state_unchanged=True,
                before_state_sha256='s' * 64, after_state_sha256='s' * 64))
    b = copy.deepcopy(a)
    b['steps'] = b['steps'][1:]
    b['loaded_checkpoint'] = b.pop('resume_checkpoint')
    b['resume_proof'] = dict(full_state_exact=True, full_state_sha256='b' * 64)
    return a, b


def test_full_learning_replay_accepts_timing_and_allocator_differences_only():
    a, b = pair()
    for step in b['steps']:
        step.update(training_seconds=2., seconds=2.5, peak_reserved_bytes=40)
    gate.check_gate_pair(a, b)


@pytest.mark.parametrize('change', ['loss', 'initial', 'final', 'checkpoint', 'restore',
    'validation_rng', 'validation_claim', 'prefix_count', 'final_count', 'missing_step', 'sampling', 'gpu'])
def test_non_equivalent_training_cannot_pass(change):
    a, b = pair()
    if change == 'loss': b['steps'][0]['total_loss'] = 2.
    elif change == 'initial': b['initial_state_sha256'] = 'changed'
    elif change == 'final': b['final_state_sha256'] = 'changed'
    elif change == 'checkpoint': b['loaded_checkpoint']['sha256'] = 'changed'
    elif change == 'restore': b['resume_proof']['full_state_exact'] = False
    elif change == 'validation_rng': b['validation_state_proof']['after_state_sha256'] = 'changed'
    elif change == 'validation_claim': b['validation_state_proof']['full_learning_state_unchanged'] = False
    elif change == 'prefix_count': a['final_progress']['prefix_count'] = b['final_progress']['prefix_count'] = 4096
    elif change == 'final_count': a['final_progress']['final_count'] = b['final_progress']['final_count'] = 899
    elif change == 'missing_step': b['steps'].pop()
    elif change == 'sampling': b['steps'][0]['sampling_attempt'] = 99
    else: b['gpu']['name'] = 'other'
    with pytest.raises(AssertionError): gate.check_gate_pair(a, b)


def test_shared_deadline_is_not_extended_by_resume(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(gate.time, 'time', lambda: 1000.)
    first, entry = gate.gate_clock(manifest, {}, False)
    monkeypatch.setattr(gate.time, 'time', lambda: 2000.)
    resumed, other_entry = gate.gate_clock(manifest, {}, True)
    assert first == resumed and entry == other_entry
    assert resumed['deadline_unix_seconds'] == 3400.
    monkeypatch.setattr(gate.time, 'time', lambda: 3401.)
    with pytest.raises(TimeoutError): gate.gate_clock(manifest, {}, True)


def test_clock_cannot_cross_manifest_or_job(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    gate.gate_clock(path, {}, False)
    monkeypatch.setenv('SLURM_JOB_ID', '456')
    with pytest.raises(AssertionError): gate.gate_clock(path, {}, True)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    path.write_text('{"changed": true}')
    with pytest.raises(AssertionError): gate.gate_clock(path, {}, True)


def test_gate_uses_same_formal_helpers_and_separate_create_only_outputs():
    source = Path(gate.__file__).read_text()
    assert 'while p[\'attempts\'] < 3:' in source
    assert 'save_bound_checkpoint' in source and 'restore_bound_checkpoint' in source
    assert 'perform_validation' in source and 'validation_limit=65' in source
    assert 'np.testing.assert_array_equal' in source
    assert "directory.mkdir(exist_ok=False)" in source
    assert 'fit_standardization' not in source and 'sbatch' not in source
