"""Execute Bash with mocked ML/Slurm commands; never allocate GPUs in unit tests."""
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'run_click_listwise_history_last_user_v1.sh'
MOCK = r'''
function /home/MMReco2021/.conda/envs/GRID/bin/python() {
  if [[ "$*" == *scripts.prepare_training_resume* ]]; then
    printf '20000\n'
  else
    printf 'CHECK=%s\n' "$@"
  fi
}
srun() { printf 'ARG=%s\n' "$@"; printf 'ENV=%s,%s,%s,%s\n' "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$TF_NUM_INTRAOP_THREADS" "$GRID_CPU_BUDGET"; }
mkdir() { :; }
cd() { :; }
date() { printf '2026-09-16\n'; }
'''


def invoke(teacher, resume, **overrides):
    env = {'PATH': os.defpath, 'USER': 'test', 'SLURM_JOB_ID': 'mock', 'SLURM_NTASKS': '3',
           'SLURM_NTASKS_PER_NODE': '3', 'SLURM_CPUS_PER_TASK': '1', 'SLURM_JOB_CPUS_PER_NODE': '3',
           'OMP_NUM_THREADS': '24', **overrides}
    return subprocess.run(['bash', '-c', MOCK + SCRIPT.read_text(), str(SCRIPT), str(teacher), '44', str(resume)],
                          env=env, capture_output=True, text=True)


@pytest.fixture
def inputs(tmp_path):
    teacher, resume = tmp_path/'teacher.ckpt', tmp_path/'checkpoint_000_020000.ckpt'
    teacher.touch(); resume.touch()
    return teacher, resume


def test_formal_resume_keeps_horizon_and_uses_new_best_copy(inputs):
    result = invoke(*inputs)
    assert result.returncode == 0, result.stderr
    text = result.stdout
    for arg in ['ARG=--cpus-per-task=1', 'ARG=--cpu-bind=verbose,cores',
                'ARG=model.load_only_weights=false', 'ARG=trainer.max_steps=120000',
                'ARG=optim.scheduler.num_training_steps=120000',
                'ARG=+callbacks.cpu_resume_audit.expected_step=20000', 'ENV=1,1,1,4']:
        assert arg in text
    assert 'click-listwise-seed44-mock/checkpoints/checkpoint_000_020000.ckpt' in text
    assert 'ARG=test=false' not in text


def test_smoke_is_separate_and_cannot_extend_training_schedule(inputs):
    result = invoke(*inputs, GRID_RESUME_SMOKE='1')
    assert result.returncode == 0, result.stderr
    for arg in ['ARG=trainer.max_steps=20100', 'ARG=trainer.limit_val_batches=2', 'ARG=test=false',
                'ARG=optim.scheduler.num_training_steps=120000', 'cpu-resume-smoke-listwise-seed44-mock']:
        assert arg in result.stdout


def test_old_24_cpu_allocation_is_rejected_before_preflight(inputs):
    result = invoke(*inputs, SLURM_CPUS_PER_TASK='8', SLURM_JOB_CPUS_PER_NODE='24')
    assert result.returncode != 0
    assert 'ARG=' not in result.stdout and 'CHECK=' not in result.stdout


def test_missing_resume_is_not_silently_replaced_by_teacher(inputs):
    result = invoke(inputs[0], str(inputs[1]) + '.missing')
    assert result.returncode == 2
    assert 'Resume checkpoint not found' in result.stderr
    assert 'ARG=' not in result.stdout
