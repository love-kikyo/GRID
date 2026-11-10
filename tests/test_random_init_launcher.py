"""Inspect real shell argument construction without importing ML or using GPUs."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'run_click_listwise_random_init.sh'
MOCK = r'''
function /home/MMReco2021/.conda/envs/GRID/bin/python() {
  printf 'CHECK=%s\n' "$@"
  if [[ "$*" == *scripts.verify_random_init* && "${FAIL_SMOKE:-0}" == 1 ]]; then return 1; fi
}
srun() { printf 'ARG=%s\n' "$@"; printf 'ENV=%s,%s,%s,%s\n' "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$TF_NUM_INTRAOP_THREADS" "$GRID_CPU_BUDGET"; }
mkdir() { :; }
cd() { :; }
date() { printf '2099-01-01\n'; }
git() {
  if [[ "$1" == diff && "${FAIL_SOURCE:-0}" == 1 ]]; then return 1; fi
  if [[ "$1" == status && "${DIRTY_SOURCE:-0}" == 1 ]]; then printf ' M src/train.py\n'; fi
  return 0
}
'''


def invoke(*args, **overrides):
    env = {'PATH': os.defpath, 'USER': 'test', 'SLURM_JOB_ID': 'mock',
           'SLURM_NTASKS': '3', 'SLURM_NTASKS_PER_NODE': '3', 'SLURM_CPUS_PER_TASK': '1',
           'SLURM_JOB_CPUS_PER_NODE': '3', 'GRID_RANDOM_INIT_COMMIT': 'a' * 40,
           'OMP_NUM_THREADS': '24', **overrides}
    return subprocess.run(['bash', '-c', MOCK + SCRIPT.read_text(), str(SCRIPT), *args],
                          env=env, capture_output=True, text=True)


def test_smoke_then_fresh_formal_keeps_comparison_settings():
    result = invoke()
    assert result.returncode == 0, result.stderr
    text = result.stdout
    for arg in ('ckpt_path=null', 'model.load_only_weights=false', 'model.manual_ckpt_path=null',
                'model.rerank_objective=listwise', 'callbacks.restart_job=null',
                '++should_skip_retry=true', 'test=false', '--cpus-per-task=1', '--cpu-bind=verbose,cores',
                'optim.scheduler.num_training_steps=120000',
                'data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false'):
        assert text.count('ARG=' + arg + '\n') == 2
    assert 'ARG=trainer.max_steps=2\n' in text
    assert 'ARG=trainer.max_steps=120000\n' in text
    assert text.count('ARG=callbacks.model_checkpoint=null\n') == 1
    assert text.count('ARG=callbacks.early_stopping=null\n') == 1
    assert text.count('ENV=1,1,1,4') == 2
    assert '--initialization-checkpoint' not in text
    assert text.index('random-init-smoke-listwise-seed42-mock') < text.index('CHECK=scripts.verify_random_init')
    assert text.index('CHECK=scripts.verify_random_init') < text.index('ARG=id=2099-01-01/click-listwise-random-seed42-mock')


def test_failed_smoke_acceptance_blocks_formal_fit():
    result = invoke(FAIL_SMOKE='1')
    assert result.returncode != 0
    assert 'ARG=trainer.max_steps=2\n' in result.stdout
    assert 'ARG=trainer.max_steps=120000\n' not in result.stdout


@pytest.mark.parametrize('change', [dict(SLURM_CPUS_PER_TASK='8'), dict(FAIL_SOURCE='1'),
                                  dict(DIRTY_SOURCE='1'), dict(GRID_RANDOM_INIT_COMMIT='')])
def test_invalid_allocation_or_source_stops_before_ml(change):
    result = invoke(**change)
    assert result.returncode != 0
    assert 'ARG=' not in result.stdout and 'CHECK=' not in result.stdout


def test_extra_seed_or_checkpoint_is_not_accepted():
    for args in [('43',), ('42', 'teacher.ckpt')]:
        result = invoke(*args)
        assert result.returncode == 2
        assert 'ARG=' not in result.stdout
