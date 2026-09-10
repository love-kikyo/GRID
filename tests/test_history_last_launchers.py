"""Exercise actual Bash launchers without running preflight, Slurm or training."""
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = {
    "sid": ("run_sid_history_last_user_v1.sh", "sid", "sid", "tiger-sid-stage"),
    "listwise": ("run_click_listwise_history_last_user_v1.sh", "click-listwise", "listwise", "recochain-listwise"),
    "pointwise": ("run_click_pointwise_history_last_user_v1.sh", "click-pointwise", "pointwise", "recochain-pointwise"),
    "separate": ("run_rerank_small_history_last_user_v1.sh", "rerank-small", "separate_small", "separate-reranker-small"),
}
MOCK_COMMANDS = r'''
function /home/MMReco2021/.conda/envs/GRID/bin/python() { printf 'PREFLIGHT_ARG=%s\n' "$@"; }
srun() { printf 'TRAIN_ARG=%s\n' "$@"; }
mkdir() { :; }
cd() { :; }
date() { printf '2026-09-10\n'; }
'''


def invoke(method, arguments, **env_overrides):
    script = ROOT / LAUNCHERS[method][0]
    env = {
        "PATH": os.defpath, "USER": "launcher-test", "SLURM_JOB_ID": "mock-job",
        "SLURM_NTASKS": "3", "SLURM_NTASKS_PER_NODE": "3", **env_overrides,
    }
    return subprocess.run(
        ["bash", "-c", MOCK_COMMANDS + script.read_text(), str(script), *arguments],
        env=env, capture_output=True, text=True, check=False,
    )


@pytest.fixture
def teacher(tmp_path):
    path = tmp_path / "teacher with spaces.ckpt"
    path.touch()
    return str(path)


def arguments(method, teacher, seed=None):
    result = [] if method == "sid" else [teacher]
    return result if seed is None else [*result, str(seed)]


@pytest.mark.parametrize("method", LAUNCHERS)
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_seed_reaches_training_and_provenance_names(method, seed, teacher):
    result = invoke(method, arguments(method, teacher, seed))
    assert result.returncode == 0, result.stderr
    train = [line.removeprefix("TRAIN_ARG=") for line in result.stdout.splitlines() if line.startswith("TRAIN_ARG=")]
    preflight = [line.removeprefix("PREFLIGHT_ARG=") for line in result.stdout.splitlines() if line.startswith("PREFLIGHT_ARG=")]
    script, run_prefix, report_prefix, tag = LAUNCHERS[method]
    assert f"seed={seed}" in train
    assert f"id=2026-09-10/{run_prefix}-seed{seed}-mock-job" in train
    assert f"tags=[taobao-history-last-user-v1,{tag},seed{seed}]" in train
    assert preflight[preflight.index("--report-path") + 1].endswith(f"/{report_prefix}_seed{seed}_mock-job.json")
    assert script in preflight
    for fixed_arg in (
        "--ntasks=3", "--ntasks-per-node=3", "--kill-on-bad-exit=1",
        "trainer.max_steps=120000", "optim.scheduler.num_training_steps=120000",
        "num_hierarchies=4", "num_embeddings_per_hierarchy=4096", "++should_skip_retry=true",
        "data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false",
        "data_loading.val_dataloader_config.dataloader.num_workers=2",
        "data_loading.test_dataloader_config.dataloader.num_workers=3",
    ):
        assert fixed_arg in train
    if method != "sid":
        assert preflight[preflight.index("--initialization-checkpoint") + 1] == teacher
        field = "model.sid_teacher_ckpt_path" if method == "separate" else "model.manual_ckpt_path"
        assert f"{field}={teacher}" in train
    else:
        assert "--initialization-checkpoint" not in preflight
        assert not any("ckpt_path=" in arg for arg in train)


@pytest.mark.parametrize("method", LAUNCHERS)
def test_default_still_means_seed42(method, teacher):
    default = invoke(method, arguments(method, teacher))
    explicit = invoke(method, arguments(method, teacher, 42))
    assert default.returncode == explicit.returncode == 0
    assert default.stdout == explicit.stdout


@pytest.mark.parametrize("method", LAUNCHERS)
@pytest.mark.parametrize("seed", ["-1", "abc", "4.3"])
def test_invalid_seed_rejected_before_any_task(method, seed, teacher):
    result = invoke(method, arguments(method, teacher, seed))
    assert result.returncode == 2
    assert "SEED must be a non-negative integer" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("method", LAUNCHERS)
def test_extra_arguments_rejected(method, teacher):
    result = invoke(method, [*arguments(method, teacher, 43), "extra"])
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("method", ["listwise", "pointwise", "separate"])
def test_missing_teacher_rejected(method, tmp_path):
    result = invoke(method, arguments(method, str(tmp_path / "absent.ckpt"), 43))
    assert result.returncode == 2
    assert "Teacher checkpoint not found" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("method", LAUNCHERS)
@pytest.mark.parametrize("field", ["SLURM_NTASKS", "SLURM_NTASKS_PER_NODE"])
def test_wrong_ddp_allocation_cannot_start_training(method, field, teacher):
    result = invoke(method, arguments(method, teacher, 43), **{field: "1"})
    assert result.returncode == 1
    assert "three-task Slurm allocation" in result.stderr
    assert "TRAIN_ARG=" not in result.stdout
