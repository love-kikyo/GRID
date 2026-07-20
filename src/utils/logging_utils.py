import json
import os
import socket
import subprocess
from pathlib import Path
from importlib.util import find_spec
from typing import Any, Dict

from dotenv import load_dotenv
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# logging constants
END_RUN = "end_run"


def convert_dict_to_json_string(data: dict) -> str:
    return json.dumps(data, indent=4)


def _safe_run_command(
    command: list[str],
    cwd: str | None = None,
    errors: list[str] | None = None,
) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        if errors is not None:
            errors.append(f"{' '.join(command)} failed in {cwd}: {exc}")
        return None
    except subprocess.CalledProcessError as exc:
        if errors is not None:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or f"exit code {exc.returncode}"
            errors.append(f"{' '.join(command)} failed in {cwd}: {detail}")
        return None
    except subprocess.SubprocessError as exc:
        if errors is not None:
            errors.append(f"{' '.join(command)} failed in {cwd}: {exc}")
        return None

    output = result.stdout.strip()
    return output or None


def _resolve_git_root(candidates: list[Path], errors: list[str] | None = None) -> Path | None:
    for candidate in candidates:
        git_root = _safe_run_command(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(candidate),
            errors=errors,
        )
        if git_root:
            return Path(git_root).resolve()

    return None


def _find_git_root_by_marker(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        current = candidate if candidate.is_dir() else candidate.parent
        for parent in [current, *current.parents]:
            if (parent / ".git").exists():
                return parent.resolve()
    return None


def _resolve_git_dir(git_root: Path) -> Path | None:
    git_marker = git_root / ".git"
    if git_marker.is_dir():
        return git_marker
    if git_marker.is_file():
        try:
            content = git_marker.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if content.startswith("gitdir:"):
            git_dir = Path(content.split(":", 1)[1].strip())
            if not git_dir.is_absolute():
                git_dir = git_root / git_dir
            return git_dir.resolve()
    return None


def _read_packed_ref(git_dir: Path, ref_name: str) -> str | None:
    packed_refs = git_dir / "packed-refs"
    if not packed_refs.exists():
        return None
    try:
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            commit, ref = line.split(" ", 1)
            if ref == ref_name:
                return commit
    except (OSError, ValueError):
        return None
    return None


def _read_git_head(git_root: Path) -> tuple[str | None, str | None]:
    git_dir = _resolve_git_dir(git_root)
    if git_dir is None:
        return None, None

    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None, None

    if not head.startswith("ref:"):
        return head or None, "HEAD"

    ref_name = head.split(":", 1)[1].strip()
    branch = ref_name.removeprefix("refs/heads/")
    ref_path = git_dir / ref_name
    try:
        commit = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        commit = _read_packed_ref(git_dir, ref_name)

    return commit or None, branch or None


def _find_slurm_artifacts(job_id: str, search_roots: list[Path]) -> list[Path]:
    patterns = [
        f"slurm-{job_id}.out",
        f"*{job_id}*.out",
        f"*{job_id}*.err",
        f"*{job_id}*.txt",
        f"*{job_id}*",
    ]
    subdirs = ["", "outputs", "errors", "slurm", "logs/slurm"]
    matches: list[Path] = []
    seen: set[Path] = set()

    for root in search_roots:
        for subdir in subdirs:
            base_dir = root / subdir
            if not base_dir.exists():
                continue
            for pattern in patterns:
                for candidate in base_dir.glob(pattern):
                    if candidate.is_file() and candidate not in seen:
                        seen.add(candidate)
                        matches.append(candidate.resolve())

    return matches


@rank_zero_only
def login_wandb():
    """
    If WANDB_API_KEY is set in the environment, login to wandb.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Now you can access the WANDB_API_KEY
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        import wandb

        wandb.login(key=wandb_api_key, relogin=True)


@rank_zero_only
def finalize_loggers(trainer: Any, status=END_RUN) -> None:
    """
    Finalize loggers after training is done.

    :param trainer: The Lightning trainer.
    """
    [
        logger.finalize(status)
        for logger in trainer.loggers
        if hasattr(logger, "finalize")
    ]

    if find_spec(
        "wandb"
    ):  # check if wandb is installed. If so, close connection to wandb.
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_run_metadata(cfg: DictConfig) -> None:
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(cfg.paths.work_dir).resolve()
    root_dir = Path(cfg.paths.root_dir).resolve()
    git_errors: list[str] = []
    git_candidates = [work_dir, root_dir, output_dir]
    git_root = _resolve_git_root(git_candidates, errors=git_errors)
    if git_root is None:
        git_root = _find_git_root_by_marker(git_candidates)
        if git_root is not None:
            git_errors.append("git root resolved from .git marker fallback")

    git_cwd = str(git_root) if git_root else None
    git_commit = (
        _safe_run_command(["git", "rev-parse", "HEAD"], cwd=git_cwd, errors=git_errors)
        if git_cwd
        else None
    )
    git_branch = (
        _safe_run_command(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=git_cwd,
            errors=git_errors,
        )
        if git_cwd
        else None
    )
    git_status = (
        _safe_run_command(["git", "status", "--short"], cwd=git_cwd, errors=git_errors)
        if git_cwd
        else None
    )
    if git_root is not None and (git_commit is None or git_branch is None):
        fallback_commit, fallback_branch = _read_git_head(git_root)
        git_commit = git_commit or fallback_commit
        git_branch = git_branch or fallback_branch
        if fallback_commit or fallback_branch:
            git_errors.append("git commit/branch resolved from .git/HEAD fallback")

    slurm_job_id = os.getenv("SLURM_JOB_ID")
    slurm_job_name = os.getenv("SLURM_JOB_NAME")
    slurm_artifacts: list[str] = []
    if slurm_job_id:
        slurm_dir = output_dir / "slurm"
        slurm_dir.mkdir(exist_ok=True)
        for artifact in _find_slurm_artifacts(slurm_job_id, [work_dir, root_dir]):
            target = slurm_dir / artifact.name
            if not target.exists():
                target.symlink_to(artifact)
            slurm_artifacts.append(str(target))

    metadata = {
        "task_name": cfg.get("task_name"),
        "id": cfg.get("id"),
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "hostname": socket.gethostname(),
        "git": {
            "root": str(git_root) if git_root else None,
            "commit": git_commit,
            "branch": git_branch,
            "status": git_status.splitlines() if git_status else [],
            "errors": git_errors,
        },
        "slurm": {
            "job_id": slurm_job_id,
            "job_name": slurm_job_name,
            "job_nodelist": os.getenv("SLURM_JOB_NODELIST"),
            "local_id": os.getenv("SLURM_LOCALID"),
            "proc_id": os.getenv("SLURM_PROCID"),
            "artifacts": slurm_artifacts,
        },
    }

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=True)

    if git_commit:
        with (output_dir / "git_commit.txt").open("w", encoding="utf-8") as file:
            file.write(f"{git_commit}\n")


@rank_zero_only
def log_hyperparameters(
    cfg: DictConfig, model: LightningModule, trainer: Trainer
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}
    # We resolve the configs to get the actual paths for logging.
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["paths"] = cfg["paths"]
    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data_loading"] = cfg["data_loading"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
