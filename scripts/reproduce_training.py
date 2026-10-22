"""Preview the four-model training plan without importing or running ML code.

This preparation-stage interface deliberately has no execution mode. Its
commands require a separately validated resource launcher and data preflight.
In particular, a single-device preview is not an equivalent reproduction of
the historical three-GPU experiments.
"""

import argparse
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "sid": {
        "experiments": ["tiger_train_flat", "tiger_train_sid"],
        "source_launcher": "run_sid_history_last_user_v1.sh",
        "tag": "tiger-sid-stage",
    },
    "listwise": {
        "experiments": ["tiger_train_flat", "tiger_train_click"],
        "source_launcher": "run_click_listwise_history_last_user_v1.sh",
        "tag": "recochain-listwise",
    },
    "pointwise": {
        "experiments": [
            "tiger_train_flat", "tiger_train_click", "tiger_train_click_pointwise"
        ],
        "source_launcher": "run_click_pointwise_history_last_user_v1.sh",
        "tag": "recochain-pointwise",
    },
    "separate": {
        "experiments": ["tiger_train_flat", "tiger_train_rerank_small"],
        "source_launcher": "run_rerank_small_history_last_user_v1.sh",
        "tag": "separate-reranker-small",
    },
}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
)
CORE_SOURCE_FILES = (
    "scripts/reproduce_training.py",
    "scripts/verify_taobao_training_data.py",
    "src/train.py",
    "src/runtime_cpu.py",
    "src/checkpoint_resume.py",
    "src/data/loading/components/iterators.py",
    "src/data/loading/components/dataloading.py",
    "src/data/loading/components/interfaces.py",
    "src/models/modules/semantic_id/tiger_generation_model.py",
)


def _path(value, name):
    """Keep explicit placeholders usable; do not touch or resolve input assets."""
    if value is None:
        raise ValueError(f"{name} must be an explicit path")
    value = str(value)
    if (not value or value.strip() != value or any(ord(c) < 32 for c in value)
            or "${" in value):
        raise ValueError(f"{name} must be a nonempty path without control characters or interpolation")
    return value


def _hydra_string(value):
    # These quotes are part of the argv element consumed by Hydra, not shell
    # quoting. shlex.join below separately renders the complete argument safely.
    return json.dumps(value, ensure_ascii=False)


def build_plan(*, model, data_root, sid_path, output_root,
               teacher_checkpoint=None, seed=42, devices=3,
               python_executable=None, project_root=None):
    """Return JSON-ready command *previews*. Never launch or create anything."""
    if model not in MODELS:
        raise ValueError("model must be sid, listwise, pointwise, or separate")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(devices) is not int or devices not in (1, 3):
        raise ValueError("devices must be 1 or 3")
    if model == "sid" and teacher_checkpoint is not None:
        raise ValueError("sid starts from scratch and must not receive a teacher checkpoint")
    if model != "sid" and teacher_checkpoint is None:
        raise ValueError(f"{model} requires --teacher-checkpoint")
    data_root = _path(data_root, "data_root")
    sid_path = _path(sid_path, "sid_path")
    output_root = _path(output_root, "output_root")
    teacher = None if teacher_checkpoint is None else _path(teacher_checkpoint, "teacher_checkpoint")
    executable = _path(python_executable or sys.executable, "python_executable")
    root = Path(project_root) if project_root is not None else ROOT
    setting = MODELS[model]
    run_id = f"{model}-seed{seed}-devices{devices}-preview"
    experiments = [f"configs/experiment/{name}.yaml" for name in setting["experiments"]]
    source_files = [*CORE_SOURCE_FILES, *experiments]
    if model == "separate":
        source_files.append("src/models/modules/semantic_id/separate_rerank_model.py")

    preflight = [
        executable, "-m", "scripts.verify_taobao_training_data",
        "--data-root", data_root, "--semantic-id-path", sid_path,
        "--expected-protocol", "history-last-user-v1",
        "--num-hierarchies", "4", "--num-embeddings-per-hierarchy", "4096",
    ]
    if teacher is not None:
        preflight += ["--initialization-checkpoint", teacher]
    for source in source_files:
        preflight += ["--source-file", source]
    preflight += ["--report-path", str(Path(output_root) / "training_preflight.json")]

    argv = [
        executable, "-m", "src.train",
        "experiment=[" + ",".join(setting["experiments"]) + "]",
        "data_dir=" + _hydra_string(data_root),
        "semantic_id_path=" + _hydra_string(sid_path),
        "hydra.run.dir=" + _hydra_string(output_root),
        "num_hierarchies=4", "num_embeddings_per_hierarchy=4096",
        f"seed={seed}", "id=" + _hydra_string(run_id),
        "++should_skip_retry=true", "train=true", "test=false", "ckpt_path=null",
        f"tags=[taobao-history-last-user-v1,{setting['tag']},seed{seed},reproduction-preview]",
        "data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false",
        "data_loading.train_dataloader_config.dataloader.num_workers=1",
        "data_loading.val_dataloader_config.dataloader.num_workers=2",
        "data_loading.test_dataloader_config.dataloader.num_workers=3",
        "trainer.max_steps=120000", "optim.scheduler.num_training_steps=120000",
        f"trainer.devices={devices}", "trainer.num_nodes=1",
        "trainer.strategy=" + ("ddp" if devices == 3 else "auto"),
    ]
    if model in ("listwise", "pointwise"):
        argv += [
            "model.manual_ckpt_path=" + _hydra_string(teacher),
            "model.load_only_weights=true", f"model.rerank_objective={model}",
        ]
    elif model == "separate":
        argv += [
            "model.sid_teacher_ckpt_path=" + _hydra_string(teacher),
            "model.load_only_weights=false",
        ]
    else:
        argv += ["model.load_only_weights=false"]

    environment = {name: "1" for name in THREAD_VARIABLES}
    environment.update(
        GRID_CPU_BUDGET="4", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE",
        TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
    )
    gates = [
        "Execution is not implemented: neither this preview nor --dry-run starts a process.",
        "Validate all input files, fixed data/SID identities, and a new empty output directory before execution.",
        "Run and accept the data preflight with an explicit release-source manifest before training.",
        "The preview binds core source files only; add the eventual launcher, environment lock and asset receipts to execution provenance.",
        "Bind the whole job, all ranks and workers to at most four CPUs; library thread limits alone do not enforce this.",
        "Training leaves test disabled; full validation, fusion selection and final test are separate audited stages.",
        "A tracked .project-root marker file must exist in the exported repository before any ML entrypoint is run.",
    ]
    if devices == 3:
        gates.append(
            "The argv is one rank's payload, not a local three-GPU launcher. "
            "The historical runtime requires an external single-node allocation "
            "with three tasks, one CPU per task, three allocated CPUs and explicit rank CPU binding."
        )
    else:
        gates.append(
            "Single-device execution is not released: the existing CPU-budget "
            "runtime only supports the historical three-task allocation. A "
            "separately validated single-device CPU wrapper is required; do not "
            "disable the guard or fabricate scheduler variables. Effective batch "
            "is 128 instead of 384, so this is not a formal-result reproduction."
        )
    return {
        "schema": "grid.training_preview.v1", "mode": "dry_run_only",
        "execution_enabled": False, "formal_reproduction_verified": False,
        "model": model, "seed": seed, "devices": devices,
        "cwd": str(root), "root_marker": str(root / ".project-root"),
        "source_launcher": setting["source_launcher"],
        "source_experiments": experiments, "preflight_source_files": source_files,
        "preflight_argv": preflight, "training_argv": argv,
        "training_command_preview": shlex.join(argv),
        "environment_preview": environment,
        "cpu_budget": {"whole_job_ceiling": 4, "library_threads": 1},
        "batch": {"per_device": 32, "accumulate_grad_batches": 4,
                  "effective": 32 * 4 * devices, "historical_effective": 384},
        "deliberate_preview_overrides": [
            "Explicit paths, output directory, run identifier, tags and device count.",
            "test=false prevents the historical limited trainer test pass after fitting.",
            "Single-device strategy=auto is a diagnostic proposal, not an approved training launcher."
            if devices == 1 else "Three-device strategy=ddp matches the historical configuration.",
        ],
        "execution_gates": gates,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sid-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, choices=(1, 3), default=3)
    parser.add_argument("--dry-run", action="store_true", help="Preview only (also the default); there is no execute mode.")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(
            model=args.model, data_root=args.data_root, sid_path=args.sid_path,
            output_root=args.output_root, teacher_checkpoint=args.teacher_checkpoint,
            seed=args.seed, devices=args.devices,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
