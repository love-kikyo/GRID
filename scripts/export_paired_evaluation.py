#!/usr/bin/env python
"""Stream per-example beam and rerank outputs to HDF5 for paired evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import h5py
import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.chdir(ROOT)

OmegaConf.register_new_resolver("sub", lambda x, y: int(x) - int(y), replace=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--training-preflight", type=Path, required=True)
    parser.add_argument("--source-compatibility", type=Path, default=None,
                        help="Explicit, preflight-bound record of reviewed source hash transitions")
    parser.add_argument("--expected-data-root", type=Path, required=True)
    parser.add_argument("--expected-protocol", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--flush-samples", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.flush_samples <= 0:
        parser.error("--flush-samples must be positive")
    if args.output.suffix != ".h5":
        parser.error("--output must use the .h5 suffix")
    return args


def load_run_config(run_dir: Path) -> Any:
    importlib.import_module("src.utils.custom_hydra_resolvers")
    config_path = run_dir.resolve() / ".hydra/config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")
    cfg = OmegaConf.load(config_path)
    cfg.model.load_only_weights = False
    return cfg


def configure_dataloader(cfg: Any, split: str, batch_size: int, num_workers: int) -> None:
    dataloader_cfg = cfg.data_loading[f"{split}_dataloader_config"].dataloader
    dataloader_cfg.batch_size_per_device = batch_size
    dataloader_cfg.num_workers = num_workers
    dataloader_cfg.persistent_workers = num_workers > 0
    dataloader_cfg.drop_last = False
    dataloader_cfg.should_shuffle_rows = False
    dataloader_cfg.assign_all_files_per_worker = False
    dataloader_cfg.dataset_config.keep_user_id = True
    # Keep IDs outside the model's sequence collator: collate_fn_train rejects
    # extra fields, and IDs must never be padded, masked, or treated as input.
    if dataloader_cfg.collate_fn._target_ != "scripts.export_paired_evaluation.collate_with_user_ids":
        sequence_collate = OmegaConf.to_container(dataloader_cfg.collate_fn, resolve=True)
        dataloader_cfg.collate_fn = {
            "_target_": "scripts.export_paired_evaluation.collate_with_user_ids",
            "_partial_": True,
            "sequence_collate": sequence_collate,
            "user_id_field": dataloader_cfg.dataset_config.user_id_field,
        }


def collate_with_user_ids(batch: Any, sequence_collate: Any, user_id_field: str, **kwargs: Any) -> Any:
    """Preserve row identity without changing the original sequence collator."""
    if isinstance(batch, list):
        values = [row[user_id_field] for row in batch]
        sequence_batch = [
            {key: value for key, value in row.items() if key != user_id_field}
            for row in batch
        ]
    else:
        values = batch[user_id_field]
        sequence_batch = {key: value for key, value in batch.items() if key != user_id_field}
    user_ids = torch.stack([torch.as_tensor(value).reshape(()) for value in values])
    if user_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("User IDs must be scalar integers")
    model_input = sequence_collate(sequence_batch, **kwargs)
    if len(user_ids) != model_input.transformed_sequences["sequence_data"].shape[0]:
        raise ValueError("Sequence collator changed the number of examples")
    model_input.user_id_list = user_ids.long()
    return model_input


def instantiate_datamodule(cfg: Any) -> Any:
    import hydra

    datamodule = hydra.utils.instantiate(cfg.data_loading.datamodule)
    datamodule.trainer = SimpleNamespace(world_size=1, global_rank=0)
    datamodule.setup()
    return datamodule


def get_loader(datamodule: Any, split: str) -> Iterable[Any]:
    loader = (datamodule.val_dataloader if split == "val" else datamodule.test_dataloader)()
    if isinstance(loader, tuple):
        loader = loader[0]
    return loader


def resolve_checkpoint(run_dir: Path, checkpoint: Optional[Path]) -> Path:
    if checkpoint is not None:
        path = checkpoint.expanduser().resolve()
    else:
        checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint under {run_dir / 'checkpoints'}")
        path = checkpoints[0].resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def instantiate_model(cfg: Any, checkpoint: Path, device: torch.device) -> Any:
    import hydra

    model = hydra.utils.instantiate(cfg.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state_dict = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model_target = str(cfg.model._target_)

    if missing:
        print(f"[model] missing keys: {missing[:20]}")
    if unexpected:
        unexpected_prefixes = sorted({key.split(".", 1)[0] for key in unexpected})
        print(
            f"[model] unexpected key prefixes={unexpected_prefixes}; "
            f"sample={unexpected[:20]}"
        )
    del state_dict, payload
    gc.collect()

    # SeparateRerankModule constructs its frozen generator teacher in setup().
    if model_target.endswith("SeparateRerankModule"):
        model.setup(stage="validate")

    print(
        f"[model] target={model_target} checkpoint={checkpoint} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    model.eval().to(device)
    return model


def unwrap_batch(batch: Any) -> Any:
    return batch[0] if isinstance(batch, tuple) else batch


def slice_batch(batch: Any, size: int) -> Any:
    from src.data.loading.components.interfaces import SequentialModelInputData

    user_id_list = batch.user_id_list
    if isinstance(user_id_list, (torch.Tensor, list, tuple)):
        user_id_list = user_id_list[:size]
    return SequentialModelInputData(
        user_id_list=user_id_list,
        transformed_sequences={key: value[:size] for key, value in batch.transformed_sequences.items()},
        mask=batch.mask[:size] if batch.mask is not None else None,
    )


@contextmanager
def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        yield
        return
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    from src.data.loading.components.interfaces import SequentialModelInputData

    user_id_list = batch.user_id_list
    if isinstance(user_id_list, torch.Tensor):
        user_id_list = user_id_list.to(device, non_blocking=True)
    return SequentialModelInputData(
        user_id_list=user_id_list,
        transformed_sequences={
            key: value.to(device, non_blocking=True)
            for key, value in batch.transformed_sequences.items()
        },
        mask=batch.mask.to(device, non_blocking=True) if batch.mask is not None else None,
    )


def infer_batch(model: Any, batch: Any, device: torch.device, amp: str) -> Dict[str, Any]:
    from src.models.modules.huggingface.transformer_base_module import ModelMode

    model_batch = move_batch_to_device(batch, device)
    with torch.inference_mode(), autocast_context(device, amp):
        if model.__class__.__name__ == "SeparateRerankModule":
            return model.model_step(model_input=model_batch, mode=ModelMode.INFER)

        sid = model_batch.transformed_sequences["sequence_data"].long()
        mask = model_batch.mask.long()
        hist_itemkey = model_batch.transformed_sequences["hist_itemkey"].long()
        target_sid = sid[:, -model.num_hierarchies :].contiguous()
        result, _ = model._infer_for_eval(
            hist_sid=sid[:, : -model.num_hierarchies].contiguous(),
            hist_mask=mask[:, : -model.num_hierarchies].contiguous(),
            hist_itemkey=hist_itemkey[:, :-1].contiguous(),
            target_sid=target_sid,
        )
        return result


def align_rerank_logits_to_beam(
    beam_ids: torch.Tensor,
    rerank_ids: torch.Tensor,
    ordered_rerank_logits: torch.Tensor,
) -> torch.Tensor:
    """Undo final rerank ordering so residual logits align with beam candidates."""
    matches = rerank_ids.unsqueeze(2).eq(beam_ids.unsqueeze(1)).all(dim=-1)
    if not torch.all(matches.sum(dim=1) == 1) or not torch.all(matches.sum(dim=2) == 1):
        raise AssertionError("Beam and rerank outputs are not a one-to-one candidate permutation")
    return torch.einsum("bj,bjk->bk", ordered_rerank_logits.float(), matches.float())


def example_hashes(batch: Any) -> np.ndarray:
    """Return stable 128-bit input fingerprints as two uint64 columns."""
    user_ids = extract_user_ids(batch)
    sid = batch.transformed_sequences["sequence_data"].detach().cpu().numpy()
    hist_itemkey = batch.transformed_sequences["hist_itemkey"].detach().cpu().numpy()
    mask = batch.mask.detach().cpu().numpy()
    fingerprints = np.empty((sid.shape[0], 2), dtype=np.uint64)
    for row in range(sid.shape[0]):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.asarray([user_ids[row]], dtype="<i8").view(np.uint8))
        digest.update(np.ascontiguousarray(sid[row]).view(np.uint8))
        digest.update(np.ascontiguousarray(hist_itemkey[row]).view(np.uint8))
        digest.update(np.ascontiguousarray(mask[row]).view(np.uint8))
        fingerprints[row] = np.frombuffer(digest.digest(), dtype="<u8")
    return fingerprints


def extract_user_ids(batch: Any) -> np.ndarray:
    """Normalize collated scalar user IDs to one int64 value per row."""
    values = batch.user_id_list
    if isinstance(values, torch.Tensor):
        user_ids = values.detach().cpu().numpy().reshape(-1)
    elif isinstance(values, (list, tuple)):
        user_ids = np.asarray(
            [int(value.item()) if isinstance(value, torch.Tensor) else int(value) for value in values],
            dtype=np.int64,
        )
    else:
        raise TypeError(f"Unsupported user_id_list type: {type(values)!r}")
    batch_size = batch.transformed_sequences["sequence_data"].shape[0]
    if user_ids.shape != (batch_size,):
        raise ValueError(
            f"Expected one user ID per example, got {user_ids.shape} for batch {batch_size}"
        )
    return user_ids.astype(np.int64, copy=False)


def itemkeys_from_sid(sid: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
    return (sid.long() * powers.long()).sum(dim=-1)


def extract_batch_arrays(model: Any, batch: Any, result: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    beam_ids = result["beam"]["ids"].detach().cpu().long()
    rerank_ids = result["rerank"]["ids"].detach().cpu().long()
    beam_scores = result["beam"]["scores"].detach().cpu().float()
    rerank_scores = result["rerank"]["scores"].detach().cpu().float()
    ordered_logits = result["rerank"]["click_logits"].detach().cpu().float()
    raw_logits = align_rerank_logits_to_beam(beam_ids, rerank_ids, ordered_logits)
    target_sid = batch.transformed_sequences["sequence_data"][:, -model.num_hierarchies :].cpu().long()

    return {
        "user_id": extract_user_ids(batch),
        "example_hash": example_hashes(batch),
        "target_sid": target_sid.numpy().astype(np.int16),
        "beam_sid": beam_ids.numpy().astype(np.int16),
        "beam_score": beam_scores.numpy().astype(np.float32),
        "raw_rerank_logit": raw_logits.numpy().astype(np.float32),
        "rerank_sid": rerank_ids.numpy().astype(np.int16),
        "rerank_score": rerank_scores.numpy().astype(np.float32),
    }


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_source_compatibility(path: Optional[Path], preflight: Path) -> Dict[str, Any]:
    """Load explicit exact-hash exceptions; never infer compatibility from Git."""
    if path is None:
        return {}
    path = path.expanduser().resolve()
    record = load_json(path)
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported source compatibility schema")
    if record.get("training_preflight_sha256") != file_sha256(preflight):
        raise ValueError("Source compatibility record is bound to a different preflight")
    changes = record.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("Source compatibility record must contain reviewed changes")
    seen = set()
    for change in changes:
        relative = Path(change["path"])
        resolved = (ROOT / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(ROOT.resolve()):
            raise ValueError("Compatibility paths must stay inside the repository")
        if str(resolved) in seen:
            raise ValueError("Duplicate source compatibility path")
        seen.add(str(resolved))
        if not change.get("reason") or not change.get("change_commit"):
            raise ValueError("Source compatibility requires a review reason and change commit")
        for key in ("training_sha256", "evaluation_sha256"):
            value = change.get(key, "")
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Source compatibility requires exact SHA256 values")
        if change["training_sha256"] == change["evaluation_sha256"]:
            raise ValueError("Source compatibility must describe a changed source")
    return {**record, "path": str(path), "sha256": file_sha256(path)}


def validate_provenance(
    cfg: Any,
    split: str,
    expected_data_root: Path,
    expected_protocol: str,
    training_preflight: Path,
    source_compatibility: Optional[Path] = None,
) -> Tuple[Dict[str, Any], int]:
    """Refuse exports whose run, preflight, or manifests do not match."""
    data_root = Path(str(cfg.data_dir)).expanduser().resolve()
    expected_data_root = expected_data_root.expanduser().resolve()
    if data_root != expected_data_root:
        raise ValueError(
            f"Run data root {data_root} does not match required root {expected_data_root}"
        )

    manifest_paths = {
        "train": data_root / "protocol_manifest_train.json",
        "test": data_root / "protocol_manifest_test.json",
        "audit": data_root / "protocol_audit.json",
    }
    manifests = {name: load_json(path) for name, path in manifest_paths.items()}
    protocols = {payload.get("protocol") for payload in manifests.values()}
    if protocols != {expected_protocol}:
        raise ValueError(
            f"Dataset protocol mismatch: expected {expected_protocol!r}, got {protocols}"
        )
    if manifests["audit"].get("status") != "passed":
        raise ValueError("Dataset audit is not passed")

    training_preflight = training_preflight.expanduser().resolve()
    preflight = load_json(training_preflight)
    if preflight.get("status") != "passed":
        raise ValueError(f"Training preflight is not passed: {training_preflight}")
    if Path(str(preflight.get("data_root"))).resolve() != data_root:
        raise ValueError("Training preflight data root does not match run data root")
    if preflight.get("protocol") != expected_protocol:
        raise ValueError("Training preflight protocol does not match expected protocol")

    current_manifest_hashes = {
        name: file_sha256(path) for name, path in manifest_paths.items()
    }
    preflight_manifests = preflight.get("source_manifests", {})
    for name, current_hash in current_manifest_hashes.items():
        if preflight_manifests.get(name, {}).get("sha256") != current_hash:
            raise ValueError(
                f"Training preflight {name} manifest hash does not match current data"
            )

    required_training_sources = [
        ROOT / "src/data/loading/components/dataloading.py",
        ROOT / "src/data/loading/components/interfaces.py",
        ROOT / "src/models/modules/semantic_id/tiger_generation_model.py",
    ]
    if str(cfg.model._target_).endswith("SeparateRerankModule"):
        required_training_sources.append(
            ROOT / "src/models/modules/semantic_id/separate_rerank_model.py"
        )
    preflight_sources = {
        str(Path(entry["path"]).resolve()): entry
        for entry in preflight.get("source_files", [])
    }
    compatibility = load_source_compatibility(source_compatibility, training_preflight)
    reviewed_changes = {
        str((ROOT / change["path"]).resolve()): change
        for change in compatibility.get("changes", [])
    }
    applied_changes = []
    verified_training_sources = {}
    for path in required_training_sources:
        resolved = path.resolve()
        entry = preflight_sources.get(str(resolved))
        if entry is None:
            raise ValueError(f"Training preflight lacks required source file: {resolved}")
        current_hash = file_sha256(resolved)
        if entry.get("sha256") != current_hash:
            change = reviewed_changes.get(str(resolved))
            if (change is None or change["training_sha256"] != entry.get("sha256")
                    or change["evaluation_sha256"] != current_hash):
                raise ValueError(f"Training source changed since preflight: {resolved}")
            applied_changes.append(change)
        verified_training_sources[str(resolved)] = current_hash
    if len(applied_changes) != len(reviewed_changes):
        raise ValueError("Unused source compatibility changes; use a record for this exact run")

    initialization = preflight.get("initialization_checkpoint")
    configured_teacher = OmegaConf.select(cfg, "model.sid_teacher_ckpt_path")
    if configured_teacher is None:
        configured_teacher = OmegaConf.select(cfg, "model.manual_ckpt_path")
    if initialization is not None:
        if configured_teacher is None:
            raise ValueError("Preflight records a teacher but run config does not")
        teacher_path = Path(str(configured_teacher)).expanduser().resolve()
        if teacher_path != Path(initialization["path"]).resolve():
            raise ValueError("Run config teacher path does not match training preflight")
        if file_sha256(teacher_path) != initialization.get("sha256"):
            raise ValueError("Teacher checkpoint changed since training preflight")
    elif configured_teacher is not None:
        raise ValueError("Run config uses a teacher absent from training preflight")

    destination = "evaluation" if split == "val" else "test"
    expected_samples = int(
        manifests["audit"]["destinations"][destination]["records"]
    )
    if manifests["audit"]["destinations"][destination]["unique_users"] != expected_samples:
        raise ValueError(f"{destination} is not one-record-per-user")
    if int(preflight.get("counts", {}).get(destination, -1)) != expected_samples:
        raise ValueError(f"Training preflight count does not match {destination}")

    provenance = {
        "data_root": str(data_root),
        "protocol": expected_protocol,
        "sample_unit": "unique_user_history",
        "expected_samples": expected_samples,
        "manifests": {
            name: {"path": str(path.resolve()), "sha256": current_manifest_hashes[name]}
            for name, path in manifest_paths.items()
        },
        "training_preflight": {
            "path": str(training_preflight),
            "sha256": file_sha256(training_preflight),
            "status": preflight["status"],
        },
        "verified_training_sources": verified_training_sources,
        "source_compatibility": compatibility or None,
        "initialization_checkpoint": initialization,
    }
    return provenance, expected_samples


def git_commit() -> Optional[str]:
    inherited = os.environ.get("SOURCE_GIT_COMMIT")
    if inherited:
        return inherited
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


class H5StreamWriter:
    """Append aligned NumPy batches without retaining the full split in RAM."""

    def __init__(self, output: Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing export: {output}")
        self.output = output
        self.handle = h5py.File(output, "w")
        self.num_samples = 0

    def append(self, arrays: Mapping[str, np.ndarray]) -> None:
        sample_counts = {value.shape[0] for value in arrays.values()}
        if len(sample_counts) != 1:
            raise AssertionError(f"Buffered arrays have inconsistent sample counts: {sample_counts}")
        count = sample_counts.pop()
        start = self.num_samples
        stop = start + count
        for key, value in arrays.items():
            if key not in self.handle:
                chunk_rows = min(2048, max(1, count))
                self.handle.create_dataset(
                    key,
                    shape=(0, *value.shape[1:]),
                    maxshape=(None, *value.shape[1:]),
                    chunks=(chunk_rows, *value.shape[1:]),
                    dtype=value.dtype,
                    compression="lzf",
                    shuffle=True,
                )
            dataset = self.handle[key]
            dataset.resize(stop, axis=0)
            dataset[start:stop] = value
        self.num_samples = stop

    def close(self) -> None:
        self.handle.attrs["num_samples"] = self.num_samples
        self.handle.flush()
        self.handle.close()


def flush_buffer(
    writer: H5StreamWriter,
    arrays: Mapping[str, List[np.ndarray]],
) -> int:
    if not next(iter(arrays.values())):
        return 0
    concatenated = {key: np.concatenate(chunks, axis=0) for key, chunks in arrays.items()}
    writer.append(concatenated)
    flushed = next(iter(concatenated.values())).shape[0]
    for chunks in arrays.values():
        chunks.clear()
    return flushed


def write_metadata(output: Path, metadata: Mapping[str, Any], num_samples: int) -> None:
    metadata_path = output.with_suffix(".json")
    with metadata_path.open("w") as handle:
        json.dump({**metadata, "num_samples": num_samples}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[done] wrote {output} and {metadata_path}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    checkpoint_root = (run_dir / "checkpoints").resolve()
    if not checkpoint.is_relative_to(checkpoint_root):
        raise ValueError(
            f"Checkpoint {checkpoint} is not inside run checkpoint directory {checkpoint_root}"
        )
    cfg = load_run_config(run_dir)
    provenance, expected_samples = validate_provenance(
        cfg=cfg,
        split=args.split,
        expected_data_root=args.expected_data_root,
        expected_protocol=args.expected_protocol,
        training_preflight=args.training_preflight,
        source_compatibility=args.source_compatibility,
    )
    configure_dataloader(cfg, args.split, args.batch_size, args.num_workers)

    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing export or metadata: {output}, {metadata_path}"
        )
    incomplete_output = output.with_name(
        f"{output.name}.incomplete.{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "src/data/loading/components/dataloading.py",
        ROOT / "src/data/loading/components/interfaces.py",
        ROOT / "src/models/modules/semantic_id/tiger_generation_model.py",
    ]
    if str(cfg.model._target_).endswith("SeparateRerankModule"):
        source_paths.append(
            ROOT / "src/models/modules/semantic_id/separate_rerank_model.py"
        )
    source_file_records = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in source_paths
    ]
    checkpoint_record = {
        "size_bytes": checkpoint.stat().st_size,
        "mtime_ns": checkpoint.stat().st_mtime_ns,
        "sha256": file_sha256(checkpoint),
    }

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    datamodule = instantiate_datamodule(cfg)
    loader = get_loader(datamodule, args.split)
    model = instantiate_model(cfg, checkpoint, device)

    arrays: Dict[str, List[np.ndarray]] = {
        key: []
        for key in (
            "user_id",
            "example_hash",
            "target_sid",
            "beam_sid",
            "beam_score",
            "raw_rerank_logit",
            "rerank_sid",
            "rerank_score",
        )
    }
    processed = 0
    buffered = 0
    writer = H5StreamWriter(incomplete_output)
    try:
        for batch_idx, wrapped_batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = unwrap_batch(wrapped_batch)
            if args.max_samples is not None:
                remaining = args.max_samples - processed
                if remaining <= 0:
                    break
                batch_size = batch.transformed_sequences["sequence_data"].size(0)
                if batch_size > remaining:
                    batch = slice_batch(batch, remaining)

            result = infer_batch(model, batch, device, args.amp)
            batch_arrays = extract_batch_arrays(model, batch, result)
            for key in arrays:
                arrays[key].append(batch_arrays[key])
            batch_count = batch_arrays["target_sid"].shape[0]
            processed += batch_count
            buffered += batch_count

            if buffered >= args.flush_samples:
                flush_buffer(writer, arrays)
                buffered = 0
            if args.progress_every and (batch_idx + 1) % args.progress_every == 0:
                print(f"[export] split={args.split} batches={batch_idx + 1} samples={processed}")
            if args.max_samples is not None and processed >= args.max_samples:
                break

        flush_buffer(writer, arrays)
    finally:
        writer.close()

    if not processed:
        raise RuntimeError("No evaluation examples were exported")
    if args.max_batches is None and args.max_samples is None and processed != expected_samples:
        raise RuntimeError(
            f"Full export count mismatch: expected {expected_samples}, got {processed}"
        )

    incomplete_output.replace(output)

    metadata = {
        "schema_version": 3,
        "storage": "hdf5_lzf_streaming",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_record["sha256"],
        "model_target": str(cfg.model._target_),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "keep_user_id": True,
        "top_k_for_generation": int(model.top_k_for_generation),
        "num_hierarchies": int(model.num_hierarchies),
        "num_embeddings_per_hierarchy": int(model.num_embeddings_per_hierarchy),
        "beam_log_prob_weight": float(model.beam_log_prob_weight),
        "rerank_logit_weight": float(model.rerank_logit_weight),
        "amp": args.amp,
        "device": str(device),
        "git_commit": git_commit(),
        "exporter_sha256": file_sha256(Path(__file__).resolve()),
        "hydra_config_sha256": file_sha256((run_dir / ".hydra/config.yaml").resolve()),
        "checkpoint_stat": {
            "size_bytes": checkpoint_record["size_bytes"],
            "mtime_ns": checkpoint_record["mtime_ns"],
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "sample_unit": "unique_user_history",
        "example_identity": "user_id plus blake2b-128(user_id,sequence_data,hist_itemkey,mask)",
        "metric_identity": "exact_semantic_id_tuple",
        "dataset_provenance": provenance,
        "source_files": source_file_records,
        "limits": {"max_batches": args.max_batches, "max_samples": args.max_samples},
        "flush_samples": args.flush_samples,
    }
    write_metadata(output, metadata, processed)


if __name__ == "__main__":
    main()
