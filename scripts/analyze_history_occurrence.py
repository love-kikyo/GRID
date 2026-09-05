#!/usr/bin/env python
"""Analyze ground-truth and prediction recurrence in user history."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.chdir(ROOT)

OmegaConf.register_new_resolver("sub", lambda x, y: int(x) - int(y), replace=True)

DEFAULT_KS = (1, 5, 10, 20)
EXPECTED_PROTOCOL = "history-last-user-v1"
EXPECTED_SPLIT_COUNTS = {"train": 6_573_674, "val": 347_416, "test": 2_061_101}
LENGTH_BUCKETS = (
    ("0", 0, 0),
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101+", 101, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dataset", "prediction"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--training-preflight", type=Path, default=None)
    parser.add_argument("--expected-data-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--semantic-id-path", type=Path, default=None)
    parser.add_argument(
        "--collision-audit-cache",
        type=Path,
        default=None,
        help="Reuse or create a standalone JSON cache for the semantic-ID collision audit.",
    )
    parser.add_argument(
        "--skip-collision-audit",
        action="store_true",
        help="Skip the potentially memory-intensive full semantic-ID mapping audit.",
    )
    args = parser.parse_args()

    if args.splits is None:
        args.splits = ["train", "val", "test"] if args.mode == "dataset" else ["test"]
    if args.mode == "prediction" and "train" in args.splits:
        parser.error("prediction mode supports val/test only; train inference is intentionally disabled")
    if args.mode == "prediction" and args.checkpoint is None:
        parser.error("prediction mode requires --checkpoint")
    if args.mode == "prediction" and args.training_preflight is None:
        parser.error("prediction mode requires --training-preflight")
    if any(k <= 0 for k in args.ks):
        parser.error("all --ks values must be positive")
    args.ks = sorted(set(args.ks))
    if args.batch_size is None:
        args.batch_size = 512 if args.mode == "dataset" else 32
    return args


def load_run_config(run_dir: Path) -> Any:
    importlib.import_module("src.utils.custom_hydra_resolvers")
    config_path = run_dir.resolve() / ".hydra/config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")
    return OmegaConf.load(config_path)


def configure_dataloaders(cfg: Any, args: argparse.Namespace) -> None:
    for split in args.splits:
        split_key = {"train": "train", "val": "val", "test": "test"}[split]
        dataloader_cfg = cfg.data_loading[f"{split_key}_dataloader_config"].dataloader
        dataloader_cfg.batch_size_per_device = args.batch_size
        dataloader_cfg.num_workers = args.num_workers
        dataloader_cfg.persistent_workers = args.num_workers > 0
        dataloader_cfg.drop_last = False
        dataloader_cfg.should_shuffle_rows = False
        dataloader_cfg.assign_all_files_per_worker = False
        if args.limit_files is not None:
            dataloader_cfg.limit_files = args.limit_files


def instantiate_datamodule(cfg: Any) -> Any:
    import hydra

    datamodule = hydra.utils.instantiate(cfg.data_loading.datamodule)
    datamodule.trainer = SimpleNamespace(world_size=1, global_rank=0)
    datamodule.setup()
    return datamodule


def get_split_loader(datamodule: Any, split: str) -> Iterable[Any]:
    loader = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }[split]()
    if isinstance(loader, tuple):
        loader = loader[0]
    # The training IterableDataset is infinite by design. Analysis needs one pass.
    loader.dataset.is_for_training = False
    return loader


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


@dataclass
class HistoryView:
    target_itemkey: torch.Tensor
    history_itemkey: torch.Tensor
    valid_mask: torch.Tensor
    repeat_mask: torch.Tensor
    occurrence_count: torch.Tensor
    recent_distance: torch.Tensor
    history_length: torch.Tensor


def itemkeys_from_sid(sid: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
    return (sid.long() * powers.long()).sum(dim=-1)


def build_history_views(
    batch: Any,
    num_hierarchies: int,
    num_embeddings_per_hierarchy: int,
    masking_token: int,
) -> Dict[str, HistoryView]:
    sid = batch.transformed_sequences["sequence_data"].long().cpu()
    attention_mask = batch.mask.long().cpu()
    hist_itemkey_with_target = batch.transformed_sequences["hist_itemkey"].long().cpu()
    powers = num_embeddings_per_hierarchy ** torch.arange(
        num_hierarchies - 1, -1, -1, dtype=torch.long
    )

    lengths = attention_mask.sum(dim=1)
    if not torch.all(lengths >= num_hierarchies):
        raise AssertionError("Every sample must contain at least one target SID block")
    if not torch.all(lengths % num_hierarchies == 0):
        raise AssertionError("Valid SID lengths must be multiples of num_hierarchies")

    target_sid = sid[:, -num_hierarchies:]
    target_itemkey = itemkeys_from_sid(target_sid, powers)
    target_from_full = hist_itemkey_with_target[:, -1]
    if not torch.equal(target_itemkey, target_from_full):
        mismatch = int((target_itemkey != target_from_full).sum().item())
        raise AssertionError(f"Target SID/itemkey mismatch for {mismatch} samples")

    full_history = hist_itemkey_with_target[:, :-1]
    full_valid = full_history != masking_token

    ar_sid = sid[:, :-num_hierarchies].reshape(sid.size(0), -1, num_hierarchies)
    ar_mask_blocks = attention_mask[:, :-num_hierarchies].reshape(
        sid.size(0), -1, num_hierarchies
    )
    if not torch.all(ar_mask_blocks.eq(ar_mask_blocks[:, :, :1])):
        raise AssertionError("AR-visible padding splits a SID block")
    ar_valid = ar_mask_blocks[:, :, 0].bool()
    ar_history = itemkeys_from_sid(ar_sid, powers)

    for row in range(sid.size(0)):
        ar_values = ar_history[row][ar_valid[row]]
        full_values = full_history[row][full_valid[row]]
        if ar_values.numel() > full_values.numel():
            raise AssertionError("AR-visible history is longer than full retrieval history")
        if ar_values.numel() and not torch.equal(ar_values, full_values[-ar_values.numel() :]):
            raise AssertionError("AR-visible history does not match the full-history suffix")

    return {
        "ar_history": make_history_view(target_itemkey, ar_history, ar_valid),
        "full_history": make_history_view(target_itemkey, full_history, full_valid),
    }


def make_history_view(
    target_itemkey: torch.Tensor,
    history_itemkey: torch.Tensor,
    valid_mask: torch.Tensor,
) -> HistoryView:
    matches = history_itemkey.eq(target_itemkey.unsqueeze(1)) & valid_mask
    occurrence_count = matches.sum(dim=1)
    repeat_mask = occurrence_count > 0
    width = history_itemkey.size(1)
    positions = torch.arange(width, dtype=torch.long).unsqueeze(0).expand_as(history_itemkey)
    last_position = positions.masked_fill(~matches, -1).max(dim=1).values
    recent_distance = torch.where(repeat_mask, width - last_position, torch.zeros_like(last_position))
    return HistoryView(
        target_itemkey=target_itemkey,
        history_itemkey=history_itemkey,
        valid_mask=valid_mask,
        repeat_mask=repeat_mask,
        occurrence_count=occurrence_count,
        recent_distance=recent_distance,
        history_length=valid_mask.sum(dim=1),
    )


def _empty_group() -> Dict[str, int]:
    return {
        "num_samples": 0,
        "history_length_sum": 0,
        "gt_occurrence_sum": 0,
        "recent_distance_sum": 0,
        "recent_distance_count": 0,
    }


class DatasetAccumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.gt_repeat_count = 0
        self.history_item_count = 0
        self.history_slot_count = 0
        self.padding_filtered_count = 0
        self.gt_occurrence_sum = 0
        self.recent_distance_sum = 0
        self.recent_distance_count = 0
        self.occurrence_histogram: Dict[str, int] = {}
        self.recent_distance_histogram: Dict[str, int] = {}
        self.buckets = {label: _empty_group() for label, _, _ in LENGTH_BUCKETS}
        self.cohorts = {"gt_repeated": _empty_group(), "gt_novel": _empty_group()}

    @staticmethod
    def _add_histogram(target: Dict[str, int], values: torch.Tensor) -> None:
        unique, counts = torch.unique(values, return_counts=True)
        for value, count in zip(unique.tolist(), counts.tolist()):
            key = str(int(value))
            target[key] = target.get(key, 0) + int(count)

    @staticmethod
    def _update_group(group: Dict[str, int], view: HistoryView, mask: torch.Tensor) -> None:
        group["num_samples"] += int(mask.sum().item())
        group["history_length_sum"] += int(view.history_length[mask].sum().item())
        group["gt_occurrence_sum"] += int(view.occurrence_count[mask].sum().item())
        repeated = mask & view.repeat_mask
        group["recent_distance_sum"] += int(view.recent_distance[repeated].sum().item())
        group["recent_distance_count"] += int(repeated.sum().item())

    def update(self, view: HistoryView) -> None:
        batch_size = view.target_itemkey.numel()
        self.num_samples += batch_size
        self.gt_repeat_count += int(view.repeat_mask.sum().item())
        self.history_item_count += int(view.valid_mask.sum().item())
        self.history_slot_count += view.valid_mask.numel()
        self.padding_filtered_count += int((~view.valid_mask).sum().item())
        self.gt_occurrence_sum += int(view.occurrence_count.sum().item())
        self.recent_distance_sum += int(view.recent_distance[view.repeat_mask].sum().item())
        self.recent_distance_count += int(view.repeat_mask.sum().item())
        self._add_histogram(self.occurrence_histogram, view.occurrence_count)
        self._add_histogram(
            self.recent_distance_histogram,
            view.recent_distance[view.repeat_mask],
        )

        for label, lower, upper in LENGTH_BUCKETS:
            bucket_mask = view.history_length >= lower
            if upper is not None:
                bucket_mask &= view.history_length <= upper
            self._update_group(self.buckets[label], view, bucket_mask)
        self._update_group(self.cohorts["gt_repeated"], view, view.repeat_mask)
        self._update_group(self.cohorts["gt_novel"], view, ~view.repeat_mask)

    def summary(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "gt_repeat_count": self.gt_repeat_count,
            "gt_repeat_rate": safe_div(self.gt_repeat_count, self.num_samples),
            "history_item_count": self.history_item_count,
            "history_slot_count": self.history_slot_count,
            "padding_filtered_count": self.padding_filtered_count,
            "mean_history_length": safe_div(self.history_item_count, self.num_samples),
            "gt_occurrence_sum": self.gt_occurrence_sum,
            "mean_gt_occurrence_count": safe_div(self.gt_occurrence_sum, self.num_samples),
            "mean_gt_occurrence_count_if_repeated": safe_div(
                self.gt_occurrence_sum, self.gt_repeat_count
            ),
            "mean_recent_distance_if_repeated": safe_div(
                self.recent_distance_sum, self.recent_distance_count
            ),
            "gt_occurrence_count_distribution": self.occurrence_histogram,
            "gt_recent_distance_distribution": self.recent_distance_histogram,
        }


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def summarize_group(group: Mapping[str, int]) -> Dict[str, Any]:
    samples = group["num_samples"]
    return {
        **group,
        "mean_history_length": safe_div(group["history_length_sum"], samples),
        "mean_gt_occurrence_count": safe_div(group["gt_occurrence_sum"], samples),
        "mean_recent_distance_if_repeated": safe_div(
            group["recent_distance_sum"], group["recent_distance_count"]
        ),
    }


class PredictionAccumulator:
    def __init__(self, ks: Sequence[int]) -> None:
        self.ks = tuple(ks)
        self.num_samples = 0
        self.actual_candidate_count: Optional[int] = None
        self.variants = {name: self._new_variant() for name in ("beam", "rerank")}
        self.cohorts = {
            cohort: {name: self._new_cohort_variant() for name in ("beam", "rerank")}
            for cohort in ("gt_repeated", "gt_novel")
        }
        self.cohort_samples = {"gt_repeated": 0, "gt_novel": 0}

    def _new_variant(self) -> Dict[str, Any]:
        return {
            "top1_repeat_count": 0,
            "candidate_repeat_count": {k: 0 for k in self.ks},
            "candidate_denominator": {k: 0 for k in self.ks},
            "any_repeat_count": {k: 0 for k in self.ks},
            "effective_k": {k: None for k in self.ks},
        }

    def _new_cohort_variant(self) -> Dict[str, Any]:
        return {
            "top1_repeat_count": 0,
            "recall_sum": {k: 0.0 for k in self.ks},
            "ndcg_sum": {k: 0.0 for k in self.ks},
            "effective_k": {k: None for k in self.ks},
        }

    def update(
        self,
        view: HistoryView,
        beam_itemkey: torch.Tensor,
        rerank_itemkey: torch.Tensor,
        beam_target_matches: Optional[torch.Tensor] = None,
        rerank_target_matches: Optional[torch.Tensor] = None,
    ) -> None:
        if beam_itemkey.shape != rerank_itemkey.shape:
            raise AssertionError("Beam and rerank candidate shapes differ")
        if not torch.equal(torch.sort(beam_itemkey, dim=1).values, torch.sort(rerank_itemkey, dim=1).values):
            raise AssertionError("Rerank changed the complete beam candidate multiset")

        candidate_count = beam_itemkey.size(1)
        if self.actual_candidate_count not in (None, candidate_count):
            raise AssertionError("Candidate count changed between batches")
        self.actual_candidate_count = candidate_count
        self.num_samples += beam_itemkey.size(0)
        if beam_target_matches is None:
            beam_target_matches = beam_itemkey.eq(view.target_itemkey.unsqueeze(1))
        if rerank_target_matches is None:
            rerank_target_matches = rerank_itemkey.eq(view.target_itemkey.unsqueeze(1))
        if beam_target_matches.shape != beam_itemkey.shape:
            raise AssertionError("Beam target-match shape differs from beam candidates")
        if rerank_target_matches.shape != rerank_itemkey.shape:
            raise AssertionError("Rerank target-match shape differs from rerank candidates")

        repeat_by_variant = {}
        for name, candidates, target_matches in (
            ("beam", beam_itemkey, beam_target_matches),
            ("rerank", rerank_itemkey, rerank_target_matches),
        ):
            repeats = (
                candidates.unsqueeze(2).eq(view.history_itemkey.unsqueeze(1))
                & view.valid_mask.unsqueeze(1)
            ).any(dim=2)
            repeat_by_variant[name] = repeats
            variant = self.variants[name]
            variant["top1_repeat_count"] += int(repeats[:, 0].sum().item())

            for requested_k in self.ks:
                k = min(requested_k, candidate_count)
                variant["effective_k"][requested_k] = k
                variant["candidate_repeat_count"][requested_k] += int(repeats[:, :k].sum().item())
                variant["candidate_denominator"][requested_k] += repeats.size(0) * k
                variant["any_repeat_count"][requested_k] += int(repeats[:, :k].any(dim=1).sum().item())

            for cohort, cohort_mask in (
                ("gt_repeated", view.repeat_mask),
                ("gt_novel", ~view.repeat_mask),
            ):
                cohort_variant = self.cohorts[cohort][name]
                cohort_variant["top1_repeat_count"] += int(
                    (repeats[:, 0] & cohort_mask).sum().item()
                )
                for requested_k in self.ks:
                    k = min(requested_k, candidate_count)
                    cohort_variant["effective_k"][requested_k] = k
                    matches = target_matches[:, :k]
                    hit = matches.any(dim=1)
                    first_rank = matches.float().argmax(dim=1) + 1
                    ndcg = torch.where(
                        hit,
                        1.0 / torch.log2(first_rank.float() + 1.0),
                        torch.zeros_like(first_rank, dtype=torch.float32),
                    )
                    cohort_variant["recall_sum"][requested_k] += float(hit[cohort_mask].sum().item())
                    cohort_variant["ndcg_sum"][requested_k] += float(ndcg[cohort_mask].sum().item())

        if int(repeat_by_variant["beam"].sum()) != int(repeat_by_variant["rerank"].sum()):
            raise AssertionError("Full-list candidate repeat count changed after reranking")
        if not torch.equal(
            repeat_by_variant["beam"].any(dim=1),
            repeat_by_variant["rerank"].any(dim=1),
        ):
            raise AssertionError("Full-list any-repeat changed after reranking")

        repeated_count = int(view.repeat_mask.sum().item())
        self.cohort_samples["gt_repeated"] += repeated_count
        self.cohort_samples["gt_novel"] += view.repeat_mask.numel() - repeated_count
        if sum(self.cohort_samples.values()) != self.num_samples:
            raise AssertionError("GT repeated and novel cohort counts do not sum to total")

    def summary(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {"num_samples": self.num_samples}
        for name, variant in self.variants.items():
            output[f"{name}_top1_repeat_count"] = variant["top1_repeat_count"]
            output[f"{name}_top1_repeat_rate"] = safe_div(
                variant["top1_repeat_count"], self.num_samples
            )
            for k in self.ks:
                output[f"{name}_effective_k_at_{k}"] = variant["effective_k"][k]
                output[f"{name}_candidate_repeat_count_at_{k}"] = variant[
                    "candidate_repeat_count"
                ][k]
                output[f"{name}_candidate_denominator_at_{k}"] = variant[
                    "candidate_denominator"
                ][k]
                output[f"{name}_candidate_repeat_rate_at_{k}"] = safe_div(
                    variant["candidate_repeat_count"][k],
                    variant["candidate_denominator"][k],
                )
                output[f"{name}_any_repeat_count_at_{k}"] = variant["any_repeat_count"][k]
                output[f"{name}_any_repeat_rate_at_{k}"] = safe_div(
                    variant["any_repeat_count"][k], self.num_samples
                )
        output["metrics_by_gt_recurrence"] = self.cohort_summary()
        return output

    def cohort_summary(self) -> Dict[str, Any]:
        result = {}
        for cohort, samples in self.cohort_samples.items():
            cohort_result: Dict[str, Any] = {"num_samples": samples}
            for name in ("beam", "rerank"):
                values = self.cohorts[cohort][name]
                cohort_result[f"{name}_top1_repeat_count"] = values["top1_repeat_count"]
                cohort_result[f"{name}_top1_repeat_rate"] = safe_div(
                    values["top1_repeat_count"], samples
                )
                for k in self.ks:
                    cohort_result[f"{name}_recall_at_{k}"] = safe_div(
                        values["recall_sum"][k], samples
                    )
                    cohort_result[f"{name}_ndcg_at_{k}"] = safe_div(
                        values["ndcg_sum"][k], samples
                    )
            for k in self.ks:
                for metric in ("recall", "ndcg"):
                    beam_value = cohort_result[f"beam_{metric}_at_{k}"]
                    rerank_value = cohort_result[f"rerank_{metric}_at_{k}"]
                    cohort_result[f"{metric}_gain_at_{k}"] = (
                        rerank_value - beam_value
                        if beam_value is not None and rerank_value is not None
                        else None
                    )
            result[cohort] = cohort_result
        return result


def resolve_checkpoint(run_dir: Path, checkpoint: Optional[Path]) -> Path:
    if checkpoint is None:
        raise ValueError("An explicit checkpoint is required for prediction analysis")
    path = checkpoint.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint_root = (run_dir / "checkpoints").resolve()
    if not path.is_relative_to(checkpoint_root):
        raise ValueError(f"Checkpoint {path} is not under {checkpoint_root}")
    return path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(cfg: Any, expected_data_root: Path) -> Dict[str, Any]:
    """Lock exploratory statistics to the audited publication dataset."""
    data_root = Path(str(cfg.data_dir)).expanduser().resolve()
    expected_data_root = expected_data_root.expanduser().resolve()
    if data_root != expected_data_root:
        raise ValueError(f"Run data root {data_root} does not match {expected_data_root}")
    paths = {
        "train": data_root / "protocol_manifest_train.json",
        "test": data_root / "protocol_manifest_test.json",
        "audit": data_root / "protocol_audit.json",
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    if {payload.get("protocol") for payload in payloads.values()} != {EXPECTED_PROTOCOL}:
        raise ValueError("Dataset protocol is not history-last-user-v1")
    if payloads["audit"].get("status") != "passed":
        raise ValueError("Dataset audit has not passed")
    actual_counts = {
        "train": int(payloads["audit"]["destinations"]["train"]["records"]),
        "val": int(payloads["audit"]["destinations"]["evaluation"]["records"]),
        "test": int(payloads["audit"]["destinations"]["test"]["records"]),
    }
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected audited split counts: {actual_counts}")
    return {
        "data_root": str(data_root),
        "protocol": EXPECTED_PROTOCOL,
        "split_counts": actual_counts,
        "manifests": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }


def instantiate_model(cfg: Any, checkpoint: Path, device: torch.device) -> Any:
    import hydra

    cfg.model.load_only_weights = False
    model = hydra.utils.instantiate(cfg.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[prediction] checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval().to(device)
    return model


@contextmanager
def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        yield
        return
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield


def infer_batch(model: Any, batch: Any, device: torch.device, amp: str) -> Dict[str, Any]:
    sid = batch.transformed_sequences["sequence_data"].long().to(device, non_blocking=True)
    mask = batch.mask.long().to(device, non_blocking=True)
    hist_itemkey = batch.transformed_sequences["hist_itemkey"].long().to(
        device, non_blocking=True
    )
    target_sid = sid[:, -model.num_hierarchies :].contiguous()
    with torch.inference_mode(), autocast_context(device, amp):
        result, _ = model._infer_for_eval(
            hist_sid=sid[:, : -model.num_hierarchies].contiguous(),
            hist_mask=mask[:, : -model.num_hierarchies].contiguous(),
            hist_itemkey=hist_itemkey[:, :-1].contiguous(),
            target_sid=target_sid,
        )
    return result


def prediction_values(
    result: Mapping[str, Any],
    powers: torch.Tensor,
    target_sid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    beam_sid = result["beam"]["ids"].detach().cpu()
    rerank_sid = result["rerank"]["ids"].detach().cpu()
    beam_itemkey = itemkeys_from_sid(beam_sid, powers)
    rerank_itemkey = itemkeys_from_sid(rerank_sid, powers)
    beam_target_matches = beam_sid.eq(target_sid.unsqueeze(1)).all(dim=2)
    rerank_target_matches = rerank_sid.eq(target_sid.unsqueeze(1)).all(dim=2)
    return beam_itemkey, rerank_itemkey, beam_target_matches, rerank_target_matches


def semantic_id_path_from_config(cfg: Any, override: Optional[Path]) -> Path:
    if override is not None:
        return override.resolve()
    configured = cfg.semantic_id_path
    return (ROOT / str(configured)).resolve() if not Path(str(configured)).is_absolute() else Path(str(configured))


def collision_audit(
    semantic_id_path: Path,
    num_hierarchies: int,
    num_embeddings_per_hierarchy: int,
    cache_path: Optional[Path],
) -> Dict[str, Any]:
    if cache_path is not None and cache_path.exists():
        with cache_path.open() as handle:
            cached = json.load(handle)
        expected_config = {
            "semantic_id_path": str(semantic_id_path.resolve()),
            "num_hierarchies": num_hierarchies,
            "num_embeddings_per_hierarchy": num_embeddings_per_hierarchy,
            "semantic_id_offset": 1,
        }
        mismatches = {
            key: (cached.get(key), value)
            for key, value in expected_config.items()
            if cached.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Collision cache configuration mismatch: {mismatches}"
            )
        return cached

    mapping = np.load(semantic_id_path, mmap_mode="r", allow_pickle=False)
    if mapping.ndim != 2 or mapping.shape[1] != num_hierarchies:
        raise ValueError(f"Unexpected semantic ID mapping shape: {mapping.shape}")
    powers = num_embeddings_per_hierarchy ** np.arange(
        num_hierarchies - 1, -1, -1, dtype=np.int64
    )
    itemkeys = np.empty(mapping.shape[0], dtype=np.int64)
    chunk_size = 1_000_000
    for start in range(0, mapping.shape[0], chunk_size):
        end = min(start + chunk_size, mapping.shape[0])
        semantic_ids = np.asarray(mapping[start:end], dtype=np.int64) + 1
        itemkeys[start:end] = semantic_ids @ powers
    _, counts = np.unique(itemkeys, return_counts=True)
    num_unique = int(counts.size)
    audit = {
        "identity_definition": "semantic_itemkey",
        "semantic_id_path": str(semantic_id_path.resolve()),
        "num_hierarchies": num_hierarchies,
        "num_embeddings_per_hierarchy": num_embeddings_per_hierarchy,
        "semantic_id_offset": 1,
        "num_raw_items": int(mapping.shape[0]),
        "num_unique_itemkeys": num_unique,
        "collision_count": int(mapping.shape[0] - num_unique),
        "num_colliding_itemkeys": int((counts > 1).sum()),
        "max_items_per_itemkey": int(counts.max()) if counts.size else 0,
        "is_one_to_one": bool(np.all(counts == 1)),
    }
    if cache_path is not None:
        write_json(cache_path, audit)
    return audit


def git_commit() -> Optional[str]:
    """Return the current commit when git is available on the compute node."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        # Slurm compute images may intentionally omit git. This is diagnostic
        # metadata and must never discard an otherwise completed analysis.
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def process_split(
    split: str,
    loader: Iterable[Any],
    args: argparse.Namespace,
    num_hierarchies: int,
    num_embeddings_per_hierarchy: int,
    masking_token: int,
    model: Optional[Any],
    device: Optional[torch.device],
) -> Tuple[Dict[str, DatasetAccumulator], Optional[Dict[str, PredictionAccumulator]], int]:
    dataset_stats = {scope: DatasetAccumulator() for scope in ("ar_history", "full_history")}
    prediction_stats = (
        {scope: PredictionAccumulator(args.ks) for scope in ("ar_history", "full_history")}
        if model is not None
        else None
    )
    processed = 0
    powers = num_embeddings_per_hierarchy ** torch.arange(
        num_hierarchies - 1, -1, -1, dtype=torch.long
    )

    for batch_idx, wrapped_batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        batch = unwrap_batch(wrapped_batch)
        if args.max_samples is not None:
            remaining = args.max_samples - processed
            if remaining <= 0:
                break
            if batch.transformed_sequences["sequence_data"].size(0) > remaining:
                batch = slice_batch(batch, remaining)

        views = build_history_views(
            batch=batch,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            masking_token=masking_token,
        )
        for scope, view in views.items():
            dataset_stats[scope].update(view)

        if model is not None:
            assert prediction_stats is not None and device is not None
            result = infer_batch(model, batch, device, args.amp)
            target_sid = batch.transformed_sequences["sequence_data"][
                :, -num_hierarchies:
            ].long().cpu()
            prediction_values_for_batch = prediction_values(result, powers, target_sid)
            for scope, view in views.items():
                prediction_stats[scope].update(view, *prediction_values_for_batch)

        processed += next(iter(views.values())).target_itemkey.numel()
        if args.progress_every and (batch_idx + 1) % args.progress_every == 0:
            print(f"[{args.mode}] split={split} batches={batch_idx + 1} samples={processed}")
        if args.max_samples is not None and processed >= args.max_samples:
            break
    return dataset_stats, prediction_stats, processed


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.touch()
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_rows(
    all_dataset_stats: Mapping[str, Mapping[str, DatasetAccumulator]],
    all_prediction_stats: Mapping[str, Optional[Mapping[str, PredictionAccumulator]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    length_rows: List[Dict[str, Any]] = []
    cohort_rows: List[Dict[str, Any]] = []
    for split, scopes in all_dataset_stats.items():
        for scope, stats in scopes.items():
            for bucket, group in stats.buckets.items():
                summarized = summarize_group(group)
                length_rows.append(
                    {
                        "split": split,
                        "history_scope": scope,
                        "history_length_bucket": bucket,
                        **summarized,
                        "gt_repeat_rate": safe_div(
                            group["recent_distance_count"], group["num_samples"]
                        ),
                    }
                )
            prediction = all_prediction_stats.get(split)
            prediction_cohorts = prediction[scope].cohort_summary() if prediction else None
            for cohort, group in stats.cohorts.items():
                row = {
                    "split": split,
                    "history_scope": scope,
                    "cohort": cohort,
                    **summarize_group(group),
                }
                if prediction_cohorts:
                    row.update(prediction_cohorts[cohort])
                cohort_rows.append(row)
    return length_rows, cohort_rows


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg = load_run_config(run_dir)
    dataset_provenance = validate_dataset(cfg, args.expected_data_root)
    if args.mode == "prediction":
        from scripts.export_paired_evaluation import validate_provenance

        # Prediction analysis currently supports val/test only; validate every
        # requested destination against the exact training preflight.
        for split in args.splits:
            validate_provenance(
                cfg=cfg,
                split=split,
                expected_data_root=args.expected_data_root,
                expected_protocol=EXPECTED_PROTOCOL,
                training_preflight=args.training_preflight,
            )
    configure_dataloaders(cfg, args)
    datamodule = instantiate_datamodule(cfg)

    num_hierarchies = int(cfg.num_hierarchies)
    num_embeddings = int(cfg.num_embeddings_per_hierarchy)
    masking_token = int(cfg.data_loading.test_dataloader_config.dataloader.masking_token)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (ROOT / "logs/analysis/history_occurrence" / args.mode).resolve()
    )
    output_paths = [
        output_dir / "summary.json",
        output_dir / "by_history_length.csv",
        output_dir / "by_gt_recurrence.csv",
    ]
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite analysis outputs: {existing_outputs}")

    semantic_path = semantic_id_path_from_config(cfg, args.semantic_id_path)
    audit = None
    if not args.skip_collision_audit:
        print(f"[audit] semantic-ID mapping: {semantic_path}")
        audit = collision_audit(
            semantic_id_path=semantic_path,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings,
            cache_path=args.collision_audit_cache.resolve() if args.collision_audit_cache else None,
        )

    checkpoint = None
    model = None
    device = None
    if args.mode == "prediction":
        checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        model = instantiate_model(cfg, checkpoint, device)

    all_dataset_stats: Dict[str, Dict[str, DatasetAccumulator]] = {}
    all_prediction_stats: Dict[str, Optional[Dict[str, PredictionAccumulator]]] = {}
    sample_counts: Dict[str, int] = {}
    for split in args.splits:
        print(f"[{args.mode}] starting split={split}")
        dataset_stats, prediction_stats, processed = process_split(
            split=split,
            loader=get_split_loader(datamodule, split),
            args=args,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings,
            masking_token=masking_token,
            model=model,
            device=device,
        )
        all_dataset_stats[split] = dataset_stats
        all_prediction_stats[split] = prediction_stats
        sample_counts[split] = processed
        if (
            args.limit_files is None
            and args.max_batches is None
            and args.max_samples is None
            and processed != EXPECTED_SPLIT_COUNTS[split]
        ):
            raise RuntimeError(
                f"Full {split} count mismatch: expected {EXPECTED_SPLIT_COUNTS[split]}, "
                f"got {processed}"
            )
        print(f"[{args.mode}] finished split={split} samples={processed}")

    summary: Dict[str, Any] = {
        "mode": args.mode,
        "splits": args.splits,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "run_dir": str(run_dir),
        "dataset_provenance": dataset_provenance,
        "training_preflight": (
            {
                "path": str(args.training_preflight.resolve()),
                "sha256": file_sha256(args.training_preflight.resolve()),
            }
            if args.training_preflight is not None
            else None
        ),
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "git_commit": git_commit(),
        "identity_definition": "semantic_itemkey",
        "recommendation_metric_identity": "exact_semantic_id_vector",
        "num_hierarchies": num_hierarchies,
        "num_embeddings_per_hierarchy": num_embeddings,
        "requested_k": args.ks,
        "sample_counts": sample_counts,
        "semantic_itemkey_collision_audit": audit,
        "results": {},
        "limits": {
            "limit_files": args.limit_files,
            "max_batches": args.max_batches,
            "max_samples": args.max_samples,
        },
    }
    for split, scopes in all_dataset_stats.items():
        summary["results"][split] = {}
        for scope, dataset_stats in scopes.items():
            scope_summary = dataset_stats.summary()
            prediction = all_prediction_stats[split]
            if prediction is not None:
                scope_summary.update(prediction[scope].summary())
            summary["results"][split][scope] = scope_summary

    length_rows, cohort_rows = output_rows(all_dataset_stats, all_prediction_stats)
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "by_history_length.csv", length_rows)
    write_csv(output_dir / "by_gt_recurrence.csv", cohort_rows)
    print(f"[done] outputs written to {output_dir}")


if __name__ == "__main__":
    main()
