#!/usr/bin/env python
"""Profile RecoChain and separate-reranker latency.

This script is intentionally separate from Lightning validation/test loops. It
measures model forward time only, using fixed batches and explicit CUDA events.
For RecoChain, the main cache ablation keeps the generated beam candidates
fixed and compares:

- beam_only: semantic-ID beam generation only
- rerank_cache: beam generation + RecoChain reranking with beam KV cache reuse
- rerank_no_cache: same beam candidates, but reranking recomputes the full
  history/candidate prefix without using the beam KV cache

For the separate small reranker, the measured path is:

- beam_only: frozen SID teacher beam generation only
- separate_rerank: frozen SID teacher beam generation + standalone reranker
  scoring over the same candidate list
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import statistics
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import rootutils
import torch
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.chdir(ROOT)

OmegaConf.register_new_resolver(
    "sub",
    lambda x, y: int(x) - int(y),
    replace=True,
)


@dataclass
class ProfileRecord:
    variant: str
    batch_idx: int
    batch_size: int
    top_k: int
    gen_ms: float
    rerank_ms: float
    total_ms: float


@dataclass
class ProfileSummary:
    variant: str
    batches: int
    users: int
    candidates: int
    batch_size: int
    top_k: int
    gen_ms_mean: float
    gen_ms_p50: float
    gen_ms_p95: float
    rerank_ms_mean: float
    rerank_ms_p50: float
    rerank_ms_p95: float
    total_ms_mean: float
    total_ms_p50: float
    total_ms_p95: float
    users_per_sec: float
    candidates_per_sec: float
    peak_mem_gb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-mode",
        choices=["recochain", "separate_rerank"],
        default="recochain",
        help="Which inference path to profile.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Training run directory containing .hydra/config.yaml. For "
            "separate_rerank, this is typically the SID teacher run dir used "
            "for data/model base settings."
        ),
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint path. Defaults to the first checkpoint in run-dir/checkpoints.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to logs/profiles/<run-name>_profile.json.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional per-batch CSV output path.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["beam_only", "rerank_cache", "rerank_no_cache"],
        choices=["beam_only", "rerank_cache", "rerank_no_cache", "separate_rerank"],
    )
    parser.add_argument(
        "--separate-rerank-config",
        default="configs/experiment/tiger_train_rerank_small.yaml",
        help="Hydra override yaml that defines SeparateRerankModule settings.",
    )
    parser.add_argument(
        "--sid-teacher-ckpt",
        default=None,
        help=(
            "SID teacher checkpoint for separate_rerank mode. Defaults to "
            "--ckpt if provided, otherwise first checkpoint in --run-dir."
        ),
    )
    parser.add_argument(
        "--reranker-ckpt",
        default=None,
        help=(
            "Optional trained SeparateRerankModule checkpoint. If omitted, "
            "only the configured architecture is instantiated; this is enough "
            "for latency and parameter profiling."
        ),
    )
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measure-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--top-k-for-generation",
        type=int,
        default=None,
        help=(
            "Optional evaluation/profiling-time override for model.top_k_for_generation. "
            "Use mainly with recochain/SID checkpoints; trained separate rerankers may "
            "have rank-embedding shapes tied to their original beam size."
        ),
    )
    parser.add_argument(
        "--top-k-for-score",
        type=int,
        default=None,
        help=(
            "Optional evaluation/profiling-time override for model.top_k_for_score. "
            "This controls how many similar historical items are retrieved for "
            "candidate-aware reranking context."
        ),
    )
    parser.add_argument(
        "--active-top-k-for-score",
        type=int,
        default=None,
        help=(
            "Optional mask-based active context length. Keep model.top_k_for_score "
            "as the fixed maximum number of historical-context slots, but mask "
            "retrieved items after this active length so the candidate token "
            "position stays fixed."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split used for profiling batches.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--amp",
        choices=["bf16", "fp16", "none"],
        default="bf16",
        help="Autocast mode for CUDA profiling.",
    )
    return parser.parse_args()


def resolve_checkpoint(run_dir: Path, ckpt_arg: Optional[str]) -> Path:
    if ckpt_arg:
        return Path(ckpt_arg).expanduser().resolve()

    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")
    return ckpts[0].resolve()


def load_config(run_dir: Path, args: argparse.Namespace, sid_teacher_ckpt: Optional[Path] = None):
    importlib.import_module("src.utils.custom_hydra_resolvers")

    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")
    cfg = OmegaConf.load(config_path)
    if args.profile_mode == "separate_rerank":
        rerank_config_path = Path(args.separate_rerank_config).expanduser()
        if not rerank_config_path.exists():
            raise FileNotFoundError(f"Missing rerank config: {rerank_config_path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(rerank_config_path))
        if sid_teacher_ckpt is None:
            raise ValueError("--sid-teacher-ckpt is required for separate_rerank mode")
        cfg.model.sid_teacher_ckpt_path = str(sid_teacher_ckpt)

    cfg.model.load_only_weights = False
    cfg.trainer.devices = 1
    cfg.trainer.strategy = "auto"
    return cfg


def configure_dataloader(cfg, args: argparse.Namespace) -> None:
    split_key = "test_dataloader_config" if args.split == "test" else "val_dataloader_config"
    dataloader_cfg = cfg.data_loading[split_key].dataloader
    dataloader_cfg.batch_size_per_device = args.batch_size
    dataloader_cfg.num_workers = args.num_workers
    dataloader_cfg.persistent_workers = args.num_workers > 0
    dataloader_cfg.drop_last = False
    dataloader_cfg.should_shuffle_rows = False


def instantiate_model(
    cfg,
    ckpt_path: Optional[Path],
    device: torch.device,
    profile_mode: str,
    reranker_ckpt_path: Optional[Path] = None,
):
    import hydra

    model = hydra.utils.instantiate(cfg.model)

    if profile_mode == "separate_rerank":
        if reranker_ckpt_path is not None:
            checkpoint = torch.load(
                reranker_ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[profile] Missing keys while loading reranker checkpoint: {len(missing)}")
            if unexpected:
                print(f"[profile] Unexpected keys while loading reranker checkpoint: {len(unexpected)}")
        else:
            print("[profile] No trained reranker checkpoint provided; profiling configured architecture.")
        model.setup(stage="validate")
    else:
        if ckpt_path is None:
            raise ValueError("RecoChain profiling requires --ckpt or a checkpoint in --run-dir.")
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[profile] Missing keys while loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"[profile] Unexpected keys while loading checkpoint: {len(unexpected)}")

    model.eval()
    model.to(device)
    return model


def instantiate_batches(cfg, args: argparse.Namespace, device: torch.device):
    import hydra

    datamodule = hydra.utils.instantiate(cfg.data_loading.datamodule)
    datamodule.trainer = SimpleNamespace(world_size=1, global_rank=0)
    datamodule.setup()
    loader = datamodule.test_dataloader() if args.split == "test" else datamodule.val_dataloader()
    if isinstance(loader, tuple):
        loader = loader[0]

    total_batches = args.warmup_batches + args.measure_batches
    batches = []
    iterator = iter(loader)
    for _ in range(total_batches):
        batch = next(iterator)
        batch = unwrap_batch(batch)
        batches.append(move_batch_to_device(batch, device))
    return batches


def unwrap_batch(batch):
    return batch[0] if isinstance(batch, tuple) else batch


def move_batch_to_device(
    batch: SequentialModelInputData,
    device: torch.device,
) -> SequentialModelInputData:
    from src.data.loading.components.interfaces import SequentialModelInputData

    transformed_sequences = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.transformed_sequences.items()
    }
    mask = batch.mask.to(device, non_blocking=True) if batch.mask is not None else None
    user_id_list = batch.user_id_list
    if isinstance(user_id_list, torch.Tensor):
        user_id_list = user_id_list.to(device, non_blocking=True)
    return SequentialModelInputData(
        user_id_list=user_id_list,
        transformed_sequences=transformed_sequences,
        mask=mask,
    )


def split_batch(model, batch: SequentialModelInputData) -> Dict[str, torch.Tensor]:
    sid = batch.transformed_sequences["sequence_data"].long()
    mask = batch.mask
    hist_itemkey = batch.transformed_sequences["hist_itemkey"].long()
    return {
        "target_sid": sid[:, -model.num_hierarchies:].contiguous(),
        "hist_sid": sid[:, :-model.num_hierarchies].contiguous(),
        "hist_mask": mask[:, :-model.num_hierarchies].contiguous(),
        "hist_itemkey": hist_itemkey[:, :-1].contiguous(),
    }


@contextmanager
def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        yield
        return
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, fn):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize(device)
        return result, float(start.elapsed_time(end))

    start_time = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return result, elapsed_ms


def run_beam(model, batch_dict: Dict[str, torch.Tensor], device: torch.device, amp: str):
    with autocast_context(device, amp):
        return model._beam_search_semantic_ids(
            sid=batch_dict["hist_sid"],
            attention_mask=batch_dict["hist_mask"],
        )


def rerank_with_cache(
    model,
    batch_dict: Dict[str, torch.Tensor],
    generated_ids: torch.Tensor,
    beam_log_prob: torch.Tensor,
    input_mask: torch.Tensor,
    past_key_values,
    device: torch.device,
    amp: str,
):
    with autocast_context(device, amp):
        return model._rerank_with_click_prediction(
            generated_ids=generated_ids,
            beam_log_prob=beam_log_prob,
            input_mask=input_mask,
            past_key_values=past_key_values,
            hist_itemkey=batch_dict["hist_itemkey"],
            batch_size=batch_dict["hist_sid"].size(0),
        )


def rerank_without_cache(
    model,
    batch_dict: Dict[str, torch.Tensor],
    generated_ids: torch.Tensor,
    beam_log_prob: torch.Tensor,
    device: torch.device,
    amp: str,
):
    batch_size = batch_dict["hist_sid"].size(0)
    top_k = model.top_k_for_generation
    row_count = generated_ids.size(0)

    hist_sid = batch_dict["hist_sid"].repeat_interleave(top_k, dim=0)
    hist_mask = batch_dict["hist_mask"].repeat_interleave(top_k, dim=0)
    hist_itemkey = batch_dict["hist_itemkey"].repeat_interleave(top_k, dim=0)

    with autocast_context(device, amp):
        history_embeds, history_mask = model.build_sid_block_embeds(
            sid=hist_sid,
            attention_mask=hist_mask,
            add_bos_token=True,
        )
        generated_itemkey = (generated_ids * model.powers).sum(dim=-1)
        similar_itemkey = model.get_similar_itemkey(
            target_itemkey=generated_itemkey,
            hist_itemkey=hist_itemkey,
        )
        similar_embeds, similar_mask = model.build_similar_item_embeds(
            similar_itemkey=similar_itemkey,
        )
        sid_embeds = [
            model.embed_token(
                token_id=generated_ids[:, h : h + 1],
                hierarchy=h,
            )
            for h in range(model.num_hierarchies)
        ]
        item_embeds = model.embed_token(token_id=generated_itemkey.unsqueeze(-1))
        input_embeds = torch.cat(
            [history_embeds, *sid_embeds, item_embeds, similar_embeds],
            dim=1,
        )
        generated_mask = torch.ones(
            row_count,
            model.num_hierarchies + 1,
            device=device,
            dtype=history_mask.dtype,
        )
        full_mask = torch.cat([history_mask, generated_mask, similar_mask], dim=-1)

        decoder_output = model.decoder(
            sequence_embedding=input_embeds,
            attention_mask=full_mask,
            use_cache=False,
            past_key_values=None,
        )
        click_logits = model.decoder.click_head(decoder_output[:, -1, :]).squeeze(-1)
        click_logits_2d = click_logits.view(batch_size, top_k)
        final_score = model._combine_rerank_scores(
            beam_log_prob=beam_log_prob,
            rerank_logits=click_logits_2d,
        )
        topk_results = torch.topk(final_score, k=top_k, dim=-1)
        rerank_scores = topk_results.values
        indices_topk = topk_results.indices
        rerank_ids = generated_ids.view(
            batch_size,
            top_k,
            model.num_hierarchies,
        )
        rerank_ids = torch.gather(
            rerank_ids,
            dim=1,
            index=indices_topk.unsqueeze(-1).expand(-1, -1, model.num_hierarchies),
        )
        rerank_click_logits = torch.gather(click_logits_2d, 1, indices_topk)
        return rerank_ids, rerank_scores, rerank_click_logits


def run_separate_beam(
    model,
    batch_dict: Dict[str, torch.Tensor],
    device: torch.device,
    amp: str,
):
    with autocast_context(device, amp):
        return model.teacher.generate_candidates(
            hist_sid=batch_dict["hist_sid"],
            hist_mask=batch_dict["hist_mask"],
            hist_itemkey=batch_dict["hist_itemkey"],
            ensure_gt_in_candidates=False,
        )


def run_separate_rerank(
    model,
    candidate_dict: Dict[str, torch.Tensor],
    batch_dict: Dict[str, torch.Tensor],
    device: torch.device,
    amp: str,
):
    with autocast_context(device, amp):
        rerank_logits = model.reranker(
            hist_itemkey=batch_dict["hist_itemkey"],
            candidate_itemkey=candidate_dict["candidate_itemkey"],
            candidate_sid=candidate_dict["candidate_sid"],
            beam_scores=candidate_dict["beam_scores"],
        )
        final_scores = model._combine_rerank_scores(
            beam_log_prob=candidate_dict["beam_scores"],
            rerank_logits=rerank_logits,
        )
        topk = torch.topk(final_scores, k=model.top_k_for_generation, dim=-1)
        gather_index = topk.indices.unsqueeze(-1).expand(
            -1,
            -1,
            model.num_hierarchies,
        )
        rerank_ids = torch.gather(candidate_dict["candidate_sid"], 1, gather_index)
        return rerank_ids, topk.values, rerank_logits


def run_recochain_profile_variant(
    model,
    batches: List[SequentialModelInputData],
    variant: str,
    device: torch.device,
    amp: str,
    warmup_batches: int,
) -> List[ProfileRecord]:
    records: List[ProfileRecord] = []
    top_k = model.top_k_for_generation

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(batches):
            batch_dict = split_batch(model, batch)
            batch_size = batch_dict["hist_sid"].size(0)

            synchronize(device)
            beam_output, gen_ms = timed_call(
                device,
                lambda: run_beam(model, batch_dict, device, amp),
            )
            generated_ids, beam_log_prob, input_mask, past_key_values = beam_output

            rerank_ms = 0.0
            if variant == "rerank_cache":
                _, rerank_ms = timed_call(
                    device,
                    lambda: rerank_with_cache(
                        model,
                        batch_dict,
                        generated_ids,
                        beam_log_prob,
                        input_mask,
                        past_key_values,
                        device,
                        amp,
                    ),
                )
            elif variant == "rerank_no_cache":
                _, rerank_ms = timed_call(
                    device,
                    lambda: rerank_without_cache(
                        model,
                        batch_dict,
                        generated_ids,
                        beam_log_prob,
                        device,
                        amp,
                    ),
                )
            elif variant != "beam_only":
                raise ValueError(f"Unknown variant: {variant}")

            if batch_idx >= warmup_batches:
                records.append(
                    ProfileRecord(
                        variant=variant,
                        batch_idx=batch_idx - warmup_batches,
                        batch_size=batch_size,
                        top_k=top_k,
                        gen_ms=gen_ms,
                        rerank_ms=rerank_ms,
                        total_ms=gen_ms + rerank_ms,
                    )
                )

    return records


def run_separate_profile_variant(
    model,
    batches: List[SequentialModelInputData],
    variant: str,
    device: torch.device,
    amp: str,
    warmup_batches: int,
) -> List[ProfileRecord]:
    records: List[ProfileRecord] = []
    top_k = model.top_k_for_generation

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(batches):
            batch_dict = split_batch(model, batch)
            batch_size = batch_dict["hist_sid"].size(0)

            synchronize(device)
            candidate_dict, gen_ms = timed_call(
                device,
                lambda: run_separate_beam(model, batch_dict, device, amp),
            )

            rerank_ms = 0.0
            if variant == "separate_rerank":
                _, rerank_ms = timed_call(
                    device,
                    lambda: run_separate_rerank(
                        model,
                        candidate_dict,
                        batch_dict,
                        device,
                        amp,
                    ),
                )
            elif variant != "beam_only":
                raise ValueError(f"Unknown separate-rerank variant: {variant}")

            if batch_idx >= warmup_batches:
                records.append(
                    ProfileRecord(
                        variant=variant,
                        batch_idx=batch_idx - warmup_batches,
                        batch_size=batch_size,
                        top_k=top_k,
                        gen_ms=gen_ms,
                        rerank_ms=rerank_ms,
                        total_ms=gen_ms + rerank_ms,
                    )
                )

    return records


def run_profile_variant(
    model,
    batches: List[SequentialModelInputData],
    variant: str,
    device: torch.device,
    amp: str,
    warmup_batches: int,
    profile_mode: str,
) -> List[ProfileRecord]:
    if profile_mode == "separate_rerank":
        return run_separate_profile_variant(
            model=model,
            batches=batches,
            variant=variant,
            device=device,
            amp=amp,
            warmup_batches=warmup_batches,
        )

    return run_recochain_profile_variant(
        model=model,
        batches=batches,
        variant=variant,
        device=device,
        amp=amp,
        warmup_batches=warmup_batches,
    )


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = int(round((pct / 100.0) * (len(values) - 1)))
    return values[index]


def summarize_records(
    records: List[ProfileRecord],
    peak_mem_gb: float,
) -> ProfileSummary:
    if not records:
        raise ValueError("No records to summarize")

    users = sum(record.batch_size for record in records)
    candidates = sum(record.batch_size * record.top_k for record in records)
    total_ms_values = [record.total_ms for record in records]
    gen_ms_values = [record.gen_ms for record in records]
    rerank_ms_values = [record.rerank_ms for record in records]
    total_seconds = sum(total_ms_values) / 1000.0

    return ProfileSummary(
        variant=records[0].variant,
        batches=len(records),
        users=users,
        candidates=candidates,
        batch_size=records[0].batch_size,
        top_k=records[0].top_k,
        gen_ms_mean=statistics.mean(gen_ms_values),
        gen_ms_p50=statistics.median(gen_ms_values),
        gen_ms_p95=percentile(gen_ms_values, 95),
        rerank_ms_mean=statistics.mean(rerank_ms_values),
        rerank_ms_p50=statistics.median(rerank_ms_values),
        rerank_ms_p95=percentile(rerank_ms_values, 95),
        total_ms_mean=statistics.mean(total_ms_values),
        total_ms_p50=statistics.median(total_ms_values),
        total_ms_p95=percentile(total_ms_values, 95),
        users_per_sec=users / total_seconds if total_seconds > 0 else 0.0,
        candidates_per_sec=candidates / total_seconds if total_seconds > 0 else 0.0,
        peak_mem_gb=peak_mem_gb,
    )


def write_csv(path: Path, records: Iterable[ProfileRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if (parameter.requires_grad or not trainable_only)
    )


def module_param_count(model: torch.nn.Module, path: str) -> Optional[int]:
    current = model
    for name in path.split("."):
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    if not isinstance(current, torch.nn.Module):
        return None
    return count_parameters(current)


def summarize_parameters(model: torch.nn.Module, profile_mode: str) -> Dict[str, Optional[int]]:
    raw_total_params = count_parameters(model)
    raw_trainable_params = count_parameters(model, trainable_only=True)
    summary: Dict[str, Optional[int]] = {
        "total_params": raw_total_params,
        "trainable_params": raw_trainable_params,
    }

    if profile_mode == "separate_rerank":
        teacher_params = module_param_count(model, "teacher.sid_model")
        reranker_params = module_param_count(model, "reranker")
        active_pipeline_params = (
            teacher_params + reranker_params
            if teacher_params is not None and reranker_params is not None
            else None
        )
        summary.update(
            {
                "raw_module_total_params": raw_total_params,
                "raw_module_trainable_params": raw_trainable_params,
                "total_params": active_pipeline_params,
                "trainable_params": reranker_params,
                "active_pipeline_params": active_pipeline_params,
                "active_trainable_params": reranker_params,
                "inactive_wrapper_params": (
                    raw_total_params - active_pipeline_params
                    if active_pipeline_params is not None
                    else None
                ),
                "teacher_sid_model_params": teacher_params,
                "reranker_params": reranker_params,
                "scorer_params": module_param_count(model, "reranker.scorer"),
            }
        )
    else:
        summary.update(
            {
                "decoder_params": module_param_count(model, "decoder"),
                "click_head_params": module_param_count(model, "decoder.click_head"),
                "sid_head_params": module_param_count(model, "decoder.sid_head"),
            }
        )

    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    ckpt_path = resolve_checkpoint(run_dir, args.ckpt)
    sid_teacher_ckpt = (
        Path(args.sid_teacher_ckpt).expanduser().resolve()
        if args.sid_teacher_ckpt
        else ckpt_path
    )
    reranker_ckpt_path = (
        Path(args.reranker_ckpt).expanduser().resolve()
        if args.reranker_ckpt
        else None
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    cfg = load_config(run_dir, args, sid_teacher_ckpt=sid_teacher_ckpt)
    if "active_top_k_for_score" not in cfg.model:
        cfg.model.active_top_k_for_score = None
    if args.top_k_for_generation is not None:
        if args.profile_mode == "separate_rerank" and reranker_ckpt_path is not None:
            print(
                "[profile] WARNING: overriding top_k_for_generation for a trained "
                "separate reranker can mismatch rank-embedding checkpoint shapes."
            )
        cfg.model.top_k_for_generation = args.top_k_for_generation
    if args.top_k_for_score is not None:
        cfg.model.top_k_for_score = args.top_k_for_score
    if args.active_top_k_for_score is not None:
        cfg.model.active_top_k_for_score = args.active_top_k_for_score
    configure_dataloader(cfg, args)

    print(f"[profile] Root: {ROOT}")
    print(f"[profile] Mode: {args.profile_mode}")
    print(f"[profile] Run dir: {run_dir}")
    print(f"[profile] Checkpoint: {ckpt_path}")
    if args.profile_mode == "separate_rerank":
        print(f"[profile] SID teacher checkpoint: {sid_teacher_ckpt}")
        print(f"[profile] Separate reranker checkpoint: {reranker_ckpt_path or 'None'}")
    print(f"[profile] Device: {device}")
    print(f"[profile] Variants: {args.variants}")
    print(f"[profile] top_k_for_generation: {cfg.model.top_k_for_generation}")
    print(f"[profile] top_k_for_score: {cfg.model.top_k_for_score}")
    print(f"[profile] active_top_k_for_score: {cfg.model.active_top_k_for_score}")

    model = instantiate_model(
        cfg=cfg,
        ckpt_path=ckpt_path,
        device=device,
        profile_mode=args.profile_mode,
        reranker_ckpt_path=reranker_ckpt_path,
    )
    parameter_summary = summarize_parameters(model, args.profile_mode)
    print(f"[profile] Parameter summary: {parameter_summary}")
    batches = instantiate_batches(cfg, args, device)

    all_records: List[ProfileRecord] = []
    summaries: List[ProfileSummary] = []

    for variant in args.variants:
        print(f"[profile] Running variant: {variant}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        records = run_profile_variant(
            model=model,
            batches=batches,
            variant=variant,
            device=device,
            amp=args.amp,
            warmup_batches=args.warmup_batches,
            profile_mode=args.profile_mode,
        )
        peak_mem_gb = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        summary = summarize_records(records, peak_mem_gb=peak_mem_gb)
        summaries.append(summary)
        all_records.extend(records)
        print(
            "[profile] "
            f"{variant}: total={summary.total_ms_mean:.3f} ms/batch, "
            f"gen={summary.gen_ms_mean:.3f}, "
            f"rerank={summary.rerank_ms_mean:.3f}, "
            f"users/sec={summary.users_per_sec:.2f}, "
            f"peak_mem={summary.peak_mem_gb:.2f} GB"
        )

    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        output_path = Path("logs/profiles") / f"{run_dir.parent.name}_{run_dir.name}_profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "profile_mode": args.profile_mode,
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "sid_teacher_ckpt": str(sid_teacher_ckpt) if args.profile_mode == "separate_rerank" else None,
        "reranker_ckpt": str(reranker_ckpt_path) if reranker_ckpt_path is not None else None,
        "device": str(device),
        "amp": args.amp,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "top_k_for_generation": int(cfg.model.top_k_for_generation),
        "top_k_for_score": int(cfg.model.top_k_for_score),
        "active_top_k_for_score": (
            int(cfg.model.active_top_k_for_score)
            if cfg.model.active_top_k_for_score is not None
            else None
        ),
        "parameter_summary": parameter_summary,
        "summaries": [asdict(summary) for summary in summaries],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[profile] Wrote summary: {output_path}")

    if args.csv_output:
        csv_path = Path(args.csv_output).expanduser()
        write_csv(csv_path, all_records)
        print(f"[profile] Wrote per-batch records: {csv_path}")


if __name__ == "__main__":
    main()
