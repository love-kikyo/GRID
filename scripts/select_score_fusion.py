#!/usr/bin/env python
"""Select a generator/reranker fusion weight on validation data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import h5py
import numpy as np

try:
    from scripts.analyze_paired_evaluation import metric_contributions, ranked_candidates
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from analyze_paired_evaluation import metric_contributions, ranked_candidates


DEFAULT_METRIC_ORDER = ("ndcg@10", "ndcg@5", "recall@10", "recall@5")


def selection_cpu_budget():
    """Reuse the single-task four-core guard; do not import training/ML code."""
    if os.environ.get("GRID_FUSION_CPU_BUDGET") != "4":
        return None
    from src.runtime_cpu import THREAD_VARIABLES
    from src.runtime_export_cpu import check_export_allocation

    # Libraries are imported above: limits must be set by the launcher, not here.
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise RuntimeError("Fusion library thread limits must be set before Python starts")
    return check_export_allocation()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Validation HDF5/NPZ export.")
    parser.add_argument("--beam-weight", type=float, default=1.0)
    parser.add_argument("--rerank-weight-start", type=Decimal, default=Decimal("0.00"))
    parser.add_argument("--rerank-weight-stop", type=Decimal, default=Decimal("0.50"))
    parser.add_argument("--rerank-weight-step", type=Decimal, default=Decimal("0.01"))
    parser.add_argument("--metric-order", nargs="+", default=list(DEFAULT_METRIC_ORDER))
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument(
        "--sample-modulus",
        type=int,
        default=1,
        help="Select rows whose first hash word modulo this value equals --sample-remainder.",
    )
    parser.add_argument("--sample-remainder", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    if args.rerank_weight_step <= 0:
        parser.error("--rerank-weight-step must be positive")
    if args.rerank_weight_stop < args.rerank_weight_start:
        parser.error("weight stop must be no smaller than weight start")
    if args.chunk_size <= 0 or args.sample_modulus <= 0:
        parser.error("chunk size and sample modulus must be positive")
    if not 0 <= args.sample_remainder < args.sample_modulus:
        parser.error("sample remainder must be in [0, sample modulus)")
    for metric in args.metric_order:
        name, separator, raw_k = metric.partition("@")
        if name not in {"recall", "ndcg"} or separator != "@" or not raw_k.isdigit():
            parser.error(f"invalid selection metric: {metric}")
    return args


def decimal_grid(start: Decimal, stop: Decimal, step: Decimal) -> List[Decimal]:
    count = int((stop - start) // step)
    values = [start + index * step for index in range(count + 1)]
    if not values or values[-1] != stop:
        values.append(stop)
    return values


def evaluate_variant(
    payload: Mapping[str, np.ndarray],
    variant: str,
    metric_order: Sequence[str],
) -> Dict[str, float]:
    ranked_items = ranked_candidates(payload, variant)
    target = payload["target_sid"] if "target_sid" in payload else payload["target_itemkey"]
    output: Dict[str, float] = {}
    for metric in metric_order:
        metric_name, raw_k = metric.split("@", 1)
        contributions = metric_contributions(target, ranked_items, int(raw_k))
        output[metric] = float(contributions[metric_name].mean())
    return output


def select_best(rows: Sequence[Mapping[str, object]], metric_order: Sequence[str]) -> int:
    if not rows:
        raise ValueError("No fusion candidates were evaluated")

    def selection_key(index: int):
        row = rows[index]
        metrics = row["metrics"]
        return tuple(float(metrics[name]) for name in metric_order) + (
            -float(row["rerank_weight"]),
        )

    return max(range(len(rows)), key=selection_key)


def build_candidates(args: argparse.Namespace) -> List[Dict[str, object]]:
    return [
        {
            "variant": f"fusion@{args.beam_weight:g}@{weight_decimal}",
            "beam_weight": args.beam_weight,
            "rerank_weight": float(weight_decimal),
            "metrics": {metric: 0.0 for metric in args.metric_order},
        }
        for weight_decimal in decimal_grid(
            args.rerank_weight_start,
            args.rerank_weight_stop,
            args.rerank_weight_step,
        )
    ]


def evaluate_h5(
    input_path: Path,
    candidates: List[Dict[str, object]],
    args: argparse.Namespace,
) -> int:
    required = ("example_hash", "target_sid", "beam_sid", "beam_score", "raw_rerank_logit")
    sample_count = 0
    with h5py.File(input_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"Validation HDF5 lacks datasets: {missing}")
        total = handle["target_sid"].shape[0]
        for start in range(0, total, args.chunk_size):
            stop = min(start + args.chunk_size, total)
            example_hash = handle["example_hash"][start:stop]
            selected = (
                example_hash[:, 0] % np.uint64(args.sample_modulus)
                == np.uint64(args.sample_remainder)
            )
            if not np.any(selected):
                continue
            payload = {
                key: handle[key][start:stop][selected]
                for key in ("target_sid", "beam_sid", "beam_score", "raw_rerank_logit")
            }
            current = int(selected.sum())
            sample_count += current
            for candidate in candidates:
                ranked = ranked_candidates(payload, str(candidate["variant"]))
                metric_cache: Dict[int, Dict[str, np.ndarray]] = {}
                for metric in args.metric_order:
                    metric_name, raw_k = metric.split("@", 1)
                    k = int(raw_k)
                    if k not in metric_cache:
                        metric_cache[k] = metric_contributions(payload["target_sid"], ranked, k)
                    candidate["metrics"][metric] += float(metric_cache[k][metric_name].sum())
            if stop % (args.chunk_size * 20) == 0 or stop == total:
                print(f"[fusion] scanned={stop}/{total} selected={sample_count}")
    if not sample_count:
        raise RuntimeError("Hash sampling selected no validation examples")
    for candidate in candidates:
        for metric in args.metric_order:
            candidate["metrics"][metric] /= sample_count
    return sample_count


def markdown_report(payload: Mapping[str, object]) -> str:
    selected = payload["selected"]
    metric_order = payload["selection_protocol"]["metric_order"]
    lines = [
        "# Validation-only Score Fusion Selection",
        "",
        f"Input split: `{payload['input_split']}`; examples: {payload['num_examples']}.",
        "",
        "Selection is lexicographic over "
        + ", ".join(f"`{metric}`" for metric in metric_order)
        + "; an exact tie selects the smaller reranker weight.",
        "",
        f"Selected frozen variant: `{selected['variant']}`.",
        "",
        "| Beam weight | Rerank weight | " + " | ".join(metric_order) + " |",
        "| ---: | ---: | " + " | ".join("---:" for _ in metric_order) + " |",
    ]
    for row in payload["candidates"]:
        values = " | ".join(f"{row['metrics'][metric]:.6f}" for metric in metric_order)
        marker = " **(selected)**" if row["variant"] == selected["variant"] else ""
        lines.append(
            f"| {row['beam_weight']:.6g} | {row['rerank_weight']:.6g}{marker} | {values} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("Refusing to overwrite an existing fusion selection")
    cpu_budget = selection_cpu_budget()
    cpu_samples = []
    if cpu_budget is not None:
        from src.runtime_export_cpu import export_cpu_snapshot
        cpu_samples.append(export_cpu_snapshot(cpu_budget["allowed_cpus"]))
        print("[cpu-budget] " + json.dumps(cpu_budget), flush=True)
    input_path = args.input.expanduser().resolve()
    metadata_path = input_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing export metadata {metadata_path}; refusing to select without split provenance"
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("split") != "val":
        raise ValueError(
            f"Fusion selection requires a validation export, got split={metadata.get('split')!r}"
        )
    if metadata.get("sample_unit") != "unique_user_history":
        raise ValueError("Fusion selection requires a unique-user history export")
    if any(value is not None for value in metadata.get("limits", {}).values()):
        raise ValueError("Fusion selection refuses limited smoke exports")
    if not metadata.get("checkpoint_sha256"):
        raise ValueError("Validation export lacks a checkpoint SHA-256")
    provenance = metadata.get("dataset_provenance", {})
    if not provenance.get("protocol") or len(provenance.get("manifests", {})) != 3:
        raise ValueError("Validation export lacks complete dataset provenance")
    if int(metadata.get("num_samples", -1)) != int(provenance.get("expected_samples", -2)):
        raise ValueError("Validation export count does not match dataset provenance")

    candidates = build_candidates(args)
    if input_path.suffix == ".h5":
        num_examples = evaluate_h5(input_path, candidates, args)
    else:
        with np.load(input_path) as archive:
            data = {key: archive[key] for key in archive.files}
        for candidate in candidates:
            candidate["metrics"] = evaluate_variant(
                data, str(candidate["variant"]), args.metric_order
            )
        target_key = "target_sid" if "target_sid" in data else "target_itemkey"
        num_examples = int(data[target_key].shape[0])
    if args.sample_modulus == 1 and num_examples != int(metadata.get("num_samples", -1)):
        raise ValueError(
            f"Full validation count mismatch: sidecar={metadata.get('num_samples')} "
            f"selected={num_examples}"
        )

    selected_index = select_best(candidates, args.metric_order)
    if cpu_budget is not None:
        cpu_samples.append(export_cpu_snapshot(cpu_budget["allowed_cpus"]))
        print("[cpu-process-tree] " + json.dumps(cpu_samples), flush=True)
    root = Path(__file__).resolve().parents[1]
    output = {
        "schema_version": 2,
        "selector_sha256": file_sha256(Path(__file__).resolve()),
        "source_git_commit": os.environ.get("SOURCE_GIT_COMMIT"),
        "source_files": [
            {"path": str(root / name), "sha256": file_sha256(root / name)}
            for name in (
                "scripts/select_score_fusion.py", "scripts/analyze_paired_evaluation.py",
                "run_select_score_fusion.sh", "src/runtime_export_cpu.py", "src/runtime_cpu.py",
            )
        ],
        "cpu_budget": cpu_budget,
        "cpu_process_tree_samples": cpu_samples,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "selection_scope": "validation_only",
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "input_sidecar_sha256": file_sha256(metadata_path),
        "input_split": metadata["split"],
        "input_exporter_sha256": metadata.get("exporter_sha256"),
        "input_checkpoint_sha256": metadata["checkpoint_sha256"],
        "input_dataset_provenance": provenance,
        "num_examples": num_examples,
        "selection_protocol": {
            "metric_order": list(args.metric_order),
            "tie_breaker": "smaller_rerank_weight",
            "beam_weight": args.beam_weight,
            "hash_sample": {
                "word": 0,
                "modulus": args.sample_modulus,
                "remainder": args.sample_remainder,
            },
            "rerank_weight_grid": {
                "start": str(args.rerank_weight_start),
                "stop": str(args.rerank_weight_stop),
                "step": str(args.rerank_weight_step),
            },
        },
        "selected": candidates[selected_index],
        "candidates": candidates,
    }
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("Refusing to overwrite an existing fusion selection")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(markdown_report(output))
    print(f"[done] selected {output['selected']['variant']}")
    print(f"[done] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
