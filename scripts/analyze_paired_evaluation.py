#!/usr/bin/env python
"""Compute paired recommendation metrics, confidence intervals, and tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
from scipy.stats import binomtest


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_value(value: str, separator: str) -> Tuple[str, str]:
    if separator not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME{separator}VALUE, got {value!r}")
    name, payload = value.split(separator, 1)
    if not name or not payload:
        raise argparse.ArgumentTypeError(f"Expected NAME{separator}VALUE, got {value!r}")
    return name, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named HDF5 (streaming) or NPZ (small legacy) export; repeat for multiple models.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        required=True,
        metavar="BASELINE,CHALLENGER",
        help=(
            "Variant pair such as listwise:beam,listwise:rerank. Variants are "
            "beam, rerank, click, selected, legacy_pointwise, or "
            "fusion@BEAM_WEIGHT@RERANK_WEIGHT."
        ),
    )
    parser.add_argument(
        "--selection",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Validation-only fusion selection JSON for a named model. Required "
            "whenever a comparison uses that model's fusion@... variant."
        ),
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=50)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--expected-split", choices=("val", "test"), default="test")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    if any(k <= 0 for k in args.ks):
        parser.error("all --ks values must be positive")
    if args.bootstrap_samples <= 0 or args.bootstrap_batch_size <= 0:
        parser.error("bootstrap sizes must be positive")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def load_exports(specs: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    exports: Dict[str, Dict[str, np.ndarray]] = {}
    for spec in specs:
        name, raw_path = parse_named_value(spec, "=")
        if name in exports:
            raise ValueError(f"Duplicate input name: {name}")
        path = Path(raw_path).expanduser().resolve()
        with np.load(path) as payload:
            exports[name] = {key: payload[key] for key in payload.files}
        exports[name]["_path"] = np.asarray(str(path))
    return exports


def stable_order(example_hash: np.ndarray) -> np.ndarray:
    if example_hash.ndim != 2 or example_hash.shape[1] != 2:
        raise ValueError(f"Unexpected example_hash shape: {example_hash.shape}")
    return np.lexsort((example_hash[:, 1], example_hash[:, 0]))


def align_exports(exports: Mapping[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    aligned: Dict[str, Dict[str, np.ndarray]] = {}
    reference_hash = None
    reference_target = None
    reference_target_key = None
    for name, payload in exports.items():
        order = stable_order(payload["example_hash"])
        current = {
            key: (value[order] if key != "_path" else value)
            for key, value in payload.items()
        }
        if reference_hash is None:
            reference_hash = current["example_hash"]
            reference_target_key = "target_sid" if "target_sid" in current else "target_itemkey"
            reference_target = current[reference_target_key]
        else:
            if not np.array_equal(current["example_hash"], reference_hash):
                raise ValueError(f"Input {name} does not contain the same evaluation examples")
            if reference_target_key not in current:
                raise ValueError(
                    f"Input {name} lacks the reference target identity {reference_target_key}"
                )
            if not np.array_equal(current[reference_target_key], reference_target):
                raise ValueError(f"Input {name} has target mismatch after example alignment")
        aligned[name] = current
    return aligned


def parse_variant(spec: str) -> Tuple[str, str]:
    return parse_named_value(spec, ":")


def ranked_candidates(payload: Mapping[str, np.ndarray], variant: str) -> np.ndarray:
    beam_key = "beam_sid" if "beam_sid" in payload else "beam_itemkey"
    rerank_key = "rerank_sid" if "rerank_sid" in payload else "rerank_itemkey"
    if variant == "beam":
        return payload[beam_key]
    if variant == "rerank":
        return payload[rerank_key]
    if variant == "click":
        order = np.argsort(-payload["raw_rerank_logit"], axis=1, kind="stable")
    elif variant == "legacy_pointwise":
        # The June pointwise checkpoint was evaluated as beam log-probability
        # plus log(sigmoid(click_logit)); preserve that historical protocol.
        click_log_prob = -np.logaddexp(0.0, -payload["raw_rerank_logit"])
        score = payload["beam_score"] + click_log_prob
        order = np.argsort(-score, axis=1, kind="stable")
    elif variant.startswith("fusion@"):
        parts = variant.split("@")
        if len(parts) != 3:
            raise ValueError(f"Expected fusion@BEAM_WEIGHT@RERANK_WEIGHT, got {variant}")
        beam_weight, rerank_weight = map(float, parts[1:])
        score = beam_weight * payload["beam_score"] + rerank_weight * payload["raw_rerank_logit"]
        order = np.argsort(-score, axis=1, kind="stable")
    else:
        raise ValueError(f"Unknown variant: {variant}")
    beam_candidates = payload[beam_key]
    gather_order = order if beam_candidates.ndim == 2 else order[..., None]
    return np.take_along_axis(beam_candidates, gather_order, axis=1)


def ranked_itemkeys(payload: Mapping[str, np.ndarray], variant: str) -> np.ndarray:
    """Backward-compatible alias; schema v2 returns exact SID tuples when present."""
    return ranked_candidates(payload, variant)


def metric_contributions(
    target_itemkey: np.ndarray,
    ranked_items: np.ndarray,
    k: int,
) -> Dict[str, np.ndarray]:
    if ranked_items.ndim not in (2, 3):
        raise ValueError(f"Expected ranked candidates [N,K] or [N,K,H], got {ranked_items.shape}")
    effective_k = min(k, ranked_items.shape[1])
    if ranked_items.ndim == 3:
        if target_itemkey.ndim != 2 or target_itemkey.shape[1] != ranked_items.shape[2]:
            raise ValueError(
                f"Target/candidate SID shape mismatch: {target_itemkey.shape}, {ranked_items.shape}"
            )
        matches = np.all(ranked_items == target_itemkey[:, None, :], axis=-1)
    else:
        if target_itemkey.ndim != 1:
            raise ValueError(
                f"Packed target must be one-dimensional, got {target_itemkey.shape}"
            )
        matches = ranked_items == target_itemkey[:, None]
    top_matches = matches[:, :effective_k]
    relevant = matches.sum(axis=1)
    denominator = np.minimum(relevant, effective_k)
    recall = top_matches.sum(axis=1) / np.maximum(denominator, 1)

    discounts = 1.0 / np.log2(np.arange(2, effective_k + 2, dtype=np.float64))
    dcg = (top_matches * discounts[None, :]).sum(axis=1)
    ideal_count = np.minimum(relevant, effective_k)
    ideal_prefix = np.concatenate(([0.0], np.cumsum(discounts)))
    idcg = ideal_prefix[ideal_count]
    ndcg = dcg / np.where(idcg == 0, 1.0, idcg)
    return {"recall": recall.astype(np.float64), "ndcg": ndcg.astype(np.float64)}


def paired_bootstrap_ci(
    delta: np.ndarray,
    samples: int,
    batch_size: int,
    confidence: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Exact nonparametric bootstrap of the mean via compressed value counts.

    Resampling N observations from the empirical distribution is equivalent to
    drawing category counts from Multinomial(N, empirical frequencies).  Metric
    deltas have few unique values, so this avoids materializing [samples, N]
    row-index matrices without changing the bootstrap distribution.
    """
    estimates = np.empty(samples, dtype=np.float64)
    n = delta.size
    values, counts = np.unique(delta, return_counts=True)
    probabilities = counts.astype(np.float64) / n
    written = 0
    while written < samples:
        current = min(batch_size, samples - written)
        sampled_counts = rng.multinomial(n, probabilities, size=current)
        estimates[written : written + current] = sampled_counts @ values / n
        written += current
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def paired_bootstrap_ci_from_counts(
    value_counts: Mapping[float, int],
    samples: int,
    batch_size: int,
    confidence: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    values = np.asarray(list(value_counts.keys()), dtype=np.float64)
    counts = np.asarray(list(value_counts.values()), dtype=np.int64)
    n = int(counts.sum())
    if n <= 0:
        raise ValueError("Cannot bootstrap an empty delta distribution")
    probabilities = counts.astype(np.float64) / n
    estimates = np.empty(samples, dtype=np.float64)
    written = 0
    while written < samples:
        current = min(batch_size, samples - written)
        sampled_counts = rng.multinomial(n, probabilities, size=current)
        estimates[written : written + current] = sampled_counts @ values / n
        written += current
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def recall_discordance_test(baseline: np.ndarray, challenger: np.ndarray) -> Dict[str, Any]:
    baseline_hit = baseline > 0
    challenger_hit = challenger > 0
    loss_to_win = int((~baseline_hit & challenger_hit).sum())
    win_to_loss = int((baseline_hit & ~challenger_hit).sum())
    discordant = loss_to_win + win_to_loss
    pvalue = float(binomtest(loss_to_win, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "baseline_miss_challenger_hit": loss_to_win,
        "baseline_hit_challenger_miss": win_to_loss,
        "discordant_pairs": discordant,
        "exact_two_sided_pvalue": pvalue,
    }


def recall_discordance_result(loss_to_win: int, win_to_loss: int) -> Dict[str, Any]:
    discordant = loss_to_win + win_to_loss
    pvalue = float(binomtest(loss_to_win, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "baseline_miss_challenger_hit": loss_to_win,
        "baseline_hit_challenger_miss": win_to_loss,
        "discordant_pairs": discordant,
        "exact_two_sided_pvalue": pvalue,
    }


def comparison_result(
    baseline_label: str,
    challenger_label: str,
    baseline_items: np.ndarray,
    challenger_items: np.ndarray,
    target_itemkey: np.ndarray,
    ks: Sequence[int],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "baseline": baseline_label,
        "challenger": challenger_label,
        "num_examples": int(target_itemkey.shape[0]),
        "metrics": {},
    }
    for k in ks:
        baseline_metrics = metric_contributions(target_itemkey, baseline_items, k)
        challenger_metrics = metric_contributions(target_itemkey, challenger_items, k)
        for metric_name in ("recall", "ndcg"):
            baseline = baseline_metrics[metric_name]
            challenger = challenger_metrics[metric_name]
            delta = challenger - baseline
            ci_low, ci_high = paired_bootstrap_ci(
                delta,
                samples=args.bootstrap_samples,
                batch_size=args.bootstrap_batch_size,
                confidence=args.confidence,
                rng=rng,
            )
            baseline_mean = float(baseline.mean())
            challenger_mean = float(challenger.mean())
            result = {
                "baseline_mean": baseline_mean,
                "challenger_mean": challenger_mean,
                "absolute_gain": float(delta.mean()),
                "relative_gain_percent": (
                    float(delta.mean() / baseline_mean * 100.0) if baseline_mean else None
                ),
                "paired_bootstrap_ci": [ci_low, ci_high],
                "confidence": args.confidence,
            }
            if metric_name == "recall":
                result["paired_binary_test"] = recall_discordance_test(baseline, challenger)
            output["metrics"][f"{metric_name}@{k}"] = result
    return output


def parse_comparison_specs(
    specs: Sequence[str],
    model_names: Sequence[str],
) -> List[Dict[str, str]]:
    comparisons = []
    known_models = set(model_names)
    for spec in specs:
        if "," not in spec:
            raise ValueError(f"Expected BASELINE,CHALLENGER, got {spec!r}")
        baseline_spec, challenger_spec = spec.split(",", 1)
        baseline_model, baseline_variant = parse_variant(baseline_spec)
        challenger_model, challenger_variant = parse_variant(challenger_spec)
        if baseline_model not in known_models or challenger_model not in known_models:
            raise ValueError(f"Unknown model in comparison: {spec}")
        comparisons.append(
            {
                "baseline": baseline_spec,
                "challenger": challenger_spec,
                "baseline_model": baseline_model,
                "baseline_variant": baseline_variant,
                "challenger_model": challenger_model,
                "challenger_variant": challenger_variant,
            }
        )
    return comparisons


def update_delta_counter(counter: Counter, delta: np.ndarray) -> None:
    values, counts = np.unique(delta, return_counts=True)
    counter.update({float(value): int(count) for value, count in zip(values, counts)})


def analyze_h5(args: argparse.Namespace) -> Dict[str, Any]:
    paths: Dict[str, Path] = {}
    for spec in args.input:
        name, raw_path = parse_named_value(spec, "=")
        if name in paths:
            raise ValueError(f"Duplicate input name: {name}")
        paths[name] = Path(raw_path).expanduser().resolve()

    expected_split = getattr(args, "expected_split", "test")
    sidecars: Dict[str, Dict[str, Any]] = {}
    dataset_identity = None
    for name, path in paths.items():
        metadata_path = path.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing export metadata for {name}: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("split") != expected_split:
            raise ValueError(
                f"Input {name} has split={metadata.get('split')!r}; "
                f"expected {expected_split!r}"
            )
        if metadata.get("sample_unit") != "unique_user_history":
            raise ValueError(
                f"Input {name} is not certified as unique-user history data"
            )
        if any(value is not None for value in metadata.get("limits", {}).values()):
            raise ValueError(f"Input {name} is a limited smoke export, not a full split")
        provenance = metadata.get("dataset_provenance", {})
        current_identity = {
            "data_root": provenance.get("data_root"),
            "protocol": provenance.get("protocol"),
            "manifest_sha256": {
                key: value.get("sha256")
                for key, value in provenance.get("manifests", {}).items()
            },
        }
        if not current_identity["protocol"] or len(current_identity["manifest_sha256"]) != 3:
            raise ValueError(f"Input {name} lacks complete dataset provenance")
        if int(metadata.get("num_samples", -1)) != int(provenance.get("expected_samples", -2)):
            raise ValueError(f"Input {name} count does not match dataset provenance")
        if dataset_identity is None:
            dataset_identity = current_identity
        elif current_identity != dataset_identity:
            raise ValueError(f"Input {name} uses different dataset provenance")
        if not metadata.get("checkpoint_sha256"):
            raise ValueError(f"Input {name} lacks a checkpoint SHA-256")
        sidecars[name] = metadata

    comparison_specs = parse_comparison_specs(args.compare, list(paths))
    selection_paths: Dict[str, Path] = {}
    for spec in getattr(args, "selection", []):
        name, raw_path = parse_named_value(spec, "=")
        if name in selection_paths:
            raise ValueError(f"Duplicate selection name: {name}")
        if name not in paths:
            raise ValueError(f"Selection refers to unknown input model: {name}")
        selection_paths[name] = Path(raw_path).expanduser().resolve()

    selections: Dict[str, Dict[str, Any]] = {}
    for name, path in selection_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing selection for {name}: {path}")
        selection = json.loads(path.read_text())
        if selection.get("selection_scope") != "validation_only" or selection.get("input_split") != "val":
            raise ValueError(f"Selection for {name} is not validation-only")
        if selection.get("input_checkpoint_sha256") != sidecars[name]["checkpoint_sha256"]:
            raise ValueError(
                f"Selection for {name} was not produced from the test checkpoint"
            )
        validation_input = Path(str(selection.get("input", ""))).expanduser().resolve()
        if not validation_input.is_file():
            raise FileNotFoundError(
                f"Validation export used by selection for {name} is missing: {validation_input}"
            )
        if file_sha256(validation_input) != selection.get("input_sha256"):
            raise ValueError(f"Validation export used by selection for {name} changed")
        selection_provenance = selection.get("input_dataset_provenance", {})
        selection_identity = {
            "data_root": selection_provenance.get("data_root"),
            "protocol": selection_provenance.get("protocol"),
            "manifest_sha256": {
                key: value.get("sha256")
                for key, value in selection_provenance.get("manifests", {}).items()
            },
        }
        if selection_identity != dataset_identity:
            raise ValueError(f"Selection for {name} uses different dataset provenance")
        selections[name] = selection

    for comparison in comparison_specs:
        for side in ("baseline", "challenger"):
            model_name = comparison[f"{side}_model"]
            variant = comparison[f"{side}_variant"]
            if variant == "selected":
                if model_name not in selections:
                    raise ValueError(
                        f"{model_name}:selected lacks --selection provenance"
                    )
                variant = selections[model_name]["selected"]["variant"]
                comparison[f"{side}_variant"] = variant
                comparison[side] = f"{model_name}:{variant}"
            if variant.startswith("fusion@"):
                if model_name not in selections:
                    raise ValueError(
                        f"Fusion comparison for {model_name} lacks --selection provenance"
                    )
                selected_variant = selections[model_name].get("selected", {}).get("variant")
                if variant != selected_variant:
                    raise ValueError(
                        f"Comparison uses {model_name}:{variant}, but validation selected "
                        f"{selected_variant}"
                    )
    ks = sorted(set(args.ks))
    aggregate: List[Dict[str, Any]] = []
    for comparison in comparison_specs:
        aggregate.append(
            {
                **comparison,
                "num_examples": 0,
                "metrics": {
                    f"{metric}@{k}": {
                        "baseline_sum": 0.0,
                        "challenger_sum": 0.0,
                        "delta_counts": Counter(),
                        "baseline_miss_challenger_hit": 0,
                        "baseline_hit_challenger_miss": 0,
                    }
                    for k in ks
                    for metric in ("recall", "ndcg")
                },
            }
        )

    required_keys = (
        "user_id",
        "example_hash",
        "target_sid",
        "beam_sid",
        "beam_score",
        "raw_rerank_logit",
        "rerank_sid",
    )
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context(h5py.File(path, "r"))
            for name, path in paths.items()
        }
        lengths = {name: handle["target_sid"].shape[0] for name, handle in handles.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"HDF5 inputs have different lengths: {lengths}")
        num_examples = next(iter(lengths.values()))
        for name, length in lengths.items():
            if int(sidecars[name].get("num_samples", -1)) != length:
                raise ValueError(
                    f"Input {name} sidecar count does not match HDF5: "
                    f"{sidecars[name].get('num_samples')} vs {length}"
                )
        for name, handle in handles.items():
            missing = [key for key in required_keys if key not in handle]
            if missing:
                raise ValueError(f"Input {name} lacks HDF5 datasets: {missing}")

        model_payloads: Dict[str, Dict[str, np.ndarray]] = {}
        for start in range(0, num_examples, args.chunk_size):
            stop = min(start + args.chunk_size, num_examples)
            model_payloads.clear()
            for name, handle in handles.items():
                model_payloads[name] = {key: handle[key][start:stop] for key in required_keys}

            reference_name = next(iter(model_payloads))
            reference = model_payloads[reference_name]
            for name, payload in model_payloads.items():
                if not np.array_equal(payload["example_hash"], reference["example_hash"]):
                    raise ValueError(
                        f"Input order mismatch for {name} in rows [{start}, {stop}); "
                        "rerun exports with one deterministic data worker"
                    )
                if not np.array_equal(payload["user_id"], reference["user_id"]):
                    raise ValueError(f"User ID mismatch for {name} in rows [{start}, {stop})")
                if not np.array_equal(payload["target_sid"], reference["target_sid"]):
                    raise ValueError(f"Target SID mismatch for {name} in rows [{start}, {stop})")

            target_sid = reference["target_sid"]
            ranked_cache: Dict[Tuple[str, str], np.ndarray] = {}
            metric_cache: Dict[Tuple[str, str, int], Dict[str, np.ndarray]] = {}
            for comparison, state in zip(comparison_specs, aggregate):
                sides = []
                for side in ("baseline", "challenger"):
                    model_name = comparison[f"{side}_model"]
                    variant = comparison[f"{side}_variant"]
                    rank_key = (model_name, variant)
                    if rank_key not in ranked_cache:
                        ranked_cache[rank_key] = ranked_candidates(
                            model_payloads[model_name], variant
                        )
                    sides.append(rank_key)

                for k in ks:
                    metrics_by_side = []
                    for rank_key in sides:
                        metric_key = (*rank_key, k)
                        if metric_key not in metric_cache:
                            metric_cache[metric_key] = metric_contributions(
                                target_sid, ranked_cache[rank_key], k
                            )
                        metrics_by_side.append(metric_cache[metric_key])

                    for metric_name in ("recall", "ndcg"):
                        baseline = metrics_by_side[0][metric_name]
                        challenger = metrics_by_side[1][metric_name]
                        delta = challenger - baseline
                        metric_state = state["metrics"][f"{metric_name}@{k}"]
                        metric_state["baseline_sum"] += float(baseline.sum())
                        metric_state["challenger_sum"] += float(challenger.sum())
                        update_delta_counter(metric_state["delta_counts"], delta)
                        if metric_name == "recall":
                            baseline_hit = baseline > 0
                            challenger_hit = challenger > 0
                            metric_state["baseline_miss_challenger_hit"] += int(
                                (~baseline_hit & challenger_hit).sum()
                            )
                            metric_state["baseline_hit_challenger_miss"] += int(
                                (baseline_hit & ~challenger_hit).sum()
                            )

                state["num_examples"] += stop - start
            if stop % (args.chunk_size * 20) == 0 or stop == num_examples:
                print(f"[analyze] rows={stop}/{num_examples}")

    rng = np.random.default_rng(args.seed)
    results = []
    for state in aggregate:
        n = state["num_examples"]
        result = {
            "baseline": state["baseline"],
            "challenger": state["challenger"],
            "num_examples": n,
            "metrics": {},
        }
        for metric_label, metric_state in state["metrics"].items():
            baseline_mean = metric_state["baseline_sum"] / n
            challenger_mean = metric_state["challenger_sum"] / n
            absolute_gain = challenger_mean - baseline_mean
            ci_low, ci_high = paired_bootstrap_ci_from_counts(
                metric_state["delta_counts"],
                samples=args.bootstrap_samples,
                batch_size=args.bootstrap_batch_size,
                confidence=args.confidence,
                rng=rng,
            )
            values = {
                "baseline_mean": baseline_mean,
                "challenger_mean": challenger_mean,
                "absolute_gain": absolute_gain,
                "relative_gain_percent": (
                    absolute_gain / baseline_mean * 100.0 if baseline_mean else None
                ),
                "paired_bootstrap_ci": [ci_low, ci_high],
                "confidence": args.confidence,
            }
            if metric_label.startswith("recall@"):
                values["paired_binary_test"] = recall_discordance_result(
                    metric_state["baseline_miss_challenger_hit"],
                    metric_state["baseline_hit_challenger_miss"],
                )
            result["metrics"][metric_label] = values
        results.append(result)

    return {
        "schema_version": 2,
        "analyzer_sha256": file_sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "sample_unit": "unique_user_history",
        "num_examples": num_examples,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_batch_size": args.bootstrap_batch_size,
        "bootstrap_method": "paired_nonparametric_multinomial_counts",
        "confidence": args.confidence,
        "seed": args.seed,
        "stream_chunk_size": args.chunk_size,
        "inputs": {name: str(path) for name, path in paths.items()},
        "input_metadata": {
            name: {
                "path": str(paths[name].with_suffix(".json")),
                "checkpoint_sha256": sidecars[name]["checkpoint_sha256"],
                "exporter_sha256": sidecars[name].get("exporter_sha256"),
            }
            for name in paths
        },
        "dataset_identity": dataset_identity,
        "fusion_selections": {
            name: {
                "path": str(selection_paths[name]),
                "sha256": file_sha256(selection_paths[name]),
                "variant": selection["selected"]["variant"],
            }
            for name, selection in selections.items()
        },
        "comparisons": results,
    }


def markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Paired Evaluation Report",
        "",
        f"Sample unit: `{payload['sample_unit']}`; bootstrap resamples: "
        f"{payload['bootstrap_samples']}; seed: {payload['seed']}.",
        "",
        "| Comparison | Metric | Baseline | Challenger | Abs. gain | Rel. gain | Paired bootstrap CI | p-value |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for comparison in payload["comparisons"]:
        label = f"{comparison['baseline']} → {comparison['challenger']}"
        for metric, values in comparison["metrics"].items():
            relative = values["relative_gain_percent"]
            relative_text = f"{relative:.3f}%" if relative is not None else "NA"
            ci_low, ci_high = values["paired_bootstrap_ci"]
            test = values.get("paired_binary_test")
            pvalue = f"{test['exact_two_sided_pvalue']:.4g}" if test else "—"
            lines.append(
                f"| {label} | {metric} | {values['baseline_mean']:.6f} | "
                f"{values['challenger_mean']:.6f} | {values['absolute_gain']:+.6f} | "
                f"{relative_text} | [{ci_low:+.6f}, {ci_high:+.6f}] | {pvalue} |"
            )
    lines.extend(
        [
            "",
            "The p-value is an exact two-sided paired binary test for Recall only. "
            "NDCG uncertainty is reported through the paired bootstrap interval.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_paths = [
        Path(parse_named_value(spec, "=")[1]).expanduser().resolve()
        for spec in args.input
    ]
    if all(path.suffix == ".h5" for path in input_paths):
        output = analyze_h5(args)
        if args.output_json.exists() or args.output_md.exists():
            raise FileExistsError("Refusing to overwrite an existing paired report")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        args.output_md.write_text(markdown_report(output))
        print(f"[done] wrote {args.output_json} and {args.output_md}")
        return
    if any(path.suffix == ".h5" for path in input_paths):
        raise ValueError("Do not mix streaming HDF5 and legacy NPZ inputs")

    exports = align_exports(load_exports(args.input))
    first_payload = next(iter(exports.values()))
    target_itemkey = (
        first_payload["target_sid"]
        if "target_sid" in first_payload
        else first_payload["target_itemkey"]
    )
    rng = np.random.default_rng(args.seed)

    comparisons = []
    for spec in args.compare:
        if "," not in spec:
            raise ValueError(f"Expected BASELINE,CHALLENGER, got {spec!r}")
        baseline_spec, challenger_spec = spec.split(",", 1)
        baseline_model, baseline_variant = parse_variant(baseline_spec)
        challenger_model, challenger_variant = parse_variant(challenger_spec)
        if baseline_model not in exports or challenger_model not in exports:
            raise ValueError(f"Unknown model in comparison: {spec}")
        baseline_items = ranked_itemkeys(exports[baseline_model], baseline_variant)
        challenger_items = ranked_itemkeys(exports[challenger_model], challenger_variant)
        comparisons.append(
            comparison_result(
                baseline_label=baseline_spec,
                challenger_label=challenger_spec,
                baseline_items=baseline_items,
                challenger_items=challenger_items,
                target_itemkey=target_itemkey,
                ks=sorted(set(args.ks)),
                args=args,
                rng=rng,
            )
        )

    output = {
        "schema_version": 1,
        "sample_unit": "evaluation_example",
        "num_examples": int(target_itemkey.shape[0]),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_batch_size": args.bootstrap_batch_size,
        "bootstrap_method": "paired_nonparametric_multinomial_counts",
        "confidence": args.confidence,
        "seed": args.seed,
        "inputs": {name: str(payload["_path"]) for name, payload in exports.items()},
        "comparisons": comparisons,
    }
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("Refusing to overwrite an existing paired report")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(markdown_report(output))
    print(f"[done] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
