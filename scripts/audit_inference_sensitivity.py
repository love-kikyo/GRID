"""Read-only, chunked acceptance of fixed-checkpoint inference sensitivity exports.

Only the final acceptance JSON is written. No model imports, inference, tuning,
or bootstrap. Runtime BF16 ordering is kept separate from FP32 selected fusion.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


FIELDS = {"user_id", "example_hash", "target_sid", "beam_sid", "beam_score",
          "raw_rerank_logit", "rerank_sid", "rerank_score"}
GROUPS = {"retrieval": [[20, 0], [20, 5], [20, 10], [20, 20]],
          "beam": [[10, 10], [40, 10]]}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def setting_name(c, m):
    return f"c{c}_m{m}"


def _packed(sid):
    require(np.issubdtype(sid.dtype, np.integer), "SID must be integer")
    require(bool(np.all((sid >= 1) & (sid <= 4096))), "SID out of range")
    # Base 4097 avoids aliasing the valid digit 4096. Four digits fit uint64.
    return (sid.astype(np.uint64) * np.array([4097**3, 4097**2, 4097, 1],
                                            dtype=np.uint64)).sum(axis=-1)


def _ranks(matches, scores=None):
    """Verify stable-sort GT ranks against an independent outranker count."""
    n, c = matches.shape
    hit = matches.any(axis=1)
    index = matches.argmax(axis=1)
    if scores is None:
        return np.where(hit, index + 1, c + 1)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = np.take_along_axis(matches, order, axis=1)
    rank = np.where(hit, ranked.argmax(axis=1) + 1, c + 1)
    positive = scores[np.arange(n), index, None]
    before = np.arange(c)[None, :] < index[:, None]
    outrank = ((scores > positive) | ((scores == positive) & before)).sum(axis=1)
    independent = np.where(hit, outrank + 1, c + 1)
    require(np.array_equal(rank, independent), "Independent GT rank mismatch")
    return rank


def _metric_sums(rank, ks):
    result = {}
    for k in ks:
        hit = rank <= k
        result[f"recall@{k}"] = int(hit.sum())
        result[f"ndcg@{k}"] = float((hit / np.log2(rank + 1)).sum(dtype=np.float64))
    return result


def _validate_run(run, manifest, phase, group, expected):
    require(run.get("status") == "complete", "Run is not complete")
    require(run.get("phase") == phase and run.get("group") == group,
            "Run phase/group mismatch")
    require(run.get("processed") == expected, "Run count mismatch")
    require(run.get("expected_samples") == manifest["expected_test_samples"],
            "Run expected test count mismatch")
    if "_manifest_sha256" in manifest:
        require(run.get("manifest_sha256") == manifest["_manifest_sha256"],
                "Run manifest binding mismatch")
    require(run.get("smoke_gate", {}).get("status") == "passed", "Missing passed smoke gate")
    require("4090" in run.get("hardware", {}).get("name", ""), "Expected RTX 4090")
    budget = run.get("cpu_budget", {})
    allowed = set(budget.get("allowed_cpus", []))
    require(0 < len(allowed) <= 4 and budget.get("allocated_cpus") == 4
            and budget.get("library_threads") == 1, "Invalid CPU budget")
    snapshots = run.get("cpu_process_tree_samples", [])
    require(len(snapshots) >= 2, "Missing CPU start/end checks")
    for snapshot in snapshots:
        require(set(snapshot.get("allowed_cpus", [])) == allowed
                and bool(snapshot.get("processes")), "Invalid CPU snapshot")


def _validate_manifest(manifest, phase, group):
    require(phase in {"smoke", "full"} and group in GROUPS, "Invalid phase/group")
    require(manifest.get("seed") == 42, "Only seed42 is authorized")
    require(manifest.get("groups") == GROUPS, "Settings differ from frozen scan")
    require(manifest.get("beta") == .5 and manifest.get("beam_weight") == 1.,
            "Frozen fusion weights changed")
    require(isinstance(manifest.get("expected_test_samples"), int)
            and manifest["expected_test_samples"] > 0, "Invalid expected test size")
    beta_record = manifest["frozen_beta"]
    require(sha(Path(beta_record["path"])) == beta_record["sha256"], "Selection hash mismatch")
    selection = json.loads(Path(beta_record["path"]).read_text())
    require(selection.get("input_split") == "val"
            and selection.get("selection_scope") == "validation_only", "Not validation-only selection")
    require(selection["selected"]["beam_weight"] == manifest["beam_weight"]
            and selection["selected"]["rerank_weight"] == manifest["beta"], "Selected weights mismatch")
    require(selection["input_checkpoint_sha256"] == manifest["checkpoint"]["sha256"],
            "Selection checkpoint mismatch")


def _shape_check(handle, count, c):
    require(set(handle.keys()) == FIELDS, "Unexpected HDF5 fields")
    shapes = {"user_id": (count,), "example_hash": (count, 2), "target_sid": (count, 4),
              "beam_sid": (count, c, 4), "beam_score": (count, c),
              "raw_rerank_logit": (count, c), "rerank_sid": (count, c, 4),
              "rerank_score": (count, c)}
    for name, shape in shapes.items():
        require(handle[name].shape == shape, f"Shape mismatch for {name}")
    require(handle.attrs.get("num_samples") == count, "HDF5 sample count mismatch")
    require(handle["user_id"].dtype == np.dtype("int64"), "User IDs must be int64")
    require(handle["example_hash"].dtype == np.dtype("uint64"), "Input hashes must be uint64")
    for name in ("beam_score", "raw_rerank_logit", "rerank_score"):
        require(handle[name].dtype == np.dtype("float32"), f"{name} must be float32")


def audit_group(output_dir: Path, manifest: dict, phase: str, group: str) -> dict:
    """Return acceptance after all checks; this API does not write any file."""
    output_dir = Path(output_dir)
    require(not any("incomplete" in p.name.lower() for p in output_dir.iterdir()), "Incomplete export present")
    _validate_manifest(manifest, phase, group)
    count = manifest["expected_test_samples"] if phase == "full" else manifest.get("smoke_samples", 65)
    require(isinstance(count, int) and 0 < count <= manifest["expected_test_samples"], "Invalid audit size")
    run_path = output_dir / "run.json"
    run = json.loads(run_path.read_text())
    _validate_run(run, manifest, phase, group, count)
    settings = [(c, m, setting_name(c, m)) for c, m in manifest["groups"][group]]
    totals = {name: {v: {} for v in ("beam", "raw", "fused")} for _, _, name in settings}
    hits = {name: 0 for _, _, name in settings}
    users = set()
    chunk_size = 4096
    with ExitStack() as stack:
        reference = stack.enter_context(h5py.File(manifest["reference_test"]["path"], "r"))
        require(reference["user_id"].shape[0] == manifest["expected_test_samples"], "Reference size mismatch")
        handles = {}
        for c, _, name in settings:
            handles[name] = stack.enter_context(h5py.File(output_dir / f"{name}.h5", "r"))
            _shape_check(handles[name], count, c)
        for start in range(0, count, chunk_size):
            end = min(count, start + chunk_size)
            identity = {k: reference[k][start:end] for k in ("user_id", "example_hash", "target_sid")}
            old_size = len(users)
            users.update(map(int, identity["user_id"]))
            require(len(users) - old_size == end - start, "Repeated evaluation user")
            shared_ids = shared_scores = None
            for c, _, name in settings:
                a = {k: handles[name][k][start:end] for k in FIELDS}
                for k, value in identity.items():
                    require(np.array_equal(a[k], value), f"Reference {k} mismatch: {name}")
                for k in ("beam_score", "raw_rerank_logit", "rerank_score"):
                    require(bool(np.isfinite(a[k]).all()), f"Nonfinite {k}: {name}")
                for k in ("beam_score", "rerank_score"):
                    require(bool((a[k][:, :-1] >= a[k][:, 1:]).all()), f"Unsorted {k}: {name}")
                beam = _packed(a["beam_sid"])
                rerank = _packed(a["rerank_sid"])
                target = _packed(a["target_sid"])
                sorted_beam = np.sort(beam, axis=1)
                require(bool((sorted_beam[:, 1:] != sorted_beam[:, :-1]).all()), f"Duplicate candidates: {name}")
                require(np.array_equal(sorted_beam, np.sort(rerank, axis=1)), f"Rerank is not a permutation: {name}")
                if group == "retrieval":
                    if shared_ids is None:
                        shared_ids, shared_scores = a["beam_sid"], a["beam_score"]
                    else:
                        require(np.array_equal(shared_ids, a["beam_sid"])
                                and np.array_equal(shared_scores, a["beam_score"]),
                                "Retrieval settings changed generated candidates/scores")
                matches = beam == target[:, None]
                hits[name] += int(matches.any(axis=1).sum())
                fused = (np.float32(manifest["beam_weight"]) * a["beam_score"]
                         + np.float32(manifest["beta"]) * a["raw_rerank_logit"])
                require(bool(np.isfinite(fused).all()), "Nonfinite fused scores")
                ks = [k for k in (5, 10, 20) if k <= c]
                for variant, scores in (("beam", None), ("raw", a["raw_rerank_logit"]), ("fused", fused)):
                    for metric, value in _metric_sums(_ranks(matches, scores), ks).items():
                        totals[name][variant][metric] = totals[name][variant].get(metric, 0) + value
    result_settings = {}
    for c, m, name in settings:
        metrics = {v: {k: value / count for k, value in values.items()} for v, values in totals[name].items()}
        result_settings[name] = {"C": c, "M": m, "coverage_at_C": hits[name] / count,
                                 "coverage_hits": hits[name], "metrics": metrics,
                                 "fusion_minus_beam": {k: metrics["fused"][k] - metrics["beam"][k]
                                                       for k in metrics["fused"]}}
    deltas = {}
    if group == "retrieval":
        center = result_settings["c20_m10"]["metrics"]
        for _, _, name in settings:
            if name != "c20_m10":
                deltas[name] = {v: {k: value - center[v][k] for k, value in values.items()}
                                for v, values in result_settings[name]["metrics"].items()}
    return {"status": "passed", "phase": phase, "group": group, "seed": 42,
            "checked_at": datetime.now(timezone.utc).isoformat(), "num_examples": count,
            "unique_users": len(users), "settings": result_settings, "minus_c20_m10": deltas,
            "candidate_sets_and_scores_shared_across_M": group == "retrieval",
            "independent_sort_and_outranker_ranks_verified": True,
            "fusion": {"beam_weight": manifest["beam_weight"], "rerank_weight": manifest["beta"],
                       "precision": "float32", "ties": "stable original beam order"},
            "artifacts": [{"path": str((output_dir / f"{name}.h5").resolve()),
                           "sha256": sha(output_dir / f"{name}.h5")} for _, _, name in settings],
            "run": {"path": str(run_path.resolve()), "sha256": sha(run_path)},
            "reference_test": manifest["reference_test"], "frozen_beta": manifest["frozen_beta"],
            "manifest_sha256": run.get("manifest_sha256"), "checkpoint": manifest["checkpoint"],
            "limitations": ["Single fixed training seed; descriptive paired mean differences, no new CI or tuning",
                            "Candidate generation/GT independence relies additionally on the runner smoke gate",
                            "Raw logits align to beam via the original exporter; no second model forward in this auditor",
                            "Runtime BF16 beta0.1 ranking is not the reported FP32 beta0.5 fusion",
                            "Reference file hash is inherited from the frozen manifest; identities/targets are reread here",
                            "Existing split overlap retained; M changes actual input length/candidate position"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--group", choices=tuple(GROUPS), required=True)
    args = parser.parse_args()
    require(not (args.output_dir / "acceptance.json").exists(), "Acceptance already exists")
    manifest = json.loads(args.manifest.read_text())
    manifest["_manifest_sha256"] = sha(args.manifest)
    report = audit_group(args.output_dir, manifest, args.phase, args.group)
    with (args.output_dir / "acceptance.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "phase": args.phase, "group": args.group,
                      "num_examples": report["num_examples"]}), flush=True)


if __name__ == "__main__":
    main()
