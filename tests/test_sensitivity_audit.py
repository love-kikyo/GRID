import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.audit_inference_sensitivity import GROUPS, _ranks, audit_group, main, sha


def fixture(tmp_path, group="retrieval", n=7):
    root = tmp_path / "outputs"
    root.mkdir()
    uid = np.arange(n, dtype=np.int64)
    hashes = np.stack([uid, uid + 20], axis=1).astype(np.uint64)
    # Rows 0..4 hit different positions; remaining rows are misses.
    target = np.ones((n, 4), dtype=np.int16)
    target[:, -1] = np.array([1, 5, 10, 15, 20] + [100] * (n - 5))[:n]
    reference = tmp_path / "reference.h5"
    with h5py.File(reference, "w") as f:
        f["user_id"] = uid
        f["example_hash"] = hashes
        f["target_sid"] = target
    beta = tmp_path / "selection.json"
    beta.write_text(json.dumps({"input_split": "val", "selection_scope": "validation_only",
                               "input_checkpoint_sha256": "checkpoint",
                               "selected": {"beam_weight": 1., "rerank_weight": .5}}))
    manifest = {"seed": 42, "groups": GROUPS, "beta": .5, "beam_weight": 1.,
                "expected_test_samples": n, "smoke_samples": n,
                "checkpoint": {"path": "/unused.ckpt", "sha256": "checkpoint"},
                "reference_test": {"path": str(reference), "sha256": sha(reference)},
                "frozen_beta": {"path": str(beta), "sha256": sha(beta)},
                "_manifest_sha256": "manifest"}
    run = {"status": "complete", "phase": "smoke", "group": group, "processed": n,
           "expected_samples": n, "manifest_sha256": "manifest", "smoke_gate": {"status": "passed"},
           "hardware": {"name": "NVIDIA RTX 4090"},
           "cpu_budget": {"allowed_cpus": [1, 2, 3, 4], "allocated_cpus": 4, "library_threads": 1},
           "cpu_process_tree_samples": [{"allowed_cpus": [1, 2, 3, 4], "processes": [{"pid": 1}]}] * 2}
    (root / "run.json").write_text(json.dumps(run))
    for c, m in GROUPS[group]:
        candidates = np.ones((n, c, 4), dtype=np.int16)
        candidates[:, :, -1] = np.arange(1, c + 1)
        beam_score = np.broadcast_to(-np.arange(c, dtype=np.float32), (n, c)).copy()
        raw = np.broadcast_to(np.arange(c, dtype=np.float32) * 3, (n, c)).copy()
        runtime = beam_score + np.float32(.1) * raw
        order = np.argsort(-runtime, axis=1, kind="stable")
        with h5py.File(root / f"c{c}_m{m}.h5", "w") as f:
            f.attrs["num_samples"] = n
            f["user_id"], f["example_hash"], f["target_sid"] = uid, hashes, target
            f["beam_sid"], f["beam_score"], f["raw_rerank_logit"] = candidates, beam_score, raw
            f["rerank_sid"] = np.take_along_axis(candidates, order[:, :, None], axis=1)
            f["rerank_score"] = np.take_along_axis(runtime, order, axis=1)
    return root, manifest


def test_retrieval_audit_metrics_and_independent_ranks(tmp_path):
    root, manifest = fixture(tmp_path)
    report = audit_group(root, manifest, "smoke", "retrieval")
    item = report["settings"]["c20_m10"]
    assert item["coverage_hits"] == 5
    assert item["metrics"]["beam"]["recall@5"] == 2 / 7
    assert item["metrics"]["fused"]["recall@5"] == 1 / 7
    assert item["metrics"]["beam"]["ndcg@5"] == pytest.approx((1 + 1 / np.log2(6)) / 7)
    assert report["independent_sort_and_outranker_ranks_verified"]
    assert all(v == 0 for d in report["minus_c20_m10"].values() for a in d.values() for v in a.values())
    assert not (root / "acceptance.json").exists()  # API is read-only.


def test_beam10_does_not_report_at20(tmp_path):
    root, manifest = fixture(tmp_path, "beam")
    report = audit_group(root, manifest, "smoke", "beam")
    assert "recall@20" not in report["settings"]["c10_m10"]["metrics"]["fused"]
    assert "recall@20" in report["settings"]["c40_m10"]["metrics"]["fused"]
    assert report["settings"]["c10_m10"]["coverage_hits"] == 3


def test_stable_tie_ranks_and_miss():
    matches = np.array([[False, True, False], [False, False, False]])
    np.testing.assert_array_equal(_ranks(matches, np.ones((2, 3), np.float32)), [2, 4])


@pytest.mark.parametrize("mutation", ["duplicate", "range", "nan", "beam_order", "permutation", "identity", "shared_beam"])
def test_reject_corrupt_export(tmp_path, mutation):
    root, manifest = fixture(tmp_path)
    with h5py.File(root / "c20_m0.h5", "r+") as f:
        if mutation == "duplicate":
            f["beam_sid"][0, 1] = f["beam_sid"][0, 0]
        elif mutation == "range":
            f["target_sid"][0, 0] = 0
        elif mutation == "nan":
            f["raw_rerank_logit"][0, 0] = np.nan
        elif mutation == "beam_order":
            f["beam_score"][0, 1] = 1
        elif mutation == "permutation":
            f["rerank_sid"][0, 0, -1] = 500
        elif mutation == "identity":
            f["user_id"][0] = 1000
        elif mutation == "shared_beam":
            f["beam_score"][0, 0] = .5  # still descending, but not shared.
    with pytest.raises(ValueError):
        audit_group(root, manifest, "smoke", "retrieval")


@pytest.mark.parametrize("key,value", [("status", "running"), ("processed", 6),
                                       ("manifest_sha256", "changed"), ("smoke_gate", {}),
                                       ("hardware", {"name": "3090"}), ("cpu_process_tree_samples", [])])
def test_reject_bad_run_metadata(tmp_path, key, value):
    root, manifest = fixture(tmp_path)
    run = json.loads((root / "run.json").read_text())
    run[key] = value
    (root / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError):
        audit_group(root, manifest, "smoke", "retrieval")


@pytest.mark.parametrize("filename", ["c20_m0.h5.incomplete.123", "INCOMPLETE"])
def test_reject_incomplete(tmp_path, filename):
    root, manifest = fixture(tmp_path)
    (root / filename).write_text("{}")
    with pytest.raises(ValueError):
        audit_group(root, manifest, "smoke", "retrieval")


def test_readonly_reaudit_allows_existing_receipt(tmp_path):
    root, manifest = fixture(tmp_path)
    report = audit_group(root, manifest, "smoke", "retrieval")
    receipt = root / "acceptance.json"
    receipt.write_text(json.dumps(report))
    previous = receipt.read_bytes()
    repeated = audit_group(root, manifest, "smoke", "retrieval")
    assert repeated["settings"] == report["settings"]
    assert receipt.read_bytes() == previous


def test_full_runner_accepts_saved_smoke_gate(tmp_path):
    from scripts.run_inference_sensitivity import require_gate

    root, manifest = fixture(tmp_path, n=65)
    report = audit_group(root, manifest, "smoke", "retrieval")
    receipt = root / "acceptance.json"
    receipt.write_text(json.dumps(report))
    previous = receipt.read_bytes()

    record, gate = require_gate(manifest, "retrieval", root)

    assert record == {"path": str(receipt), "sha256": sha(receipt)}
    assert gate["status"] == "passed"
    assert receipt.read_bytes() == previous


def test_cli_refuses_existing_receipt_without_touching_it(tmp_path, monkeypatch):
    root, manifest = fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    receipt = root / "acceptance.json"
    receipt.write_text('{"preserved":true}\n')
    previous = receipt.read_bytes()
    monkeypatch.setattr("sys.argv", ["audit_inference_sensitivity", "--manifest", str(manifest_path),
                                    "--output-dir", str(root), "--phase", "smoke", "--group", "retrieval"])
    with pytest.raises(ValueError, match="Acceptance already exists"):
        main()
    assert receipt.read_bytes() == previous


def test_selection_hash_and_checkpoint_are_bound(tmp_path):
    root, manifest = fixture(tmp_path)
    manifest["checkpoint"]["sha256"] = "another-checkpoint"
    with pytest.raises(ValueError, match="checkpoint"):
        audit_group(root, manifest, "smoke", "retrieval")
    manifest["checkpoint"]["sha256"] = "checkpoint"
    Path(manifest["frozen_beta"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="Selection hash"):
        audit_group(root, manifest, "smoke", "retrieval")


def test_duplicate_reference_users_rejected(tmp_path):
    root, manifest = fixture(tmp_path)
    with h5py.File(manifest["reference_test"]["path"], "r+") as f:
        f["user_id"][1] = f["user_id"][0]
    with pytest.raises(ValueError, match="Repeated evaluation user"):
        audit_group(root, manifest, "smoke", "retrieval")


def test_full_phase_requires_full_count(tmp_path):
    root, manifest = fixture(tmp_path)
    run = json.loads((root / "run.json").read_text())
    run["phase"] = "full"
    (root / "run.json").write_text(json.dumps(run))
    assert audit_group(root, manifest, "full", "retrieval")["num_examples"] == 7
    manifest["expected_test_samples"] = 8
    with pytest.raises(ValueError, match="count mismatch"):
        audit_group(root, manifest, "full", "retrieval")
