#!/usr/bin/env python3
"""Refuse training unless the frozen TAOBAO-MM data protocol is fully audited."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


EXPECTED_PROTOCOL = "history-last-user-v1"
EXPECTED_FIELDS = ["sequence_data", "user_id"]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantic-id-path", type=Path, required=True)
    parser.add_argument("--expected-protocol", default=EXPECTED_PROTOCOL)
    parser.add_argument("--num-hierarchies", type=int, required=True)
    parser.add_argument("--num-embeddings-per-hierarchy", type=int, required=True)
    parser.add_argument(
        "--source-file",
        type=Path,
        action="append",
        default=[],
        help="Relevant source/config file to fingerprint; may be repeated.",
    )
    parser.add_argument(
        "--initialization-checkpoint",
        type=Path,
        help="Optional teacher checkpoint to fingerprint for a staged run.",
    )
    parser.add_argument("--report-path", type=Path)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_with_sha256(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File to fingerprint is missing: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _assert_protocol_manifest(
    manifest: dict[str, Any], split: str, expected_protocol: str
) -> None:
    assert manifest["status"] == "complete", f"{split} manifest is not complete"
    assert manifest["split"] == split, f"Unexpected split in {split} manifest"
    assert manifest["protocol"] == expected_protocol, f"Wrong {split} protocol"
    assert manifest["partial_run"] is False, f"{split} manifest is a partial run"
    assert manifest["source"]["columns_read"] == ["129_1", "150_2_180"]

    rules = manifest["protocol_rules"]
    assert rules["sample_unit"] == "one user per official split"
    assert rules["history_item_filter"] == (
        "remove items missing from item2row before target selection"
    )
    assert rules["ground_truth"] == "last retained sequence_data item"
    assert rules["model_input"] == "all retained sequence_data items except the last"
    assert rules["exposure_target_used"] is False
    assert rules["click_label_used"] is False
    assert sorted(rules["tfrecord_fields"]) == EXPECTED_FIELDS
    assert int(rules["minimum_sequence_length_including_target"]) == 2
    if split == "train":
        assert rules["validation_split"] == "MD5(user_id) bucket, ratio=0.05"


def _assert_file_identity(entry: dict[str, Any]) -> Path:
    path = Path(entry["path"]).resolve()
    assert path.is_file(), f"Manifest output is missing: {path}"
    stat = path.stat()
    assert stat.st_size == int(entry["size_bytes"]), f"Size changed: {path}"
    assert stat.st_mtime_ns == int(entry["mtime_ns"]), f"mtime changed: {path}"
    return path


def _assert_manifest_outputs(manifest: dict[str, Any], data_root: Path) -> None:
    for output in manifest["outputs"].values():
        files = output["files"]
        assert len(files) == int(output["shards"])
        for entry in files:
            path = _assert_file_identity(entry)
            assert path.is_relative_to(data_root), f"Output escapes data root: {path}"


def _assert_audit(
    audit: dict[str, Any],
    train_manifest: dict[str, Any],
    test_manifest: dict[str, Any],
    expected_protocol: str,
) -> None:
    assert audit["status"] == "passed", "Protocol audit did not pass"
    assert audit["protocol"] == expected_protocol
    assert audit["partial_run"] is False, "Protocol audit is only partial"
    checks = audit["checks"]
    assert checks["manifest_consistency"] is True
    assert checks["exact_schema"] == EXPECTED_FIELDS
    assert checks["minimum_sequence_length"] == 2
    assert checks["train_validation_user_disjoint"] is True
    assert checks["within_destination_user_unique"] is True

    expected_samples = {
        "train": int(train_manifest["outputs"]["train"]["samples"]),
        "evaluation": int(train_manifest["outputs"]["evaluation"]["samples"]),
        "test": int(test_manifest["outputs"]["test"]["samples"]),
    }
    for destination, expected in expected_samples.items():
        result = audit["destinations"][destination]
        assert int(result["records"]) == expected
        assert int(result["unique_users"]) == expected
        assert int(result["duplicate_users"]) == 0
        assert int(result["below_minimum_length"]) == 0


def verify(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    semantic_id_path = args.semantic_id_path.resolve()
    if not semantic_id_path.is_file():
        raise FileNotFoundError(f"Semantic-ID mapping is missing: {semantic_id_path}")

    incomplete = sorted(data_root.glob(".protocol_manifest_*.json.incomplete"))
    assert not incomplete, f"Incomplete conversion marker(s) found: {incomplete}"

    train_manifest_path = data_root / "protocol_manifest_train.json"
    test_manifest_path = data_root / "protocol_manifest_test.json"
    audit_path = data_root / "protocol_audit.json"
    train_manifest = _load_json(train_manifest_path)
    test_manifest = _load_json(test_manifest_path)
    audit = _load_json(audit_path)
    _assert_protocol_manifest(train_manifest, "train", args.expected_protocol)
    _assert_protocol_manifest(test_manifest, "test", args.expected_protocol)
    _assert_manifest_outputs(train_manifest, data_root)
    _assert_manifest_outputs(test_manifest, data_root)
    _assert_audit(audit, train_manifest, test_manifest, args.expected_protocol)

    assert train_manifest["converter"]["sha256"] == test_manifest["converter"]["sha256"]
    train_mapping = train_manifest["mapping"]
    test_mapping = test_manifest["mapping"]
    assert train_mapping == test_mapping, "Train/test used different item2row mappings"
    mapping_path = _assert_file_identity(train_mapping)

    audit_mtime = audit_path.stat().st_mtime_ns
    assert audit_mtime >= train_manifest_path.stat().st_mtime_ns
    assert audit_mtime >= test_manifest_path.stat().st_mtime_ns

    semantic_ids = np.load(semantic_id_path, mmap_mode="r", allow_pickle=False)
    assert semantic_ids.ndim == 2, semantic_ids.shape
    assert semantic_ids.shape[1] == args.num_hierarchies, semantic_ids.shape
    semantic_minimum = int(semantic_ids.min())
    semantic_maximum = int(semantic_ids.max())
    assert semantic_minimum >= 0
    assert semantic_maximum < args.num_embeddings_per_hierarchy

    destination_ranges: dict[str, dict[str, int]] = {}
    for destination, result in audit["destinations"].items():
        minimum_item_row = result.get("minimum_item_row")
        maximum_item_row = result.get("maximum_item_row")
        assert minimum_item_row is not None, f"Audit lacks item range for {destination}"
        assert maximum_item_row is not None, f"Audit lacks item range for {destination}"
        assert int(minimum_item_row) >= 0
        assert int(maximum_item_row) < semantic_ids.shape[0]
        destination_ranges[destination] = {
            "minimum_item_row": int(minimum_item_row),
            "maximum_item_row": int(maximum_item_row),
        }

    semantic_stat = semantic_id_path.stat()
    source_files = [
        _identity_with_sha256(Path(path))
        for path in getattr(args, "source_file", [])
    ]
    initialization_checkpoint = getattr(args, "initialization_checkpoint", None)
    report = {
        "status": "passed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": args.expected_protocol,
        "data_root": str(data_root),
        "counts": {
            name: int(result["records"])
            for name, result in audit["destinations"].items()
        },
        "item_row_ranges": destination_ranges,
        "semantic_id": {
            "path": str(semantic_id_path),
            "sha256": _sha256(semantic_id_path),
            "size_bytes": semantic_stat.st_size,
            "mtime_ns": semantic_stat.st_mtime_ns,
            "shape": list(semantic_ids.shape),
            "dtype": str(semantic_ids.dtype),
            "minimum_code": semantic_minimum,
            "maximum_code": semantic_maximum,
        },
        "item2row_mapping": {
            **train_mapping,
            "path": str(mapping_path),
            "sha256": _sha256(mapping_path),
        },
        "source_manifests": {
            "train": {
                "path": str(train_manifest_path),
                "sha256": _sha256(train_manifest_path),
            },
            "test": {
                "path": str(test_manifest_path),
                "sha256": _sha256(test_manifest_path),
            },
            "audit": {
                "path": str(audit_path),
                "sha256": _sha256(audit_path),
            },
        },
        "source_files": source_files,
        "initialization_checkpoint": (
            _identity_with_sha256(Path(initialization_checkpoint))
            if initialization_checkpoint is not None
            else None
        ),
    }
    if args.report_path:
        report_path = args.report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite preflight report: {report_path}")
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    verify(_parse_args())


if __name__ == "__main__":
    main()
