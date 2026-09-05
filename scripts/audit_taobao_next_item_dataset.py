#!/usr/bin/env python3
"""Audit manifests, TFRecord schema, lengths, counts, and user uniqueness."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import tensorflow as tf


EXPECTED_FIELDS = {"user_id", "sequence_data"}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-protocol", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--require-partial-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def _load_manifest(data_root: Path, split: str) -> dict[str, Any]:
    path = data_root / f"protocol_manifest_{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing protocol manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_manifest(
    manifest: dict[str, Any],
    expected_protocol: str,
    require_partial_run: bool,
) -> None:
    assert manifest["status"] == "complete"
    assert manifest["protocol"] == expected_protocol
    assert bool(manifest["partial_run"]) is require_partial_run
    rules = manifest["protocol_rules"]
    assert rules["sample_unit"] == "one user per official split"
    assert rules["exposure_target_used"] is False
    assert rules["click_label_used"] is False
    assert set(rules["tfrecord_fields"]) == EXPECTED_FIELDS
    assert int(rules["minimum_sequence_length_including_target"]) >= 2

    counts = manifest["counts"]
    assert counts["source_rows_read"] == (
        counts["unique_users_seen"] + counts["duplicate_exposure_rows_skipped"]
    )
    assert counts["written_total"] == (
        counts["written_train"]
        + counts["written_evaluation"]
        + counts["written_test"]
    )
    output_total = sum(int(output["samples"]) for output in manifest["outputs"].values())
    assert output_total == counts["written_total"]


def _manifest_files(output: dict[str, Any]) -> list[Path]:
    files = [Path(entry["path"]) for entry in output["files"]]
    assert len(files) == int(output["shards"])
    for path, entry in zip(files, output["files"]):
        assert path.is_file(), path
        stat = path.stat()
        assert stat.st_size == int(entry["size_bytes"]), path
        assert stat.st_mtime_ns == int(entry["mtime_ns"]), path
    return files


def _assert_raw_schema(files: list[Path]) -> None:
    for path in files:
        dataset = tf.data.TFRecordDataset([str(path)], compression_type="GZIP")
        first_record = next(iter(dataset.take(1)), None)
        assert first_record is not None, f"Empty TFRecord shard: {path}"
        example = tf.train.Example.FromString(bytes(first_record.numpy()))
        assert set(example.features.feature) == EXPECTED_FIELDS, path


def _scan_records(
    files: list[Path],
    expected_count: int,
    minimum_length: int,
    batch_size: int,
    forbidden_users: Optional[set[int]] = None,
) -> tuple[dict[str, int], set[int]]:
    _assert_raw_schema(files)
    dataset = tf.data.TFRecordDataset(
        [str(path) for path in files],
        compression_type="GZIP",
        num_parallel_reads=tf.data.AUTOTUNE,
    ).batch(batch_size)
    description = {
        "user_id": tf.io.FixedLenFeature([], tf.int64),
        "sequence_data": tf.io.VarLenFeature(tf.int64),
    }

    users_seen: set[int] = set()
    num_records = 0
    duplicate_users = 0
    forbidden_user_hits = 0
    below_minimum = 0
    sequence_items = 0
    minimum_item_row: Optional[int] = None
    maximum_item_row: Optional[int] = None
    for serialized_batch in dataset:
        parsed = tf.io.parse_example(serialized_batch, description)
        user_ids = parsed["user_id"].numpy().astype(np.int64, copy=False)
        sparse_sequence = parsed["sequence_data"]
        row_ids = sparse_sequence.indices[:, 0].numpy()
        item_rows = sparse_sequence.values.numpy().astype(np.int64, copy=False)
        lengths = np.bincount(row_ids, minlength=len(user_ids))
        below_minimum += int(np.count_nonzero(lengths < minimum_length))
        sequence_items += int(item_rows.size)
        if item_rows.size:
            batch_minimum = int(item_rows.min())
            batch_maximum = int(item_rows.max())
            minimum_item_row = (
                batch_minimum
                if minimum_item_row is None
                else min(minimum_item_row, batch_minimum)
            )
            maximum_item_row = (
                batch_maximum
                if maximum_item_row is None
                else max(maximum_item_row, batch_maximum)
            )

        for user_id_value in user_ids:
            user_id = int(user_id_value)
            if user_id in users_seen:
                duplicate_users += 1
            else:
                users_seen.add(user_id)
            if forbidden_users is not None and user_id in forbidden_users:
                forbidden_user_hits += 1
        num_records += len(user_ids)

    assert num_records == expected_count, (num_records, expected_count)
    assert duplicate_users == 0, duplicate_users
    assert forbidden_user_hits == 0, forbidden_user_hits
    assert below_minimum == 0, below_minimum
    return {
        "records": num_records,
        "unique_users": len(users_seen),
        "duplicate_users": duplicate_users,
        "forbidden_user_hits": forbidden_user_hits,
        "below_minimum_length": below_minimum,
        "sequence_items": sequence_items,
        "minimum_item_row": minimum_item_row,
        "maximum_item_row": maximum_item_row,
    }, users_seen


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    train_manifest = _load_manifest(data_root, "train")
    test_manifest = _load_manifest(data_root, "test")
    _validate_manifest(train_manifest, args.expected_protocol, args.require_partial_run)
    _validate_manifest(test_manifest, args.expected_protocol, args.require_partial_run)

    min_train = int(
        train_manifest["protocol_rules"]["minimum_sequence_length_including_target"]
    )
    min_test = int(
        test_manifest["protocol_rules"]["minimum_sequence_length_including_target"]
    )
    train_files = _manifest_files(train_manifest["outputs"]["train"])
    eval_files = _manifest_files(train_manifest["outputs"]["evaluation"])
    test_files = _manifest_files(test_manifest["outputs"]["test"])

    train_result, train_users = _scan_records(
        train_files,
        int(train_manifest["outputs"]["train"]["samples"]),
        min_train,
        args.batch_size,
    )
    eval_result, _ = _scan_records(
        eval_files,
        int(train_manifest["outputs"]["evaluation"]["samples"]),
        min_train,
        args.batch_size,
        forbidden_users=train_users,
    )
    del train_users
    test_result, _ = _scan_records(
        test_files,
        int(test_manifest["outputs"]["test"]["samples"]),
        min_test,
        args.batch_size,
    )

    report = {
        "status": "passed",
        "protocol": args.expected_protocol,
        "partial_run": args.require_partial_run,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "checks": {
            "manifest_consistency": True,
            "exact_schema": sorted(EXPECTED_FIELDS),
            "minimum_sequence_length": min(min_train, min_test),
            "train_validation_user_disjoint": True,
            "within_destination_user_unique": True,
        },
        "destinations": {
            "train": train_result,
            "evaluation": eval_result,
            "test": test_result,
        },
    }
    report_name = "protocol_audit_partial.json" if args.require_partial_run else "protocol_audit.json"
    report_path = data_root / report_name
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), **report}, indent=2))
    return report


def main() -> None:
    run_audit(_parse_args())


if __name__ == "__main__":
    main()
