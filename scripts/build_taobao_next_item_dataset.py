#!/usr/bin/env python3
"""Build a versioned TAOBAO-MM next-item dataset from joined Parquet shards.

Protocol ``history-last-user-v1`` is deliberately independent of the exposure
target (column ``205``) and click label (column ``label_0``):

1. keep at most one row per user in each official split;
2. remove history items without a Semantic-ID/SCL mapping;
3. keep the most recent ``max_sequence_length`` mapped items;
4. require at least two mapped items;
5. write the full retained sequence.  The model uses its final item as the
   target and all preceding items as input history.

The converter never overwrites an existing split and writes a protocol
manifest alongside the generated TFRecords.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf


PROTOCOL_NAME = "history-last-user-v1"
DEFAULT_MAX_SEQUENCE_LENGTH = 1000
DEFAULT_MIN_SEQUENCE_LENGTH = 2
DEFAULT_EVAL_RATIO = 0.05
DEFAULT_SAMPLES_PER_SHARD = 200_000
DEFAULT_BATCH_SIZE = 8192


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deduplicated TAOBAO-MM history-last next-item dataset."
    )
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing train/ and test/ Parquet shard directories.",
    )
    parser.add_argument(
        "--item2row-path",
        type=Path,
        required=True,
        help="Pickle mapping raw history item IDs to Semantic-ID row indices.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New versioned root; training/evaluation/testing are created below it.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Keep this many most recent mapped history items at most.",
    )
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=DEFAULT_MIN_SEQUENCE_LENGTH,
        help="Minimum retained length including the final target item (must be >=2).",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=DEFAULT_EVAL_RATIO,
        help="User-hash validation fraction carved from the official train split.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=DEFAULT_SAMPLES_PER_SHARD,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-input-shards",
        type=int,
        default=None,
        help="Testing only: stop after this many input shards.",
    )
    parser.add_argument(
        "--max-input-rows",
        type=int,
        default=None,
        help="Testing only: stop after this many source rows.",
    )
    args = parser.parse_args(argv)
    _validate_args(args, parser.error)
    return args


def _validate_args(args: argparse.Namespace, fail) -> None:
    if args.min_sequence_length < 2:
        fail("--min-sequence-length must be at least 2 (one input plus one target).")
    if args.max_sequence_length < args.min_sequence_length:
        fail("--max-sequence-length must be >= --min-sequence-length.")
    if args.samples_per_shard <= 0:
        fail("--samples-per-shard must be positive.")
    if args.batch_size <= 0:
        fail("--batch-size must be positive.")
    if not 0.0 <= args.eval_ratio < 1.0:
        fail("--eval-ratio must be in [0, 1).")
    if args.split == "test" and args.eval_ratio != DEFAULT_EVAL_RATIO:
        fail("--eval-ratio only applies to --split train.")
    if args.max_input_shards is not None and args.max_input_shards <= 0:
        fail("--max-input-shards must be positive.")
    if args.max_input_rows is not None and args.max_input_rows <= 0:
        fail("--max-input-rows must be positive.")


def _is_eval_user(user_id: int, eval_ratio: float) -> bool:
    if eval_ratio <= 0.0:
        return False
    digest = hashlib.md5(str(user_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10_000
    return bucket < int(eval_ratio * 10_000)


def map_and_truncate_history(
    raw_history: Optional[Sequence[int]],
    item2row: Mapping[int, int],
    max_sequence_length: int,
) -> tuple[np.ndarray, int]:
    """Apply H1: filter unmapped items, then retain the most recent suffix."""
    if not raw_history:
        return np.empty(0, dtype=np.int64), 0

    mapped: list[int] = []
    removed = 0
    for raw_item in raw_history:
        row = item2row.get(int(raw_item))
        if row is None:
            removed += 1
        else:
            mapped.append(int(row))

    if len(mapped) > max_sequence_length:
        mapped = mapped[-max_sequence_length:]
    return np.asarray(mapped, dtype=np.int64), removed


def make_tfrecord_example(user_id: int, sequence_data: np.ndarray) -> tf.train.Example:
    """Serialize only fields needed by the frozen next-item protocol."""
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "user_id": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(user_id)])
                ),
                "sequence_data": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=sequence_data.astype(np.int64, copy=False).tolist()
                    )
                ),
            }
        )
    )


class ShardedTFRecordWriter:
    def __init__(self, output_dir: Path, prefix: str, samples_per_shard: int):
        self.output_dir = output_dir
        self.prefix = prefix
        self.samples_per_shard = samples_per_shard
        self.options = tf.io.TFRecordOptions(compression_type="GZIP")
        self.writer: Optional[tf.io.TFRecordWriter] = None
        self.num_shards = 0
        self.samples_in_shard = 0
        self.total_samples = 0
        self.paths: list[Path] = []

    def _open_next(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.prefix}_{self.num_shards:06d}.tfrecord.gz"
        self.writer = tf.io.TFRecordWriter(str(path), options=self.options)
        self.paths.append(path)
        self.num_shards += 1
        self.samples_in_shard = 0

    def write(self, serialized: bytes) -> None:
        if self.writer is None or self.samples_in_shard >= self.samples_per_shard:
            self.close_current()
            self._open_next()
        assert self.writer is not None
        self.writer.write(serialized)
        self.samples_in_shard += 1
        self.total_samples += 1

    def close_current(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self) -> None:
        self.close_current()


def _input_shards(data_dir: Path, split: str) -> list[Path]:
    return sorted((data_dir / split).glob(f"{split}-shard-*.parquet"))


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _percentile_from_histogram(histogram: np.ndarray, percentile: float) -> int:
    total = int(histogram.sum())
    if total == 0:
        return 0
    rank = max(1, math.ceil(total * percentile))
    return int(np.searchsorted(np.cumsum(histogram), rank, side="left"))


def _manifest_name(split: str) -> str:
    return f"protocol_manifest_{split}.json"


def _preflight_output(output_root: Path, split: str) -> Path:
    targets = (
        ((output_root / "training"), "grid_train_*.tfrecord.gz"),
        ((output_root / "evaluation"), "grid_eval_*.tfrecord.gz"),
    ) if split == "train" else (
        ((output_root / "testing"), "grid_test_*.tfrecord.gz"),
    )

    existing: list[Path] = []
    for directory, pattern in targets:
        if directory.exists():
            existing.extend(directory.glob(pattern))
    manifest_path = output_root / _manifest_name(split)
    incomplete_path = output_root / f".{_manifest_name(split)}.incomplete"
    if manifest_path.exists():
        existing.append(manifest_path)
    if incomplete_path.exists():
        existing.append(incomplete_path)
    if existing:
        preview = ", ".join(str(path) for path in sorted(existing)[:5])
        raise FileExistsError(
            f"Refusing to overwrite existing {split} outputs under {output_root}: {preview}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return incomplete_path


def run_conversion(args: argparse.Namespace) -> dict[str, Any]:
    """Run one official split conversion and return the written manifest."""
    _validate_args(args, lambda message: (_ for _ in ()).throw(ValueError(message)))
    data_dir = Path(args.data_dir).resolve()
    item2row_path = Path(args.item2row_path).resolve()
    output_root = Path(args.output_root).resolve()
    if not item2row_path.is_file():
        raise FileNotFoundError(f"item2row mapping not found: {item2row_path}")

    shards = _input_shards(data_dir, args.split)
    if not shards:
        raise FileNotFoundError(f"No {args.split} Parquet shards under {data_dir}")
    if args.max_input_shards is not None:
        shards = shards[: args.max_input_shards]

    incomplete_path = _preflight_output(output_root, args.split)
    started_at = datetime.now(timezone.utc).isoformat()
    incomplete_path.write_text(
        json.dumps(
            {"status": "incomplete", "protocol": PROTOCOL_NAME, "split": args.split},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with item2row_path.open("rb") as handle:
        item2row = pickle.load(handle)

    writers: dict[str, ShardedTFRecordWriter]
    if args.split == "train":
        writers = {
            "train": ShardedTFRecordWriter(
                output_root / "training", "grid_train", args.samples_per_shard
            ),
            "evaluation": ShardedTFRecordWriter(
                output_root / "evaluation", "grid_eval", args.samples_per_shard
            ),
        }
    else:
        writers = {
            "test": ShardedTFRecordWriter(
                output_root / "testing", "grid_test", args.samples_per_shard
            )
        }

    seen_users: set[int] = set()
    sequence_histogram = np.zeros(args.max_sequence_length + 1, dtype=np.int64)
    counters = {
        "source_rows_read": 0,
        "unique_users_seen": 0,
        "duplicate_exposure_rows_skipped": 0,
        "empty_raw_history_dropped": 0,
        "below_minimum_after_mapping_dropped": 0,
        "unmapped_history_items_removed": 0,
        "written_total": 0,
        "written_train": 0,
        "written_evaluation": 0,
        "written_test": 0,
        "retained_sequence_items": 0,
    }

    stop = False
    try:
        for shard_index, shard_path in enumerate(shards, start=1):
            parquet_file = pq.ParquetFile(shard_path)
            for batch in parquet_file.iter_batches(
                batch_size=args.batch_size,
                columns=["129_1", "150_2_180"],
            ):
                if args.max_input_rows is not None:
                    remaining = args.max_input_rows - counters["source_rows_read"]
                    if remaining <= 0:
                        stop = True
                        break
                    if len(batch) > remaining:
                        batch = batch.slice(0, remaining)
                        stop = True

                user_ids = batch.column(0).to_numpy(zero_copy_only=False)
                raw_histories = batch.column(1)
                counters["source_rows_read"] += len(batch)

                for row_index, raw_user_id in enumerate(user_ids):
                    user_id = int(raw_user_id)
                    if user_id in seen_users:
                        counters["duplicate_exposure_rows_skipped"] += 1
                        continue
                    seen_users.add(user_id)

                    raw_history = raw_histories[row_index].as_py()
                    if not raw_history:
                        counters["empty_raw_history_dropped"] += 1
                        continue
                    sequence_data, removed = map_and_truncate_history(
                        raw_history,
                        item2row,
                        args.max_sequence_length,
                    )
                    counters["unmapped_history_items_removed"] += removed
                    if len(sequence_data) < args.min_sequence_length:
                        counters["below_minimum_after_mapping_dropped"] += 1
                        continue

                    example = make_tfrecord_example(user_id, sequence_data)
                    serialized = example.SerializeToString()
                    if args.split == "train":
                        destination = (
                            "evaluation"
                            if _is_eval_user(user_id, args.eval_ratio)
                            else "train"
                        )
                    else:
                        destination = "test"
                    writers[destination].write(serialized)
                    counters[f"written_{destination}"] += 1
                    counters["written_total"] += 1
                    counters["retained_sequence_items"] += len(sequence_data)
                    sequence_histogram[len(sequence_data)] += 1

                if stop:
                    break
            print(
                f"Finished input shard {shard_index}/{len(shards)}: {shard_path.name}; "
                f"rows={counters['source_rows_read']:,}, users={len(seen_users):,}, "
                f"written={counters['written_total']:,}",
                flush=True,
            )
            if stop:
                break
    finally:
        for writer in writers.values():
            writer.close()

    counters["unique_users_seen"] = len(seen_users)
    output_files = [path for writer in writers.values() for path in writer.paths]
    mean_length = (
        counters["retained_sequence_items"] / counters["written_total"]
        if counters["written_total"]
        else 0.0
    )
    script_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "status": "complete",
        "protocol": PROTOCOL_NAME,
        "protocol_rules": {
            "sample_unit": "one user per official split",
            "history_item_filter": "remove items missing from item2row before target selection",
            "ground_truth": "last retained sequence_data item",
            "model_input": "all retained sequence_data items except the last",
            "minimum_sequence_length_including_target": args.min_sequence_length,
            "maximum_sequence_length_keep_tail": args.max_sequence_length,
            "exposure_target_used": False,
            "click_label_used": False,
            "tfrecord_fields": ["user_id", "sequence_data"],
            "validation_split": (
                f"MD5(user_id) bucket, ratio={args.eval_ratio}"
                if args.split == "train"
                else None
            ),
        },
        "split": args.split,
        "partial_run": bool(
            args.max_input_shards is not None or args.max_input_rows is not None
        ),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "data_dir": str(data_dir),
            "num_parquet_shards_processed": shard_index,
            "parquet_shards": [_file_identity(path) for path in shards[:shard_index]],
            "columns_read": ["129_1", "150_2_180"],
        },
        "mapping": _file_identity(item2row_path),
        "converter": {
            **_file_identity(script_path),
            "sha256": _sha256(script_path),
        },
        "parameters": {
            "samples_per_shard": args.samples_per_shard,
            "batch_size": args.batch_size,
            "max_input_shards": args.max_input_shards,
            "max_input_rows": args.max_input_rows,
        },
        "counts": counters,
        "sequence_length": {
            "mean": mean_length,
            "p50": _percentile_from_histogram(sequence_histogram, 0.50),
            "p95": _percentile_from_histogram(sequence_histogram, 0.95),
        },
        "outputs": {
            name: {
                "samples": writer.total_samples,
                "shards": writer.num_shards,
                "files": [_file_identity(path) for path in writer.paths],
            }
            for name, writer in writers.items()
        },
    }

    manifest_path = output_root / _manifest_name(args.split)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    incomplete_path.unlink()
    print(json.dumps({"manifest": str(manifest_path), "counts": counters}, indent=2))
    return manifest


def main() -> None:
    run_conversion(_parse_args())


if __name__ == "__main__":
    main()
