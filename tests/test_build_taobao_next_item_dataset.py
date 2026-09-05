import json
import pickle
from argparse import Namespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf

from scripts.build_taobao_next_item_dataset import (
    _is_eval_user,
    make_tfrecord_example,
    map_and_truncate_history,
    run_conversion,
)


def test_h1_filters_before_selecting_last_retained_item():
    sequence, removed = map_and_truncate_history(
        [10, 999, 20, 30],
        {10: 0, 20: 1, 30: 2},
        max_sequence_length=2,
    )
    np.testing.assert_array_equal(sequence, [1, 2])
    assert removed == 1
    assert sequence[-1] == 2


def test_l2_rejects_target_only_sequence_in_conversion(tmp_path):
    data_dir = tmp_path / "parquet"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    table = pa.table(
        {
            "129_1": pa.array([1, 1, 2, 2, 3], type=pa.int64()),
            "150_2_180": pa.array(
                [
                    [10, 999, 20],
                    [10, 999, 20],
                    [999, 10],
                    [999, 10],
                    [30],
                ],
                type=pa.list_(pa.int64()),
            ),
        }
    )
    pq.write_table(table, train_dir / "train-shard-000000.parquet")
    mapping_path = tmp_path / "item2row.pkl"
    with mapping_path.open("wb") as handle:
        pickle.dump({10: 0, 20: 1, 30: 2}, handle)

    output_root = tmp_path / "output"
    manifest = run_conversion(
        Namespace(
            split="train",
            data_dir=data_dir,
            item2row_path=mapping_path,
            output_root=output_root,
            max_sequence_length=1000,
            min_sequence_length=2,
            eval_ratio=0.0,
            samples_per_shard=2,
            batch_size=3,
            max_input_shards=None,
            max_input_rows=None,
        )
    )

    assert manifest["counts"]["source_rows_read"] == 5
    assert manifest["counts"]["unique_users_seen"] == 3
    assert manifest["counts"]["duplicate_exposure_rows_skipped"] == 2
    assert manifest["counts"]["below_minimum_after_mapping_dropped"] == 2
    assert manifest["counts"]["written_train"] == 1
    assert manifest["protocol_rules"]["exposure_target_used"] is False
    assert manifest["protocol_rules"]["click_label_used"] is False

    records = list(
        tf.data.TFRecordDataset(
            sorted(str(path) for path in (output_root / "training").glob("*.gz")),
            compression_type="GZIP",
        ).as_numpy_iterator()
    )
    assert len(records) == 1
    parsed = tf.train.Example.FromString(records[0]).features.feature
    assert set(parsed) == {"user_id", "sequence_data"}
    assert list(parsed["user_id"].int64_list.value) == [1]
    assert list(parsed["sequence_data"].int64_list.value) == [0, 1]

    saved_manifest = json.loads(
        (output_root / "protocol_manifest_train.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["protocol"] == "history-last-user-v1"


def test_example_schema_cannot_carry_exposure_target_or_click():
    example = make_tfrecord_example(7, np.asarray([3, 4], dtype=np.int64))
    fields = example.features.feature
    assert set(fields) == {"user_id", "sequence_data"}
    assert list(fields["sequence_data"].int64_list.value) == [3, 4]


def test_eval_assignment_is_deterministic_and_user_level():
    first = [_is_eval_user(user_id, 0.05) for user_id in range(1000)]
    second = [_is_eval_user(user_id, 0.05) for user_id in range(1000)]
    assert first == second
    assert 20 < sum(first) < 80


def test_converter_refuses_to_overwrite_existing_split(tmp_path):
    data_dir = tmp_path / "parquet"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "129_1": pa.array([1], type=pa.int64()),
                "150_2_180": pa.array([[10, 20]], type=pa.list_(pa.int64())),
            }
        ),
        train_dir / "train-shard-000000.parquet",
    )
    mapping_path = tmp_path / "item2row.pkl"
    with mapping_path.open("wb") as handle:
        pickle.dump({10: 0, 20: 1}, handle)
    args = Namespace(
        split="train",
        data_dir=data_dir,
        item2row_path=mapping_path,
        output_root=tmp_path / "output",
        max_sequence_length=1000,
        min_sequence_length=2,
        eval_ratio=0.0,
        samples_per_shard=2,
        batch_size=2,
        max_input_shards=None,
        max_input_rows=None,
    )
    run_conversion(args)

    try:
        run_conversion(args)
    except FileExistsError as error:
        assert "Refusing to overwrite" in str(error)
    else:
        raise AssertionError("Second conversion unexpectedly overwrote existing data")
