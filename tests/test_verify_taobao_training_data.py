import json
from argparse import Namespace

import numpy as np
import pytest

from scripts.verify_taobao_training_data import verify


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(split, outputs, mapping_path):
    return {
        "status": "complete",
        "split": split,
        "protocol": "history-last-user-v1",
        "partial_run": False,
        "source": {"columns_read": ["129_1", "150_2_180"]},
        "converter": {"sha256": "converter-sha"},
        "mapping": {
            "path": str(mapping_path.resolve()),
            "size_bytes": mapping_path.stat().st_size,
            "mtime_ns": mapping_path.stat().st_mtime_ns,
        },
        "protocol_rules": {
            "sample_unit": "one user per official split",
            "history_item_filter": (
                "remove items missing from item2row before target selection"
            ),
            "ground_truth": "last retained sequence_data item",
            "model_input": "all retained sequence_data items except the last",
            "exposure_target_used": False,
            "click_label_used": False,
            "tfrecord_fields": ["user_id", "sequence_data"],
            "minimum_sequence_length_including_target": 2,
            "validation_split": "MD5(user_id) bucket, ratio=0.05",
        },
        "outputs": outputs,
    }


def _make_ready_dataset(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    semantic_id_path = tmp_path / "semantic_ids.npy"
    np.save(semantic_id_path, np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]]))
    mapping_path = tmp_path / "item2row.pkl"
    mapping_path.write_bytes(b"mapping")

    outputs = {}
    for destination, count in (("train", 2), ("evaluation", 1), ("test", 1)):
        output_path = data_root / f"{destination}.tfrecord.gz"
        output_path.write_bytes(destination.encode("utf-8"))
        outputs[destination] = {
            "samples": count,
            "shards": 1,
            "files": [
                {
                    "path": str(output_path.resolve()),
                    "size_bytes": output_path.stat().st_size,
                    "mtime_ns": output_path.stat().st_mtime_ns,
                }
            ],
        }

    train_manifest = _manifest(
        "train",
        {"train": outputs["train"], "evaluation": outputs["evaluation"]},
        mapping_path,
    )
    test_manifest = _manifest("test", {"test": outputs["test"]}, mapping_path)
    audit = {
        "status": "passed",
        "protocol": "history-last-user-v1",
        "partial_run": False,
        "checks": {
            "manifest_consistency": True,
            "exact_schema": ["sequence_data", "user_id"],
            "minimum_sequence_length": 2,
            "train_validation_user_disjoint": True,
            "within_destination_user_unique": True,
        },
        "destinations": {
            "train": {
                "records": 2,
                "unique_users": 2,
                "duplicate_users": 0,
                "below_minimum_length": 0,
                "minimum_item_row": 0,
                "maximum_item_row": 1,
            },
            "evaluation": {
                "records": 1,
                "unique_users": 1,
                "duplicate_users": 0,
                "below_minimum_length": 0,
                "minimum_item_row": 1,
                "maximum_item_row": 1,
            },
            "test": {
                "records": 1,
                "unique_users": 1,
                "duplicate_users": 0,
                "below_minimum_length": 0,
                "minimum_item_row": 0,
                "maximum_item_row": 1,
            },
        },
    }
    _write_json(data_root / "protocol_manifest_train.json", train_manifest)
    _write_json(data_root / "protocol_manifest_test.json", test_manifest)
    _write_json(data_root / "protocol_audit.json", audit)
    return data_root, semantic_id_path


def _args(data_root, semantic_id_path, report_path=None):
    return Namespace(
        data_root=data_root,
        semantic_id_path=semantic_id_path,
        expected_protocol="history-last-user-v1",
        num_hierarchies=4,
        num_embeddings_per_hierarchy=4096,
        source_file=[],
        initialization_checkpoint=None,
        report_path=report_path,
    )


def test_preflight_accepts_only_complete_full_audit(tmp_path):
    data_root, semantic_id_path = _make_ready_dataset(tmp_path)
    report_path = tmp_path / "reports" / "preflight.json"

    report = verify(_args(data_root, semantic_id_path, report_path))

    assert report["status"] == "passed"
    assert report["counts"] == {"train": 2, "evaluation": 1, "test": 1}
    assert report["semantic_id"]["shape"] == [2, 4]
    assert report_path.is_file()


def test_preflight_rejects_incomplete_marker(tmp_path):
    data_root, semantic_id_path = _make_ready_dataset(tmp_path)
    (data_root / ".protocol_manifest_train.json.incomplete").write_text(
        "incomplete", encoding="utf-8"
    )

    with pytest.raises(AssertionError, match="Incomplete conversion marker"):
        verify(_args(data_root, semantic_id_path))


def test_preflight_rejects_out_of_range_item_row(tmp_path):
    data_root, semantic_id_path = _make_ready_dataset(tmp_path)
    audit_path = data_root / "protocol_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["destinations"]["test"]["maximum_item_row"] = 2
    _write_json(audit_path, audit)

    with pytest.raises(AssertionError):
        verify(_args(data_root, semantic_id_path))
