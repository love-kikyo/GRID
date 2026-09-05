import json
import hashlib

import numpy as np
import pytest
import torch
from argparse import Namespace
from decimal import Decimal
from omegaconf import OmegaConf

from scripts.analyze_paired_evaluation import (
    align_exports,
    analyze_h5,
    metric_contributions,
    paired_bootstrap_ci,
    ranked_candidates,
    ranked_itemkeys,
    recall_discordance_test,
)
from scripts.export_paired_evaluation import (
    H5StreamWriter,
    ROOT,
    align_rerank_logits_to_beam,
    file_sha256,
    validate_provenance,
)
from scripts.select_score_fusion import decimal_grid, select_best


def test_align_rerank_logits_to_original_beam_order():
    beam = torch.tensor([[[1, 1], [2, 2], [3, 3]]])
    rerank = torch.tensor([[[3, 3], [1, 1], [2, 2]]])
    ordered_logits = torch.tensor([[0.9, 0.4, -0.2]])
    aligned = align_rerank_logits_to_beam(beam, rerank, ordered_logits)
    assert torch.allclose(aligned, torch.tensor([[0.4, -0.2, 0.9]]))


def test_metric_contributions_match_single_relevant_item_ranking():
    target = np.array([7, 8, 9])
    ranked = np.array([[7, 1, 2], [1, 8, 2], [1, 2, 3]])
    metrics = metric_contributions(target, ranked, k=2)
    np.testing.assert_allclose(metrics["recall"], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(metrics["ndcg"], [1.0, 1.0 / np.log2(3.0), 0.0])


def test_fusion_variants_reorder_original_beam():
    payload = {
        "beam_itemkey": np.array([[10, 20, 30]]),
        "rerank_itemkey": np.array([[30, 20, 10]]),
        "beam_score": np.array([[3.0, 2.0, 1.0]]),
        "raw_rerank_logit": np.array([[0.0, 0.0, 10.0]]),
    }
    np.testing.assert_array_equal(ranked_itemkeys(payload, "beam"), [[10, 20, 30]])
    np.testing.assert_array_equal(ranked_itemkeys(payload, "click"), [[30, 10, 20]])
    np.testing.assert_array_equal(ranked_itemkeys(payload, "fusion@1@1"), [[30, 10, 20]])


def test_schema_v2_metrics_use_exact_sid_tuples():
    payload = {
        "beam_sid": np.array([[[1, 2], [2, 1], [3, 3]]], dtype=np.int16),
        "rerank_sid": np.array([[[3, 3], [1, 2], [2, 1]]], dtype=np.int16),
        "beam_score": np.array([[3.0, 2.0, 1.0]]),
        "raw_rerank_logit": np.array([[0.0, 0.0, 10.0]]),
    }
    ranked = ranked_candidates(payload, "fusion@1@1")
    np.testing.assert_array_equal(ranked, [[[3, 3], [1, 2], [2, 1]]])
    metrics = metric_contributions(np.array([[1, 2]], dtype=np.int16), ranked, k=2)
    np.testing.assert_allclose(metrics["recall"], [1.0])
    np.testing.assert_allclose(metrics["ndcg"], [1.0 / np.log2(3.0)])


def test_legacy_pointwise_uses_logsigmoid_click_score():
    payload = {
        "beam_itemkey": np.array([[10, 20]]),
        "rerank_itemkey": np.array([[10, 20]]),
        "beam_score": np.array([[0.0, -0.5]]),
        "raw_rerank_logit": np.array([[-10.0, 0.0]]),
    }
    np.testing.assert_array_equal(ranked_candidates(payload, "legacy_pointwise"), [[20, 10]])


def test_exports_align_by_input_hash_not_loader_order():
    first = {
        "example_hash": np.array([[2, 0], [1, 0]], dtype=np.uint64),
        "target_itemkey": np.array([20, 10]),
        "beam_itemkey": np.array([[20], [10]]),
        "_path": np.asarray("first.npz"),
    }
    second = {
        "example_hash": np.array([[1, 0], [2, 0]], dtype=np.uint64),
        "target_itemkey": np.array([10, 20]),
        "beam_itemkey": np.array([[10], [20]]),
        "_path": np.asarray("second.npz"),
    }
    aligned = align_exports({"first": first, "second": second})
    np.testing.assert_array_equal(aligned["first"]["target_itemkey"], [10, 20])
    np.testing.assert_array_equal(aligned["second"]["target_itemkey"], [10, 20])


def test_exact_paired_binary_discordance_counts():
    result = recall_discordance_test(
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 0.0]),
    )
    assert result["baseline_miss_challenger_hit"] == 2
    assert result["baseline_hit_challenger_miss"] == 1
    assert result["discordant_pairs"] == 3


def test_decimal_grid_includes_non_divisible_stop():
    assert decimal_grid(Decimal("0"), Decimal("0.05"), Decimal("0.02")) == [
        Decimal("0.00"),
        Decimal("0.02"),
        Decimal("0.04"),
        Decimal("0.05"),
    ]


def test_fusion_selection_uses_metric_order_then_smaller_weight():
    rows = [
        {"rerank_weight": 0.2, "metrics": {"ndcg@10": 0.3, "ndcg@5": 0.2}},
        {"rerank_weight": 0.1, "metrics": {"ndcg@10": 0.3, "ndcg@5": 0.2}},
        {"rerank_weight": 0.3, "metrics": {"ndcg@10": 0.3, "ndcg@5": 0.21}},
    ]
    assert select_best(rows, ["ndcg@10", "ndcg@5"]) == 2
    assert select_best(rows[:2], ["ndcg@10", "ndcg@5"]) == 1


def test_compressed_paired_bootstrap_is_reproducible_and_contains_mean():
    delta = np.array([-1.0, 0.0, 0.0, 1.0, 1.0])
    first = paired_bootstrap_ci(delta, 2000, 100, 0.95, np.random.default_rng(9))
    second = paired_bootstrap_ci(delta, 2000, 100, 0.95, np.random.default_rng(9))
    assert first == second
    assert first[0] <= delta.mean() <= first[1]


def test_h5_streaming_analysis_matches_exact_sid_ranking(tmp_path):
    output = tmp_path / "tiny.h5"
    arrays = {
        "user_id": np.array([101, 102, 103, 104], dtype=np.int64),
        "example_hash": np.array([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.uint64),
        "target_sid": np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int16),
        "beam_sid": np.array(
            [
                [[1, 1], [9, 9]],
                [[9, 9], [2, 2]],
                [[3, 3], [9, 9]],
                [[9, 9], [4, 4]],
            ],
            dtype=np.int16,
        ),
        "beam_score": np.array([[2.0, 1.0]] * 4, dtype=np.float32),
        "raw_rerank_logit": np.array([[0.0, 20.0]] * 4, dtype=np.float32),
        "rerank_sid": np.array(
            [
                [[9, 9], [1, 1]],
                [[2, 2], [9, 9]],
                [[9, 9], [3, 3]],
                [[4, 4], [9, 9]],
            ],
            dtype=np.int16,
        ),
        "rerank_score": np.array([[3.0, 2.0]] * 4, dtype=np.float32),
    }
    writer = H5StreamWriter(output)
    writer.append({key: value[:2] for key, value in arrays.items()})
    writer.append({key: value[2:] for key, value in arrays.items()})
    writer.close()
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "split": "test",
                "num_samples": 4,
                "sample_unit": "unique_user_history",
                "limits": {"max_batches": None, "max_samples": None},
                "checkpoint_sha256": "checkpoint-hash",
                "exporter_sha256": "exporter-hash",
                "dataset_provenance": {
                    "data_root": "/frozen/data",
                    "protocol": "history-last-user-v1",
                    "expected_samples": 4,
                    "manifests": {
                        "train": {"sha256": "train-hash"},
                        "test": {"sha256": "test-hash"},
                        "audit": {"sha256": "audit-hash"},
                    },
                },
            }
        )
    )

    args = Namespace(
        input=[f"model={output}"],
        compare=["model:beam,model:rerank"],
        ks=[1, 2],
        bootstrap_samples=500,
        bootstrap_batch_size=100,
        confidence=0.95,
        seed=2027,
        chunk_size=2,
        expected_split="test",
        selection=[],
    )
    report = analyze_h5(args)
    assert report["num_examples"] == 4
    metrics = report["comparisons"][0]["metrics"]
    assert metrics["recall@1"]["baseline_mean"] == 0.5
    assert metrics["recall@1"]["challenger_mean"] == 0.5
    assert metrics["recall@2"]["baseline_mean"] == 1.0
    assert metrics["recall@2"]["challenger_mean"] == 1.0


def test_selected_variant_requires_matching_validation_export(tmp_path):
    output = tmp_path / "test.h5"
    arrays = {
        "user_id": np.array([7], dtype=np.int64),
        "example_hash": np.array([[1, 2]], dtype=np.uint64),
        "target_sid": np.array([[1, 1]], dtype=np.int16),
        "beam_sid": np.array([[[9, 9], [1, 1]]], dtype=np.int16),
        "beam_score": np.array([[2.0, 1.0]], dtype=np.float32),
        "raw_rerank_logit": np.array([[0.0, 2.0]], dtype=np.float32),
        "rerank_sid": np.array([[[1, 1], [9, 9]]], dtype=np.int16),
        "rerank_score": np.array([[3.0, 2.0]], dtype=np.float32),
    }
    writer = H5StreamWriter(output)
    writer.append(arrays)
    writer.close()
    provenance = {
        "data_root": "/frozen/data",
        "protocol": "history-last-user-v1",
        "expected_samples": 1,
        "manifests": {
            "train": {"sha256": "train-hash"},
            "test": {"sha256": "test-hash"},
            "audit": {"sha256": "audit-hash"},
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "split": "test",
                "num_samples": 1,
                "sample_unit": "unique_user_history",
                "limits": {"max_batches": None, "max_samples": None},
                "checkpoint_sha256": "same-checkpoint",
                "dataset_provenance": provenance,
            }
        )
    )
    validation_input = tmp_path / "validation.h5"
    validation_input.write_bytes(b"immutable-validation-export")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selection_scope": "validation_only",
                "input_split": "val",
                "input": str(validation_input),
                "input_sha256": hashlib.sha256(validation_input.read_bytes()).hexdigest(),
                "input_checkpoint_sha256": "same-checkpoint",
                "input_dataset_provenance": provenance,
                "selected": {"variant": "fusion@1@2"},
            }
        )
    )
    args = Namespace(
        input=[f"model={output}"],
        compare=["model:beam,model:selected"],
        ks=[1],
        bootstrap_samples=100,
        bootstrap_batch_size=20,
        confidence=0.95,
        seed=2027,
        chunk_size=1,
        expected_split="test",
        selection=[f"model={selection}"],
    )

    report = analyze_h5(args)

    comparison = report["comparisons"][0]
    assert comparison["challenger"] == "model:fusion@1@2"
    assert comparison["metrics"]["recall@1"]["challenger_mean"] == 1.0


def test_export_provenance_locks_manifests_and_training_sources(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest_paths = {
        "train": data_root / "protocol_manifest_train.json",
        "test": data_root / "protocol_manifest_test.json",
        "audit": data_root / "protocol_audit.json",
    }
    manifest_paths["train"].write_text(json.dumps({"protocol": "history-last-user-v1"}))
    manifest_paths["test"].write_text(json.dumps({"protocol": "history-last-user-v1"}))
    manifest_paths["audit"].write_text(
        json.dumps(
            {
                "protocol": "history-last-user-v1",
                "status": "passed",
                "destinations": {
                    "evaluation": {"records": 3, "unique_users": 3},
                    "test": {"records": 5, "unique_users": 5},
                },
            }
        )
    )
    source_paths = [
        ROOT / "src/data/loading/components/dataloading.py",
        ROOT / "src/data/loading/components/interfaces.py",
        ROOT / "src/models/modules/semantic_id/tiger_generation_model.py",
    ]
    preflight = {
        "status": "passed",
        "protocol": "history-last-user-v1",
        "data_root": str(data_root),
        "counts": {"evaluation": 3, "test": 5},
        "source_manifests": {
            name: {"sha256": file_sha256(path)}
            for name, path in manifest_paths.items()
        },
        "source_files": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in source_paths
        ],
        "initialization_checkpoint": None,
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight))
    cfg = OmegaConf.create(
        {
            "data_dir": str(data_root),
            "model": {
                "_target_": "src.models.SemanticIDEncoderDecoder",
                "manual_ckpt_path": None,
            },
        }
    )

    provenance, expected = validate_provenance(
        cfg,
        split="val",
        expected_data_root=data_root,
        expected_protocol="history-last-user-v1",
        training_preflight=preflight_path,
    )

    assert expected == 3
    assert provenance["expected_samples"] == 3
    preflight["source_manifests"]["audit"]["sha256"] = "changed"
    preflight_path.write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="audit manifest hash"):
        validate_provenance(
            cfg,
            split="val",
            expected_data_root=data_root,
            expected_protocol="history-last-user-v1",
            training_preflight=preflight_path,
        )
