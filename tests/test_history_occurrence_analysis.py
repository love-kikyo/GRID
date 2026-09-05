import json
from types import SimpleNamespace

import numpy as np
import torch

from scripts.analyze_history_occurrence import (
    DatasetAccumulator,
    PredictionAccumulator,
    build_history_views,
    collision_audit,
    git_commit,
)


def synthetic_batch():
    return SimpleNamespace(
        transformed_sequences={
            "sequence_data": torch.tensor(
                [
                    [1, 2, 3, 4, 1, 2],
                    [0, 0, 0, 0, 5, 6],
                ]
            ),
            "hist_itemkey": torch.tensor(
                [
                    [-1, -1, 12, 34, 12],
                    [-1, -1, -1, -1, 56],
                ]
            ),
        },
        mask=torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1],
            ]
        ),
    )


def test_history_views_exclude_target_and_recover_ar_suffix():
    views = build_history_views(
        synthetic_batch(),
        num_hierarchies=2,
        num_embeddings_per_hierarchy=10,
        masking_token=-1,
    )

    full = views["full_history"]
    ar = views["ar_history"]
    assert full.target_itemkey.tolist() == [12, 56]
    assert full.history_length.tolist() == [2, 0]
    assert ar.history_length.tolist() == [2, 0]
    assert full.repeat_mask.tolist() == [True, False]
    assert full.occurrence_count.tolist() == [1, 0]
    assert full.recent_distance.tolist() == [2, 0]
    assert full.history_itemkey[0][full.valid_mask[0]].tolist() == [12, 34]


def test_dataset_and_prediction_denominators_and_cohorts():
    views = build_history_views(
        synthetic_batch(),
        num_hierarchies=2,
        num_embeddings_per_hierarchy=10,
        masking_token=-1,
    )
    view = views["full_history"]

    dataset = DatasetAccumulator()
    dataset.update(view)
    summary = dataset.summary()
    assert summary["num_samples"] == 2
    assert summary["gt_repeat_rate"] == 0.5
    assert summary["padding_filtered_count"] == 6

    prediction = PredictionAccumulator((1, 2, 3))
    prediction.update(
        view,
        beam_itemkey=torch.tensor([[12, 78, 34], [56, 90, 91]]),
        rerank_itemkey=torch.tensor([[78, 12, 34], [90, 56, 91]]),
    )
    pred_summary = prediction.summary()
    assert pred_summary["beam_top1_repeat_rate"] == 0.5
    assert pred_summary["rerank_top1_repeat_rate"] == 0.0
    assert pred_summary["beam_candidate_denominator_at_3"] == 6
    assert pred_summary["beam_candidate_repeat_rate_at_3"] == 2 / 6
    assert pred_summary["rerank_candidate_repeat_rate_at_3"] == 2 / 6
    assert pred_summary["beam_any_repeat_rate_at_3"] == 0.5
    assert pred_summary["rerank_any_repeat_rate_at_3"] == 0.5

    repeated = pred_summary["metrics_by_gt_recurrence"]["gt_repeated"]
    novel = pred_summary["metrics_by_gt_recurrence"]["gt_novel"]
    assert repeated["num_samples"] == 1
    assert novel["num_samples"] == 1
    assert repeated["beam_recall_at_1"] == 1.0
    assert repeated["rerank_recall_at_1"] == 0.0
    assert repeated["rerank_recall_at_2"] == 1.0
    assert novel["beam_recall_at_1"] == 1.0


def test_collision_audit_and_cache(tmp_path):
    semantic_ids = np.array([[1, 2], [1, 2], [3, 4]], dtype=np.int16)
    semantic_path = tmp_path / "sid.npy"
    cache_path = tmp_path / "audit.json"
    np.save(semantic_path, semantic_ids)

    audit = collision_audit(
        semantic_id_path=semantic_path,
        num_hierarchies=2,
        num_embeddings_per_hierarchy=10,
        cache_path=cache_path,
    )
    assert audit["num_raw_items"] == 3
    assert audit["num_unique_itemkeys"] == 2
    assert audit["collision_count"] == 1
    assert audit["max_items_per_itemkey"] == 2
    assert json.loads(cache_path.read_text()) == audit


def test_git_metadata_is_optional(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("scripts.analyze_history_occurrence.subprocess.run", missing_git)
    assert git_commit() is None
