import pickle
from bisect import bisect_right
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from src.data.loading.components.interfaces import RerankCandidateDatasetConfig


class RerankCandidateDataset(Dataset):
    """Map-style dataset backed by one or more SID candidate cache shards."""

    def __init__(self, dataset_config: RerankCandidateDatasetConfig):
        self.dataset_config = dataset_config
        candidate_cache_paths = dataset_config.candidate_cache_paths
        candidate_cache_lengths = dataset_config.candidate_cache_lengths
        if candidate_cache_paths is None:
            if dataset_config.candidate_cache_path is None:
                raise ValueError("One of candidate_cache_path or candidate_cache_paths must be set.")
            candidate_cache_paths = [dataset_config.candidate_cache_path]

        self.shard_paths = candidate_cache_paths
        self.shard_lengths: List[int] = []
        self.cumulative_lengths: List[int] = []
        cumulative = 0
        if candidate_cache_lengths is not None:
            if len(candidate_cache_lengths) != len(self.shard_paths):
                raise ValueError("candidate_cache_lengths must align with candidate_cache_paths.")
            shard_lengths_iter = candidate_cache_lengths
        else:
            shard_lengths_iter = []
            for shard_path in self.shard_paths:
                with open(shard_path, "rb") as handle:
                    shard_rows: List[Dict] = pickle.load(handle)
                shard_lengths_iter.append(len(shard_rows))

        for shard_length in shard_lengths_iter:
            self.shard_lengths.append(shard_length)
            cumulative += shard_length
            self.cumulative_lengths.append(cumulative)

        self._loaded_shard_index = None
        self._loaded_rows: List[Dict] = []

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _get_row(self, index: int) -> Dict:
        shard_index = bisect_right(self.cumulative_lengths, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_lengths[shard_index - 1]
        row_index = index - shard_start

        if self._loaded_shard_index != shard_index:
            with open(self.shard_paths[shard_index], "rb") as handle:
                self._loaded_rows = pickle.load(handle)
            self._loaded_shard_index = shard_index

        return self._loaded_rows[row_index]

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self._get_row(index)
        num_candidates = self.dataset_config.num_candidates

        candidate_sids = row["beam_candidate_sids"][:num_candidates]
        candidate_itemkeys = row["beam_candidate_itemkeys"][:num_candidates]
        beam_scores = row["beam_scores"][:num_candidates]
        candidate_labels = row["candidate_labels"][:num_candidates]

        label_index = row.get("label_index", -1)
        if label_index >= num_candidates:
            label_index = -1

        return {
            "user_id": row["user_id"],
            "hist_itemkey": row["hist_itemkey"],
            "target_itemkey": row["target_itemkey"],
            "target_sid": row["target_sid"],
            "candidate_itemkey": candidate_itemkeys,
            "candidate_sid": candidate_sids,
            "beam_score": beam_scores,
            "candidate_labels": candidate_labels,
            "label_index": label_index,
        }
