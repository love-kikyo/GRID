from typing import Dict, Iterable, List

import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.data.loading.components.interfaces import RerankCandidateDatasetConfig


class RerankCandidateDataset(IterableDataset):
    """Shard-aware iterable dataset backed by tensorized `.pt` shards."""

    def __init__(self, dataset_config: RerankCandidateDatasetConfig):
        self.dataset_config = dataset_config
        candidate_cache_paths = dataset_config.candidate_cache_paths
        if candidate_cache_paths is None:
            if dataset_config.candidate_cache_path is None:
                raise ValueError("One of candidate_cache_path or candidate_cache_paths must be set.")
            candidate_cache_paths = [dataset_config.candidate_cache_path]

        self.shard_paths = list(candidate_cache_paths)
        self.total_workers = 1
        self.global_worker_id = 0

    def set_distributed_params(self, total_workers: int, global_worker_id: int):
        self.total_workers = total_workers
        self.global_worker_id = global_worker_id

    def _iter_assigned_shards(self) -> Iterable[str]:
        worker_info = get_worker_info()
        if worker_info is None:
            local_worker_id = 0
            num_local_workers = 1
        else:
            local_worker_id = worker_info.id
            num_local_workers = worker_info.num_workers

        combined_worker_id = self.global_worker_id * num_local_workers + local_worker_id
        combined_num_workers = self.total_workers * num_local_workers

        for shard_index in range(combined_worker_id, len(self.shard_paths), combined_num_workers):
            yield self.shard_paths[shard_index]

    def __iter__(self):
        for shard_path in self._iter_assigned_shards():
            shard_data: Dict[str, torch.Tensor] = torch.load(shard_path, map_location="cpu")
            shard_rows = shard_data["target_itemkey"].size(0)
            for row_index in range(shard_rows):
                yield {
                    "user_id": shard_data["user_id"][row_index],
                    "hist_itemkey": shard_data["hist_itemkey"][row_index],
                    "target_itemkey": shard_data["target_itemkey"][row_index],
                    "target_sid": shard_data["target_sid"][row_index],
                    "candidate_itemkey": shard_data["candidate_itemkey"][row_index],
                    "candidate_sid": shard_data["candidate_sid"][row_index],
                    "beam_score": shard_data["beam_score"][row_index],
                    "candidate_labels": shard_data["candidate_labels"][row_index],
                    "label_index": shard_data["label_index"][row_index],
                }
