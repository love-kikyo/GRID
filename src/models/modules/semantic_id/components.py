"""
Shared components for Semantic ID generative recommender models.
"""

from enum import Enum, auto

import torch
from torch import nn


class ModelMode(Enum):
    """Mode for model forward pass."""
    TRAIN = auto()
    INFER = auto()
    EVAL = auto()


class DualHashEmbedding(nn.Module):
    """
    Hash-based embedding for item keys.
    Uses double hashing to reduce collisions.
    """

    def __init__(
        self,
        num_buckets: int = 2**26,
        embedding_dim: int = 1024,
        prime1: int = 1315423911,
        prime2: int = 2654435761,
        masking_token: int = -1,
    ):
        super().__init__()

        self.num_buckets = num_buckets
        self.prime1 = prime1
        self.prime2 = prime2
        self.masking_token = masking_token

        self.embedding1 = nn.Embedding(num_buckets, 128)
        self.embedding2 = nn.Embedding(num_buckets, 128)
        self.item_proj = nn.Linear(256, embedding_dim)

    def forward(self, item_key: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_key: (B,) or (B, L) - item keys

        Returns:
            embeddings: (B, D) or (B, L, D)
        """
        is_valid = (item_key != self.masking_token).long()
        safe_key = item_key * is_valid

        idx1 = (safe_key * self.prime1) % self.num_buckets
        idx2 = (safe_key * self.prime2) % self.num_buckets

        emb1 = self.embedding1(idx1)
        emb2 = self.embedding2(idx2)
        combined = torch.cat([emb1, emb2], dim=-1)
        final_emb = self.item_proj(combined)
        final_emb = final_emb * is_valid.unsqueeze(-1)

        return final_emb