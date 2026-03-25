"""
Base class for Semantic ID generative recommender models.
"""

import logging
import numpy as np
import time
from typing import Any, Optional, Tuple, Union

import torch
from lightning import LightningModule
from torch import nn
import torch.nn.functional as F
from torchmetrics.aggregation import MeanMetric

from src.models.modules.base_module import BaseModule
from src.models.modules.semantic_id.components import DualHashEmbedding, ModelMode


class SemanticIDBaseRecommender(BaseModule):
    """
    Base class for Semantic ID generative recommender models.

    This class provides common functionality for:
    - Embedding tables (SID, item, segment)
    - Item key lookup and similarity computation
    - Beam search utilities
    - Training/evaluation lifecycle hooks

    Subclasses should implement:
    - model_step(): Core forward pass and loss computation
    - build_sequence_embeds(): Build input embeddings from SID sequence
    - generate(): Generate predictions during inference
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        embedding_dim: int,
        top_k_for_generation: int,
        top_k_for_score: int,
        padding_token: int,
        masking_token: int,
        sorted_item_keys: torch.Tensor,
        sorted_row_indices: torch.Tensor,
        scl_embedding_path: str,
        beam_search_interval: int = 100,
        num_user_bins: Optional[int] = None,
        should_check_prefix: bool = False,
        # BaseModule required params
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        evaluator = None,
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDBaseRecommender.

        Args:
            codebooks: Codebooks for semantic ID validation.
            num_hierarchies: Number of hierarchy levels in the codebooks.
            num_embeddings_per_hierarchy: Number of embeddings per hierarchy.
            embedding_dim: Dimension of embeddings.
            top_k_for_generation: Number of top-k candidates for generation.
            top_k_for_score: Number of top-k candidates for scoring/reranking.
            padding_token: Token ID used for padding.
            masking_token: Token ID used for masking.
            sorted_item_keys: Sorted item keys for lookup.
            sorted_row_indices: Sorted row indices for lookup.
            scl_embedding_path: Path to SCL embedding file.
            beam_search_interval: Run beam search training every N steps.
            num_user_bins: Number of user bins for user embedding table.
            should_check_prefix: Whether to check prefix validity during generation.
        """
        super().__init__(
            model=None,  # Model is built in subclasses
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=kwargs.get("loss_function", {}),
            evaluator=evaluator,
        )

        self.save_hyperparameters(logger=False)

        # ===== Basic Configuration =====
        self.padding_token = padding_token
        self.masking_token = masking_token
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.embedding_dim = embedding_dim
        self.num_hierarchies = num_hierarchies
        self.top_k_for_generation = top_k_for_generation
        self.top_k_for_score = top_k_for_score
        self.beam_search_interval = beam_search_interval
        self.should_check_prefix = should_check_prefix

        # ===== Codebooks =====
        if codebooks is not None:
            if isinstance(codebooks, np.ndarray):
                codebooks = torch.from_numpy(codebooks)
            self.codebooks = codebooks + 1
            assert self.codebooks.size(1) == num_hierarchies, \
                "codebooks should be of shape (-1, num_hierarchies)"
        else:
            self.codebooks = None
            logging.warning(
                "Not using pre-cached codebooks, please ensure:\n"
                "1) dataset is properly pre-processed\n"
                "2) num_hierarchies and num_embeddings_per_hierarchy are properly set"
            )

        # ===== Pre-computed Buffers =====
        self.register_buffer(
            "powers",
            self.num_embeddings_per_hierarchy **
            torch.arange(self.num_hierarchies - 1, -1, -1)
        )
        self.register_buffer(
            "hierarchy_offsets",
            torch.arange(0, self.num_hierarchies * self.num_embeddings_per_hierarchy,
                         self.num_embeddings_per_hierarchy),
            persistent=False,
        )

        # ===== Sorted Indices for Lookup =====
        if isinstance(sorted_item_keys, np.ndarray):
            sorted_item_keys = torch.from_numpy(sorted_item_keys)
        self.register_buffer("sorted_item_keys", sorted_item_keys.long(), persistent=False)

        if isinstance(sorted_row_indices, np.ndarray):
            sorted_row_indices = torch.from_numpy(sorted_row_indices)
        self.register_buffer("sorted_row_indices", sorted_row_indices.long(), persistent=False)

        # ===== SCL Embedding =====
        self.scl_embedding_path = scl_embedding_path
        self.scl_embedding = None

        # ===== Embedding Tables =====
        self.sid_embedding_table = self._spawn_embedding_tables(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies + 1,
            embedding_dim=self.embedding_dim,
        )
        self.item_embedding_table = DualHashEmbedding(
            num_buckets=1_000_000,
            embedding_dim=self.embedding_dim,
            masking_token=self.masking_token,
        )
        self.segment_embedding_table = nn.Embedding(3, self.embedding_dim)

        # ===== User Embedding (Optional) =====
        self.user_embedding: Optional[nn.Embedding] = (
            self._spawn_embedding_tables(
                num_embeddings=num_user_bins,
                embedding_dim=self.embedding_dim,
            )
            if num_user_bins
            else None
        )

        # ===== Metrics Accumulators =====
        self.click_log_prob_accumulator = MeanMetric()
        self.beam_log_prob_accumulator = MeanMetric()

        # ===== Step Counter =====
        self.register_buffer("training_step_counter", torch.tensor(0, dtype=torch.long))

        # ===== Timing Variables =====
        self.batch_start_time = None
        self.prev_batch_end_time = None
        self.total_val_time = 0
        self.val_start_time = 0

    def setup(self, stage: str = None) -> None:
        """Load SCL embedding if needed."""
        if self.scl_embedding is None and self.scl_embedding_path is not None:
            arr = np.load(self.scl_embedding_path, mmap_mode="r")
            self.scl_embedding = torch.from_numpy(arr)

    def _spawn_embedding_tables(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> nn.Embedding:
        """Create an embedding table with padding token."""
        return nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=self.padding_token,
        )

    def compute_itemkey(self, sids: torch.Tensor) -> torch.Tensor:
        """
        Compute itemkey from SID tokens.

        Args:
            sids: (B, num_hierarchies) - SID tokens

        Returns:
            itemkey: (B,) - computed item key
        """
        return (sids * self.powers).sum(dim=-1)

    def itemkey_lookup(self, key: torch.Tensor) -> torch.Tensor:
        """
        Look up item embeddings by key using pre-sorted index arrays.

        Note: scl_embedding remains on CPU due to memory constraints.

        Args:
            key: Tensor of item keys to look up

        Returns:
            Tensor of embeddings for the requested keys, moved to model device
        """
        pos = torch.bucketize(key, self.sorted_item_keys)

        mask = pos < self.sorted_item_keys.numel()
        pos_safe = torch.clamp(pos, max=self.sorted_item_keys.numel() - 1)
        valid = mask & (self.sorted_item_keys[pos_safe] == key)
        row_idx = torch.where(valid, self.sorted_row_indices[pos_safe], -1).cpu()

        emb = self.scl_embedding[row_idx]
        emb[row_idx == -1] = 0

        return emb.float().to(self.device)

    def get_similar_itemkey(
        self,
        target_itemkey: torch.Tensor,
        hist_itemkey: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find similar item keys from history based on embedding similarity.

        Args:
            target_itemkey: (B,) or (B * num_beam,) - target item keys
            hist_itemkey: (B, num_items) or (B * num_beam, num_items) - historical item keys

        Returns:
            similar_itemkey: (B, top_k_for_score + 1) - similar item keys + target
        """
        target_emb = self.itemkey_lookup(target_itemkey)  # (B, dim)
        hist_emb = self.itemkey_lookup(hist_itemkey)  # (B, num_items, dim)

        # Cosine similarity
        sim = torch.einsum("bd,bhd->bh", target_emb, hist_emb)  # (B, num_items)

        pad_mask = hist_itemkey == self.masking_token
        sim = sim.masked_fill(pad_mask, -1e9)

        topk_idx = torch.topk(sim, k=self.top_k_for_score, dim=-1).indices
        topk_idx = torch.sort(topk_idx, dim=-1).values

        similar_itemkey = torch.gather(
            hist_itemkey,
            dim=1,
            index=topk_idx
        )

        return torch.cat([similar_itemkey, target_itemkey.unsqueeze(-1)], dim=-1)

    def embed_sid_token(self, token_id: torch.Tensor, hierarchy: int) -> torch.Tensor:
        """
        Embed a single SID token with hierarchy offset.

        Args:
            token_id: (B,) or (B, L) - SID token IDs (1-based)
            hierarchy: Hierarchy level (0 to num_hierarchies-1)

        Returns:
            embeddings: (B, D) or (B, L, D)
        """
        offset = hierarchy * self.num_embeddings_per_hierarchy
        return self.sid_embedding_table(token_id + offset)

    def embed_item_token(self, itemkey: torch.Tensor) -> torch.Tensor:
        """
        Embed item key(s).

        Args:
            itemkey: (B,) or (B, L) - item keys

        Returns:
            embeddings: (B, D) or (B, L, D)
        """
        return self.item_embedding_table(itemkey)

    def _check_valid_prefix(
        self,
        prefix: torch.Tensor,
        batch_size: int = 100000,
    ) -> torch.Tensor:
        """
        Check if a given prefix is a valid prefix of the codebooks.

        Args:
            prefix: (batch_size, hierarchy_level) - prefix tokens
            batch_size: Batch size for processing

        Returns:
            (batch_size,) boolean tensor indicating validity
        """
        current_hierarchy = prefix.shape[1]
        num_prefixes = prefix.shape[0]
        results = []

        if prefix.device != self.codebooks.device:
            self.codebooks = self.codebooks.to(prefix.device)

        trimmed_codebooks = self.codebooks[:, :current_hierarchy]

        for i in range(0, num_prefixes, batch_size):
            batch_prefix = prefix[i:i + batch_size]

            comparison = trimmed_codebooks.unsqueeze(1) == batch_prefix.unsqueeze(0)
            all_match = comparison.all(dim=2)
            any_match = all_match.any(dim=0)

            results.append(any_match)

        return torch.cat(results)

    def repeat_kv_cache(self, past_key_values, beam_size: int):
        """Repeat KV cache for beam search.

        Handles both tuple format (legacy) and Cache object format (newer HuggingFace).
        """
        if past_key_values is None:
            return None

        # Check if it's a Cache object with key_cache/value_cache attributes
        if hasattr(past_key_values, 'key_cache'):
            for layer in range(len(past_key_values.key_cache)):
                key = past_key_values.key_cache[layer]
                value = past_key_values.value_cache[layer]

                B = key.shape[0]

                key = key.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
                value = value.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)

                key = key.view(B * beam_size, *key.shape[2:])
                value = value.view(B * beam_size, *value.shape[2:])

                past_key_values.key_cache[layer] = key
                past_key_values.value_cache[layer] = value

            return past_key_values
        else:
            # Handle tuple format (legacy HuggingFace format)
            # past_key_values is a tuple of (key, value) tuples for each layer
            new_cache = []
            for layer_cache in past_key_values:
                key, value = layer_cache
                B = key.shape[0]

                key = key.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
                value = value.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)

                key = key.view(B * beam_size, *key.shape[2:])
                value = value.view(B * beam_size, *value.shape[2:])

                new_cache.append((key, value))

            return tuple(new_cache)

    def reorder_kv_cache(self, past_key_values, beam_idx: torch.Tensor):
        """Reorder KV cache for beam search.

        Handles both tuple format (legacy) and Cache object format (newer HuggingFace).

        Args:
            past_key_values: KV cache (tuple or Cache object)
            beam_idx: Indices to reorder the cache

        Returns:
            Reordered KV cache
        """
        if past_key_values is None:
            return None

        # Check if it's a Cache object with reorder_cache method
        if hasattr(past_key_values, 'reorder_cache'):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        else:
            # Handle tuple format (legacy HuggingFace format)
            # past_key_values is a tuple of (key, value) tuples for each layer
            new_cache = []
            for layer_cache in past_key_values:
                key, value = layer_cache
                key = key.index_select(0, beam_idx)
                value = value.index_select(0, beam_idx)
                new_cache.append((key, value))

            return tuple(new_cache)

    def _beam_search_one_step(
        self,
        candidate_logits: torch.Tensor,
        generated_ids: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        beam_log_prob: Optional[torch.Tensor],
        past_key_values,
        hierarchy: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Perform one step of beam search.

        Args:
            candidate_logits: Logits for the next token
            generated_ids: Generated IDs so far (None for first step)
            attention_mask: Attention mask for input
            beam_log_prob: Accumulated log probabilities (None for first step)
            past_key_values: KV cache from previous steps
            hierarchy: Current hierarchy level
            batch_size: Original batch size

        Returns:
            generated_ids: Updated generated IDs
            beam_log_prob: Updated log probabilities
            attention_mask: Updated attention mask
            past_key_values: Updated KV cache
        """
        candidate_logits = F.log_softmax(candidate_logits, dim=-1)
        proba, indices = torch.sort(candidate_logits, descending=True)

        if hierarchy == 0:
            proba_topk, indices_topk = (
                proba[:, :self.top_k_for_generation],
                indices[:, :self.top_k_for_generation],
            )

            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.top_k_for_generation, 1)

            next_sid = indices_topk.unsqueeze(-1) + 1
            next_mask = torch.ones(
                batch_size, self.top_k_for_generation, 1,
                device=attention_mask.device, dtype=attention_mask.dtype
            )

            generated_ids = next_sid.view(batch_size * self.top_k_for_generation, -1)
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1).view(
                batch_size * self.top_k_for_generation, -1
            )
            past_key_values = self.repeat_kv_cache(past_key_values, self.top_k_for_generation)
        else:
            proba = proba.view(-1, self.top_k_for_generation * self.num_embeddings_per_hierarchy)
            indices = indices.view(-1, self.top_k_for_generation * self.num_embeddings_per_hierarchy)

            proba = beam_log_prob.repeat_interleave(self.num_embeddings_per_hierarchy, dim=-1) + proba
            topk_results = torch.topk(proba, k=self.top_k_for_generation, dim=-1)
            proba_topk, indices_topk = topk_results.values, topk_results.indices

            replace_indices = (
                (indices_topk // self.num_embeddings_per_hierarchy)
                + torch.arange(indices_topk.size(0), device=proba.device).unsqueeze(1)
                * self.top_k_for_generation
            ).flatten()

            generated_ids = generated_ids[replace_indices]
            attention_mask = attention_mask[replace_indices]

            next_sid = torch.gather(indices, 1, indices_topk).reshape(-1).unsqueeze(-1) + 1
            next_mask = torch.ones(
                replace_indices.size(0), 1,
                device=attention_mask.device, dtype=attention_mask.dtype
            )

            generated_ids = torch.cat([generated_ids, next_sid], dim=-1)
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1)
            past_key_values = self.reorder_kv_cache(past_key_values, replace_indices.long())

        return generated_ids, proba_topk, attention_mask, past_key_values

    # ===== Lifecycle Hooks =====

    def on_train_start(self) -> None:
        """Called at the start of training."""
        super().on_train_start()

    def on_train_batch_start(self, batch, batch_idx) -> None:
        """Called at the start of each training batch."""
        now = time.time()
        if self.prev_batch_end_time is not None:
            data_wait = now - self.prev_batch_end_time
            self.log(
                "perf/next_batch_wait_time", data_wait,
                on_step=True, on_epoch=False, logger=True, sync_dist=False
            )

        self.batch_start_time = now
        if not hasattr(self, 'train_start_time'):
            self.train_start_time = now
            self.total_samples = 0

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Called at the end of each training batch."""
        now = time.time()
        step_time = now - self.batch_start_time

        model_input = batch[0][0] if isinstance(batch[0], (list, tuple)) else batch[0]
        local_batch_size = model_input.mask.size(0)
        global_batch_size = (
            local_batch_size
            * self.trainer.num_devices
            * self.trainer.accumulate_grad_batches
        )

        self.total_samples += global_batch_size
        running_time = now - self.train_start_time - self.total_val_time

        self.log(
            "perf/train_samples_per_sec_e2e", global_batch_size / step_time,
            on_step=True, on_epoch=False, logger=True, sync_dist=False
        )
        self.log(
            "perf/train_samples_per_sec_avg", self.total_samples / running_time,
            on_step=True, on_epoch=False, logger=True, sync_dist=False
        )
        self.prev_batch_end_time = time.time()

    def on_validation_start(self) -> None:
        """Called at the start of validation."""
        super().on_validation_start()
        self.val_start_time = time.time()

    def on_validation_epoch_start(self) -> None:
        """Called at the start of each validation epoch."""
        super().on_validation_epoch_start()
        self.click_log_prob_accumulator.reset()
        self.beam_log_prob_accumulator.reset()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        super().on_validation_epoch_end()
        self.log(
            "val/click_log_prob", self.click_log_prob_accumulator,
            sync_dist=True, prog_bar=False, logger=True
        )
        self.log(
            "val/beam_log_prob", self.beam_log_prob_accumulator,
            sync_dist=True, prog_bar=False, logger=True
        )

    def on_validation_end(self) -> None:
        """Called at the end of validation."""
        super().on_validation_end()
        val_duration = time.time() - self.val_start_time
        self.total_val_time += val_duration

        if self.prev_batch_end_time is not None:
            self.prev_batch_end_time += val_duration

    # ===== Abstract Methods =====

    def model_step(
        self,
        model_input: Any,
        mode: ModelMode,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Perform a forward pass and compute loss.

        Args:
            model_input: Input data
            mode: TRAIN or INFER

        Returns:
            output: Model output
            loss: Loss tensor
            metrics: Dictionary of metrics
        """
        raise NotImplementedError("Subclasses must implement model_step")

    def generate(
        self,
        attention_mask: torch.Tensor,
        sid: torch.Tensor,
        hist_itemkey: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Generate predictions using beam search.

        Args:
            attention_mask: Attention mask
            sid: Input SID sequence
            hist_itemkey: Historical item keys for similarity

        Returns:
            Dictionary with beam and rerank results
        """
        raise NotImplementedError("Subclasses must implement generate")