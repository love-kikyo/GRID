"""
Semantic ID Generative Recommender with MTP Head.

This module implements a generative recommender that uses:
- A LLaMA backbone for processing block-level sequence
- An MTP (Multi-Token Prediction) head for predicting complete semantic ID blocks

Key differences from the original TIGER model:
1. Backbone processes sum-pooled block embeddings instead of individual tokens
2. MTP head autoregressively predicts [sid0, sid1, sid2, sid3] from backbone hidden state
3. itemkey is computed deterministically from predicted sids

Training: Teacher forcing on MTP head
Inference: Beam search through MTP head
"""

import time
from typing import Any, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.aggregation import BaseAggregator
from torchmetrics.functional.classification import binary_auroc

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.modules.semantic_id.base import SemanticIDBaseRecommender
from src.models.modules.semantic_id.components import ModelMode
from src.models.modules.semantic_id.mtp_head import MTPHead, MTPConfig


class SemanticIDMTPRecommender(SemanticIDBaseRecommender):
    """
    Generative recommender using MTP (Multi-Token Prediction) head.

    Architecture:
        - Backbone: LLaMA decoder processes block-level sequence
        - MTP Head: Small Transformer that predicts complete SID blocks

    Sequence format:
        [bos, block1_embed, block2_embed, ..., blockN_embed]
        where block_embed = sum_pool(sid0, sid1, sid2, sid3, itemkey)
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        top_k_for_generation: int = 20,
        top_k_for_score: int = 10,
        padding_token: int = 0,
        masking_token: int = -1,
        sorted_item_keys: torch.Tensor = None,
        sorted_row_indices: torch.Tensor = None,
        scl_embedding_path: str = None,
        beam_search_interval: int = 50,
        # MTP specific config
        mtp_hidden_size: int = 256,
        num_mtp_layers: int = 2,
        num_mtp_attention_heads: int = 4,
        mtp_intermediate_size: int = 512,
        mtp_dropout: float = 0.1,
        # torch.compile config
        compile_mtp: bool = False,
        compile_mode: str = "reduce-overhead",
        compile_fullgraph: bool = False,
        # BaseModule required params
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        evaluator = None,
        decoder: torch.nn.Module = None,
        **kwargs,
    ) -> None:
        embedding_dim = decoder.config.hidden_size
        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            top_k_for_score=top_k_for_score,
            padding_token=padding_token,
            masking_token=masking_token,
            sorted_item_keys=sorted_item_keys,
            sorted_row_indices=sorted_row_indices,
            scl_embedding_path=scl_embedding_path,
            beam_search_interval=beam_search_interval,
            optimizer=optimizer,
            scheduler=scheduler,
            evaluator=evaluator,
            **kwargs,
        )

        # ===== MTP Head =====
        mtp_config = MTPConfig(
            num_hierarchies=num_hierarchies,
            vocab_size=num_embeddings_per_hierarchy,
            hidden_size=embedding_dim,
            mtp_hidden_size=mtp_hidden_size,
            num_mtp_layers=num_mtp_layers,
            num_attention_heads=num_mtp_attention_heads,
            intermediate_size=mtp_intermediate_size,
            dropout=mtp_dropout,
        )

        self.mtp_head = MTPHead(
            config=mtp_config,
            sid_embedding_table=self.sid_embedding_table,
            padding_token=padding_token,
            loss_fn=self.sid_loss_fn,  # Use loss function from BaseModule
        )

        # Store compile settings for lazy compilation in setup
        self.compile_mtp = compile_mtp
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph

        # ===== Click Head =====
        self.click_head = nn.Linear(embedding_dim, 1, bias=False)

        # ===== Loss Functions (inherited from BaseModule) =====
        # self.sid_loss_fn and self.click_loss_fn are set in BaseModule.__init__

        # Placeholder for decoder (will be set by config or externally)
        self.decoder = decoder

        # Remove embedding table in the decoder to save space and avoid DDP unused parameter error
        # The model uses inputs_embeds instead of input_ids, so embed_tokens is never used
        self.decoder.embed_tokens = nn.Identity()

    def setup(self, stage=None):
        """Load SCL embedding and setup decoder if needed."""
        super().setup(stage)

        # Compile MTP head after setup (on GPU)
        if self.compile_mtp and self.mtp_head is not None:
            import logging
            logging.info(f"Compiling MTP head with mode={self.compile_mode}")
            self.mtp_head.compile(mode=self.compile_mode, fullgraph=self.compile_fullgraph)

    def training_step(
        self,
        batch: Tuple[SequentialModelInputData],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step."""
        batch = batch[0]
        model_input: SequentialModelInputData = batch
        _, loss, metrics = self.model_step(
            model_input=model_input, mode=ModelMode.TRAIN
        )

        self.log(
            "train/loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for k, v in metrics.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        lr = self.lr_schedulers().get_last_lr()[0]
        self.log("train/lr", lr, on_step=True, on_epoch=False, sync_dist=False)

        return loss

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate: BaseAggregator,
    ):
        with torch.inference_mode():
            _, loss, __ = self.model_step(model_input=batch, mode=ModelMode.TRAIN)
        loss_to_aggregate(loss)

        sid = batch.transformed_sequences["sequence_data"].long()
        hist_itemkey = batch.transformed_sequences["hist_itemkey"]
        attention_mask = batch.mask

        labels = sid[:, -self.num_hierarchies:]
        hist_sid = sid[:, :-self.num_hierarchies]
        hist_itemkey = hist_itemkey[:, :-1]
        hist_mask = attention_mask[:, :-self.num_hierarchies]

        infer_input = SequentialModelInputData(
            user_id_list=batch.user_id_list,
            transformed_sequences={
                "sequence_data": hist_sid,
                "hist_itemkey": hist_itemkey,
            },
            mask=hist_mask,
        )

        with torch.inference_mode():
            result_dict = self.model_step(model_input=infer_input, mode=ModelMode.INFER)

        self.evaluators["beam"](
            marginal_probs=result_dict["beam"]["scores"].detach(),
            generated_ids=result_dict["beam"]["ids"].detach(),
            labels=labels.detach(),
        )

        self.evaluators["rerank"](
            marginal_probs=result_dict["rerank"]["scores"].detach(),
            generated_ids=result_dict["rerank"]["ids"].detach(),
            labels=labels.detach(),
        )

    def predict_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        batch_idx: int,
    ):
        """Perform a single prediction step."""
        # TODO: Implement predict_step for MTP model
        pass

    def build_block_embed(
        self,
        sids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build sum-pooled block embedding from SID tokens.

        Args:
            sids: (B, num_hierarchies) - SID tokens for one block
            attention_mask: (B,) - valid position mask

        Returns:
            block_embed: (B, D) - sum-pooled embedding
            mask: (B,) - attention mask
        """
        B, H = sids.shape
        device = sids.device

        # Vectorized: compute all SID embeddings at once
        # Create hierarchy offsets: (H,)
        hierarchy_offsets = self.hierarchy_offsets
        # Add offsets: (B, H) + (H,) -> (B, H)
        sids_with_offset = sids + hierarchy_offsets.unsqueeze(0)
        # Embed all tokens: (B, H) -> (B, H, D)
        sid_embeds = self.sid_embedding_table(sids_with_offset)
        # Sum across hierarchies: (B, H, D) -> (B, D)
        sid_embed_sum = sid_embeds.sum(dim=1)

        # Compute itemkey and embed
        itemkey = self.compute_itemkey(sids)  # (B,)
        item_embed = self.item_embedding_table(itemkey)  # (B, D)

        # Sum pool all embeddings
        block_embed = sid_embed_sum + item_embed  # (B, D)

        # Apply mask if provided
        if attention_mask is not None:
            block_embed = block_embed * attention_mask.unsqueeze(-1)

        return block_embed, attention_mask if attention_mask is not None else torch.ones(B, device=device)

    def build_sequence_embeds(
        self,
        sid_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build backbone input sequence from SID blocks.
        Backbone does NOT use BOS token - only block embeddings.

        Args:
            sid_sequence: (B, S) where S = num_blocks * num_hierarchies
            attention_mask: (B, S)

        Returns:
            sequence_embeds: (B, num_blocks, D)
            mask: (B, num_blocks)
        """
        B, S = sid_sequence.shape
        H = self.num_hierarchies
        D = self.embedding_dim
        device = sid_sequence.device
        num_blocks = S // H

        # Reshape to blocks: (B, num_blocks, H)
        sid_blocks = sid_sequence.view(B, num_blocks, H)
        mask_blocks = attention_mask.view(B, num_blocks, H)

        # Vectorized: compute all SID embeddings at once
        # Use pre-computed hierarchy_offsets from base class (avoid re-creating tensor)
        # hierarchy_offsets: (H,) - [0, num_embeddings_per_hierarchy, 2*num_embeddings_per_hierarchy, ...]
        hierarchy_offsets = self.hierarchy_offsets
        # Add offsets: (B, num_blocks, H) + (H,) -> (B, num_blocks, H)
        sids_with_offset = sid_blocks + hierarchy_offsets.unsqueeze(0).unsqueeze(0)
        # Embed all tokens: (B, num_blocks, H) -> (B, num_blocks, H, D)
        sid_embeds = self.sid_embedding_table(sids_with_offset)
        # Sum across hierarchies: (B, num_blocks, H, D) -> (B, num_blocks, D)
        sid_embed_sum = sid_embeds.sum(dim=2)

        # Compute itemkey for all blocks: (B, num_blocks, H) -> (B, num_blocks)
        itemkeys = self.compute_itemkey(sid_blocks)
        # Embed all items: (B, num_blocks) -> (B, num_blocks, D)
        item_embeds = self.item_embedding_table(itemkeys)

        # Sum SID and item embeddings
        sequence_embeds = sid_embed_sum + item_embeds  # (B, num_blocks, D)

        # Compute block masks: (B, num_blocks, H) -> (B, num_blocks)
        sequence_mask = mask_blocks.any(dim=-1).float()

        # Apply mask: zero out invalid blocks
        sequence_embeds = sequence_embeds * sequence_mask.unsqueeze(-1)

        # Add segment 0 embedding to all positions
        segment_emb = self.segment_embedding_table.weight[0]  # (D,)
        sequence_embeds = sequence_embeds + segment_emb.unsqueeze(0).unsqueeze(0)

        return sequence_embeds, sequence_mask

    def build_similar_item_embeds(
        self,
        similar_itemkey: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build embeddings for similar items."""
        B, S = similar_itemkey.shape
        device = similar_itemkey.device

        similar_item_embeds = self.item_embedding_table(similar_itemkey)
        similar_item_mask = similar_itemkey != self.masking_token
        L2 = similar_item_embeds.size(1)

        s1 = torch.ones(B, L2 - 1, dtype=torch.long, device=device)
        s2 = torch.full((B, 1), 2, dtype=torch.long, device=device)
        segment_ids = torch.cat([s1, s2], dim=1)

        segment_emb = self.segment_embedding_table(segment_ids)
        similar_item_embeds = similar_item_embeds + segment_emb

        return similar_item_embeds, similar_item_mask

    def forward_backbone(
        self,
        sequence_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
        past_key_values = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Forward pass through the backbone decoder.

        Args:
            sequence_embeds: (B, L, D)
            attention_mask: (B, L)
            use_cache: whether to use KV cache
            past_key_values: cached KV from previous steps
            position_ids: (B, L) position indices for RoPE. If None, will be inferred
                          (correct for no-cache case, but MUST be provided when using cache).

        Returns:
            hidden_states: (B, L, D)
            past_key_values: updated cache if use_cache
        """
        if self.decoder is None:
            raise ValueError("Decoder not set. Please configure the decoder in the model config.")

        outputs = self.decoder(
            inputs_embeds=sequence_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            position_ids=position_ids,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        if use_cache:
            return hidden_states, outputs.past_key_values
        return hidden_states, None

    def model_step(
        self,
        model_input: SequentialModelInputData,
        mode: ModelMode,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Perform a forward pass and calculate loss.

        Args:
            model_input: Input data
            mode: TRAIN, EVAL, or INFER

        Returns:
            output: Model output
            loss: Loss tensor
            metrics: Dictionary of metrics
        """
        if mode == ModelMode.INFER:
            return self._infer_step(model_input)

        sid = model_input.transformed_sequences["sequence_data"].long()
        mask = model_input.mask.clone()
        B, S = sid.shape

        # Training mode
        click_label = model_input.transformed_sequences.get("click_label")
        similar_itemkey = model_input.transformed_sequences.get("similar_context")
        hist_itemkey = model_input.transformed_sequences.get("hist_itemkey")
        click_label = click_label.squeeze(-1).float()

        num_pos_samples = click_label.sum().item()
        use_beam_search = (
            (self.training_step_counter.item() % self.beam_search_interval) == 0
            and num_pos_samples > 0
            and mode != ModelMode.EVAL
        )

        if use_beam_search:
            return self._train_with_beam_search(sid, mask, click_label, hist_itemkey, B, S)
        else:
            return self._train_fast(sid, mask, click_label, similar_itemkey, B, S)

    def _train_fast(
        self,
        sid: torch.Tensor,
        mask: torch.Tensor,
        click_label: torch.Tensor,
        similar_itemkey: torch.Tensor,
        B: int,
        S: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Fast training with teacher forcing."""
        H = self.num_hierarchies
        num_blocks = S // H

        # Build sequence embeddings for backbone
        sequence_embeds, sequence_mask = self.build_sequence_embeds(sid, mask)
        similar_context_embeds, similar_context_mask = self.build_similar_item_embeds(similar_itemkey)
        input_embeds = torch.cat([sequence_embeds, similar_context_embeds], dim=1)
        input_mask = torch.cat([sequence_mask, similar_context_mask], dim=1)

        # Forward through backbone
        backbone_hidden, _ = self.forward_backbone(input_embeds, input_mask)

        metrics = {}
        # ===== Click Loss =====
        last_hidden = backbone_hidden[:, -1, :]  # (B, D)
        click_logits = self.click_head(last_hidden).squeeze(-1)  # (B,)

        click_loss = self.click_loss_fn(click_logits, click_label)
        click_probs = torch.sigmoid(click_logits)
        with torch.no_grad():
            metrics["click_auc"] = binary_auroc(click_probs, click_label.long())
            metrics["click_log_prob"] = F.logsigmoid(click_logits).mean()

        # ===== MTP Loss (vectorized across all blocks) =====
        # Extract hidden states for all blocks except the last one
        mtp_hidden = backbone_hidden[:, :num_blocks]

        # Vectorized: process all blocks at once instead of loop
        num_valid_blocks = num_blocks - 1  # Last block has no target
        assert num_valid_blocks > 0, "num_valid_blocks should > 0"
        # hidden states: (B, num_valid_blocks, D) -> (B * num_valid_blocks, D)
        all_hidden = mtp_hidden[:, :num_valid_blocks].reshape(B * num_valid_blocks, -1)

        # target sids: (B, num_valid_blocks * H) -> (B * num_valid_blocks, H)
        # sid[:, H:] starts from block 1, each block has H tokens
        all_target_sids = sid[:, H:(num_blocks * H)].reshape(B * num_valid_blocks, H)

        # valid mask: (B, num_valid_blocks * H) -> (B, num_valid_blocks, H) -> (B, num_valid_blocks) -> (B * num_valid_blocks,)
        all_valid_mask = mask[:, H:(num_blocks * H)].view(B, num_valid_blocks, H).all(dim=-1).float().view(-1)

        # Single batched MTP forward pass
        _, total_mtp_loss, mtp_metrics = self.mtp_head(
            backbone_hidden=all_hidden,
            target_sids=all_target_sids,
            attention_mask=all_valid_mask,
        )
        metrics.update(mtp_metrics)

        total_loss = total_mtp_loss + click_loss
        metrics["mtp_loss"] = total_mtp_loss
        self.training_step_counter.add_(1)

        return backbone_hidden, total_loss, metrics

    def _train_with_beam_search(
        self,
        sid: torch.Tensor,
        mask: torch.Tensor,
        click_label: torch.Tensor,
        hist_itemkey: torch.Tensor,
        B: int,
        S: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Training with beam search for hard negatives."""
        H = self.num_hierarchies

        # Filter positive samples
        pos_mask = (click_label == 1)

        sid_pos = sid[pos_mask]
        mask_pos = mask[pos_mask]
        hist_itemkey_pos = hist_itemkey[pos_mask]
        B_pos = pos_mask.sum().item()

        # Build input for backbone (exclude last block)
        input_sid = sid_pos[:, :-H]
        input_mask = mask_pos[:, :-H]

        sequence_embeds, sequence_mask = self.build_sequence_embeds(input_sid, input_mask)
        seq_len = sequence_embeds.size(1)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(B_pos, -1)
        backbone_hidden, past_key_values = self.forward_backbone(
            sequence_embeds, sequence_mask, use_cache=True, position_ids=position_ids
        )

        # ===== MTP Loss (vectorized across all blocks) =====
        num_input_blocks = backbone_hidden.size(1)
        metrics = {}

        # Vectorized: process all blocks at once instead of loop
        num_valid_blocks = num_input_blocks - 1 # Last input block has no target
        assert num_valid_blocks > 0, "num_valid_blocks should > 0"
        # hidden states: (B_pos, num_valid_blocks, D) -> (B_pos * num_valid_blocks, D)
        all_hidden = backbone_hidden[:, :num_valid_blocks].reshape(B_pos * num_valid_blocks, -1)

        # target sids: (B_pos, num_valid_blocks * H) -> (B_pos * num_valid_blocks, H)
        # Targets start from block 1 (index H) to block num_input_blocks-1
        all_target_sids = sid_pos[:, H:(num_input_blocks * H)].reshape(B_pos * num_valid_blocks, H)

        # valid mask: (B_pos, num_valid_blocks * H) -> (B_pos, num_valid_blocks, H) -> (B_pos, num_valid_blocks) -> (B_pos * num_valid_blocks,)
        all_valid_mask = mask_pos[:, H:(num_input_blocks * H)].view(B_pos, num_valid_blocks, H).any(dim=-1).float().view(-1)

        # Single batched MTP forward pass
        _, total_mtp_loss, mtp_metrics = self.mtp_head(
            backbone_hidden=all_hidden,
            target_sids=all_target_sids,
            attention_mask=all_valid_mask,
        )
        metrics.update(mtp_metrics)

        # Beam search for last block prediction
        last_hidden = backbone_hidden[:, -1, :]  # (B_pos, D)
        beam_size = self.top_k_for_generation

        generated_sids, beam_log_probs = self.mtp_head.generate(
            backbone_hidden=last_hidden,
            beam_size=beam_size,
        )  # generated_sids: (B_pos, beam_size, H)

        # ===== Compute pos_flag: which beam matches the target block =====
        target_block = sid_pos[:, -H:]  # (B_pos, H) - real last block
        target_block_expanded = target_block.unsqueeze(1).expand(-1, beam_size, -1)  # (B_pos, beam_size, H)
        pos_flag = (generated_sids == target_block_expanded).all(dim=-1).float()  # (B_pos, beam_size)

        # ===== Flatten beam dimension for batch processing =====
        generated_sids_flat = generated_sids.view(B_pos * beam_size, H)  # (B_pos * beam_size, H)
        generated_itemkey_flat = self.compute_itemkey(generated_sids_flat)  # (B_pos * beam_size,)

        # Build block embeddings for generated sids
        generated_block_embeds, generated_block_mask = self.build_block_embed(generated_sids_flat)  # (B_pos * beam_size, D), (B_pos * beam_size,)

        # Expand hist_itemkey for beam search: (B_pos, num_hist) -> (B_pos * beam_size, num_hist)
        num_hist = hist_itemkey_pos.size(1)
        hist_itemkey_expanded = hist_itemkey_pos.unsqueeze(1).repeat(1, beam_size, 1).view(B_pos * beam_size, num_hist)

        # Get similar items for each generated item
        similar_context = self.get_similar_itemkey(
            target_itemkey=generated_itemkey_flat,
            hist_itemkey=hist_itemkey_expanded,
        )  # (B_pos * beam_size, top_k_for_score + 1)
        similar_context_embeds, similar_context_mask = self.build_similar_item_embeds(similar_context)

        # Expand past KV cache and sequence mask for beam
        past_key_values = self.repeat_kv_cache(past_key_values, beam_size)
        num_input_blocks = sequence_mask.size(1)
        sequence_mask_expanded = sequence_mask.unsqueeze(1).repeat(1, beam_size, 1).view(B_pos * beam_size, num_input_blocks)

        # Concatenate generated block embed with similar context
        input_embeds = torch.cat([generated_block_embeds.unsqueeze(1), similar_context_embeds], dim=1)
        input_mask = torch.cat([
            sequence_mask_expanded,
            generated_block_mask.unsqueeze(1),
            similar_context_mask,
        ], dim=1)

        # Position IDs for continuation with KV cache (start from num_input_blocks)
        new_seq_len = input_embeds.size(1)
        new_position_ids = torch.arange(
            num_input_blocks, num_input_blocks + new_seq_len, device=self.device
        ).unsqueeze(0).expand(B_pos * beam_size, -1)

        # Forward through backbone with generated block (using KV cache)
        backbone_hidden, _ = self.forward_backbone(
            input_embeds, input_mask, use_cache=True,
            past_key_values=past_key_values, position_ids=new_position_ids
        )

        # ===== Click Loss =====
        last_hidden = backbone_hidden[:, -1, :]  # (B_pos * beam_size, D)
        click_logits = self.click_head(last_hidden).squeeze(-1)  # (B_pos * beam_size,)

        # Flatten pos_flag for loss computation
        pos_flag_flat = pos_flag.view(B_pos * beam_size)  # (B_pos * beam_size,)

        click_loss = self.click_loss_fn(click_logits, pos_flag_flat)
        click_probs = torch.sigmoid(click_logits)
        with torch.no_grad():
            metrics["click_auc"] = binary_auroc(click_probs, pos_flag_flat.long())
            metrics["click_log_prob"] = F.logsigmoid(click_logits).mean()

        metrics["beam_log_prob"] = beam_log_probs.mean()

        total_loss = total_mtp_loss + click_loss
        metrics["mtp_loss"] = total_mtp_loss

        self.training_step_counter.add_(1)

        return backbone_hidden, total_loss, metrics
    
    def _infer_step(
        self,
        model_input: SequentialModelInputData,
    ) -> dict:
        """
        Inference step with beam search.

        Returns:
            Dictionary with beam and rerank results
        """
        sid = model_input.transformed_sequences.get("sequence_data")
        mask = model_input.mask
        hist_itemkey = model_input.transformed_sequences.get("hist_itemkey")

        H = self.num_hierarchies
        B, S = sid.shape

        sequence_embeds, sequence_mask = self.build_sequence_embeds(sid, mask)
        seq_len = sequence_embeds.size(1)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(B, -1)

        # Forward through backbone with cache
        backbone_hidden, past_key_values = self.forward_backbone(
            sequence_embeds, sequence_mask, use_cache=True, position_ids=position_ids
        )

        # Beam search for last block prediction
        last_hidden = backbone_hidden[:, -1, :]  # (B, D)
        beam_size = self.top_k_for_generation

        generated_sids, beam_log_probs = self.mtp_head.generate(
            backbone_hidden=last_hidden,
            beam_size=beam_size,
        )  # generated_sids: (B, beam_size, H)

        # ===== Flatten beam dimension for batch processing =====
        generated_sids_flat = generated_sids.view(B * beam_size, H)  # (B * beam_size, H)
        generated_itemkey_flat = self.compute_itemkey(generated_sids_flat)  # (B * beam_size,)

        # Build block embeddings for generated sids
        generated_block_embeds, generated_block_mask = self.build_block_embed(generated_sids_flat)  # (B * beam_size, D), (B * beam_size,)

        # Expand hist_itemkey for beam search: (B, num_hist) -> (B * beam_size, num_hist)
        num_hist = hist_itemkey.size(1)
        hist_itemkey_expanded = hist_itemkey.unsqueeze(1).repeat(1, beam_size, 1).view(B * beam_size, num_hist)

        # Get similar items for each generated item
        similar_context = self.get_similar_itemkey(
            target_itemkey=generated_itemkey_flat,
            hist_itemkey=hist_itemkey_expanded,
        )  # (B * beam_size, top_k_for_score + 1)
        similar_context_embeds, similar_context_mask = self.build_similar_item_embeds(similar_context)

        # Expand past KV cache and sequence mask for beam
        past_key_values = self.repeat_kv_cache(past_key_values, beam_size)
        num_input_blocks = sequence_mask.size(1)
        sequence_mask_expanded = sequence_mask.unsqueeze(1).repeat(1, beam_size, 1).view(B * beam_size, num_input_blocks)

        # Concatenate generated block embed with similar context
        input_embeds = torch.cat([generated_block_embeds.unsqueeze(1), similar_context_embeds], dim=1)
        input_mask = torch.cat([
            sequence_mask_expanded,
            generated_block_mask.unsqueeze(1),
            similar_context_mask,
        ], dim=1)

        # Position IDs for continuation with KV cache (start from num_input_blocks)
        new_seq_len = input_embeds.size(1)
        new_position_ids = torch.arange(
            num_input_blocks, num_input_blocks + new_seq_len, device=self.device
        ).unsqueeze(0).expand(B * beam_size, -1)

        # Forward through backbone with generated block (using KV cache)
        backbone_hidden, _ = self.forward_backbone(
            input_embeds, input_mask, use_cache=True,
            past_key_values=past_key_values, position_ids=new_position_ids
        )

        last_hidden = backbone_hidden[:, -1, :]  # (B * beam_size, D)
        click_logits = self.click_head(last_hidden).squeeze(-1)  # (B * beam_size,)
        click_log_probs = F.logsigmoid(click_logits).view(B, beam_size)  # (B, beam_size)
        self.click_log_prob_accumulator(click_log_probs.mean())
        self.beam_log_prob_accumulator(beam_log_probs.mean())

        final_score = click_log_probs + beam_log_probs  # (B, beam_size)
        topk_results = torch.topk(final_score, k=self.top_k_for_score, dim=-1)
        rerank_scores, indices_topk = topk_results.values, topk_results.indices

        replace_indices = (
            indices_topk
            + (torch.arange(B, device=self.device) * self.top_k_for_generation).unsqueeze(-1)
        ).flatten()
        rerank_ids = generated_sids_flat[replace_indices].view(B, self.top_k_for_score, -1)

        return {
            "beam": {
                "ids": generated_sids,
                "scores": beam_log_probs,
            },
            "rerank": {
                "ids": rerank_ids,
                "scores": rerank_scores,
            }
        }