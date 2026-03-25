"""
TIGER-style Semantic ID Encoder-Decoder Model.

This module implements the encoder-decoder module proposed in TIGER paper,
See Figure 2.b in https://arxiv.org/pdf/2305.05065.

Key features:
- Hierarchical beam search for semantic ID generation
- Click prediction for reranking
- Constrained beam search with prefix validation
- Separation tokens between items
"""

import logging
import numpy as np
import time
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.aggregation import BaseAggregator, MeanMetric
from torchmetrics.functional.classification import binary_auroc
from transformers.cache_utils import DynamicCache

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.modules.semantic_id.base import SemanticIDBaseRecommender
from src.models.modules.semantic_id.components import ModelMode


class SemanticIDEncoderDecoder(SemanticIDBaseRecommender):
    """
    TIGER-style encoder-decoder for semantic ID generation.

    This implementation includes:
    - Hierarchical beam search for generating semantic IDs
    - Click prediction head for reranking beam candidates
    - Constrained beam search to ensure valid semantic ID prefixes
    - Separation tokens between items in the sequence
    """

    def __init__(
        self,
        top_k_for_generation: int = 20,
        top_k_for_score: int = 10,
        codebooks: torch.Tensor = None,
        sorted_item_keys: torch.Tensor = None,
        sorted_row_indices: torch.Tensor = None,
        scl_embedding_path: torch.Tensor = None,
        embedding_dim: int = None,
        num_hierarchies: int = None,
        num_embeddings_per_hierarchy: int = None,
        num_user_bins: Optional[int] = None,
        should_check_prefix: bool = False,
        prediction_key_name: str = "user_id",
        prediction_value_name: str = "semantic_ids",
        decoder: torch.nn.Module = None,
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDEncoderDecoder.

        Args:
            codebooks: Codebooks for semantic ID.
            num_hierarchies: Number of hierarchies in the codebooks.
            top_k_for_generation: Number of top-k candidates for generation.
            top_k_for_score: Number of top-k candidates for scoring.
            num_user_bins: Number of user bins for user embedding.
            embedding_dim: Dimension of embeddings (inferred from decoder if not provided).
            should_check_prefix: Whether to check prefix validity.
            prediction_key_name: Key name for prediction output.
            prediction_value_name: Value name for prediction output.
            decoder: Decoder module (e.g., LlamaModel).
        """
        if embedding_dim is None:
            embedding_dim = decoder.config.hidden_size

        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            top_k_for_score=top_k_for_score,
            padding_token=kwargs.get("padding_token", 0),
            masking_token=kwargs.get("masking_token", -1),
            sorted_item_keys=sorted_item_keys,
            sorted_row_indices=sorted_row_indices,
            scl_embedding_path=scl_embedding_path,
            beam_search_interval=kwargs.get("beam_search_interval", 50),
            num_user_bins=num_user_bins,
            should_check_prefix=should_check_prefix,
            **kwargs,
        )

        # ===== BOS Token =====
        bos_token = nn.Parameter(
            torch.randn(1, self.embedding_dim), requires_grad=True
        )

        # ===== Decoder Module =====
        self.decoder = SemanticIDDecoderModule(
            decoder=decoder,
            bos_token=bos_token,
            sid_head=nn.ModuleList([
                nn.Linear(self.embedding_dim, self.num_embeddings_per_hierarchy, bias=False)
                for _ in range(self.num_hierarchies)
            ]),
            click_head=nn.Linear(self.embedding_dim, 1, bias=False),
        )

        # ===== Prediction Output Names =====
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name

    def _make_deterministic(self, is_training: bool) -> None:
        """
        Set model to deterministic mode for inference.

        This is needed because Lightning's default hooks don't properly
        handle the encoder/decoder sub-modules.
        """
        if is_training:
            if self.decoder is not None:
                self.decoder.decoder.is_training = True
                self.decoder.decoder.train()
        else:
            if self.decoder is not None:
                self.decoder.decoder.is_training = False
                self.decoder.decoder.eval()

    def on_predict_start(self) -> None:
        super().on_predict_start()
        self._make_deterministic(is_training=False)

    def on_predict_end(self) -> None:
        super().on_predict_end()
        self._make_deterministic(is_training=True)

    def on_validation_start(self) -> None:
        super().on_validation_start()
        self._make_deterministic(is_training=False)

    def on_validation_end(self) -> None:
        super().on_validation_end()
        self._make_deterministic(is_training=True)

    def on_test_start(self) -> None:
        super().on_test_start()
        self._make_deterministic(is_training=False)

    def on_test_end(self) -> None:
        super().on_test_end()
        self._make_deterministic(is_training=True)

    def on_train_start(self) -> None:
        super().on_train_start()
        self._make_deterministic(is_training=True)

    # ===== Core Methods =====

    def training_step(
        self,
        batch: Tuple[SequentialModelInputData],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step."""
        batch = batch[0]
        model_input: SequentialModelInputData = batch
        model_output, loss, metrics = self.model_step(
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

        if self.training_loop_function is not None:
            self.training_loop_function(self, loss)

        lr = self.lr_schedulers().get_last_lr()[0]
        self.log("train/lr", lr, on_step=True, on_epoch=False, sync_dist=False)

        return loss

    def eval_step(
        self,
        batch: SequentialModelInputData,
        loss_to_aggregate: BaseAggregator,
    ) -> None:
        """Perform a single evaluation step."""
        with torch.inference_mode():
            _, loss, __ = self.model_step(model_input=batch, mode=ModelMode.TRAIN)
        loss_to_aggregate(loss)

        sid = batch.transformed_sequences["sequence_data"].long()
        hist_itemkey = batch.transformed_sequences["hist_itemkey"]
        attention_mask = batch.mask

        lengths = attention_mask.sum(dim=1)
        assert torch.all(lengths >= self.num_hierarchies), (
            f"Found sequence length < num_hierarchies={self.num_hierarchies}: {lengths}"
        )
        assert torch.all(lengths % self.num_hierarchies == 0), (
            f"Input lengths must be multiples of block={self.num_hierarchies}, got {lengths}"
        )

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

    def predict_step(self, batch: SequentialModelInputData, batch_idx: int):
        """Perform a single prediction step."""
        generated_sids, _ = self.model_step(batch)
        ids = [
            id.item() if isinstance(id, torch.Tensor) else id
            for id in batch.user_id_list
        ]
        model_output = OneKeyPerPredictionOutput(
            keys=ids,
            predictions=generated_sids,
            key_name=self.prediction_key_name,
            prediction_name=self.prediction_value_name,
        )
        return model_output

    def model_step(
        self,
        model_input: SequentialModelInputData,
        mode: ModelMode,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Perform a forward pass and calculate the loss.

        Args:
            model_input: The input data.
            mode: TRAIN or INFER mode.

        Returns:
            Tuple of (output, loss, metrics).
        """
        sid = model_input.transformed_sequences["sequence_data"].long()
        mask = model_input.mask.clone()
        hist_itemkey = model_input.transformed_sequences["hist_itemkey"].long()

        B, S = sid.shape

        if mode is ModelMode.INFER:
            result_dict = self.generate(
                attention_mask=mask,
                sid=sid,
                hist_itemkey=hist_itemkey,
            )
            return result_dict

        similar_itemkey = model_input.transformed_sequences["similar_context"].long()
        click_label = model_input.transformed_sequences["click_label"].squeeze(-1).float()

        num_pos_samples = (click_label == 1).sum().item()
        use_beam_search = (
            (self.training_step_counter.item() % self.beam_search_interval) == 0
            and num_pos_samples > 0
        )

        if use_beam_search:
            return self._model_step_with_beam_search(
                sid=sid,
                mask=mask,
                hist_itemkey=hist_itemkey,
                click_label=click_label,
                B=B,
                S=S,
            )
        else:
            return self._model_step_fast(
                sid=sid,
                mask=mask,
                similar_itemkey=similar_itemkey,
                click_label=click_label,
                B=B,
                S=S,
            )

    def _model_step_with_beam_search(
        self,
        sid: torch.Tensor,
        mask: torch.Tensor,
        hist_itemkey: torch.Tensor,
        click_label: torch.Tensor,
        B: int,
        S: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Training step with beam search for hard negative sampling."""
        pos_mask = (click_label == 1)
        num_pos = pos_mask.sum().item()
        assert num_pos > 0, "Beam search training requires at least one positive sample."

        sid_pos = sid[pos_mask]
        mask_pos = mask[pos_mask]
        hist_itemkey_pos = hist_itemkey[pos_mask]
        B_pos = int(num_pos)

        # Part 1: SID loss on full sequence
        input_embeds_sid, mask_sid = self.build_sid_block_embeds(sid=sid_pos, attention_mask=mask_pos)

        sid_output = self.decoder(
            sequence_embedding=input_embeds_sid,
            attention_mask=mask_sid,
            use_cache=False,
            past_key_values=None,
        )

        num_block = S // self.num_hierarchies
        sid_hidden_blk = sid_output.view(B_pos, num_block, -1, self.embedding_dim)
        sid_targets = sid_pos.view(B_pos, num_block, self.num_hierarchies).clone()
        mask_blk = mask_pos.view(B_pos, num_block, self.num_hierarchies).clone()

        total_sid_loss = 0
        metrics = {}

        for h in range(self.num_hierarchies):
            h_logits = self.decoder.sid_head[h](sid_hidden_blk[:, :, h, :])
            h_targets = sid_targets[:, :, h].long() - 1
            h_mask = mask_blk[:, :, h]

            h_loss = self.sid_loss_fn(
                h_logits.reshape(-1, h_logits.size(-1)),
                h_targets.reshape(-1)
            )
            total_sid_loss += h_loss

            with torch.no_grad():
                _, top10_indices = h_logits.topk(10, dim=-1)
                hit = (top10_indices == h_targets.unsqueeze(-1)).any(dim=-1)
                hit_rate = (hit * h_mask).sum() / h_mask.sum().clamp_min(1)
                metrics[f"hit@10_sid{h}"] = hit_rate

        # Part 2: Click loss with beam search
        target_itemkey = hist_itemkey_pos[:, -1]
        hist_sid = sid_pos[:, :-self.num_hierarchies]
        hist_mask = mask_pos[:, :-self.num_hierarchies]
        hist_itemkey_for_beam = hist_itemkey_pos[:, :-1]

        _, loss_bce, click_metrics = self._compute_click_loss_with_beam_search(
            hist_sid=hist_sid,
            hist_mask=hist_mask,
            hist_itemkey_for_beam=hist_itemkey_for_beam,
            target_itemkey=target_itemkey,
            batch_size=B_pos,
        )
        metrics["beam_log_prob"] = click_metrics["beam_log_prob"]
        metrics["click_auc"] = click_metrics["click_auc"]
        metrics["click_log_prob"] = click_metrics["click_log_prob"]

        self.training_step_counter.add_(1)

        return sid_output, total_sid_loss + loss_bce, metrics

    def _model_step_fast(
        self,
        sid: torch.Tensor,
        mask: torch.Tensor,
        similar_itemkey: torch.Tensor,
        click_label: torch.Tensor,
        B: int,
        S: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Fast training step using teacher forcing."""
        input_embeds_sid, mask_sid = self.build_sid_block_embeds(sid=sid, attention_mask=mask)
        input_embeds_similar_item, mask_similar_item = self.build_similar_item_embeds(similar_itemkey)
        input_embeds = torch.concat([input_embeds_sid, input_embeds_similar_item], dim=1)
        input_mask = torch.concat([mask_sid, mask_similar_item], dim=1)

        model_output = self.decoder(
            sequence_embedding=input_embeds,
            attention_mask=input_mask,
            past_key_values=None,
        )

        click_logits = self.decoder.click_head(model_output[:, -1, :]).squeeze(-1)
        click_probs = torch.sigmoid(click_logits)
        loss_bce = self.click_loss_fn(click_logits, click_label)

        with torch.no_grad():
            auc = binary_auroc(click_probs, click_label.long())
            click_log_prob = F.logsigmoid(click_logits).mean()

        # SID loss
        L2 = similar_itemkey.size(1)
        sid_hidden = model_output[:, :model_output.size(1) - L2, :]
        num_block = S // self.num_hierarchies
        sid_hidden_blk = sid_hidden.view(B, num_block, -1, self.embedding_dim)
        sid_targets = sid.view(B, num_block, self.num_hierarchies).clone()
        mask_blk = mask.view(B, num_block, self.num_hierarchies).clone()

        neg_mask = (click_label == 0)
        sid_targets[neg_mask, -1, :] = self.padding_token
        mask_blk[neg_mask, -1, :] = 0

        total_sid_loss = 0
        metrics = {
            "click_auc": auc,
            "click_log_prob": click_log_prob,
        }

        for h in range(self.num_hierarchies):
            h_logits = self.decoder.sid_head[h](sid_hidden_blk[:, :, h, :])
            h_targets = sid_targets[:, :, h].long() - 1
            h_mask = mask_blk[:, :, h]

            h_loss = self.sid_loss_fn(
                h_logits.reshape(-1, h_logits.size(-1)),
                h_targets.reshape(-1)
            )
            total_sid_loss += h_loss

            with torch.no_grad():
                _, top10_indices = h_logits.topk(10, dim=-1)
                hit = (top10_indices == h_targets.unsqueeze(-1)).any(dim=-1)
                hit_rate = (hit * h_mask).sum() / h_mask.sum().clamp_min(1)
                metrics[f"hit@10_sid{h}"] = hit_rate

        self.training_step_counter.add_(1)

        return model_output, total_sid_loss + loss_bce, metrics

    def _compute_click_loss_with_beam_search(
        self,
        hist_sid: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_itemkey_for_beam: torch.Tensor,
        target_itemkey: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute click loss using beam search generated candidates."""
        B = batch_size

        generated_ids, beam_log_prob, input_mask, past_key_values = self._beam_search_semantic_ids(
            sid=hist_sid,
            attention_mask=hist_mask,
        )

        generated_itemkey = (generated_ids * self.powers).sum(dim=-1)

        hist_itemkey_expanded = hist_itemkey_for_beam.repeat_interleave(self.top_k_for_generation, dim=0)
        similar_itemkey = self.get_similar_itemkey(
            target_itemkey=generated_itemkey,
            hist_itemkey=hist_itemkey_expanded
        )

        input_embeds_similar_item, mask_similar_item = self.build_similar_item_embeds(similar_itemkey)

        input_embeds = torch.concat(
            [
                self.embed_sid_token(generated_ids[:, -1:], self.num_hierarchies - 1),
                self.embed_item_token(generated_itemkey.unsqueeze(-1)),
                input_embeds_similar_item,
            ],
            dim=1,
        )
        segment_mask = torch.concat(
            [
                torch.ones(B * self.top_k_for_generation, 1, device=self.device),
                torch.ones(B * self.top_k_for_generation, 1, device=self.device),
                mask_similar_item,
            ],
            dim=-1,
        )
        full_input_mask = torch.concat([input_mask, segment_mask], dim=-1)

        decoder_output = self.decoder(
            sequence_embedding=input_embeds,
            attention_mask=full_input_mask,
            use_cache=False,
            past_key_values=past_key_values,
        )

        click_logits = self.decoder.click_head(decoder_output[:, -1, :]).squeeze(-1)
        click_logits = click_logits.view(B, self.top_k_for_generation)

        target_itemkey_expanded = target_itemkey.unsqueeze(1).expand(B, self.top_k_for_generation)
        generated_itemkey_2d = generated_itemkey.view(B, self.top_k_for_generation)
        click_labels = (generated_itemkey_2d == target_itemkey_expanded).float()

        loss_bce = self.click_loss_fn(click_logits, click_labels)

        with torch.no_grad():
            click_probs = torch.sigmoid(click_logits)
            auc = binary_auroc(click_probs.view(-1), click_labels.view(-1).long())
            click_log_prob = F.logsigmoid(click_logits).mean()

        return decoder_output, loss_bce, {
            "click_auc": auc,
            "click_log_prob": click_log_prob,
            "beam_log_prob": beam_log_prob.mean(),
        }

    # ===== Beam Search Methods =====

    def _beam_search_semantic_ids(
        self,
        sid: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DynamicCache]:
        """
        Perform hierarchical beam search to generate semantic IDs.

        Args:
            sid: Input semantic ID sequence (B, S)
            attention_mask: Attention mask for the input (B, S)

        Returns:
            generated_ids: Generated semantic IDs (B * top_k, num_hierarchies)
            beam_log_prob: Log probabilities of beams (B, top_k)
            input_mask: Attention mask for generated sequences (B * top_k, S')
            past_key_values: KV cache from the decoder
        """
        B, S = sid.shape
        past_key_values = DynamicCache()
        generated_ids = None
        beam_log_prob = None

        for hierarchy in range(self.num_hierarchies):
            if generated_ids is not None:
                input_embeds = self.embed_sid_token(generated_ids[:, -1:], hierarchy - 1)
            else:
                input_embeds, input_mask = self.build_sid_block_embeds(
                    sid=sid, attention_mask=attention_mask, add_bos_token=True
                )

            decoder_output, past_key_values = self.decoder(
                sequence_embedding=input_embeds,
                attention_mask=input_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

            latest_output_representation = decoder_output[:, -1, :]
            candidate_logits = self.decoder.sid_head[hierarchy](latest_output_representation)

            (
                generated_ids,
                beam_log_prob,
                input_mask,
                past_key_values,
            ) = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                attention_mask=input_mask,
                beam_log_prob=beam_log_prob,
                past_key_values=past_key_values,
                hierarchy=hierarchy,
                batch_size=B,
            )

        return generated_ids, beam_log_prob, input_mask, past_key_values

    def _rerank_with_click_prediction(
        self,
        generated_ids: torch.Tensor,
        beam_log_prob: torch.Tensor,
        input_mask: torch.Tensor,
        past_key_values: DynamicCache,
        hist_itemkey: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rerank beam candidates using click prediction.

        Args:
            generated_ids: Generated semantic IDs (B * top_k, num_hierarchies)
            beam_log_prob: Log probabilities of beams (B, top_k)
            input_mask: Attention mask (B * top_k, S')
            past_key_values: KV cache from beam search
            hist_itemkey: Historical item keys (B, num_items)
            batch_size: Original batch size B

        Returns:
            rerank_ids: Reranked semantic IDs (B, top_k_for_score, num_hierarchies)
            rerank_scores: Reranked scores (B, top_k_for_score)
        """
        B = batch_size
        target_itemkey = (generated_ids * self.powers).sum(dim=-1)
        hist_itemkey = hist_itemkey.repeat_interleave(self.top_k_for_generation, dim=0)

        similar_itemkey = self.get_similar_itemkey(target_itemkey=target_itemkey, hist_itemkey=hist_itemkey)
        input_embeds_similar_item, mask_similar_item = self.build_similar_item_embeds(similar_itemkey)

        input_embeds = torch.concat(
            [
                self.embed_sid_token(generated_ids[..., -1:], self.num_hierarchies - 1),
                self.embed_item_token(target_itemkey.unsqueeze(-1)),
                input_embeds_similar_item,
            ],
            dim=1,
        )
        input_mask = torch.concat(
            [input_mask, torch.ones(B * self.top_k_for_generation, 1, device=self.device), mask_similar_item],
            dim=-1
        )

        decoder_output = self.decoder(
            sequence_embedding=input_embeds,
            attention_mask=input_mask,
            use_cache=False,
            past_key_values=past_key_values,
        )

        click_logits = self.decoder.click_head(decoder_output[:, -1, :]).view(B, self.top_k_for_generation)
        click_log_prob = F.logsigmoid(click_logits)
        self.click_log_prob_accumulator(click_log_prob.mean())
        self.beam_log_prob_accumulator(beam_log_prob.mean())

        final_score = click_log_prob + beam_log_prob
        topk_results = torch.topk(final_score, k=self.top_k_for_score, dim=-1)
        rerank_scores, indices_topk = topk_results.values, topk_results.indices

        replace_indices = (
            indices_topk
            + (torch.arange(B, device=self.device) * self.top_k_for_generation).unsqueeze(-1)
        ).flatten()

        rerank_ids = generated_ids[replace_indices].view(B, self.top_k_for_score, -1)

        return rerank_ids, rerank_scores

    def generate(
        self,
        attention_mask: torch.Tensor,
        sid: torch.Tensor,
        hist_itemkey: torch.Tensor,
    ) -> dict:
        """
        Generate the semantic id given the current model using beam search.

        Args:
            attention_mask: The attention mask for the decoder
            sid: Input semantic ID sequence
            hist_itemkey: Historical item keys for similarity computation

        Returns:
            Dictionary containing beam search results and reranked results
        """
        B, S = sid.shape

        # Step 1: Beam search to generate semantic IDs
        generated_ids, beam_log_prob, input_mask, past_key_values = self._beam_search_semantic_ids(
            sid=sid,
            attention_mask=attention_mask,
        )

        beam_ids = generated_ids.view(B, self.top_k_for_generation, -1)
        beam_scores = beam_log_prob

        # Step 2: Rerank using click prediction
        rerank_ids, rerank_scores = self._rerank_with_click_prediction(
            generated_ids=generated_ids,
            beam_log_prob=beam_log_prob,
            input_mask=input_mask,
            past_key_values=past_key_values,
            hist_itemkey=hist_itemkey,
            batch_size=B,
        )

        return {
            "beam": {
                "ids": beam_ids,
                "scores": beam_scores
            },
            "rerank": {
                "ids": rerank_ids,
                "scores": rerank_scores
            }
        }

    # ===== Embedding Building Methods =====

    def build_sid_block_embeds(
        self,
        sid: torch.Tensor,
        attention_mask: torch.Tensor,
        add_bos_token: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build block-level embeddings from SID sequence.

        Block format: [bos, sid0, sid1, sid2, sid3, itemkey]
        """
        B, S = sid.shape
        device = sid.device
        H = self.num_hierarchies

        sid_blk = sid.view(B, -1, H)
        item_key = (sid_blk * self.powers).sum(dim=-1)

        sid_indexed = sid_blk + self.hierarchy_offsets.to(device)
        sid_indexed = sid_indexed.masked_fill(sid_blk == self.padding_token, self.padding_token)
        sid_embeds = self.sid_embedding_table(sid_indexed)
        item_embeds = self.item_embedding_table(item_key).unsqueeze(2)

        bos_token = self.decoder.bos_token.view(1, 1, 1, -1).expand(B, sid_blk.size(1), 1, -1)
        full_block_embeds = torch.cat([bos_token, sid_embeds, item_embeds], dim=2)
        inputs_embeds = full_block_embeds.view(B, -1, self.embedding_dim)

        att_blk = attention_mask.view(B, -1, H)
        item_mask = att_blk[..., -1:]
        bos_mask = att_blk[..., 0].unsqueeze(-1)
        mask = torch.cat([bos_mask, att_blk, item_mask], dim=-1).view(B, -1)

        if add_bos_token:
            bos = self.decoder.bos_token.view(1, 1, -1).expand(B, 1, -1)
            inputs_embeds = torch.cat([inputs_embeds, bos], dim=1)

            bos_mask = torch.ones(B, 1, device=device)
            mask = torch.cat([mask, bos_mask], dim=1)

        L1 = inputs_embeds.size(1)
        segment_emb = self.segment_embedding_table.weight[0].view(1, 1, -1).expand(B, L1, -1)
        inputs_embeds = inputs_embeds + segment_emb

        return inputs_embeds, mask

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
        inputs_embeds = similar_item_embeds + segment_emb

        return inputs_embeds, similar_item_mask


class SemanticIDDecoderModule(nn.Module):
    """
    Decoder module for Semantic ID generation.

    This wraps a HuggingFace decoder (e.g., LlamaModel) with:
    - SID prediction heads (one per hierarchy level)
    - Click prediction head
    - BOS token for sequence initiation
    """

    def __init__(
        self,
        decoder,
        sid_head: Optional[nn.Module] = None,
        click_head: Optional[nn.Module] = None,
        bos_token: Optional[nn.Parameter] = None,
    ) -> None:
        """
        Initialize the SemanticIDDecoderModule.

        Args:
            decoder: The encoder model (e.g., transformers.LlamaModel).
            sid_head: MLP layers for SID prediction per hierarchy.
            click_head: Linear layer for click prediction.
            bos_token: BOS token parameter for decoder prompting.
        """
        super().__init__()

        self.decoder = decoder
        self.bos_token = bos_token
        self.sid_head = sid_head
        self.click_head = click_head

        # Remove embedding table in the decoder to save space
        self.decoder.embed_tokens = nn.Identity()

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
        use_cache: bool = False,
        past_key_values: DynamicCache = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder."""
        if past_key_values is None:
            past_key_values = DynamicCache()

        outputs = self.decoder(
            inputs_embeds=sequence_embedding,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        if use_cache:
            return hidden_states, outputs.past_key_values

        return hidden_states