import logging
import numpy as np
import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import transformers
from torch import nn
import torch.nn.functional as F
from torchmetrics.aggregation import BaseAggregator, MeanMetric
from torchmetrics.functional.classification import binary_auroc
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import Seq2SeqModelOutput

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.components.network_blocks.mlp import MLP
from src.models.modules.huggingface.transformer_base_module import TransformerBaseModule, ModelMode


class SemanticIDGenerativeRecommender(TransformerBaseModule):
    """
    This is a base class for the generative recommender model.
    It is used to generate the semantic ID for the given input.
    It does not contain any specific implementation for the encoder or decoder.
    The encoder and decoder are defined in the subclasses.
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        embedding_dim: int,
        should_check_prefix: bool,
        top_k_for_generation: int,
        top_k_for_score: int,
        padding_token: int,
        masking_token: int,
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDGenerativeRecommender module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        num_embeddings_per_hierarchy (int): the number of embeddings per hierarchy.
        embedding_dim (int): the dimension of the embeddings.
        top_k_for_generation (int): the number of top-k candidates for generation.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """
        super().__init__(**kwargs)
        self.padding_token = padding_token
        self.masking_token = masking_token
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.embedding_dim = embedding_dim
        self.num_hierarchies = num_hierarchies
        self.should_check_prefix = should_check_prefix
        if codebooks != None:
            self.codebooks = codebooks + 1
            assert (
                self.codebooks.size(1) == num_hierarchies
            ), "codebooks should be of shape (-1, num_hierarchies)"
        else:
            logging.warning(
                "Not using pre-cached codebooks, \
            please make sure that \n \
                            1) dataset is properly pre-processed \n \
                            2) num_hierarchies and  num_embeddings_per_hierarchy are proerly set\
            "
            )

        self.top_k_for_generation = top_k_for_generation
        self.top_k_for_score = top_k_for_score
        self.batch_start_time = None
        self.prev_batch_end_time = None
        self.total_val_time = 0
        self.val_start_time = 0

        # Metrics accumulators for validation logging
        self.click_log_prob_accumulator = MeanMetric()
        self.beam_log_prob_accumulator = MeanMetric()

    def _spawn_embedding_tables(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> torch.nn.Embedding:
        """
        Spawn an embedding table with the given number of embeddings and embedding dimension.

        Parameters:
        num_embeddings (int): the number of embeddings in the table.
        embedding_dim (int): the dimension of the embeddings.
        """
        table = torch.nn.Embedding(
            num_embeddings=num_embeddings,  # type: ignore
            embedding_dim=embedding_dim,  # type: ignore
            padding_idx=self.padding_token,
        )
        return table

    def _beam_search_one_step(
        self,
        candidate_logits: torch.Tensor,
        generated_ids: Union[torch.Tensor, None],
        attention_mask: torch.Tensor, 
        beam_log_prob: Union[torch.Tensor, None],
        past_key_values: Union[DynamicCache, None],
        hierarchy: int,
        batch_size: int,
    ):
        """
        Perform one step of beam search.

        Args:
            candidate_logits: The logits for the next token.
            generated_ids: The generated IDs so far.
            beam_log_prob: Accumulated (joint) log probabilities of current beams.  (B, top_k)
            attention_mask: The attention mask for input.
            hierarchy: The current hierarchy level.
            batch_size: The size of the batch.

        Returns:
            The updated generated IDs and the marginal probabilities.
        """

        candidate_logits = F.log_softmax(candidate_logits, dim=-1)
        proba, indices = torch.sort(candidate_logits, descending=True)

        if hierarchy == 0:
            proba_topk, indices_topk = (
                proba[:, : self.top_k_for_generation],
                indices[:, : self.top_k_for_generation],
            )  # (B, top_k)

            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.top_k_for_generation, 1) # (B, top_k, S)

            next_sid = indices_topk.unsqueeze(-1) + 1  # (B, top_k, 1)
            next_mask = torch.ones(batch_size, self.top_k_for_generation, 1, device=attention_mask.device, dtype=attention_mask.dtype)  # (B, top_k, 1)
            
            generated_ids = next_sid.view(batch_size * self.top_k_for_generation, -1)  # (B * top_k, S + 1)
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1).view(batch_size * self.top_k_for_generation, -1)  # (B * top_k, S + 1)
            past_key_values = self.repeat_kv_cache(past_key_values, self.top_k_for_generation)
        else:
            proba, indices = proba.view(
                -1, self.top_k_for_generation * self.num_embeddings_per_hierarchy
            ), indices.view(
                -1, self.top_k_for_generation * self.num_embeddings_per_hierarchy
            )  # (B, top_k * num_candidates)
            # calculating the marginal probability
            proba =  beam_log_prob.repeat_interleave(self.num_embeddings_per_hierarchy, dim=-1) + proba  # (B, top_k * num_candidates)
            topk_results = torch.topk(proba, k=self.top_k_for_generation, dim=-1)  # (B, top_k)
            proba_topk, indices_topk = topk_results.values, topk_results.indices  # (B, top_k)
            # getting indices of winning beams in the original beams
            replace_indices = (
                (indices_topk // self.num_embeddings_per_hierarchy)
                + torch.arange(indices_topk.size(0), device=proba.device).unsqueeze(1)
                * self.top_k_for_generation
            ).flatten()  # (B * top_k)

            generated_ids = generated_ids[replace_indices]  # (B * top_k, S)
            attention_mask = attention_mask[replace_indices] # (B * top_k, S)

            next_sid = torch.gather(indices, 1, indices_topk).reshape(-1).unsqueeze(-1) + 1  # (B * top_k, 1)
            next_mask = torch.ones(replace_indices.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype)  # (B * top_k, 1)

            generated_ids = torch.cat([generated_ids, next_sid], dim=-1)  # (B * top_k, S + 1)
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1)  # (B * top_k, S + 1)
            past_key_values.reorder_cache(replace_indices.long())

        return generated_ids, proba_topk, attention_mask, past_key_values
    
    def repeat_kv_cache(self, past_key_values, beam_size):
        if past_key_values is None:
            return None

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

    def _make_deterministic(self, is_training: bool):
        """
        Make the model deterministic by turning off some flags.
        This is needed as the default functions in lightning such as
        on_validation_start on_predict_start cannnot properly set the flags
        for the encoder and decoder.
        (TODO) clark: in the future we can revisit this and make it more generic

        Args:
            is_training (bool): Whether the model is in training mode or not.
        """
        if is_training:
            if self.decoder != None:
                self.decoder.decoder.is_training = True
                self.decoder.decoder.train()
            if self.encoder != None:
                self.encoder.encoder.is_training = True
                self.encoder.encoder.train()
        else:
            if self.decoder != None:
                self.decoder.decoder.is_training = False
                self.decoder.decoder.eval()
            if self.encoder != None:
                self.encoder.encoder.is_training = False
                self.encoder.encoder.eval()

    def on_predict_start(self):
        super().on_predict_start()
        self._make_deterministic(is_training=False)

    def on_predict_end(self):
        super().on_predict_end()
        self._make_deterministic(is_training=True)

    def on_validation_start(self):
        super().on_validation_start()
        self._make_deterministic(is_training=False)
        self.val_start_time = time.time()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.click_log_prob_accumulator.reset()
        self.beam_log_prob_accumulator.reset()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self.log(
            "val/click_log_prob",
            self.click_log_prob_accumulator,
            sync_dist=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val/beam_log_prob",
            self.beam_log_prob_accumulator,
            sync_dist=True,
            prog_bar=False,
            logger=True,
        )

    def on_validation_end(self):
        super().on_validation_end()
        self._make_deterministic(is_training=True)
        val_duration = time.time() - self.val_start_time
        self.total_val_time += val_duration
        
        if self.prev_batch_end_time is not None:
            self.prev_batch_end_time += val_duration

    def on_test_start(self):
        super().on_test_start()
        self._make_deterministic(is_training=False)

    def on_test_end(self):
        super().on_test_end()
        self._make_deterministic(is_training=True)

    def on_train_start(self):
        super().on_train_start()
        self._make_deterministic(is_training=True)

    def on_train_batch_start(self, batch, batch_idx):
        now = time.time()
        if self.prev_batch_end_time is not None:
            data_wait = now - self.prev_batch_end_time
            self.log("perf/next_batch_wait_time", data_wait, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        
        self.batch_start_time = now
        if not hasattr(self, 'train_start_time'):
            self.train_start_time = now
            self.total_samples = 0

    def on_train_batch_end(self, outputs, batch, batch_idx):
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
        
        self.log("perf/train_samples_per_sec_e2e", global_batch_size / step_time, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        self.log("perf/train_samples_per_sec_avg", self.total_samples / running_time, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        self.prev_batch_end_time = time.time()

class SemanticIDEncoderDecoder(SemanticIDGenerativeRecommender):
    """
    This is an in-house implementation of the encoder-decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    We added some additional features and modifications to the original architecture.
    (e.g., constrained beam search, separation tokens, etc)
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
        training_task: str = "sid",
        load_only_weights: bool = False,
        manual_ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDEncoderDecoder module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings_per_hierarchy).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        top_k_for_generation (int): the number of top-k candidates for generation.
        num_user_bins (Optional[int]): the number of bins for user in the dataset (this number equals to the number of rows in the embedding table ).
        embedding_dim (Optional[int]): the dimension of the embeddings.
        should_check_prefix (bool): whether to check if the prefix is valid.
        manual_ckpt_path (Optional[str]): Path to checkpoint for manual weight loading.
            When load_only_weights=True and manual_ckpt_path is set, weights are loaded
            manually in setup() to avoid Lightning's automatic state restoration.
        """
        if isinstance(codebooks, np.ndarray):
            codebooks = torch.from_numpy(codebooks)
        if isinstance(sorted_item_keys, np.ndarray):
            sorted_item_keys = torch.from_numpy(sorted_item_keys)
        if isinstance(sorted_row_indices, np.ndarray):
            sorted_row_indices = torch.from_numpy(sorted_row_indices)
        if embedding_dim is None:
            embedding_dim = kwargs["decoder"].config.hidden_size

        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            top_k_for_score=top_k_for_score,
            should_check_prefix=should_check_prefix,
            **kwargs,
        )

        self.training_task = training_task
        self.load_only_weights = load_only_weights
        self._manual_ckpt_path = manual_ckpt_path

        # bos_token used to prompt the decoder to generate the first token
        bos_token = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim), requires_grad=True
        )

        self.decoder = SemanticIDDecoderModule(
            decoder=self.decoder,
            bos_token=bos_token,
            sid_head=torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.embedding_dim,
                        self.num_embeddings_per_hierarchy,
                        bias=False,
                    )
                    for _ in range(self.num_hierarchies)
                ]
            ),
            click_head=torch.nn.Linear(self.embedding_dim, 1, bias=False),
        )

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings
        self.item_sid_embedding_table_encoder = self._spawn_embedding_tables(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies + 1,
            embedding_dim=self.embedding_dim,
        )
        self.item_embedding_table = DualHashEmbedding(
            num_buckets=300_000,
            embedding_dim=self.embedding_dim,
            masking_token=self.masking_token,
        )
        self.register_buffer(
            "powers",
            self.num_embeddings_per_hierarchy **
            torch.arange(self.num_hierarchies-1, -1, -1),
            persistent=False,
        )
        # Pre-compute hierarchy offsets for SID embedding indexing
        # Avoids repeated tensor creation in build_sid_block_embeds
        self.register_buffer(
            "hierarchy_offsets",
            torch.arange(0, self.num_hierarchies * self.num_embeddings_per_hierarchy,
                         self.num_embeddings_per_hierarchy),
            persistent=False,
        )
        self.register_buffer(
            "sorted_item_keys",
            sorted_item_keys.long(),
            persistent=False,
        )
        self.register_buffer(
            "sorted_row_indices",
            sorted_row_indices.long(),
            persistent=False,
        )
        self.scl_embedding_path = scl_embedding_path
        self.scl_embedding = None

        # generating user embedding table
        self.user_embedding: torch.nn.Embedding = (
            self._spawn_embedding_tables(
                num_embeddings=num_user_bins,
                embedding_dim=self.embedding_dim,
            )
            if num_user_bins
            else None
        )

        self.segment_embedding_table = nn.Embedding(3, self.embedding_dim)

        # the key value names for the prediction output
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name
    
    def setup(self, stage=None):
        if self.scl_embedding is None:
            arr = np.load(self.scl_embedding_path, mmap_mode="r")
            self.scl_embedding = torch.from_numpy(arr)

        if stage == "fit":
            self._set_trainable_params()
            # Manually load weights if load_only_weights is True and ckpt_path is set
            # This avoids Lightning's automatic state restoration
            if self.load_only_weights and hasattr(self, '_manual_ckpt_path') and self._manual_ckpt_path:
                self._load_weights_only()

    def _load_weights_only(self):
        """Load only model weights from checkpoint, skipping optimizer/trainer states."""
        import os
        ckpt_path = self._manual_ckpt_path
        if not os.path.exists(ckpt_path):
            logging.warning(f"Checkpoint not found: {ckpt_path}")
            return

        logging.info(f"[load_only_weights] Loading model weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)

        # Load only model state dict
        if 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("[load_only_weights] Model weights loaded successfully")
        else:
            logging.warning("[load_only_weights] No state_dict found in checkpoint")
            return

        # Mark that we need to reset trainer states in on_fit_start
        self._needs_trainer_reset = True

    def on_fit_start(self) -> None:
        """Reset trainer step/epoch/optimizer/scheduler when weights were manually loaded."""
        super().on_fit_start()
        if getattr(self, '_needs_trainer_reset', False):
            logging.info("[load_only_weights] Resetting trainer global_step and current_epoch to 0")
            # Use internal attributes to reset read-only properties
            self.trainer._global_step = 0
            self.trainer._current_epoch = 0
            self.trainer.fit_loop.epoch_progress.current.reset()
            self.trainer.fit_loop.epoch_progress.current.completed = 0
            self.trainer.fit_loop.epoch_loop.batch_progress.current.reset()
            self.trainer.fit_loop.epoch_loop.batch_progress.current.completed = 0

            # Reset callbacks that track step/epoch state
            for callback in self.trainer.callbacks:
                if hasattr(callback, '_current_epoch'):
                    callback._current_epoch = 0
                if hasattr(callback, '_global_step'):
                    callback._global_step = 0

            # Reset optimizer states (re-initialize instead of clear)
            for optimizer in self.trainer.optimizers:
                # Reset all state values but keep parameter keys
                for param in optimizer.state:
                    optimizer.state[param] = {}
                # Re-initialize state for all parameters
                optimizer.zero_grad(set_to_none=True)

            # Reset scheduler states
            lr_schedulers = getattr(self.trainer, 'lr_scheduler_configs', getattr(self.trainer, 'lr_schedulers', []))
            for scheduler_config in lr_schedulers:
                scheduler_obj = scheduler_config['scheduler'] if isinstance(scheduler_config, dict) else scheduler_config
                if hasattr(scheduler_obj, '_reset'):
                    scheduler_obj._reset()
                elif hasattr(scheduler_obj, 'last_epoch'):
                    scheduler_obj.last_epoch = 0
                if hasattr(scheduler_obj, '_last_lr'):
                    initial_lr = self.trainer.optimizers[0].param_groups[0].get('initial_lr', self.trainer.optimizers[0].param_groups[0]['lr'])
                    scheduler_obj._last_lr = [initial_lr]

            # Clear the flag
            self._needs_trainer_reset = False

    def _set_trainable_params(self):
        """Set requires_grad for head parameters based on training_task.

        This must be called in setup("fit") before DDP wrapper is created.

        New two-stage training flow:
        - Stage 1 (training_task=sid): Train sid_head only (freeze click_head)
        - Stage 2 (training_task=click): Joint training of sid_head and click_head (both trainable)
        """
        if self.training_task == "sid":
            # Stage 1: SID only - only sid_head is trainable
            for p in self.decoder.sid_head.parameters():
                p.requires_grad = True
            for p in self.decoder.click_head.parameters():
                p.requires_grad = False
            logging.info("Stage 1: Training sid_head only (click_head frozen)")
        elif self.training_task == "click":
            # Stage 2: Joint training - both heads are trainable
            for p in self.decoder.sid_head.parameters():
                p.requires_grad = True
            for p in self.decoder.click_head.parameters():
                p.requires_grad = True
            logging.info("Stage 2: Joint training sid_head and click_head")

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
                input_embeds = self.embed_token(token_id=generated_ids[:, -1:], hierarchy=hierarchy - 1)
            else:
                input_embeds, input_mask = self.build_sid_block_embeds(sid=sid, attention_mask=attention_mask, add_bos_token=True)

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

        Note: This method uses the KV cache from beam search to maintain context
        from the user's historical sequence, which is important for understanding
        user preferences when predicting click probability.

        Args:
            generated_ids: Generated semantic IDs (B * top_k, num_hierarchies)
            beam_log_prob: Log probabilities of beams (B, top_k)
            input_mask: Attention mask (B * top_k, S')
            past_key_values: KV cache from beam search (contains historical context)
            hist_itemkey: Historical item keys (B, num_items)
            batch_size: Original batch size B

        Returns:
            rerank_ids: Reranked semantic IDs (B, top_k_for_generation, num_hierarchies)
            rerank_scores: Reranked scores (B, top_k_for_generation)
        """
        B = batch_size
        target_itemkey = (generated_ids * self.powers).sum(dim=-1)
        hist_itemkey = hist_itemkey.repeat_interleave(self.top_k_for_generation, dim=0)

        similar_itemkey = self.get_similar_itemkey(target_itemkey=target_itemkey, hist_itemkey=hist_itemkey)
        input_embeds_similar_item, mask_similar_item = self.build_similar_item_embeds(similar_itemkey=similar_itemkey)

        input_embeds = torch.concat(
            [
                self.embed_token(token_id=generated_ids[..., -1:], hierarchy=self.num_hierarchies-1),
                self.embed_token(token_id=target_itemkey.unsqueeze(-1)),
                input_embeds_similar_item,
            ],
            dim=1,
        )
        input_mask = torch.concat([input_mask, torch.ones(B * self.top_k_for_generation, 1, device=self.device), mask_similar_item], dim=-1)

        # Use KV cache from beam search (use_cache=False means don't update cache, but still use existing cache)
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
        topk_results = torch.topk(final_score, k=self.top_k_for_generation, dim=-1)
        rerank_scores, indices_topk = topk_results.values, topk_results.indices

        replace_indices = (
            indices_topk
            + (torch.arange(B, device=self.device) * self.top_k_for_generation).unsqueeze(-1)
        ).flatten()

        rerank_ids = generated_ids[replace_indices].view(B, self.top_k_for_generation, -1)

        return rerank_ids, rerank_scores

    def generate(
        self,
        attention_mask: torch.Tensor = None,
        sid: torch.Tensor = None,
        hist_itemkey: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate the semantic id given the current model in the sequence using beam search.

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

        # Step 2: Rerank using click prediction (uses KV cache to maintain historical context)
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

    def get_embedding_table(self, table_name: str, hierarchy: Optional[int] = None):
        """
        Get the embedding table for the given table name and hierarchy.
        Args:
            table_name: The name of the table to get the embedding for.
            hierarchy: The hierarchy level to get the embedding for.
        """
        # here we assume the encoder and decoder share the same embedding table
        # we can have flexible embedding table in the future
        if table_name == "sid":
            embedding_table = self.item_sid_embedding_table_encoder
        elif table_name == "item":
            embedding_table = self.item_embedding_table

        if hierarchy is not None:
            return embedding_table(
                torch.arange(
                    hierarchy * self.num_embeddings_per_hierarchy,
                    (hierarchy + 1) * self.num_embeddings_per_hierarchy,
                ).to(self.device)
            )
        return embedding_table

    def predict_step(self, batch: SequentialModelInputData):
        pass

    def eval_step(
        self,
        batch: SequentialModelInputData,
        loss_to_aggregate: BaseAggregator,
    ):
        with torch.inference_mode():
            _, loss, __ = self.model_step(model_input=batch, mode=ModelMode.TRAIN)
        loss_to_aggregate(loss)

        sid = batch.transformed_sequences["sequence_data"].long()  # (B, S)
        hist_itemkey = batch.transformed_sequences["hist_itemkey"]
        attention_mask = batch.mask  # (B, S)

        lengths = attention_mask.sum(dim=1)  # (B)
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

        # Create a new model input to avoid inplace modification of the original batch
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

    def training_step(
        self,
        batch: Tuple[SequentialModelInputData],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data of data (tuple). Because of lightning, the tuple is wrapped in another tuple,
        and the actual batch is at position 0. The batch is a tuple of data where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        batch = batch[0]
        model_input: SequentialModelInputData = batch

        _, loss, metrics = self.model_step(
            model_input=model_input, mode=ModelMode.TRAIN
        )

        # checks logging interval and logs the loss
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

    def model_step(
        self,
        model_input: SequentialModelInputData,
        mode: ModelMode,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Perform a forward pass of the model and calculate the loss if mode is INFER.

        Args:
            model_input: The input data to the model.

        Returns:
            Tuple of (output, loss, metrics) for training mode.
            For INFER mode, returns the generation result dict.
        """
        # ===== Extract all fields from model_input =====
        sid = model_input.transformed_sequences["sequence_data"].long()
        mask = model_input.mask
        hist_itemkey = model_input.transformed_sequences["hist_itemkey"].long()

        B, S = sid.shape
        if mode is ModelMode.INFER:
            result_dict = self.generate(
                attention_mask=mask,
                sid=sid,
                hist_itemkey=hist_itemkey,
            )
            return result_dict

        # ===== Training mode: call appropriate training step based on task =====
        # Use self.training_task to determine which heads to train:
        # "sid": Stage 1 - train sid_head only (freeze click_head)
        # "click": Stage 2 - joint training of sid_head and click_head
        training_task = getattr(self, "training_task", "sid")

        if training_task == "sid":
            return self._model_step_sid(
                sid=sid,
                mask=mask,
                batch_size=B,
                seq_len=S,
            )
        else:  # training_task == "click"
            return self._model_step_click_head(
                sid=sid,
                mask=mask,
                hist_itemkey=hist_itemkey,
                batch_size=B,
                seq_len=S,
            )

    def _model_step_click_head(
        self,
        sid: torch.Tensor,
        mask: torch.Tensor,
        hist_itemkey: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Stage 2: Joint training of click_head and sid_head.
        Both heads are trained together for 90k steps.
        """
        target_itemkey = hist_itemkey[:, -1]
        # Extract target SID (last block of the sequence)
        target_sid = sid[:, -self.num_hierarchies:].contiguous()
        # Remove last block for input (history for beam search)
        hist_sid = sid[:, :-self.num_hierarchies]
        hist_mask = mask[:, :-self.num_hierarchies]
        hist_itemkey_for_beam = hist_itemkey[:, :-1]

        # ===== Compute SID loss on full sequence (to train sid_head) =====
        input_embeds_sid, mask_sid = self.build_sid_block_embeds(sid=sid, attention_mask=mask)

        sid_output = self.decoder(
            sequence_embedding=input_embeds_sid,
            attention_mask=mask_sid,
            use_cache=False,
            past_key_values=None,
        )

        num_block = seq_len // self.num_hierarchies
        sid_hidden_blk = sid_output.view(batch_size, num_block, -1, self.embedding_dim)
        sid_targets = sid.view(batch_size, num_block, self.num_hierarchies).clone()
        mask_blk = mask.view(batch_size, num_block, self.num_hierarchies).clone()

        total_sid_loss = 0
        sid_metrics = {}

        for h in range(self.num_hierarchies):
            h_logits = self.decoder.sid_head[h](sid_hidden_blk[:, :, h, :])  # (B, num_block, C)
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
                sid_metrics[f"hit@10_sid{h}"] = hit_rate

        # ===== Beam search to generate candidates, then compute click loss =====
        _, loss_bce, click_metrics = self._compute_click_loss_with_beam_search(
            hist_sid=hist_sid,
            hist_mask=hist_mask,
            hist_itemkey_for_beam=hist_itemkey_for_beam,
            target_itemkey=target_itemkey,
            target_sid=target_sid,
            batch_size=batch_size,
        )

        # ===== Joint loss: sum of SID loss and click loss =====
        total_loss = total_sid_loss + loss_bce

        metrics = {
            "beam_log_prob": click_metrics["beam_log_prob"],
            "click_auc": click_metrics["click_auc"],
            "click_log_prob": click_metrics["click_log_prob"],
            **sid_metrics,
        }

        return sid_output, total_loss, metrics

    def _model_step_sid(
        self,
        sid: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Stage 2: SID (Semantic ID) generation training.
        Only sid_head is trained for additional 30k steps from joint checkpoint.
        The checkpoint is loaded with full trainer state for LR continuity.
        """
        # ===== SID loss on full sequence =====
        input_embeds_sid, mask_sid = self.build_sid_block_embeds(sid=sid, attention_mask=mask)

        sid_output = self.decoder(
            sequence_embedding=input_embeds_sid,
            attention_mask=mask_sid,
            use_cache=False,
            past_key_values=None,
        )

        num_block = seq_len // self.num_hierarchies
        sid_hidden_blk = sid_output.view(batch_size, num_block, -1, self.embedding_dim)
        sid_targets = sid.view(batch_size, num_block, self.num_hierarchies).clone()
        mask_blk = mask.view(batch_size, num_block, self.num_hierarchies).clone()

        total_sid_loss = 0
        metrics = {}

        for h in range(self.num_hierarchies):
            h_logits = self.decoder.sid_head[h](sid_hidden_blk[:, :, h, :])  # (B, num_block, C)
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

        return sid_output, total_sid_loss, metrics

    def _compute_click_loss_with_beam_search(
        self,
        hist_sid: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_itemkey_for_beam: torch.Tensor,
        target_itemkey: torch.Tensor,
        target_sid: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute click loss using beam search generated candidates (hard negatives).

        Returns:
            decoder_output: The decoder output
            loss_bce: BCE loss for click prediction
            metrics: Dictionary containing click metrics
        """
        B = batch_size

        # Beam search to generate candidates
        generated_ids, beam_log_prob, input_mask, past_key_values = self._beam_search_semantic_ids(
            sid=hist_sid,
            attention_mask=hist_mask,
        )

        # generated_ids: (B * top_k, H)
        # beam_log_prob: (B, top_k)
        generated_itemkey = (generated_ids * self.powers).sum(dim=-1)  # (B * top_k,)

        # Check which samples have no positive sample in beam
        generated_itemkey_2d = generated_itemkey.view(B, self.top_k_for_generation)
        target_itemkey_expanded = target_itemkey.unsqueeze(1).expand(B, self.top_k_for_generation)
        has_hit = (generated_itemkey_2d == target_itemkey_expanded).any(dim=1)  # (B,)
        no_hit_mask = ~has_hit  # (B,)

        if no_hit_mask.any():
            # For samples with no hit, randomly select a beam to replace with GT
            num_no_hit = no_hit_mask.sum().item()
            random_beam_indices = torch.randint(
                0, self.top_k_for_generation, (num_no_hit,), device=generated_ids.device
            )

            # Compute indices in flattened generated_ids
            no_hit_indices = torch.where(no_hit_mask)[0]  # (num_no_hit,)
            replace_indices = no_hit_indices * self.top_k_for_generation + random_beam_indices

            # Replace selected beams with target SID
            generated_ids[replace_indices] = target_sid[no_hit_indices]

            # Recompute generated_itemkey after replacement
            generated_itemkey = (generated_ids * self.powers).sum(dim=-1)  # (B * top_k,)

        # Build segment2 (similar items) for each candidate
        hist_itemkey_expanded = hist_itemkey_for_beam.repeat_interleave(self.top_k_for_generation, dim=0)
        similar_itemkey = self.get_similar_itemkey(
            target_itemkey=generated_itemkey,
            hist_itemkey=hist_itemkey_expanded
        )  # (B * top_k, top_k_for_score + 1)

        input_embeds_similar_item, mask_similar_item = self.build_similar_item_embeds(similar_itemkey=similar_itemkey)

        # Continue forward with segment2 and segment3
        input_embeds = torch.concat(
            [
                self.embed_token(token_id=generated_ids[:, -1:], hierarchy=self.num_hierarchies - 1),
                self.embed_token(token_id=generated_itemkey.unsqueeze(-1)),
                input_embeds_similar_item,
            ],
            dim=1,
        )
        segment_mask = torch.concat(
            [
                torch.ones(B * self.top_k_for_generation, 2, device=self.device),
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

        # Click prediction for all candidates
        click_logits = self.decoder.click_head(decoder_output[:, -1, :]).squeeze(-1)  # (B * top_k,)
        click_logits = click_logits.view(B, self.top_k_for_generation)  # (B, top_k)

        # Create labels - correct candidate gets 1, others get 0
        # Now GT is guaranteed to be in the candidates (either from beam or replaced)
        target_itemkey_expanded = target_itemkey.unsqueeze(1).expand(B, self.top_k_for_generation)
        generated_itemkey_2d = generated_itemkey.view(B, self.top_k_for_generation)
        click_labels = (generated_itemkey_2d == target_itemkey_expanded).float()  # (B, top_k)

        # BCE loss for click prediction
        loss_bce = self.click_loss_fn(click_logits, click_labels)

        with torch.no_grad():
            click_probs = torch.sigmoid(click_logits)
            # AUC: treat as binary classification over all candidates
            auc = binary_auroc(click_probs.view(-1), click_labels.view(-1).long())

            # Mean click log prob for all candidates (same as eval)
            click_log_prob = F.logsigmoid(click_logits).mean()

        return decoder_output, loss_bce, {
            "click_auc": auc,
            "click_log_prob": click_log_prob,
            "beam_log_prob": beam_log_prob.mean(),
        }

    def build_sid_block_embeds(
        self,
        sid: torch.Tensor,
        attention_mask: torch.Tensor,
        add_bos_token: bool = False,
    ):
        """
        block = [bos, sid, ..., sid, itemkey]
        """
        B, S = sid.shape
        device = sid.device
        H = self.num_hierarchies

        sid_table = self.get_embedding_table(table_name="sid")
        item_table = self.get_embedding_table(table_name="item")

        sid_blk = sid.view(B, -1, H)  # (B, num_block, H)
        item_key = (sid_blk * self.powers).sum(dim=-1) # (B, num_block)

        # Use pre-computed hierarchy_offsets buffer instead of creating new tensor
        sid_indexed = sid_blk + self.hierarchy_offsets.to(device)
        sid_indexed = sid_indexed.masked_fill(sid_blk == self.padding_token, self.padding_token)
        sid_embeds = sid_table(sid_indexed)  # (B, num_block, H, D)
        item_embeds = item_table(item_key).unsqueeze(2)  # (B, num_block, 1, D)

        bos_token = self.decoder.bos_token.view(1, 1, 1, -1).expand(B, sid_blk.size(1), 1, -1)
        full_block_embeds = torch.cat([bos_token, sid_embeds, item_embeds], dim=2)
        inputs_embeds = full_block_embeds.view(B, -1, self.embedding_dim)

        att_blk = attention_mask.view(B, -1, H)  # (B, num_block, H)
        item_mask = att_blk[..., -1:]  # (B, num_block)
        bos_mask = att_blk[..., 0].unsqueeze(-1)
        mask = torch.cat([bos_mask, att_blk, item_mask], dim=-1).view(B, -1)

        if add_bos_token:
            bos = self.decoder.bos_token.view(1, 1, -1).expand(B, 1, -1)  # (B, 1, D)
            inputs_embeds = torch.cat([inputs_embeds, bos], dim=1)  # (B, , D)

            bos_mask = torch.ones(B, 1, device=device)
            mask = torch.cat([mask, bos_mask], dim=1)

        L1 = inputs_embeds.size(1)
        # For segment_id=0, use weight directly to avoid creating zero tensor
        segment_emb = self.segment_embedding_table.weight[0].view(1, 1, -1).expand(B, L1, -1)
        inputs_embeds = inputs_embeds + segment_emb

        return inputs_embeds, mask

    def build_similar_item_embeds(
        self,
        similar_itemkey: torch.Tensor,
    ):
        B, S = similar_itemkey.shape
        device = similar_itemkey.device

        item_table = self.get_embedding_table(table_name="item")
        similar_item_embeds = item_table(similar_itemkey)
        similar_item_mask = similar_itemkey != self.masking_token
        L2 = similar_item_embeds.size(1)

        inputs_embeds = similar_item_embeds
        mask = similar_item_mask
        
        s1 = torch.ones(B, L2 - 1, dtype=torch.long, device=device)
        s2 = torch.full((B, 1), 2, dtype=torch.long, device=device)
        segment_ids = torch.cat([s1, s2], dim=1)

        segment_emb = self.segment_embedding_table(segment_ids)
        inputs_embeds = inputs_embeds + segment_emb

        return inputs_embeds, mask

    def embed_token(
        self,
        token_id,
        hierarchy=None,
    ):
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)

        # For segment_id=0, use weight directly to avoid creating zero tensor
        segment_emb = self.segment_embedding_table.weight[0].view(1, 1, -1).expand(
            token_id.size(0), token_id.size(1), -1
        )

        if hierarchy == None:
            item_table = self.get_embedding_table(table_name="item")
            emb = item_table(token_id)
        else:
            assert hierarchy < self.num_hierarchies
            sid_table = self.get_embedding_table(table_name="sid")
            offset = hierarchy * self.num_embeddings_per_hierarchy
            token_id = token_id + offset
            emb = sid_table(token_id)

        return emb + segment_emb

    def itemkey_lookup(self, key):
        """
        Look up item embeddings by key using pre-sorted index arrays.
        Uses pure PyTorch operations to minimize CPU-GPU sync overhead.
        Note: scl_embedding remains on CPU due to memory constraints with large item catalogs.

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

        # Use pure PyTorch indexing instead of numpy conversion
        # This avoids creating a new tensor and reduces synchronization overhead
        emb = self.scl_embedding[row_idx]
        emb[row_idx == -1] = 0

        return emb.float().to(self.device)

    def get_similar_itemkey(self, target_itemkey, hist_itemkey):
        """
        target_itemkey.shape = (B * num_beam)
        hist_itemkey.shape = (B * num_beam, num_items)
        """
        target_emb = self.itemkey_lookup(target_itemkey)  # (B * num_beam, dim)
        hist_emb = self.itemkey_lookup(hist_itemkey)  # (B * num_beam, num_items, dim)

        # cosine similarity
        sim = torch.einsum(
            "bd,bhd->bh",
            target_emb,
            hist_emb
        )  # (B * num_beam, num_items)

        pad_mask = hist_itemkey == self.masking_token
        sim = sim.masked_fill(pad_mask, -1e9)

        topk_idx = torch.topk(sim, k=self.top_k_for_score, dim=-1).indices
        topk_idx = torch.sort(topk_idx, dim=-1).values

        similar_itemkey = torch.gather(
            hist_itemkey,
            dim=1,
            index=topk_idx
        )

        return torch.cat(
            [similar_itemkey, target_itemkey.unsqueeze(-1)],
            dim=-1
        )

class SemanticIDDecoderModule(torch.nn.Module):
    """
    This is an in-house replication of the decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        decoder: transformers.PreTrainedModel,
        sid_head: Optional[torch.nn.Module] = None,
        click_head: Optional[torch.nn.Module] = None,
        bos_token: Optional[torch.nn.Parameter] = None,
    ) -> None:
        """
        Initialize the SemanticIDDecoderModule.

        Parameters:
        decoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        sid_head (torch.nn.Module): the mlp layers used to project the decoder output to the embedding table.
        click_head (torch.nn.Module): the head used for click prediction.
        bos_token (Optional[torch.nn.Parameter]):
            the bos token used to prompt the decoder.
            if None, then this means the decoder is used standalone without an encoder.
        """

        super().__init__()
        # some sanity checks

        self.decoder = decoder
        # this bos token is prompt for the decoder
        self.bos_token = bos_token
        self.sid_head = sid_head
        self.click_head = click_head

        # deleting embedding table in the decoder to save space
        self.decoder.embed_tokens = torch.nn.Identity()

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
        use_cache: bool = False,
        past_key_values: DynamicCache = DynamicCache(),
    ) -> torch.Tensor:
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

class DualHashEmbedding(nn.Module):
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

    def forward(self, item_key: torch.Tensor):
        """
        item_key: (B, num_block) int64
        return: (B, num_block, D)
        """
        # mask shape: (B, num_block)
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