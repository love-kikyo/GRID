import logging
import numpy as np
import time
from typing import Any, Optional, Tuple, Union

import torch
import transformers
from torch import nn
import torch.nn.functional as F
from torchmetrics.aggregation import BaseAggregator
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.models.t5.modeling_t5 import T5Config, T5LayerNorm

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.components.network_blocks.mlp import MLP
from src.models.modules.huggingface.transformer_base_module import TransformerBaseModule, ModelMode
from src.utils.utils import (
    delete_module,
    find_module_shape,
    get_parent_module_and_attr,
    reset_parameters,
)


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
        padding_token: int,
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
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.embedding_dim = embedding_dim
        self.num_hierarchies = num_hierarchies
        self.should_check_prefix = should_check_prefix
        if codebooks != None:
            self.codebooks = codebooks
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
        self.batch_start_time = None

    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject a separator token into the ID embeddings and attention mask.

        Parameters:
        id_embeddings (torch.Tensor): The ID embeddings of shape (batch_size, seq_len, emb_dim).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
        sep_token (torch.Tensor): The separator token of shape (1, emb_dim).
        num_hierarchies (int): The number of hierarchies in the codebooks.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified ID embeddings and attention mask.
        id_embeddings: The ID embeddings with the separator token injected of shape (batch_size, seq_len + num_items, emb_dim).
        attention_mask: The attention mask with the separator token injected of shape (batch_size, seq_len + num_items).

        An intuitive example of the input and output:
        input:
        id_embeddings: [[1, 2, 3, 4], [5, 6, 7, 8]]
        attention_mask: [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
        output:
        id_embeddings: [[1, 2, 3, 4, sep_token], [5, 6, 7, 8, sep_token]]
        attention_mask: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
        """
        batch_size, seq_len, emb_dim = id_embeddings.size()
        item_count_per_sequence = seq_len // num_hierarchies

        reshaped_id_embeddings = id_embeddings.view(
            batch_size, item_count_per_sequence, num_hierarchies, -1
        )
        reshaped_attention_mask = attention_mask.view(
            batch_size, item_count_per_sequence, num_hierarchies
        )
        reshaped_sep_token_for_concat = (
            sep_token.unsqueeze(0)
            .expand(batch_size, item_count_per_sequence, -1)
            .unsqueeze(-2)
        )
        id_embeddings = torch.cat(
            [reshaped_id_embeddings, reshaped_sep_token_for_concat], dim=-2
        )
        attention_mask = torch.cat(
            [reshaped_attention_mask, reshaped_attention_mask[:, :, [-1]]],
            dim=-1,
        )
        id_embeddings = id_embeddings.reshape(batch_size, -1, emb_dim)
        attention_mask = attention_mask.reshape(batch_size, -1)
        return id_embeddings, attention_mask

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

    def _is_kv_cache_valid(
        self, kv_cache: Union[Tuple, DynamicCache, EncoderDecoderCache]
    ) -> bool:

        if isinstance(kv_cache, (EncoderDecoderCache, DynamicCache)):
            return len(kv_cache) > 0
        elif isinstance(kv_cache, Tuple):
            return True
        else:
            return False

    def _add_repeating_offset_to_rows(
        self,
        input_sids: torch.Tensor,
        codebook_size: int,
        num_hierarchies: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Adds repeating offsets to each element in each row of input_sids.
        we use a single embedding table for multiple code books.
        for example if each codebook has 300 embeddings and we have 3 codebooks,
        the input sequence will be transformed from [0, 1, 2] -> to [0, 301, 602]

        Parameters:
            input_sids (torch.Tensor): A 2D PyTorch tensor.
            codebook_size (int): The number of elements in the codebook.
            num_hierarchies (int): The number of hierarchy levels.
        """

        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")

        num_rows, num_cols = input_sids.shape
        offsets = (
            torch.arange(num_hierarchies, device=input_sids.device) * codebook_size
        )

        # Calculate how many times the full offset pattern needs to repeat
        num_repeats = (
            num_cols + num_hierarchies - 1
        ) // num_hierarchies  # Integer division to handle cases where num_cols is not a multiple of num_hierarchies

        # Repeat the offsets and slice to match the number of columns
        repeated_offsets = offsets.repeat(num_repeats)[:num_cols]

        # Add the repeated offsets to each row using broadcasting
        input_sids_with_offsets = input_sids + repeated_offsets
        if attention_mask is not None:
            input_sids_with_offsets = input_sids_with_offsets * attention_mask
        return input_sids_with_offsets

    def _check_valid_prefix(
        self, prefix: torch.Tensor, batch_size: int = 100000
    ) -> torch.Tensor:
        """
        Checks if a given prefix is a valid prefix of the codebooks.

        Args:
            prefix: A tensor of shape [batch_size, hierarchy_level].
            batch_size: The size of the batch to process.

        Returns:
            A boolean tensor of shape [batch_size] indicating the validity of each prefix.
        """
        # TODO (clark): this is a temporary solution, we should use a more efficient way to do this
        # like pre-sorting the codebook and implementing a tree strcture

        current_hierarchy = prefix.shape[1]
        num_prefixes = prefix.shape[0]
        results = []

        # Ensure codebooks are on the correct device.  Do this *once* outside the loop.
        if prefix.device != self.codebooks.device:
            self.codebooks = self.codebooks.to(prefix.device)

        # Trim the codebooks to the relevant hierarchy *once* outside the loop.
        trimmed_codebooks = self.codebooks[:, :current_hierarchy]

        for i in range(0, num_prefixes, batch_size):
            # Get the current batch of prefixes.
            batch_prefix = prefix[
                i : i + batch_size
            ]  # Shape: [batch_size, hierarchy_level]

            # Perform the comparison.  Broadcasting is now limited by batch_size.
            # trimmed_codebooks shape: [C, H] -> unsqueezed [C, 1, H]
            # batch_prefix shape   : [b, H] -> unsqueezed [1, b, H]
            # comparison result    : [C, b, H]
            comparison = trimmed_codebooks.unsqueeze(1) == batch_prefix.unsqueeze(0)

            # Reduce along the hierarchy dimension (H). Shape: [C, b]
            all_match = comparison.all(dim=2)

            # Reduce along the codebook dimension (C).  Shape: [b]
            any_match = all_match.any(dim=0)

            # Append the results for this batch.
            results.append(any_match)

        # Concatenate the results from all batches.
        return torch.cat(results)

    def _beam_search_one_step(
        self,
        candidate_logits: torch.Tensor,
        sid: Union[torch.Tensor, None],
        attention_mask: torch.Tensor, 
        beam_log_prob: Union[torch.Tensor, None],
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

        candidate_logits = torch.nn.functional.log_softmax(candidate_logits, dim=-1)
        proba, indices = torch.sort(candidate_logits, descending=True)

        if hierarchy == 0:
            proba_topk, indices_topk = (
                proba[:, : self.top_k_for_generation],
                indices[:, : self.top_k_for_generation],
            )  # (B, top_k)

            sid = sid.unsqueeze(1).repeat(1, self.top_k_for_generation, 1)  # (B, top_k, S)
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.top_k_for_generation, 1) # (B, top_k, S)

            next_sid = indices_topk.unsqueeze(-1) + 1  # (B, top_k, 1)
            next_mask = torch.ones(batch_size, self.top_k_for_generation, 1, device=attention_mask.device, dtype=attention_mask.dtype)  # (B, top_k, 1)
            
            sid = torch.cat([sid, next_sid], dim=-1).view(batch_size * self.top_k_for_generation, -1)  # (B * top_k, S + 1)
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1).view(batch_size * self.top_k_for_generation, -1)  # (B * top_k, S + 1)
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

            sid = sid[replace_indices]  # (B * top_k, S)
            attention_mask = attention_mask[replace_indices] # (B * top_k, S)

            next_sid = torch.gather(indices, 1, indices_topk).reshape(-1).unsqueeze(-1) + 1  # (B * top_k, 1)
            next_mask = torch.ones(replace_indices.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype)  # (B * top_k, 1)

            sid = torch.cat([sid, next_sid], dim=-1)  # (B * top_k, S + 1)
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1)  # (B * top_k, S + 1)

        return sid, proba_topk, attention_mask

    def eval_step(
        self,
        batch: SequentialModelInputData,
        loss_to_aggregate: BaseAggregator,
    ):
        model_input: SequentialModelInputData = batch
        _, loss, metrics = self.model_step(model_input=model_input, mode=ModelMode.TRAIN)
        loss_to_aggregate(loss)

        sid = model_input.transformed_sequences["sequence_data"].long()  # (B, S)
        attention_mask = model_input.mask  # (B, S)

        lengths = attention_mask.sum(dim=1)  # (B)
        B, S = sid.shape
        device = sid.device

        assert torch.all(lengths >= self.num_hierarchies), (
            f"Found sequence length < num_hierarchies={self.num_hierarchies}: {lengths}"
        )
        assert torch.all(lengths % self.num_hierarchies == 0), (
            f"Input lengths must be multiples of block={self.num_hierarchies}, got {lengths}"
        )

        # ===== Construct label ===== 
        idx = lengths[:, None] - self.num_hierarchies + torch.arange(self.num_hierarchies, device=device)  # (B, num_hierarchies)
        labels = sid.gather(1, idx)  # (B, num_hierarchies)
        
        # ===== Left padding =====
        hist_lengths = lengths - self.num_hierarchies  # (B,)
        max_hist_len = hist_lengths.max().item()

        hist_sid = torch.full(
            (B, max_hist_len),
            self.padding_token,
            device=device,
            dtype=sid.dtype,
        )  # (B, max_hist_len)
        
        pos = torch.arange(max_hist_len, device=device)  # (max_hist_len,)
        start = max_hist_len - hist_lengths[:, None]  # (B, max_hist_len)

        hist_mask = pos >= start    # (B, max_hist_len)
        hist_mask = hist_mask.to(attention_mask.dtype)

        src_idx = pos - start  # (B, max_hist_len)
        safe_src_idx = src_idx.clamp(min=0)

        hist_sid[hist_mask.bool()] = sid.gather(1, safe_src_idx)[hist_mask.bool()]

        model_input.transformed_sequences["sequence_data"] = hist_sid
        model_input.mask = hist_mask

        generated_ids, marginal_probs = self.model_step(model_input=model_input, mode=ModelMode.INFER)
        self.evaluator(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            labels=labels.to(marginal_probs.device),
        )

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

    def on_validation_end(self):
        super().on_validation_end()
        self._make_deterministic(is_training=True)

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
        self.batch_start_time = time.time()
        if not hasattr(self, 'train_start_time'):
            self.train_start_time = time.time()
            self.total_samples = 0

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step_time = time.time() - self.batch_start_time
        model_input = batch[0]
        local_batch_size = model_input.mask.size(0)
        world_size = self.trainer.world_size
        global_batch_size = local_batch_size * world_size

        self.total_samples += global_batch_size
        running_time = time.time() - self.train_start_time
        running_samples_per_sec = self.total_samples / running_time

        self.log(
            "train/samples_per_sec_e2e",
            global_batch_size / step_time,  # 每 step
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )

        self.log(
            "train/samples_per_sec_avg",
            running_samples_per_sec,  # 累积平均
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )

class SemanticIDEncoderDecoder(SemanticIDGenerativeRecommender):
    """
    This is an in-house implementation of the encoder-decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    We added some additional features and modifications to the original architecture.
    (e.g., constrained beam search, separation tokens, etc)
    """

    def __init__(
        self,
        top_k_for_generation: int = 10,
        codebooks: torch.Tensor = None,
        embedding_dim: int = None,
        num_hierarchies: int = None,
        num_embeddings_per_hierarchy: int = None,
        num_user_bins: Optional[int] = None,
        should_check_prefix: bool = False,
        should_add_sep_token: bool = True,
        prediction_key_name: str = "user_id",
        prediction_value_name: str = "semantic_ids",
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
        """
        if isinstance(codebooks, np.ndarray):
            codebooks = torch.from_numpy(codebooks)
        if num_hierarchies is None or num_embeddings_per_hierarchy is None:
            num_hierarchies, num_embeddings_per_hierarchy = (
                codebooks.shape[1],
                codebooks.max().item() + 1,
            )
        if embedding_dim is None:
            embedding_dim = kwargs["decoder"].config.hidden_size

        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            should_check_prefix=should_check_prefix,
            **kwargs,
        )

        if self.encoder != None:
            self.encoder = SemanticIDEncoderModule(
                encoder=self.encoder,
            )

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
        )

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings
        self.item_sid_embedding_table_encoder = self._spawn_embedding_tables(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies + 1,
            embedding_dim=self.embedding_dim,
        )

        # generating user embedding table
        self.user_embedding: torch.nn.Embedding = (
            self._spawn_embedding_tables(
                num_embeddings=num_user_bins,
                embedding_dim=self.embedding_dim,
            )
            if num_user_bins
            else None
        )

        # separation token for the encoder to differentiate between items
        if self.encoder != None:
            self.sep_token = (
                torch.nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)
                if should_add_sep_token
                else None
            )
        # the key value names for the prediction output
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name
    
    def decoder_forward_pass(
        self,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # TODO (clark): in the future we should support variable length semantic id
        sid: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[DynamicCache] = None,
        add_bos_token: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (Optional[DynamicCache]): The cache for past key values.
        """

        # ===== Convert to embedding =====
        B, S = sid.shape
        assert attention_mask.shape == sid.shape
        sid_table = self.get_embedding_table(table_name="decoder")

        pad_len = (-S) % self.num_hierarchies
        sid_pad = nn.functional.pad(sid, (0, pad_len), value=self.padding_token)  # (B, S + pad_len)
        attention_mask_pad = F.pad(attention_mask, (0, pad_len), value=0)  # (B, S + pad_len)

        sid_blk = sid_pad.view(B, -1, self.num_hierarchies)  # (B, num_block, in_block_len)
        attention_mask_blk = attention_mask_pad.view(B, -1, self.num_hierarchies)  # (B, num_block, in_block_len)

        offsets = (
            torch.arange(self.num_hierarchies, device=sid_blk.device)
            * self.num_embeddings_per_hierarchy
        )
        sid_blk_raw = sid_blk
        sid_blk = sid_blk + offsets
        sid_blk = torch.where(
            sid_blk_raw == self.padding_token,
            sid_blk_raw,
            sid_blk
        )
        sid_embeds = sid_table(sid_blk)  # (B, num_block, hirerachies, D)

        bos = self.decoder.bos_token
        bos = bos.view(1, 1, 1, -1).expand(B, sid_embeds.size(1), 1, -1)  # (B, num_block, 1, D)
        mask = torch.ones(B, sid_embeds.size(1), 1, device=sid.device)  # (B, num_block, 1)

        inputs_embeds = torch.cat([bos, sid_embeds], dim=-2)  # (B, num_block, out_block_len, D)
        mask = torch.cat([mask, attention_mask_blk], dim=-1)  # (B, num_block, out_block_len)

        inputs_embeds = inputs_embeds.view(B, -1, self.embedding_dim)  # (B, S + pad_len + num_block, D)
        mask = mask.view(B, -1)  # (B, S + pad + num_block)

        if pad_len > 0:
            inputs_embeds = inputs_embeds[:, :-pad_len]
            mask = mask[:, :-pad_len]

        if add_bos_token:
            bos = self.decoder.bos_token.view(1, 1, -1).expand(B, 1, -1)  # (B, 1, D)
            inputs_embeds = torch.cat([inputs_embeds, bos], dim=1)  # (B, S + num_block + 1, D)

            bos_mask = torch.ones(B, 1, device=mask.device)  
            mask = torch.cat([mask, bos_mask], dim=1)

        decoder_output = self.decoder(
            sequence_embedding=inputs_embeds,
            attention_mask=mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        return decoder_output

    def generate(
        self,
        attention_mask: torch.Tensor = None,
        sid: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate the semantic id given the current model in the sequence using beam search.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
        """
        # initilize cached generated ids to None
        B, S = sid.shape
        beam_log_prob = None

        for hierarchy in range(self.num_hierarchies):
            # feeding the decoder with the generated ids
            add_bos_token = hierarchy == 0
            decoder_output = self.decoder_forward_pass(
                sid=sid,
                attention_mask=attention_mask,
                use_cache=False,
                add_bos_token=add_bos_token
            )  # (B, T, D) if hierarchy == 0 else (B * top_k, T, D)

            # decoder_output[:, -1, :] is the embedding for the next token
            latest_output_representation = decoder_output[:, -1, :]  # (B, D) if hierarchy == 0 else (B * top_k, D)

            # calculating the logits for the next token
            candidate_logits = self.decoder.sid_head[hierarchy](
                latest_output_representation
            )  # (B, num_candidates) if hierarchy == 0 else (B * top_k, num_candidates)

            (
                sid,  # (B * top_k, S)
                beam_log_prob,  # (B, top_k)
                attention_mask,  # (B * top_k, S)
            ) = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                sid=sid,
                attention_mask=attention_mask,
                beam_log_prob=beam_log_prob,
                hierarchy=hierarchy,
                batch_size=B,
            )

        sid = sid.view(B, self.top_k_for_generation, -1)  # (B, top_k, S)
        sid = sid[:, :, -self.num_hierarchies:]  # (B, top_k, self.num_hierarchies)
        return sid, beam_log_prob

    def get_embedding_table(self, table_name: str, hierarchy: Optional[int] = None):
        """
        Get the embedding table for the given table name and hierarchy.
        Args:
            table_name: The name of the table to get the embedding for.
            hierarchy: The hierarchy level to get the embedding for.
        """
        # here we assume the encoder and decoder share the same embedding table
        # we can have flexible embedding table in the future
        if table_name == "encoder":
            embedding_table = self.item_sid_embedding_table_encoder
        elif table_name == "decoder":
            embedding_table = self.item_sid_embedding_table_encoder

        if hierarchy is not None:
            return embedding_table(
                torch.arange(
                    hierarchy * self.num_embeddings_per_hierarchy,
                    (hierarchy + 1) * self.num_embeddings_per_hierarchy,
                ).to(self.device)
            )
        return embedding_table

    def predict_step(self, batch: SequentialModelInputData):
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the model and calculate the loss if mode is INFER.

        Args:
            model_input: The input data to the model.
        """

        # ===== Splicing the sid and item_id together =====
        sid = model_input.transformed_sequences["sequence_data"].long()
        mask = model_input.mask

        B, S = sid.shape
        assert S % self.num_hierarchies == 0, (
            f"S={S} not divisible by num_hierarchies={self.num_hierarchies}"
        )

        if mode is ModelMode.INFER:
            generated_ids, marginal_probs = self.generate(
                attention_mask=mask,
                sid=sid,
            )
            return generated_ids, marginal_probs
        
        # ===== Forward =====
        model_output = self.decoder_forward_pass(
            attention_mask=mask,
            sid=sid,
        )

        # ===== Calculate loss =====
        num_block = S // self.num_hierarchies
        sid = sid.view(B, num_block, -1)
        model_output_blk = model_output.view(B, num_block, -1, self.embedding_dim)
        mask_blk = mask.view(B, num_block, self.num_hierarchies)

        loss = 0
        hit_list = []

        for cur in range(self.num_hierarchies):
            logits = self.decoder.sid_head[cur](
                model_output_blk[..., cur, :]
            )  # (B, num_block, num_candidates)
            targets = sid[..., cur].long() - 1  # (B, num_block)

            loss_cur = self.loss_function(
                logits.reshape(-1, logits.size(-1)),  # (B * num_block, num_candidates)
                targets.reshape(-1).long(), # (B * num_block)
            )  # (B * num_block)
            loss = loss + loss_cur

            with torch.no_grad():
                topk = torch.topk(logits, k=10, dim=-1).indices  # (B, num_block, 10)
                hit = (
                    (topk == targets.unsqueeze(-1))
                    .any(dim=-1)
                    .float()
                )  # (B, num_block)

                cur_mask = mask_blk[..., cur]
                hit = hit * cur_mask
                hit_rate = hit.sum() / cur_mask.sum().clamp_min(1)
                hit_list.append(hit_rate)
        metrics = {
            f"hit@10_sid{h}": hit_list[h]
            for h in range(self.num_hierarchies)
        }
        return model_output, loss, metrics

class SemanticIDDecoderModule(torch.nn.Module):
    """
    This is an in-house replication of the decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        decoder: transformers.PreTrainedModel,
        sid_head: Optional[torch.nn.Module] = None,
        bos_token: Optional[torch.nn.Parameter] = None,
    ) -> None:
        """
        Initialize the SemanticIDDecoderModule.

        Parameters:
        decoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        sid_head (torch.nn.Module): the mlp layers used to project the decoder output to the embedding table.
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


class SemanticIDEncoderModule(torch.nn.Module):
    """
    This is an in-house replication of the encoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        encoder: transformers.PreTrainedModel,
    ) -> None:
        """
        Initialize the SemanticIDEncoderModule module.

        Paremeters:
        encoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        """
        super().__init__()

        self.encoder = encoder
        embedding_table_dim = find_module_shape(self.encoder, "embed_tokens")
        num_embeddings, embedding_dim = embedding_table_dim

        self.num_embeddings_per_hierarchy = num_embeddings
        self.embedding_dim = embedding_dim
        # TODO (clark): take care of chunky position encoding

        # deleting embedding table in the encoder to save space
        delete_module(self.encoder, "embed_tokens")
        delete_module(self.encoder, "shared")
        reset_parameters(self.encoder)

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
    ) -> torch.Tensor:

        encoder_output = self.encoder(
            inputs_embeds=sequence_embedding,
            attention_mask=attention_mask,
        )
        embeddings = encoder_output.last_hidden_state
        return embeddings