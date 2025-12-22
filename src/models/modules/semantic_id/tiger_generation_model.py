import logging
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
        num_items: int,
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

        self.num_embeddings_per_hierarchy = int(num_embeddings_per_hierarchy)
        self.padding_token = padding_token
        self.embedding_dim = embedding_dim
        self.num_items = num_items
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=padding_token)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.num_hierarchies = num_hierarchies
        self.should_check_prefix = should_check_prefix
        if codebooks != None:
            self.codebooks = codebooks.t()
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
        generated_ids: Union[torch.Tensor, None],
        attention_mask: torch.Tensor, 
        marginal_log_prob: Union[torch.Tensor, None],
        hierarchy: int,
        batch_size: int,
    ):
        """
        Perform one step of beam search.

        Args:
            candidate_logits: The logits for the next token.
            generated_ids: The generated IDs so far.
            marginal_log_prob: The marginal log probabilities.  (B, top_k)
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

            generated_ids = generated_ids.unsqueeze(1).repeat(1, self.top_k_for_generation, 1)  # (B, top_k, S)
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.top_k_for_generation, 1) # (B, top_k, S)

            next_tok = indices_topk.unsqueeze(-1)  # (B, top_k, 1)
            next_mask = torch.ones(batch_size, self.top_k_for_generation, 1, device=attention_mask.device, dtype=attention_mask.dtype)  # (B, top_k, 1)
            
            generated_ids = torch.cat([generated_ids, next_tok], dim=-1).view(batch_size * self.top_k_for_generation, -1)  # (B * top_k, S + 1)
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1).view(batch_size * self.top_k_for_generation, -1)  # (B * top_k, S + 1)
        else:
            num_candidates = self.num_embeddings_per_hierarchy if hierarchy < self.num_hierarchies else self.num_items
            proba, indices = proba.view(
                -1, self.top_k_for_generation * num_candidates
            ), indices.view(
                -1, self.top_k_for_generation * num_candidates
            )  # (B, top_k * num_candidates)
            # calculating the marginal probability
            proba =  marginal_log_prob.repeat_interleave(num_candidates, dim=-1) + proba  # (B, top_k * num_candidates)
            topk_results = torch.topk(proba, k=self.top_k_for_generation, dim=-1)  # (B, top_k)
            proba_topk, indices_topk = topk_results.values, topk_results.indices  # (B, top_k)
            # getting indices of winning beams in the original beams
            replace_indices = (
                (indices_topk // num_candidates)
                + torch.arange(indices_topk.size(0), device=proba.device).unsqueeze(1)
                * self.top_k_for_generation
            ).flatten()  # (B * top_k)

            indices_topk = torch.gather(indices, 1, indices_topk)  # (B, top_k)
            generated_ids = torch.cat(
                [
                    generated_ids[replace_indices],  # (B * top_k, S)
                    indices_topk.reshape(-1).unsqueeze(-1),  # (B * top_k, 1)
                ],
                dim=-1,
            )  # (B * top_k, S + 1)

            next_mask = torch.ones(replace_indices.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype)  # (B * top_k, 1)
            attention_mask = torch.cat(
                [
                    attention_mask[replace_indices],  # (B * top_k, S)
                    next_mask  # (B * top_k, 1)
                ],
                dim=-1
            )  # (B * top_k, S + 1)

        return generated_ids, proba_topk, attention_mask

    def eval_step(
        self,
        batch: SequentialModelInputData,
        loss_to_aggregate: BaseAggregator,
    ):
        model_input = SequentialModelInputData(
            transformed_sequences={
                k: v.clone() for k, v in batch.transformed_sequences.items()
            },
            mask=batch.mask.clone(),
        )
        _, loss = self.model_step(model_input=model_input, mode=ModelMode.TRAIN)

        sid = model_input.transformed_sequences["sequence_data"].long()  # (B, S)
        item_id = model_input.transformed_sequences["sequence_data_item_id"].long()  # (B, S)
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
        labels_sid = sid.gather(1, idx)  # (B, num_hierarchies)
        labels_item = item_id.gather(1, idx)  # (B, num_hierarchies)
        label = torch.cat([labels_sid, labels_item[:, -1:]], dim=1)  # (B, num_hierarchies + 1)
        
        # ===== Left padding =====
        hist_lengths = lengths - self.num_hierarchies  # (B,)
        max_hist_len = hist_lengths.max().item()

        hist_sid = torch.full(
            (B, max_hist_len),
            self.padding_token,
            device=device,
            dtype=sid.dtype,
        )  # (B, max_hist_len)
        hist_item = torch.full_like(hist_sid, self.padding_token)  # (B, max_hist_len)
        
        pos = torch.arange(max_hist_len, device=device)  # (max_hist_len,)
        start = max_hist_len - hist_lengths[:, None]  # (B, max_hist_len)

        hist_mask = pos >= start    # (B, max_hist_len)
        hist_mask = hist_mask.to(attention_mask.dtype)

        src_idx = pos - start  # (B, max_hist_len)
        safe_src_idx = src_idx.clamp(min=0)

        hist_sid[hist_mask.bool()] = sid.gather(1, safe_src_idx)[hist_mask.bool()]
        hist_item[hist_mask.bool()] = item_id.gather(1, safe_src_idx)[hist_mask.bool()]

        model_input.transformed_sequences["sequence_data"] = hist_sid
        model_input.transformed_sequences["sequence_data_item_id"] = hist_item
        model_input.mask = hist_mask

        generated_ids, marginal_probs = self.model_step(model_input=model_input, mode=ModelMode.INFER)
        self.evaluator(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            labels=label.to(marginal_probs.device),
        )

        loss_to_aggregate(loss["total"])

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
        item_feat_tensor: torch.Tensor = None,
        sid2item_container: torch.Tensor = None,
        replace_prob: float = 0.1,
        embedding_dim: int = None,
        num_hierarchies: int = None,
        num_embeddings_per_hierarchy: int = None,
        num_user_bins: Optional[int] = None,
        mlp_layers: Optional[int] = None,
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
        mlp_layers (Optional[int]): the number of mlp layers in the encoder and decoder.
        embedding_dim (Optional[int]): the dimension of the embeddings.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """

        if num_hierarchies is None or num_embeddings_per_hierarchy is None:
            num_hierarchies, num_embeddings_per_hierarchy = (
                codebooks.shape[0],
                codebooks.max().item() + 1,
            )
        if embedding_dim is None:
            embedding_dim = (
                kwargs["huggingface_model"]
                .encoder.block[0]
                .layer[0]
                .SelfAttention.q.in_features
            )

        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_items=item_feat_tensor.size(0),
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            should_check_prefix=should_check_prefix,
            **kwargs,
        )

        self.encoder = SemanticIDEncoderModule(
            encoder=self.encoder,
        )

        # bos_token used to prompt the decoder to generate the first token
        bos_token = torch.nn.Parameter(
            torch.randn(self.embedding_dim), requires_grad=True
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
            item_head=torch.nn.Linear(self.embedding_dim, item_feat_tensor.size(0)),
            binary_head=torch.nn.Linear(self.embedding_dim, 1)
        )

        if mlp_layers is not None:
            # bloating the mlp layers in both encoder and decoder
            # TODO (clark): this currently only works for T5
            for name, module in self.named_modules():
                if isinstance(module, transformers.models.t5.modeling_t5.T5LayerFF):
                    parent_module, attr_name = get_parent_module_and_attr(self, name)
                    setattr(
                        parent_module,
                        attr_name,
                        T5MultiLayerFF(
                            config=self.encoder.encoder.config, num_layers=mlp_layers
                        ),
                    )

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings
        self.item_sid_embedding_table_encoder = self._spawn_embedding_tables(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies,
            embedding_dim=self.embedding_dim,
        )
        self.item_feature_table = nn.Embedding.from_pretrained(
            item_feat_tensor,
            freeze=True,
        )
        self.feat_proj = nn.Linear(item_feat_tensor.size(1), self.embedding_dim)

        self.register_buffer("sid_code", sid2item_container["sid_code"])
        self.register_buffer("offsets",  sid2item_container["offsets"])
        self.register_buffer("items",    sid2item_container["items"])

        self.replace_prob = replace_prob

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
        self.sep_token = (
            torch.nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)
            if should_add_sep_token
            else None
        )
        # the key value names for the prediction output
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name

    def encoder_forward_pass(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder module.

        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
        """

        # we shift the IDs here to match the hierarchy structure
        # so that we can use a single embedding table to store the embeddigns for all hierarchies
        shifted_sids = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds_for_encoder = self.get_embedding_table(table_name="encoder")(
            shifted_sids
        )

        if self.sep_token is not None:
            (
                inputs_embeds_for_encoder,
                attention_mask,
            ) = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds_for_encoder,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        # we enter this loop if we want to use user_id
        if user_id is not None and self.user_embedding is not None:
            # preprocessing function pad user_id with zeros
            # so we only need to take the first column
            user_id = user_id[:, 0]

            # TODO (clark): here we assume remainder hashing, which is different from LSH hashing used in TIGER.
            user_embeds = self.user_embedding(
                torch.remainder(user_id, self.user_embedding.num_embeddings)
            )

            # prepending the user_id embedding to the input senquence
            inputs_embeds_for_encoder = torch.cat(
                [
                    user_embeds.unsqueeze(1),
                    inputs_embeds_for_encoder,
                ],
                dim=1,
            )
            # prepending 1 to attention mask as we introduce user embedding in the first column
            user_attention_mask = torch.ones(
                attention_mask.size(0), 1, device=attention_mask.device
            )
            attention_mask_for_encoder = torch.cat(
                [
                    user_attention_mask,
                    attention_mask,
                ],
                dim=1,
            )
        else:
            attention_mask_for_encoder = attention_mask

        encoder_output = self.encoder(
            sequence_embedding=inputs_embeds_for_encoder,
            attention_mask=attention_mask_for_encoder,
        )
        return encoder_output, attention_mask_for_encoder

    def decoder_forward_pass(
        self,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # TODO (clark): in the future we should support variable length semantic id
        future_ids: Optional[torch.Tensor] = None,
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
        B, S = future_ids.shape
        assert attention_mask.shape == future_ids.shape
        sid_table = self.get_embedding_table(table_name="decoder")
        feat_table = self.get_embedding_table(table_name="item_feature")

        in_block_len = self.num_hierarchies + 1
        out_block_len = self.num_hierarchies + 2

        pad_len = (-S) % in_block_len
        future_ids_pad = nn.functional.pad(future_ids, (0, pad_len), value=0)  # (B, S + pad_len)
        attention_mask_pad = F.pad(attention_mask, (0, pad_len), value=0)  # (B, S + pad_len)

        future_ids_blk = future_ids_pad.view(B, -1, in_block_len)  # (B, num_block, in_block_len)
        mask_blk = attention_mask_pad.view(B, -1, in_block_len)  # (B, num_block, in_block_len)

        block_pad = mask_blk.sum(dim=-1) == 0   # (B, num_block)
        fake_pos = (mask_blk[..., -1] == 0) & (~block_pad)  # (B, num_block)

        sid_ids = future_ids_blk[..., :-1].masked_fill(block_pad.unsqueeze(-1), 0)  # (B, num_block, hirerachies)
        offsets = (
            torch.arange(self.num_hierarchies, device=sid_ids.device)
            * self.num_embeddings_per_hierarchy
        )
        sid_ids = sid_ids + offsets
        sid_embeds = sid_table(sid_ids)  # (B, num_block, hirerachies, D)
        sid_embeds = sid_embeds.masked_fill(block_pad[..., None, None], 0.0)

        item_ids = future_ids_blk[..., -1].masked_fill(block_pad, 0)  # (B, num_block)
        item_embeds = self.feat_proj(feat_table(item_ids))  # (B, num_block, D)
        item_embeds = item_embeds.masked_fill(block_pad[..., None], 0.0)

        bos = self.decoder.bos_token
        bos = bos.view(1, 1, 1, -1).expand(B, sid_embeds.size(1), 1, -1)  # (B, num_block, 1, D)

        out_blk = torch.cat([bos, sid_embeds, item_embeds.unsqueeze(-2)], dim=-2)  # (B, num_block, out_block_len, D)
        out_mask = torch.ones(B, out_blk.size(1), out_block_len, device=future_ids.device, dtype=attention_mask.dtype)  # (B, num_block, out_block_len)
        out_mask[block_pad] = 0

        fake_pos_lists = [
            (torch.nonzero(fake_pos[b]).squeeze(-1) * out_block_len + (out_block_len - 1)).tolist()
            for b in range(B)
        ]

        inputs_embeds = out_blk.view(B, -1, out_blk.size(-1))  # (B, S + pad_len + num_block, D)
        padding_mask = out_mask.view(B, -1)  # (B, S + pad + num_block)

        if pad_len > 0:
            inputs_embeds = inputs_embeds[:, :-pad_len]
            padding_mask = padding_mask[:, :-pad_len]

        if add_bos_token:
            bos = self.decoder.bos_token.view(1, 1, -1).expand(B, 1, -1)  # (B, 1, D)
            inputs_embeds = torch.cat([inputs_embeds, bos], dim=1)  # (B, S + num_block + 1, D)

            bos_mask = torch.ones(B, 1, device=padding_mask.device)  
            padding_mask = torch.cat([padding_mask, bos_mask], dim=1)

        # ===== Construct mask matrix =====
        B, L, _ = inputs_embeds.shape
        device = inputs_embeds.device
        causal_mask = torch.tril(torch.ones(L, L, device=device)) # (L, L)
        final_mask = causal_mask.unsqueeze(0).repeat(B, 1, 1) # (B, L, L)
        final_mask = final_mask * padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)

        for b in range(B):
            for fp in fake_pos_lists[b]:
                if fp < L:
                    final_mask[b, fp+1:, fp] = 0

        # BART expects (B, 1, L, L)
        attn_mask_4d = final_mask.unsqueeze(1)
        attn_mask_4d = (1.0 - attn_mask_4d) * -1e4

        decoder_output = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_4d,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        return decoder_output

    def generate(
        self,
        attention_mask: torch.Tensor = None,
        input_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate the semantic id given the current model in the sequence using beam search.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            input_ids (torch.Tensor): The input IDs for the decoder.
        """
        # initilize cached generated ids to None
        generated_ids = input_ids
        decoder_attention_mask = attention_mask
        marginal_log_prob = None

        for hierarchy in range(self.num_hierarchies + 1):
            # feeding the decoder with the generated ids
            add_bos_token = hierarchy == 0
            decoder_output = self.decoder_forward_pass(
                future_ids=generated_ids,
                attention_mask=decoder_attention_mask,
                use_cache=False,
                add_bos_token=add_bos_token
            )  # (B, T, D) if hierarchy == 0 else (B * top_k, T, D)

            # decoder_output[:, -1, :] is the embedding for the next token
            latest_output_representation = decoder_output[:, -1, :]  # (B, D) if hierarchy == 0 else (B * top_k, D)

            # calculating the logits for the next token
            if hierarchy < self.num_hierarchies:
                candidate_logits = self.decoder.sid_head[hierarchy](
                    latest_output_representation
                )  # (B, num_candidates) if hierarchy == 0 else (B * top_k, num_candidates)
            else:
                candidate_logits = self.decoder.item_head(
                    latest_output_representation
                )  # (B, num_candidates) if hierarchy == 0 else (B * top_k, num_candidates)

            (
                generated_ids,  # (B * top_k, S)
                marginal_log_prob,  # (B, top_k)
                decoder_attention_mask,  # (B * top_k, S)
            ) = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                attention_mask=decoder_attention_mask,
                marginal_log_prob=marginal_log_prob,
                hierarchy=hierarchy,
                batch_size=input_ids.size(0),
            )

        if self.replace_prob > 0.0:
            decoder_output = self.decoder_forward_pass(
                future_ids=generated_ids,
                attention_mask=decoder_attention_mask,
                use_cache=False,
                add_bos_token=False
            )  # (B * top_k, T, D)
            logits = self.decoder.binary_head(decoder_output[:, -1, :])  # (B * top_k, 1)
            log_prob = F.logsigmoid(logits)  # (B * top_k, 1)
            marginal_log_prob = marginal_log_prob + log_prob.view(input_ids.size(0), self.top_k_for_generation)  # (B, top_k)
            marginal_log_prob, sorted_idx = torch.sort(marginal_log_prob, dim=1, descending=True)  # (B, top_k)

            generated_ids = generated_ids.view(input_ids.size(0), self.top_k_for_generation, -1)  # (B, top_k, S)
            generated_ids = torch.gather(
                generated_ids,  # (B, top_k, S)
                dim=1,
                index=sorted_idx.unsqueeze(-1).expand(-1, -1, generated_ids.size(-1))  # (B, top_k, S)
            )
        else:
            generated_ids = generated_ids.view(input_ids.size(0), self.top_k_for_generation, -1)  # (B, top_k, S)
    
        generated_ids = generated_ids[:, :, -(self.num_hierarchies + 1):]  # (B, top_k, self.num_hierarchies + 1)
        return generated_ids, marginal_log_prob

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
        elif table_name == "item_feature":
            embedding_table = self.item_feature_table

        if hierarchy is not None:
            return embedding_table(
                torch.arange(
                    hierarchy * self.num_embeddings_per_hierarchy,
                    (hierarchy + 1) * self.num_embeddings_per_hierarchy,
                ).to(self.device)
            )
        return embedding_table

    def predict_step(self, batch: SequentialModelInputData):
        generated_sids, _ = self.model_step(batch, mode=ModelMode.INFER)
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
        item_id = model_input.transformed_sequences["sequence_data_item_id"].long()

        B, S = sid.shape
        assert S % self.num_hierarchies == 0, (
            f"S={S} not divisible by num_hierarchies={self.num_hierarchies}"
        )
        sid = sid.reshape(B, S // self.num_hierarchies, self.num_hierarchies)
        item_id = item_id[:, ::self.num_hierarchies].unsqueeze(2)
        fut_ids = torch.cat([sid, item_id], dim=2)
        fut_ids = fut_ids.reshape(B, -1)

        mask = model_input.mask
        mask_sid = mask.reshape(B, S // self.num_hierarchies, self.num_hierarchies)
        item_mask = mask_sid[..., -1:]
        fut_mask = torch.cat([mask_sid, item_mask], dim=2)
        fut_mask = fut_mask.reshape(B, -1)

        if mode is ModelMode.INFER:
            generated_ids, marginal_probs = self.generate(
                attention_mask=fut_mask,
                input_ids=fut_ids
            )
            return generated_ids, marginal_probs
        
        # ===== Build labels and sample items =====
        in_block_len = self.num_hierarchies + 1          
        # sid1 + ... + sidn + item
        out_block_len = self.num_hierarchies + 2         
        # bos + sid1 + ... + sidn + item -> sid1 + ... sidn + item + flag
        B, T = fut_ids.shape
        num_block = T // in_block_len

        fut_ids_blk  = fut_ids.view(B, -1, in_block_len)  # (B, num_block, in_block_len)
        fut_mask_blk = fut_mask.view(B, -1, in_block_len)  # (B, num_block, in_block_len)
        block_valid = fut_mask_blk[..., 0].bool()   # (B, num_block)

        label_blk = torch.zeros(
            B, num_block, out_block_len,
            device=fut_ids.device,
            dtype=fut_ids.dtype,
        )  # (B, num_block, out_len)
        label_blk[..., :in_block_len] = fut_ids_blk

        item_tokens = fut_ids_blk[..., -1]  # (B, num_block)
        replace_items = item_tokens.clone()
        replace_items.fill_(-1)  # (B, num_block)

        if block_valid.any():
            replace_items[block_valid] = self.sample_same_sid(item_tokens[block_valid])

        replace_mask = ( (torch.rand(B, num_block, device=fut_ids.device) < self.replace_prob) & block_valid )  # (B, num_block)
        can_replace = replace_items >= 0  # (B, num_block)
        replace_mask = replace_mask & can_replace  # (B, num_block)

        flag = torch.ones(B, num_block, device=fut_ids.device, dtype=fut_ids.dtype)
        flag = flag * (~replace_mask)
        label_blk[..., -1] = flag

        fut_ids_blk[..., -1] = torch.where(
            replace_mask,
            replace_items,
            fut_ids_blk[..., -1]
        )
        fut_mask_blk[..., -1] = fut_mask_blk[..., -1] * (~replace_mask)

        fut_ids  = fut_ids_blk.view(B, T)
        fut_mask = fut_mask_blk.view(B, T)
        label    = label_blk.view(B, num_block * out_block_len)

        # ===== Forward =====
        model_output = self.decoder_forward_pass(
            attention_mask=fut_mask,
            future_ids=fut_ids
        )

        # ===== Calculate loss =====
        B, T, D = model_output.shape
        model_output = model_output.view(B, -1, out_block_len, D)  # (B, num_block, out_block_len, D)
        label = label.view(B, -1, out_block_len)  # (B, num_block, out_block_len)

        fut_mask_block = fut_mask.view(B, -1, in_block_len)   # (B, num_block, in_block_len)
        block_valid = fut_mask_block[:, :, 0]   # (B, num_block)

        sid_losses = []
        for cur in range(self.num_hierarchies):
            logits = self.decoder.sid_head[cur](
                model_output[..., cur, :]
            )  # (B, num_block, num_candidates)
            loss = self.ce_loss(
                logits.reshape(-1, logits.size(-1)),  # (B * num_block, num_candidates)
                label[..., cur].reshape(-1).long(), # (B * num_block)
            )  # (B * num_block)
            sid_losses.append(loss.view(B, -1))  # [(B, num_block)]
        sid_loss = (torch.stack(sid_losses, dim=-1) * block_valid.unsqueeze(-1)).sum()

        item_pos = self.num_hierarchies
        item_logits = self.decoder.item_head(
            model_output[..., item_pos, :]
        )  # (B, num_block, num_candidates)
        item_loss = self.ce_loss(
            item_logits.reshape(-1, item_logits.size(-1)),  # (B * num_block, num_candidates)
            label[..., item_pos].reshape(-1).long(),  # (B * num_block, num_candidates)
        ).view(B, -1)  # (B, num_block)
        item_loss = (item_loss * block_valid).sum()

        flag_pos = self.num_hierarchies + 1
        flag_logits = self.decoder.binary_head(
            model_output[..., flag_pos, :]
        ).squeeze(-1)  # (B, num_block)
        flag_loss = self.bce_loss(
            flag_logits.reshape(-1), # (B * num_block)
            label[..., flag_pos].float().reshape(-1),  # (B * num_block)
        ).view(B, -1)  # (B, num_block)
        flag_loss = (flag_loss * block_valid).sum()

        block_cnt = block_valid.sum().clamp_min(1)
        loss = (sid_loss + item_loss + flag_loss) / block_cnt
        return model_output, {
            "total": loss,
            "sid_loss": sid_loss / block_cnt,
            "item_loss": item_loss / block_cnt,
            "flag_loss": flag_loss / block_cnt,
        }

    def sample_same_sid(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        item_ids: (N,)
        return:   (N,), invalid -> -1
        """
        # sid code
        codes = self.sid_code[item_ids]      # (N,)

        # offsets
        l = self.offsets[codes, 0] # (N,)
        r = self.offsets[codes, 1] # (N,)
        length = r - l                       # (N,)

        # valid candidates excluding self
        valid_len = length - 1
        no_candidate = valid_len <= 0

        # output init
        out = torch.full_like(item_ids, -1)

        # sample for valid positions only
        valid = ~no_candidate
        if valid.any():
            l_v = l[valid]
            valid_len_v = valid_len[valid]
            flat_v = item_ids[valid]

            # global max (Tensor, no .item())
            max_len = valid_len_v.max()

            # sample [0, max_len)
            rand = torch.randint(
                0,
                max_len,
                (valid_len_v.size(0),),
                device=item_ids.device,
            )

            # map to [0, valid_len)
            rand = rand % valid_len_v

            sampled_idx = l_v + rand
            sampled_item = self.items[sampled_idx]

            # skip self safely
            hit_self = sampled_item == flat_v
            sampled_idx = sampled_idx + hit_self.long()

            out[valid] = self.items[sampled_idx]

        return out


class SemanticIDDecoderModule(torch.nn.Module):
    """
    This is an in-house replication of the decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        decoder: transformers.PreTrainedModel,
        sid_head: Optional[torch.nn.Module] = None,
        item_head: Optional[torch.nn.Module] = None,
        binary_head: Optional[torch.nn.Module] = None,
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

        self.decoder = decoder
        # this bos token is prompt for the decoder
        self.bos_token = bos_token
        self.sid_head = sid_head
        self.item_head = item_head
        self.binary_head = binary_head

    def forward(
        self,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        use_cache: bool = False,
        past_key_values: DynamicCache = DynamicCache(),
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            sequence_embedding (torch.Tensor): The input sequence embedding for the decoder.
            encoder_output (torch.Tensor): The output from the encoder.
            encoder_attention_mask (torch.Tensor): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (DynamicCache): The cache for past key values.
        """

        decoder_outputs: Seq2SeqModelOutput = self.decoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        embeddings = decoder_outputs.last_hidden_state

        if use_cache:
            return embeddings, decoder_outputs.past_key_values
        return embeddings


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


# TODO (clark): this is a T5 specific implementation
# this class is used for bloating the mlp layers in the encoder and decoder
# original T5 implementation only has one layer
class T5MultiLayerFF(nn.Module):
    def __init__(self, config: T5Config, num_layers: int):
        """
        Initialize the T5MultiLayerFF module.
        This module is a multi-layer feed-forward network (MLP) used in the T5 model.
        It consists of a series of linear layers with ReLU activation and dropout.
        And it also includes layer normalization and residual connections.
        Parameters:
            config (T5Config): The T5 configuration object.
            num_layers (int): The number of layers in the MLP.
        """
        super().__init__()
        self.mlp = MLP(
            input_dim=config.d_model,
            output_dim=config.d_model,
            hidden_dim_list=[config.d_ff for _ in range(num_layers)],
            activation=nn.ReLU,
            dropout=config.dropout_rate,
        )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the T5MultiLayerFF module.
        Parameters:
            hidden_states (torch.Tensor): The input hidden states for the MLP.
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states
