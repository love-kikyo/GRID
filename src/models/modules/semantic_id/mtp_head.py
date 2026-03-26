"""
Multi-Token Prediction (MTP) Head for Generative Recommendation.

This module implements a lightweight Transformer-based head that predicts
the complete semantic ID block (sid0, sid1, sid2, sid3) in an autoregressive
manner, inspired by DeepSeek's MTP architecture.

Architecture:
    - Input: Hidden state from backbone (at block boundary positions)
    - Internal: Small Transformer decoder with causal attention
    - Output: Autoregressive prediction of [sid0, sid1, sid2, sid3]
    - itemkey is computed deterministically from predicted sids

Optimizations:
    - Merged projection layers for batched operations
    - torch.compile support for additional speedup
"""

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


class MTPConfig:
    """Configuration for MTP Head."""

    def __init__(
        self,
        num_hierarchies: int = 4,
        vocab_size: int = 256,  # per hierarchy vocab size
        hidden_size: int = 1024,  # should match backbone hidden size
        mtp_hidden_size: int = 256,  # internal MTP dimension
        num_mtp_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 512,
        dropout: float = 0.1,
        **kwargs,
    ):
        self.num_hierarchies = num_hierarchies
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mtp_hidden_size = mtp_hidden_size
        self.num_mtp_layers = num_mtp_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout


class MTPTransformerBlock(nn.Module):
    """
    A single Transformer block for MTP head.
    Uses Pre-LN architecture with causal self-attention.
    """

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.hidden_size = config.mtp_hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Self-attention with causal mask
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # FFN
        self.gate_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, self.hidden_size, bias=False)

        # Layer norms
        self.input_layernorm = nn.RMSNorm(self.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(self.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: (B, S, D) - sequence of token embeddings
            attention_mask: (B, S) - causal mask will be applied internally
            past_key_value: cached KV for incremental generation
            use_cache: whether to return updated cache

        Returns:
            hidden_states: (B, S, D)
            present_key_value: updated cache if use_cache
        """
        B, S, D = hidden_states.shape
        residual = hidden_states

        # Pre-LN
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            k_cache, v_cache = past_key_value
            k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        else:
            k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        present_key_value = (k, v) if use_cache else None

        # Attention with causal mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        seq_len = k.size(2)
        causal_mask = torch.triu(
            torch.ones(S, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=seq_len - S + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        attn_output = self.o_proj(attn_output)

        hidden_states = residual + self.dropout(attn_output)

        # FFN with SwiGLU
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states, present_key_value


class MTPHead(nn.Module):
    """
    Multi-Token Prediction Head using a small Transformer decoder.

    This module takes hidden states from the backbone (at block boundary positions)
    and autoregressively predicts the complete semantic ID block.

    Architecture flow:
        1. Input: backbone_hidden (B, D) - hidden state at block boundary
        2. Project to MTP hidden dimension
        3. Autoregressively generate sid0, sid1, sid2, sid3 through MTP Transformer
        4. Project each position to vocabulary logits

    Training (teacher forcing):
        - Input: [backbone_proj, sid0, sid1, sid2, sid3]
        - Target: [sid0, sid1, sid2, sid3]
        - backbone_proj predicts sid0, sid0 predicts sid1, etc.

    Inference (beam search):
        - Start with backbone_proj
        - Beam search through the MTP Transformer
        - Output top-k complete blocks

    Optimization:
        - Merged sid_projs into a single linear layer for batched projection
        - Merged output_heads into a single linear layer for batched logits
    """

    def __init__(
        self,
        config: MTPConfig,
        sid_embedding_table: nn.Embedding,
        padding_token: int = 0,
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.num_hierarchies = config.num_hierarchies
        self.vocab_size = config.vocab_size
        self.padding_token = padding_token
        self.loss_fn = loss_fn  # External loss function (e.g., CrossEntropyLoss with ignore_index)

        # Shared embedding table with backbone (for SID tokens)
        self.sid_embedding_table = sid_embedding_table

        # Project backbone hidden to MTP hidden dimension (only for backbone hidden)
        self.backbone_proj = nn.Linear(config.hidden_size, config.mtp_hidden_size, bias=False)

        # OPTIMIZATION: Merged SID projection into single layer
        # Projects all hierarchies at once: (B, H, hidden_size) -> (B, H, mtp_hidden_size)
        self.sid_proj_merged = nn.Linear(config.hidden_size, config.mtp_hidden_size, bias=False)

        # MTP Transformer layers
        self.layers = nn.ModuleList([
            MTPTransformerBlock(config) for _ in range(config.num_mtp_layers)
        ])

        # OPTIMIZATION: Merged output heads into single layer
        # Projects to vocab logits for all hierarchies: (B, H, mtp_hidden_size) -> (B, H, vocab_size)
        self.output_head_merged = nn.Linear(config.mtp_hidden_size, config.vocab_size, bias=False)

        # Final layer norm
        self.final_norm = nn.RMSNorm(config.mtp_hidden_size)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # torch.compile optimization
        self._compiled_forward = None
        self._is_compiled = False

    def compile(self, mode: str = "reduce-overhead", fullgraph: bool = False):
        """
        Compile the MTP head for faster inference/training.

        Args:
            mode: Compilation mode - "default", "reduce-overhead", or "max-autotune"
            fullgraph: Whether to require the full graph to be compiled
        """
        if not self._is_compiled:
            logging.info(f"Compiling MTPHead with mode={mode}, fullgraph={fullgraph}")
            self._compiled_forward = torch.compile(
                self._forward_impl,
                mode=mode,
                fullgraph=fullgraph,
            )
            self._is_compiled = True
            logging.info("MTPHead compilation complete")
        return self

    def _forward_impl(
        self,
        backbone_proj: torch.Tensor,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Core forward implementation for torch.compile.

        Args:
            backbone_proj: (B, mtp_hidden) - projected backbone hidden
            input_embeds: (B, H, mtp_hidden) - projected SID embeddings

        Returns:
            logits: (B, H, vocab_size)
        """
        # Prepend backbone hidden as context
        backbone_context = backbone_proj.unsqueeze(1)  # (B, 1, mtp_hidden)
        sequence = torch.cat([backbone_context, input_embeds], dim=1)  # (B, H+1, mtp_hidden)

        # Forward through MTP Transformer
        hidden_states = sequence
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, use_cache=False)

        hidden_states = self.final_norm(hidden_states)  # (B, H+1, mtp_hidden)

        # Get logits for prediction positions
        H = input_embeds.size(1)
        pred_hidden = hidden_states[:, :H, :]  # (B, H, mtp_hidden)
        logits = self.output_head_merged(pred_hidden)  # (B, H, vocab_size)

        return logits

    def embed_sid_token(self, token_id: torch.Tensor, hierarchy: int) -> torch.Tensor:
        """Embed a single SID token with hierarchy offset."""
        # Apply hierarchy offset (assuming shared embedding table)
        offset = hierarchy * self.vocab_size
        return self.sid_embedding_table(token_id + offset)

    def embed_sid_tokens_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed multiple SID tokens with hierarchy offsets (batched version).

        Args:
            token_ids: (B, H) - token IDs for each hierarchy

        Returns:
            embeddings: (B, H, D_backbone)
        """
        H = self.num_hierarchies
        offsets = torch.arange(H, device=token_ids.device) * self.vocab_size  # (H,)
        offset_tokens = token_ids + offsets.unsqueeze(0)  # (B, H)
        return self.sid_embedding_table(offset_tokens)  # (B, H, D_backbone)

    def forward(
        self,
        backbone_hidden: torch.Tensor,
        target_sids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training with teacher forcing.

        Args:
            backbone_hidden: (B, D) - hidden state from backbone at block boundary
            target_sids: (B, num_hierarchies) - target SID tokens for next block
            attention_mask: (B,) - mask for valid positions

        Returns:
            logits: (B, num_hierarchies, vocab_size) - logits for each hierarchy
            loss: scalar - cross-entropy loss if target_sids provided
            metrics: dict - per-hierarchy hit rates and losses
        """
        B, D = backbone_hidden.shape
        H = self.num_hierarchies

        # Project backbone hidden to MTP dimension
        backbone_proj = self.backbone_proj(backbone_hidden)  # (B, mtp_hidden)

        if target_sids is not None:
            # ===== Teacher Forcing =====
            # Build input sequence: [backbone_proj, sid0, sid1, sid2, sid3]
            # Target sequence: [sid0, sid1, sid2, sid3]
            # Position i predicts sid_i (backbone_proj predicts sid0, sid0 predicts sid1, ...)

            # Embed all target SIDs as input (vectorized)
            sid_embeds = self.embed_sid_tokens_batch(target_sids)  # (B, H, D_backbone)

            # OPTIMIZATION: Single batched projection instead of loop
            input_embeds = self.sid_proj_merged(sid_embeds)  # (B, H, mtp_hidden)

            # Use compiled forward if available
            if self._is_compiled and self._compiled_forward is not None:
                # Mark step begin for CUDA graphs compatibility (required for reduce-overhead mode)
                torch.compiler.cudagraph_mark_step_begin()
                logits = self._compiled_forward(backbone_proj, input_embeds)
            else:
                logits = self._forward_impl(backbone_proj, input_embeds)

            # Calculate loss and metrics
            loss, metrics = self._compute_loss(logits, target_sids, attention_mask)

            return logits, loss, metrics

        else:
            # ===== Inference Mode =====
            # Will be handled by generate() method
            raise ValueError("Use generate() for inference mode")

    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_sids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute cross-entropy loss for each hierarchy.

        Args:
            logits: (B, H, vocab_size) - 0-based predictions
            target_sids: (B, H) - 1-based SID tokens (0 is padding, 1-vocab_size are valid)
            attention_mask: (B,) - valid position mask

        Note:
            target_sids are 1-based (from preprocessing where codebooks are +1).
            We subtract 1 to convert to 0-based for cross-entropy loss.
            Padding token (0) becomes -1 after subtraction, which matches ignore_index=-1.

        Returns:
            total_loss: scalar tensor
            metrics: dict containing per-hierarchy hit rates and loss
        """
        B, H, V = logits.shape
        total_loss = 0.0
        metrics = {}

        for h in range(H):
            h_logits = logits[:, h, :]  # (B, V)
            # Convert 1-based target to 0-based for cross entropy
            # padding_token=0 becomes -1, which is ignore_index
            h_targets = target_sids[:, h].long() - 1  # (B,)

            # Use external loss function if provided, otherwise default to F.cross_entropy
            if self.loss_fn is not None:
                loss = self.loss_fn(h_logits, h_targets)
                if attention_mask is not None and loss.dim() > 0:
                    # If loss_fn returns per-sample loss (reduction='none')
                    loss = (loss * attention_mask).sum() / attention_mask.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(h_logits, h_targets, reduction='none')
                if attention_mask is not None:
                    loss = loss * attention_mask
                    loss = loss.sum() / attention_mask.sum().clamp(min=1)
                else:
                    loss = loss.mean()

            total_loss = total_loss + loss
            # metrics[f'loss_sid{h}'] = loss.item()

            # Calculate hit@10 rate for this hierarchy
            with torch.no_grad():
                _, top10_indices = h_logits.topk(10, dim=-1)
                hit = (top10_indices == h_targets.unsqueeze(-1)).any(dim=-1)
                valid_mask = h_targets >= 0  # Exclude padding tokens (target=-1)
                if attention_mask is not None:
                    valid_mask = valid_mask & attention_mask.bool()
                h_mask = valid_mask.float()
                hit_rate = (hit * h_mask).sum() / h_mask.sum().clamp_min(1)
                metrics[f"hit@10_sid{h}"] = hit_rate.item()

        return total_loss, metrics

    def generate(
        self,
        backbone_hidden: torch.Tensor,
        beam_size: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate SID tokens using beam search.

        Args:
            backbone_hidden: (B, D) - hidden state from backbone
            beam_size: number of beams

        Returns:
            generated_sids: (B, beam_size, num_hierarchies) - generated SID tokens (1-based)
            log_probs: (B, beam_size) - log probabilities of each beam

        Note:
            Returns 1-based SID tokens to match the data format (0 is padding, 1-vocab_size are valid).
            Internal predictions are 0-based; we add 1 before returning.
        """
        B, D = backbone_hidden.shape
        H = self.num_hierarchies
        device = backbone_hidden.device

        # Project backbone hidden
        backbone_proj = self.backbone_proj(backbone_hidden)  # (B, mtp_hidden)

        # Start with backbone context (no BOS)
        backbone_context = backbone_proj.unsqueeze(1)  # (B, 1, mtp_hidden)

        # KV cache for each layer
        past_key_values = [None for _ in range(len(self.layers))]

        generated_tokens = []
        beam_log_probs = torch.zeros(B, beam_size, device=device)

        # Expand for beam search
        # (B, 1, mtp_hidden) -> (B * beam_size, 1, mtp_hidden)
        current_seq = backbone_context.unsqueeze(1).expand(B, beam_size, -1, -1).reshape(B * beam_size, 1, -1)

        for h in range(H):
            # Forward through MTP Transformer with caching
            hidden_states = current_seq

            new_past_key_values = []
            for i, layer in enumerate(self.layers):
                hidden_states, pkv = layer(
                    hidden_states,
                    past_key_value=past_key_values[i],
                    use_cache=True
                )
                new_past_key_values.append(pkv)

            past_key_values = new_past_key_values

            # Get logits for last position using merged output head
            last_hidden = self.final_norm(hidden_states[:, -1, :])  # (B * beam_size, mtp_hidden)
            logits = self.output_head_merged(last_hidden)  # (B * beam_size, vocab_size)

            log_probs = F.log_softmax(logits, dim=-1)  # (B * beam_size, vocab_size)

            if h == 0:
                # First hierarchy: select top beam_size tokens
                log_probs_first = log_probs[:B]  # (B, vocab_size)
                topk_result = log_probs_first.topk(beam_size, dim=-1)  # (B, beam_size)
                beam_log_probs = topk_result.values  # (B, beam_size)
                topk_indices = topk_result.indices  # (B, beam_size)

                # Create new tokens
                new_tokens = topk_indices.view(B * beam_size, 1)  # (B * beam_size, 1)
                generated_tokens.append(new_tokens)

                # Update current_seq with new token embeddings for next iteration
                # Convert 0-based to 1-based for embedding lookup
                new_tokens_1based = topk_indices.view(B * beam_size) + 1  # (B * beam_size,)
                new_token_embed = self.embed_sid_token(new_tokens_1based, h)  # (B * beam_size, D_backbone)
                # Use merged projection
                new_token_embed_proj = self.sid_proj_merged(new_token_embed).unsqueeze(1)  # (B * beam_size, 1, mtp_hidden)
                current_seq = new_token_embed_proj  # (B * beam_size, 1, mtp_hidden)

            else:
                # Subsequent hierarchies: beam search
                # Combine with previous beam log probs
                combined_log_probs = beam_log_probs.unsqueeze(-1) + log_probs.view(B, beam_size, -1)
                # (B, beam_size, vocab_size)

                # Flatten and select top-k
                flat_log_probs = combined_log_probs.view(B, -1)  # (B, beam_size * vocab_size)
                topk_log_probs, topk_indices = flat_log_probs.topk(beam_size, dim=-1)

                beam_log_probs = topk_log_probs  # (B, beam_size)

                # Determine which beams to keep
                beam_indices = topk_indices // self.vocab_size  # (B, beam_size)
                token_indices = topk_indices % self.vocab_size  # (B, beam_size)

                # Reorder past key values according to beam selection
                beam_indices_flat = beam_indices.view(B * beam_size)  # (B * beam_size,)

                # OPTIMIZATION: Compute gather indices once for all layers
                batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, beam_size).reshape(-1)  # (B * beam_size,)
                gather_indices = batch_indices * beam_size + beam_indices_flat  # (B * beam_size,)

                # Reorder KV cache for each layer
                for i in range(len(past_key_values)):
                    k, v = past_key_values[i]
                    past_key_values[i] = (k[gather_indices], v[gather_indices])

                # Reorder current_seq
                current_seq = current_seq[gather_indices]

                # Add new tokens
                new_tokens = token_indices.view(B * beam_size, 1)
                generated_tokens.append(new_tokens)

                # Prepare next token embedding
                if h < H - 1:
                    new_tokens_1based = token_indices.view(B * beam_size) + 1  # (B * beam_size,)
                    new_token_embed = self.embed_sid_token(new_tokens_1based, h)  # (B * beam_size, D_backbone)
                    # Use merged projection
                    new_token_embed_proj = self.sid_proj_merged(new_token_embed).unsqueeze(1)  # (B * beam_size, 1, mtp_hidden)
                    current_seq = new_token_embed_proj  # (B * beam_size, 1, mtp_hidden)

        # Stack generated tokens (0-based from model predictions)
        generated_sids = torch.cat(generated_tokens, dim=-1).view(B, beam_size, H)

        # Convert 0-based to 1-based to match data format
        generated_sids = generated_sids + 1

        return generated_sids, beam_log_probs