from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.aggregation import BaseAggregator

from src.data.loading.components.interfaces import RerankCandidateBatch
from src.models.components.network_blocks.mlp import MLP
from src.models.modules.base_module import BaseModule


class SIDRerankSmallModel(BaseModule):
    def __init__(
        self,
        model: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_function: Dict[str, nn.Module],
        evaluator,
        num_hierarchies: int,
        num_candidates: int,
        num_embeddings_per_hierarchy: int,
        num_item_buckets: int = 300_000,
        item_embedding_dim: int = 128,
        sid_embedding_dim: int = 32,
        hidden_dim_list: Optional[list[int]] = None,
        padding_idx: int = -1,
        use_beam_score: bool = True,
    ) -> None:
        super().__init__(
            model=model if model is not None else nn.Identity(),
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            evaluator=evaluator,
        )
        self.save_hyperparameters(logger=False, ignore=["model", "loss_function"])

        self.num_hierarchies = num_hierarchies
        self.num_candidates = num_candidates
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.padding_idx = padding_idx
        self.use_beam_score = use_beam_score
        self.num_item_buckets = num_item_buckets

        self.item_embedding = nn.Embedding(num_item_buckets + 1, item_embedding_dim)
        self.sid_embedding = nn.Embedding(
            num_hierarchies * num_embeddings_per_hierarchy + 1,
            sid_embedding_dim,
        )

        feature_dim = item_embedding_dim * 2 + sid_embedding_dim + 1
        if not use_beam_score:
            feature_dim -= 1
        self.scorer = MLP(
            input_dim=feature_dim,
            output_dim=1,
            hidden_dim_list=hidden_dim_list or [256, 128],
            dropout=0.1,
        )

    def _normalize_itemkey(self, itemkeys: torch.Tensor) -> torch.Tensor:
        safe_itemkeys = torch.where(itemkeys < 0, torch.zeros_like(itemkeys), itemkeys)
        return (safe_itemkeys % self.num_item_buckets) + 1

    def _encode_history(self, hist_itemkey: torch.Tensor) -> torch.Tensor:
        hist_index = self._normalize_itemkey(hist_itemkey)
        hist_mask = (hist_itemkey >= 0).float()
        hist_emb = self.item_embedding(hist_index)
        denom = hist_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (hist_emb * hist_mask.unsqueeze(-1)).sum(dim=1) / denom

    def _encode_candidate_sid(self, candidate_sid: torch.Tensor) -> torch.Tensor:
        offsets = (
            torch.arange(self.num_hierarchies, device=candidate_sid.device)
            .view(1, 1, self.num_hierarchies)
            * self.num_embeddings_per_hierarchy
        )
        sid_index = torch.where(
            candidate_sid > 0,
            candidate_sid + offsets,
            torch.zeros_like(candidate_sid),
        )
        sid_emb = self.sid_embedding(sid_index)
        sid_mask = (candidate_sid > 0).float()
        denom = sid_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        return (sid_emb * sid_mask.unsqueeze(-1)).sum(dim=2) / denom

    def forward(self, batch: RerankCandidateBatch) -> torch.Tensor:
        hist_repr = self._encode_history(batch.hist_itemkey)  # (B, D)
        candidate_index = self._normalize_itemkey(batch.candidate_itemkey)
        candidate_repr = self.item_embedding(candidate_index)  # (B, K, D)
        sid_repr = self._encode_candidate_sid(batch.candidate_sid)  # (B, K, D_sid)

        hist_repr = hist_repr.unsqueeze(1).expand(-1, self.num_candidates, -1)
        features = [hist_repr, candidate_repr, sid_repr]
        if self.use_beam_score:
            features.append(batch.beam_score.unsqueeze(-1))
        scorer_input = torch.cat(features, dim=-1)
        logits = self.scorer(scorer_input).squeeze(-1)
        return logits

    def _compute_metrics(
        self,
        batch: RerankCandidateBatch,
        rerank_scores: torch.Tensor,
    ) -> None:
        self.evaluators["beam"](
            marginal_probs=batch.beam_score.detach(),
            generated_ids=batch.candidate_sid.detach(),
            labels=batch.target_sid.detach(),
        )
        self.evaluators["rerank"](
            marginal_probs=rerank_scores.detach(),
            generated_ids=batch.candidate_sid.detach(),
            labels=batch.target_sid.detach(),
        )

    def model_step(
        self,
        batch: RerankCandidateBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self.forward(batch)
        mask = (batch.candidate_itemkey >= 0).float()
        masked_logits = logits.masked_fill(mask == 0, -1e9)

        target_distribution = batch.candidate_labels
        target_distribution = target_distribution / target_distribution.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        log_probs = F.log_softmax(masked_logits, dim=1)
        loss = -(target_distribution * log_probs).sum(dim=1).mean()

        metrics = {
            "beam_log_prob": batch.beam_score.mean(),
            "rerank_log_prob": log_probs.mean(),
        }
        return masked_logits, loss, metrics

    def training_step(self, batch: RerankCandidateBatch, batch_idx: int) -> torch.Tensor:
        scores, loss, metrics = self.model_step(batch)
        self.log(
            "train/loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        for key, value in metrics.items():
            self.log(
                f"train/{key}",
                value,
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
        batch: RerankCandidateBatch,
        loss_to_aggregate: BaseAggregator,
    ):
        with torch.inference_mode():
            scores, loss, _ = self.model_step(batch)
        loss_to_aggregate(loss)
        self._compute_metrics(batch, scores)

    def get_primary_evaluator_name(self) -> Optional[str]:
        return "rerank"

    def should_log_rerank_gain(self) -> bool:
        return True
