import copy
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.aggregation import BaseAggregator

from src.data.loading.components.interfaces import SequentialModelInputData
from src.models.components.network_blocks.mlp import MLP
from src.models.modules.huggingface.transformer_base_module import ModelMode
from src.models.modules.semantic_id.tiger_generation_model import (
    DualHashEmbedding,
    SemanticIDEncoderDecoder,
)


class FrozenSIDTeacher(nn.Module):
    def __init__(
        self,
        sid_model: SemanticIDEncoderDecoder,
        top_k_for_generation: int,
        num_hierarchies: int,
    ) -> None:
        super().__init__()
        self.sid_model = sid_model
        self.top_k_for_generation = top_k_for_generation
        self.num_hierarchies = num_hierarchies
        self.sid_model.eval()
        for param in self.sid_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def generate_candidates(
        self,
        hist_sid: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_itemkey: torch.Tensor,
        target_sid: Optional[torch.Tensor] = None,
        target_itemkey: Optional[torch.Tensor] = None,
        ensure_gt_in_candidates: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size = hist_sid.size(0)
        generated_ids, beam_scores, _, _ = self.sid_model._beam_search_semantic_ids(
            sid=hist_sid,
            attention_mask=hist_mask,
        )
        beam_scores = beam_scores.clone()

        candidate_sid = generated_ids.view(
            batch_size, self.top_k_for_generation, self.num_hierarchies
        )
        candidate_itemkey = (candidate_sid * self.sid_model.powers).sum(dim=-1)

        if (
            ensure_gt_in_candidates
            and target_sid is not None
            and target_itemkey is not None
        ):
            target_itemkey_expanded = target_itemkey.unsqueeze(1).expand_as(candidate_itemkey)
            has_hit = (candidate_itemkey == target_itemkey_expanded).any(dim=1)
            no_hit_indices = torch.where(~has_hit)[0]
            if no_hit_indices.numel() > 0:
                replace_indices = torch.randint(
                    low=0,
                    high=self.top_k_for_generation,
                    size=(no_hit_indices.numel(),),
                    device=hist_sid.device,
                )
                gt_beam_scores = self.sid_model._compute_target_sid_log_prob(
                    hist_sid=hist_sid[no_hit_indices],
                    hist_mask=hist_mask[no_hit_indices],
                    target_sid=target_sid[no_hit_indices],
                )
                candidate_sid[no_hit_indices, replace_indices] = target_sid[no_hit_indices]
                beam_scores[no_hit_indices, replace_indices] = gt_beam_scores
                candidate_itemkey = (candidate_sid * self.sid_model.powers).sum(dim=-1)

        if target_itemkey is None:
            labels = torch.zeros_like(candidate_itemkey, dtype=torch.float32)
        else:
            labels = (candidate_itemkey == target_itemkey.unsqueeze(1)).float()

        return {
            "candidate_sid": candidate_sid,
            "candidate_itemkey": candidate_itemkey,
            "beam_scores": beam_scores,
            "labels": labels,
        }


class SmallReranker(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        rank_vocab_size: int,
        rank_embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_hash_buckets: int,
        masking_token: int = -1,
    ) -> None:
        super().__init__()
        self.masking_token = masking_token
        self.item_embedding_table = DualHashEmbedding(
            num_buckets=num_hash_buckets,
            embedding_dim=embedding_dim,
            masking_token=masking_token,
        )
        self.num_hierarchies = num_hierarchies
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.sid_embedding_table = nn.Embedding(
            num_hierarchies * num_embeddings_per_hierarchy + 1,
            embedding_dim,
            padding_idx=0,
        )
        self.rank_embedding = nn.Embedding(rank_vocab_size, rank_embedding_dim)
        hidden_dim_list = [hidden_dim] * max(num_layers - 1, 0)
        input_dim = embedding_dim * 7 + rank_embedding_dim + 2
        self.scorer = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim_list=hidden_dim_list,
            dropout=dropout,
        )

    def _candidate_aware_user_interest(
        self,
        hist_itemkey: torch.Tensor,
        cand_vec: torch.Tensor,
    ) -> torch.Tensor:
        hist_emb = self.item_embedding_table(hist_itemkey)
        hist_mask = hist_itemkey != self.masking_token
        attn_scores = torch.einsum("bkd,bld->bkl", cand_vec, hist_emb)
        attn_scores = attn_scores.masked_fill(
            ~hist_mask.unsqueeze(1),
            torch.finfo(attn_scores.dtype).min,
        )
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights * hist_mask.unsqueeze(1).float()
        denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn_weights = attn_weights / denom
        return torch.einsum("bkl,bld->bkd", attn_weights, hist_emb)

    def _encode_candidate_sid(self, candidate_sid: torch.Tensor) -> torch.Tensor:
        device = candidate_sid.device
        offsets = torch.arange(
            0,
            self.num_hierarchies * self.num_embeddings_per_hierarchy,
            self.num_embeddings_per_hierarchy,
            device=device,
        )
        sid_index = candidate_sid + offsets.view(1, 1, -1)
        sid_emb = self.sid_embedding_table(sid_index)
        return sid_emb.mean(dim=2)

    def forward(
        self,
        hist_itemkey: torch.Tensor,
        candidate_itemkey: torch.Tensor,
        candidate_sid: torch.Tensor,
        beam_scores: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, top_k = candidate_itemkey.shape
        cand_vec = self.item_embedding_table(candidate_itemkey)
        hist_vec = self._candidate_aware_user_interest(hist_itemkey, cand_vec)
        sid_vec = self._encode_candidate_sid(candidate_sid)

        rank_ids = torch.arange(top_k, device=candidate_itemkey.device)
        rank_emb = self.rank_embedding(rank_ids).unsqueeze(0).expand(batch_size, -1, -1)
        sid_alignment = (sid_vec * cand_vec).sum(dim=-1, keepdim=True)

        features = torch.cat(
            [
                hist_vec,
                cand_vec,
                sid_vec,
                hist_vec * cand_vec,
                torch.abs(hist_vec - cand_vec),
                sid_vec * cand_vec,
                torch.abs(sid_vec - cand_vec),
                beam_scores.unsqueeze(-1),
                sid_alignment,
                rank_emb,
            ],
            dim=-1,
        )
        logits = self.scorer(features).squeeze(-1)
        return logits


class SeparateRerankModule(SemanticIDEncoderDecoder):
    def __init__(
        self,
        sid_teacher_ckpt_path: str,
        reranker_embedding_dim: int,
        reranker_hidden_dim: int,
        reranker_num_layers: int,
        reranker_dropout: float,
        reranker_rank_embedding_dim: int,
        reranker_num_hash_buckets: int = 300000,
        rerank_logit_weight: float = 1.0,
        beam_log_prob_weight: float = 1.0,
        use_beam_score_in_rerank_loss: bool = False,
        inject_gt_for_rerank_loss: bool = False,
        rerank_loss_weight: float = 1.0,
        load_only_weights: bool = False,
        manual_ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            rerank_logit_weight=rerank_logit_weight,
            beam_log_prob_weight=beam_log_prob_weight,
            use_beam_score_in_rerank_loss=use_beam_score_in_rerank_loss,
            rerank_loss_weight=rerank_loss_weight,
            load_only_weights=load_only_weights,
            manual_ckpt_path=manual_ckpt_path,
            **kwargs,
        )
        self.save_hyperparameters(
            logger=False,
            ignore=["loss_function", "decoder", "codebooks"],
        )
        self.training_task = "rerank_small"
        self.sid_teacher_ckpt_path = sid_teacher_ckpt_path
        self.inject_gt_for_rerank_loss = inject_gt_for_rerank_loss
        self.reranker = SmallReranker(
            embedding_dim=reranker_embedding_dim,
            num_hierarchies=self.num_hierarchies,
            num_embeddings_per_hierarchy=self.num_embeddings_per_hierarchy,
            rank_vocab_size=self.top_k_for_generation,
            rank_embedding_dim=reranker_rank_embedding_dim,
            hidden_dim=reranker_hidden_dim,
            num_layers=reranker_num_layers,
            dropout=reranker_dropout,
            num_hash_buckets=reranker_num_hash_buckets,
            masking_token=self.masking_token,
        )
        self.teacher_model: Optional[SemanticIDEncoderDecoder] = None
        self.teacher: Optional[FrozenSIDTeacher] = None

    def _set_trainable_params(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.reranker.parameters():
            param.requires_grad = True

    def setup(self, stage=None):
        if self.teacher is None:
            teacher_model = copy.deepcopy(self)
            teacher_model.reranker = nn.Identity()
            teacher_model.teacher = None
            teacher_model.teacher_model = None
            teacher_model.training_task = "sid"
            teacher_model.load_only_weights = False
            teacher_model._manual_ckpt_path = None
            ckpt = torch.load(
                self.sid_teacher_ckpt_path,
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
            state_dict = ckpt.get("state_dict", ckpt)
            teacher_model.load_state_dict(state_dict, strict=False)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model = teacher_model
            self.teacher = FrozenSIDTeacher(
                sid_model=teacher_model,
                top_k_for_generation=self.top_k_for_generation,
                num_hierarchies=self.num_hierarchies,
            )

        if stage == "fit":
            self._set_trainable_params()
            if self.load_only_weights and self._manual_ckpt_path:
                self._load_weights_only()

    def get_primary_evaluator_name(self) -> Optional[str]:
        if "rerank" in self.evaluators:
            return "rerank"
        return super().get_primary_evaluator_name()

    def should_log_rerank_gain(self) -> bool:
        return "beam" in self.evaluators and "rerank" in self.evaluators

    def _split_batch(
        self, model_input: SequentialModelInputData
    ) -> Dict[str, torch.Tensor]:
        sid = model_input.transformed_sequences["sequence_data"].long()
        mask = model_input.mask
        hist_itemkey = model_input.transformed_sequences["hist_itemkey"].long()

        return {
            "target_sid": sid[:, -self.num_hierarchies:].contiguous(),
            "hist_sid": sid[:, :-self.num_hierarchies].contiguous(),
            "hist_mask": mask[:, :-self.num_hierarchies].contiguous(),
            "target_itemkey": hist_itemkey[:, -1].contiguous(),
            "hist_itemkey_ctx": hist_itemkey[:, :-1].contiguous(),
        }

    def _run_rerank(
        self,
        hist_sid: torch.Tensor,
        hist_mask: torch.Tensor,
        hist_itemkey_ctx: torch.Tensor,
        target_sid: Optional[torch.Tensor] = None,
        target_itemkey: Optional[torch.Tensor] = None,
        ensure_gt_in_candidates: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self.teacher is None:
            raise RuntimeError("Teacher model has not been initialized. Call setup() first.")

        candidate_dict = self.teacher.generate_candidates(
            hist_sid=hist_sid,
            hist_mask=hist_mask,
            hist_itemkey=hist_itemkey_ctx,
            target_sid=target_sid,
            target_itemkey=target_itemkey,
            ensure_gt_in_candidates=ensure_gt_in_candidates,
        )
        rerank_logits = self.reranker(
            hist_itemkey=hist_itemkey_ctx,
            candidate_itemkey=candidate_dict["candidate_itemkey"],
            candidate_sid=candidate_dict["candidate_sid"],
            beam_scores=candidate_dict["beam_scores"],
        )
        final_scores = self._combine_rerank_scores(
            beam_log_prob=candidate_dict["beam_scores"],
            rerank_logits=rerank_logits,
        )

        topk = torch.topk(final_scores, k=self.top_k_for_generation, dim=-1)
        rerank_scores = topk.values
        rerank_indices = topk.indices
        gather_index = rerank_indices.unsqueeze(-1).expand(-1, -1, self.num_hierarchies)
        rerank_ids = torch.gather(candidate_dict["candidate_sid"], 1, gather_index)
        rerank_itemkey = torch.gather(candidate_dict["candidate_itemkey"], 1, rerank_indices)
        rerank_labels = torch.gather(candidate_dict["labels"], 1, rerank_indices)

        candidate_dict.update(
            {
                "rerank_logits": rerank_logits,
                "final_scores": final_scores,
                "rerank_scores": rerank_scores,
                "rerank_indices": rerank_indices,
                "rerank_ids": rerank_ids,
                "rerank_itemkey": rerank_itemkey,
                "rerank_labels": rerank_labels,
            }
        )
        return candidate_dict

    def model_step(
        self,
        model_input: SequentialModelInputData,
        mode: ModelMode,
    ):
        batch_dict = self._split_batch(model_input)
        result_dict = self._run_rerank(
            hist_sid=batch_dict["hist_sid"],
            hist_mask=batch_dict["hist_mask"],
            hist_itemkey_ctx=batch_dict["hist_itemkey_ctx"],
            target_sid=batch_dict["target_sid"],
            target_itemkey=batch_dict["target_itemkey"],
            # Always retain the natural teacher beam here.  If it misses the
            # target, a separate GT-injected list is built below for the
            # training-only auxiliary loss; inference candidates never change.
            ensure_gt_in_candidates=False,
        )

        if mode is ModelMode.INFER:
            return {
                "beam": {
                    "ids": result_dict["candidate_sid"],
                    "scores": result_dict["beam_scores"],
                },
                "rerank": {
                    "ids": result_dict["rerank_ids"],
                    "scores": result_dict["rerank_scores"],
                    "click_logits": torch.gather(
                        result_dict["rerank_logits"],
                        1,
                        result_dict["rerank_indices"],
                    ),
                },
            }

        main_loss = self._compute_listwise_rerank_loss(
            rerank_logits=result_dict["rerank_logits"],
            labels=result_dict["labels"],
            beam_log_prob=result_dict["beam_scores"],
        )

        positive_counts = result_dict["labels"].sum(dim=-1, keepdim=True)
        valid_rows = positive_counts.squeeze(-1) > 0
        no_hit_mask = ~valid_rows
        aux_loss = result_dict["rerank_logits"].sum() * 0.0

        # Match RecoChain's no-hit supervision so that the separate-reranker
        # comparison isolates model sharing rather than candidate construction.
        # The natural beam above remains untouched and is the only inference
        # candidate set.
        if self.use_no_hit_aux_loss and no_hit_mask.any():
            no_hit_indices = torch.where(no_hit_mask)[0]
            aux_candidate_sid = result_dict["candidate_sid"][no_hit_indices].clone()
            aux_candidate_itemkey = result_dict["candidate_itemkey"][no_hit_indices].clone()
            aux_beam_scores = result_dict["beam_scores"][no_hit_indices].clone()
            insert_indices = torch.randint(
                low=0,
                high=self.top_k_for_generation,
                size=(no_hit_indices.numel(),),
                device=aux_candidate_sid.device,
            )
            aux_rows = torch.arange(no_hit_indices.numel(), device=aux_candidate_sid.device)

            aux_candidate_sid[aux_rows, insert_indices] = batch_dict["target_sid"][no_hit_indices]
            aux_candidate_itemkey[aux_rows, insert_indices] = batch_dict["target_itemkey"][no_hit_indices]
            if self.teacher is None:
                raise RuntimeError("Teacher model has not been initialized. Call setup() first.")
            with torch.no_grad():
                gt_beam_scores = self.teacher.sid_model._compute_target_sid_log_prob(
                    hist_sid=batch_dict["hist_sid"][no_hit_indices],
                    hist_mask=batch_dict["hist_mask"][no_hit_indices],
                    target_sid=batch_dict["target_sid"][no_hit_indices],
                )
            aux_beam_scores[aux_rows, insert_indices] = gt_beam_scores

            aux_logits = self.reranker(
                hist_itemkey=batch_dict["hist_itemkey_ctx"][no_hit_indices],
                candidate_itemkey=aux_candidate_itemkey,
                candidate_sid=aux_candidate_sid,
                beam_scores=aux_beam_scores,
            )
            aux_labels = F.one_hot(
                insert_indices,
                num_classes=self.top_k_for_generation,
            ).to(aux_logits.dtype)
            aux_loss = self._compute_listwise_rerank_loss(
                rerank_logits=aux_logits,
                labels=aux_labels,
                beam_log_prob=aux_beam_scores,
            )

        loss = main_loss + self.no_hit_aux_loss_weight * aux_loss
        with torch.no_grad():
            click_probs = torch.sigmoid(result_dict["rerank_logits"])
            auc = self._safe_binary_auroc(click_probs, result_dict["labels"])
            target_dist = result_dict["labels"] / positive_counts.clamp_min(1.0)
            per_row_log_prob = (
                target_dist
                * F.log_softmax(result_dict["final_scores"], dim=-1)
            ).sum(dim=-1)
            click_log_prob = per_row_log_prob[valid_rows].mean() if torch.any(valid_rows) else per_row_log_prob.sum() * 0.0
            rerank_hit_rate_in_beam = valid_rows.float().mean()

        metrics = {
            "beam_log_prob": result_dict["beam_scores"].mean(),
            "click_auc": auc,
            "click_log_prob": click_log_prob,
            "rerank_hit_rate_in_beam": rerank_hit_rate_in_beam,
            "rerank_no_hit_rate": no_hit_mask.float().mean(),
            "rerank_aux_loss": aux_loss.detach(),
        }
        return result_dict["rerank_logits"], loss, metrics

    def _unwrap_batch(self, batch):
        return batch[0] if isinstance(batch, tuple) else batch

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        batch = self._unwrap_batch(batch)
        _, loss, metrics = self.model_step(model_input=batch, mode=ModelMode.TRAIN)

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

        if self.training_loop_function is not None:
            self.training_loop_function(self, loss)

        lr = self.lr_schedulers().get_last_lr()[0]
        self.log("train/lr", lr, on_step=True, on_epoch=False, sync_dist=False)
        return loss

    def eval_step(
        self,
        batch: SequentialModelInputData,
        loss_to_aggregate: BaseAggregator,
    ):
        batch = self._unwrap_batch(batch)
        with torch.inference_mode():
            _, loss, _ = self.model_step(model_input=batch, mode=ModelMode.TRAIN)
        loss_to_aggregate(loss)

        with torch.inference_mode():
            result_dict = self.model_step(model_input=batch, mode=ModelMode.INFER)

        sid = batch.transformed_sequences["sequence_data"].long()
        labels = sid[:, -self.num_hierarchies:]

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

        rerank_click_logits = result_dict["rerank"]["click_logits"]
        rerank_ids = result_dict["rerank"]["ids"]
        target_itemkey = (labels * self.powers).sum(dim=-1)
        rerank_itemkey = (rerank_ids * self.powers).sum(dim=-1)
        click_labels = (rerank_itemkey == target_itemkey.unsqueeze(1)).float()
        click_probs = torch.sigmoid(rerank_click_logits)
        auc = self._safe_binary_auroc(click_probs, click_labels)
        self.click_auc_accumulator(auc)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.click_auc_accumulator.reset()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.click_auc_accumulator.reset()
