"""Frozen SID + ID-only SIM-soft adapter; legacy Separate is unchanged."""

import torch
from torch.nn import functional as F

from src.baselines.sim_soft import SIMSoftScorer, supervised_bce
from src.models.modules.huggingface.transformer_base_module import ModelMode
from src.models.modules.semantic_id.separate_rerank_model import SeparateRerankModule
from src.models.modules.semantic_id.tiger_generation_model import DualHashEmbedding, SemanticIDEncoderDecoder


class SIMRerankModule(SeparateRerankModule):
    def __init__(self, sid_teacher_ckpt_path, sim_embedding_dim=128,
                 sim_num_hash_buckets=300000, sim_top_k=10, sim_num_heads=4,
                 sim_gsu_loss_weight=1.0, **kwargs):
        # Reuse the training/evaluation shell without allocating SmallReranker.
        SemanticIDEncoderDecoder.__init__(self, **kwargs)
        self.save_hyperparameters(logger=False, ignore=['loss_function', 'decoder', 'codebooks'])
        if self.use_beam_score_in_rerank_loss or self.rerank_objective != 'pointwise':
            raise ValueError('SIM requires independent binary prediction losses, without beam features')
        if sim_gsu_loss_weight <= 0:
            raise ValueError('SIM-soft requires a positive GSU auxiliary weight')
        self.training_task = 'rerank_small'
        self.sid_teacher_ckpt_path = sid_teacher_ckpt_path
        self.sim_gsu_loss_weight = sim_gsu_loss_weight
        self.inject_gt_for_rerank_loss = False
        self.reranker = SIMSoftScorer(
            DualHashEmbedding(num_buckets=sim_num_hash_buckets, embedding_dim=sim_embedding_dim),
            sim_embedding_dim, sim_top_k, sim_num_heads)
        self.teacher_model = self.teacher = None
        self._set_trainable_params()

    def setup(self, stage=None):
        super().setup(stage)
        # FrozenSIDTeacher is newly constructed in setup, after train() may
        # already have run. Freeze the wrapper's mode as well as its SID model.
        self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        # Lightning's recursive train() must never reactivate teacher dropout.
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def _run_rerank(self, hist_sid, hist_mask, hist_itemkey_ctx, target_sid=None,
                    target_itemkey=None, ensure_gt_in_candidates=False, auxiliary=False):
        if self.teacher is None:
            raise RuntimeError('Call setup before generating candidates')
        if ensure_gt_in_candidates:
            raise ValueError('SIM natural candidate path forbids GT insertion')
        self.teacher.eval()
        result = self.teacher.generate_candidates(
            hist_sid, hist_mask, hist_itemkey_ctx, target_sid, target_itemkey,
            ensure_gt_in_candidates=False)
        scored = self.reranker.score(hist_itemkey_ctx, result['candidate_itemkey'], auxiliary=auxiliary)
        logits = scored['esu_logits']
        final = self._combine_rerank_scores(result['beam_scores'], logits)
        top = final.topk(self.top_k_for_generation, dim=-1)
        result.update(rerank_logits=logits, final_scores=final, rerank_scores=top.values,
                      rerank_indices=top.indices,
                      rerank_ids=result['candidate_sid'].gather(1, top.indices.unsqueeze(-1).expand(-1, -1, self.num_hierarchies)),
                      rerank_itemkey=result['candidate_itemkey'].gather(1, top.indices),
                      rerank_labels=result['labels'].gather(1, top.indices))
        if auxiliary:
            result['gsu_logits'] = scored['gsu_logits']
        return result

    def model_step(self, model_input, mode):
        batch = self._split_batch(model_input)
        result = self._run_rerank(**batch, auxiliary=mode is not ModelMode.INFER)
        if mode is ModelMode.INFER:
            return {
                'beam': {'ids': result['candidate_sid'], 'scores': result['beam_scores']},
                'rerank': {'ids': result['rerank_ids'], 'scores': result['rerank_scores'],
                           'click_logits': result['rerank_logits'].gather(1, result['rerank_indices'])},
            }
        labels = result['labels']
        main_esu = supervised_bce(result['rerank_logits'], labels)
        main_gsu = supervised_bce(result['gsu_logits'], labels)
        hits = labels.sum(-1) > 0
        aux_esu, aux_gsu = main_esu * 0, main_gsu * 0
        if self.use_no_hit_aux_loss and (~hits).any():
            # No changes to the natural beam; only a private loss-only copy.
            candidates = result['candidate_itemkey'][~hits].clone()
            rows = torch.arange(candidates.shape[0], device=candidates.device)
            slots = torch.randint(self.top_k_for_generation, (candidates.shape[0],), device=candidates.device)
            candidates[rows, slots] = batch['target_itemkey'][~hits]
            aux_labels = F.one_hot(slots, self.top_k_for_generation).float()
            scored = self.reranker.score(batch['hist_itemkey_ctx'][~hits], candidates, auxiliary=True)
            aux_esu = supervised_bce(scored['esu_logits'], aux_labels)
            aux_gsu = supervised_bce(scored['gsu_logits'], aux_labels)
        esu = main_esu + self.no_hit_aux_loss_weight * aux_esu
        gsu = main_gsu + self.no_hit_aux_loss_weight * aux_gsu
        loss = esu + self.sim_gsu_loss_weight * gsu
        with torch.no_grad():
            target_dist = labels / labels.sum(-1, keepdim=True).clamp_min(1)
            log_prob = (target_dist * F.log_softmax(result['final_scores'], -1)).sum(-1)
            metrics = {
                'beam_log_prob': result['beam_scores'].mean(),
                'click_auc': self._safe_binary_auroc(result['rerank_logits'].sigmoid(), labels),
                'click_log_prob': log_prob[hits].mean() if hits.any() else log_prob.sum() * 0,
                'rerank_hit_rate_in_beam': hits.float().mean(),
                'rerank_no_hit_rate': (~hits).float().mean(),
                'rerank_aux_loss': aux_esu + self.sim_gsu_loss_weight * aux_gsu,
                'sim_esu_loss': esu.detach(), 'sim_gsu_loss': gsu.detach(),
            }
        return result['rerank_logits'], loss, metrics
