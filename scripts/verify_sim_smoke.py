"""SIM checks inside the real, CPU-bounded three-rank training entry."""
import json

import torch

from scripts.verify_stage_smoke import StageSmokeAudit
from src.runtime_cpu import process_tree_snapshot


class SIMSmokeAudit(StageSmokeAudit):
    def __init__(self, teacher_checkpoint, report_dir):
        super().__init__('separate', teacher_checkpoint, report_dir)
        self.sim_gradients = []
        self.cpu_samples = []
        self.candidates_checked = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.candidates_checked:
            return
        inputs = pl_module._split_batch(pl_module._unwrap_batch(batch))
        inputs = {k: v[:2] for k, v in inputs.items()}
        if pl_module.teacher.training or pl_module.teacher_model.training:
            raise AssertionError('Frozen teacher was put into training mode')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            direct = pl_module.teacher.generate_candidates(
                inputs['hist_sid'], inputs['hist_mask'], inputs['hist_itemkey_ctx'],
                inputs['target_sid'], inputs['target_itemkey'], ensure_gt_in_candidates=False)
            result = pl_module._run_rerank(**inputs)
            for key in ('candidate_sid', 'candidate_itemkey', 'beam_scores'):
                torch.testing.assert_close(result[key], direct[key], rtol=0, atol=0)
            changed = dict(inputs)
            changed['target_sid'] = (inputs['target_sid'] + 1) % pl_module.num_embeddings_per_hierarchy
            changed['target_itemkey'] = (changed['target_sid'] * pl_module.powers).sum(-1)
            other = pl_module._run_rerank(**changed)
            for key in ('candidate_sid', 'candidate_itemkey', 'beam_scores', 'rerank_logits', 'rerank_scores'):
                torch.testing.assert_close(result[key], other[key], rtol=0, atol=0)
        self.candidates_checked = True

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        super().on_before_optimizer_step(trainer, pl_module, optimizer)
        groups = dict.fromkeys(('item_embedding', 'gsu_history', 'gsu_query', 'gsu_head',
                                'esu_history', 'esu_query', 'esu_head'), 0)
        for name, param in pl_module.reranker.named_parameters():
            if param.grad is not None and torch.count_nonzero(param.grad):
                groups[name.split('.')[0]] += 1
        if not all(groups.values()):
            raise AssertionError(f'Missing SIM gradient group: {groups}')
        self.sim_gradients.append(groups)
        self.cpu_samples.append(process_tree_snapshot())

    def on_fit_end(self, trainer, pl_module):
        if not self.candidates_checked or len(self.sim_gradients) != 2:
            raise AssertionError('Missing candidate/gradient audit')
        super().on_fit_end(trainer, pl_module)
        report = dict(status='passed', method='sid_sim_soft', global_rank=trainer.global_rank,
                      natural_candidates_equal_teacher=True, target_invariance=True,
                      sim_gradient_checks=self.sim_gradients, cpu_samples=self.cpu_samples,
                      scope='two-step smoke; no quality claim or formal checkpoint')
        with (self.report_dir / f'sim_rank_{trainer.global_rank}.json').open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
