"""Audit actual staged initialization and gradients inside the normal DDP trainer."""

import json
from pathlib import Path

import torch
from lightning.pytorch import Callback


def compare_teacher_parameters(module, state):
    checked = 0
    for name, parameter in module.named_parameters():
        if name not in state or not torch.equal(parameter.detach().cpu(), state[name]):
            raise AssertionError(f"Teacher parameter was not loaded exactly: {name}")
        checked += parameter.numel()
    if not checked:
        raise AssertionError("No teacher parameters were compared")
    return checked


def check_gradient_groups(module, method):
    positive = {key: 0 for key in (['reranker'] if method == 'separate' else ['backbone', 'sid', 'click'])}
    for name, parameter in module.named_parameters():
        expected_trainable = name.startswith('reranker.') if method == 'separate' else True
        if parameter.requires_grad != expected_trainable:
            raise AssertionError(f"Unexpected trainable flag: {name}")
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise AssertionError(f"Frozen parameter has a gradient: {name}")
            continue
        grad = parameter.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise AssertionError(f"Nonfinite gradient: {name}")
        if torch.count_nonzero(grad).item():
            group = ('reranker' if method == 'separate' else
                     'sid' if name.startswith('decoder.sid_head.') else
                     'click' if name.startswith('decoder.click_head.') else 'backbone')
            positive[group] += 1
    if not all(positive.values()):
        raise AssertionError(f"Missing nonzero gradient group: {positive}")
    return positive


class StageSmokeAudit(Callback):
    def __init__(self, method, teacher_checkpoint, report_dir, expected_world_size=3):
        self.method = method
        self.teacher_checkpoint = teacher_checkpoint
        self.report_dir = Path(report_dir)
        self.expected_world_size = expected_world_size
        self.gradient_checks = []
        self.validation_batches = 0
        self.initial_parameters = 0

    def on_fit_start(self, trainer, pl_module):
        if trainer.world_size != self.expected_world_size or trainer.global_step != 0:
            raise AssertionError("Expected a fresh three-rank staged optimization run")
        payload = torch.load(self.teacher_checkpoint, map_location='cpu', weights_only=False, mmap=True)
        target = pl_module.teacher_model if self.method == 'separate' else pl_module
        self.initial_parameters = compare_teacher_parameters(target, payload['state_dict'])
        del payload
        print(f'[stage-smoke] rank={trainer.global_rank} exact_teacher_parameters={self.initial_parameters}', flush=True)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.gradient_checks.append(check_gradient_groups(pl_module, self.method))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_batches += 1

    def on_fit_end(self, trainer, pl_module):
        if trainer.global_step != 2 or len(self.gradient_checks) != 2 or not self.validation_batches:
            raise AssertionError('Smoke must complete two optimizer steps and real validation')
        if self.method == 'separate':
            payload = torch.load(self.teacher_checkpoint, map_location='cpu', weights_only=False, mmap=True)
            compare_teacher_parameters(pl_module.teacher_model, payload['state_dict'])
            del payload
        metrics = {}
        for key in ('val/loss', 'val/recall@5', 'val/ndcg@5'):
            value = trainer.callback_metrics.get(key)
            if value is None or not torch.isfinite(value).all():
                raise AssertionError(f'Missing or nonfinite validation metric: {key}')
            metrics[key] = float(value)
        report = {
            'status': 'passed', 'method': self.method, 'global_rank': trainer.global_rank,
            'world_size': trainer.world_size, 'optimizer_steps': trainer.global_step,
            'exact_initial_teacher_parameters': self.initial_parameters,
            'gradient_checks': self.gradient_checks, 'validation_batches': self.validation_batches,
            'teacher_checkpoint': self.teacher_checkpoint,
            'frozen_teacher_unchanged': True if self.method == 'separate' else None,
            'validation_metrics': metrics,
            'scope': 'two-step smoke only; not a trained model or a quality result',
        }
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / f'rank_{trainer.global_rank}.json'
        with path.open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
        print(f'[stage-smoke] passed: {path}', flush=True)
