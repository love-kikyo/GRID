"""In-job evidence for exact training-state restoration and bounded CPU use."""

import json
import math
from pathlib import Path

import torch
from lightning.pytorch import Callback

from src.runtime_cpu import check_affinity, process_tree_snapshot


def assert_equal(actual, expected, path='state'):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.detach().cpu(), expected.cpu()):
            raise AssertionError(f'Restored tensor mismatch: {path}')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f'Restored mapping mismatch: {path}')
        for key in expected:
            assert_equal(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f'Restored sequence mismatch: {path}')
        for i, value in enumerate(expected):
            assert_equal(actual[i], value, f'{path}[{i}]')
    elif actual != expected:
        raise AssertionError(f'Restored value mismatch: {path}: {actual!r} != {expected!r}')


class CPUResumeAudit(Callback):
    def __init__(self, prepared_checkpoint, report_dir, expected_step, smoke_steps=0):
        self.prepared_checkpoint = prepared_checkpoint
        self.report_dir = Path(report_dir)
        self.expected_step = int(expected_step)
        self.smoke_steps = int(smoke_steps)
        self.validation_batches = 0
        self.samples = []
        self.loss_checks = 0

    def sample(self, trainer, phase):
        sample = process_tree_snapshot()
        sample.update({'phase': phase, 'global_step': trainer.global_step})
        self.samples.append(sample)

    def write(self, trainer, name, payload):
        self.report_dir.mkdir(parents=True, exist_ok=True)
        with (self.report_dir / f'rank_{trainer.global_rank}_{name}.json').open('x') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')

    def on_train_start(self, trainer, pl_module):
        allocation = check_affinity(rank=True)
        if trainer.world_size != 3 or trainer.global_step != self.expected_step:
            raise AssertionError('Expected three ranks and the original optimizer step')
        if pl_module.load_only_weights or getattr(pl_module, '_needs_trainer_reset', False):
            raise AssertionError('Weights-only/reset mode is incompatible with full resume')
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise AssertionError('PyTorch thread pools are not bounded')
        payload = torch.load(self.prepared_checkpoint, map_location='cpu', mmap=True, weights_only=False)
        assert_equal(pl_module.state_dict(), payload['state_dict'], 'model')
        assert_equal([o.state_dict() for o in trainer.optimizers], payload['optimizer_states'], 'optimizer')
        assert_equal([s.scheduler.state_dict() for s in trainer.lr_scheduler_configs], payload['lr_schedulers'], 'scheduler')
        checked_callbacks = []
        for callback in trainer.callbacks:
            if callback.state_key.startswith(('EarlyStopping{', 'ModelCheckpoint{')):
                actual = dict(callback.state_dict())
                expected = dict(payload['callbacks'][callback.state_key])
                # Lightning restores best/top-k state, not the last validation's
                # transient current_score; it is recomputed at the next validation.
                actual.pop('current_score', None)
                expected.pop('current_score', None)
                assert_equal(actual, expected, callback.state_key)
                checked_callbacks.append(callback.state_key)
        if len(checked_callbacks) != 2:
            raise AssertionError('Early stopping and best selection must both be restored')
        del payload
        self.sample(trainer, 'restored')
        self.write(trainer, 'restored', {
            'status': 'passed', 'world_size': trainer.world_size, 'global_rank': trainer.global_rank,
            'global_step': trainer.global_step, 'allocation': allocation,
            'model_optimizer_scheduler_exact': True, 'callbacks_checked': checked_callbacks,
            'sample': self.samples[-1], 'data_cursor_restored': False,
        })
        print(f'[cpu-resume] rank={trainer.global_rank} restored step={trainer.global_step} exactly', flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if loss is None or not torch.isfinite(loss).all():
            raise AssertionError('Missing/nonfinite resumed loss')
        self.loss_checks += 1
        interval = 50 if self.smoke_steps else 2000
        if self.loss_checks % interval == 0:
            self.sample(trainer, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_batches += 1
        if batch_idx == 0:
            self.sample(trainer, 'validation')

    def on_fit_end(self, trainer, pl_module):
        if self.smoke_steps and trainer.global_step != self.expected_step + self.smoke_steps:
            raise AssertionError('Smoke did not complete the intended resumed optimizer steps')
        if not self.loss_checks or not self.validation_batches:
            raise AssertionError('Need actual training and validation, not a configuration-only run')
        metrics = {}
        for name in ('val/loss', 'val/recall@5', 'val/ndcg@5'):
            value = trainer.callback_metrics.get(name)
            if value is None or not math.isfinite(float(value)):
                raise AssertionError(f'Missing/nonfinite validation metric: {name}')
            metrics[name] = float(value)
        self.sample(trainer, 'finished')
        self.write(trainer, 'finished', {
            'status': 'passed', 'world_size': trainer.world_size, 'global_rank': trainer.global_rank,
            'initial_step': self.expected_step, 'final_step': trainer.global_step,
            'loss_checks': self.loss_checks, 'validation_batches': self.validation_batches,
            'metrics': metrics, 'samples': self.samples,
            'scope': 'resource/resume smoke only' if self.smoke_steps else 'resumed formal training',
        })
        print(f'[cpu-resume] rank={trainer.global_rank} passed at step={trainer.global_step}', flush=True)
