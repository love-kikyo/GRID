"""In-job proof of fresh random Joint-listwise initialization and actual updates.

The installed Lightning version calls callback.setup before model.setup/restore.
It restores optimizer/loop state after on_fit_start, so on_train_start repeats the
fresh-state checks. Full parameter hashes avoid retaining a second large model.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from lightning.pytorch import Callback

from scripts.verify_stage_smoke import check_gradient_groups
from src.runtime_cpu import check_affinity, process_tree_snapshot


PHASES = ('initialization', 'train_start', 'updates', 'finished')
GROUPS = ('backbone', 'sid', 'click')


def parameter_group(name):
    return ('sid' if name.startswith('decoder.sid_head.') else
            'click' if name.startswith('decoder.click_head.') else 'backbone')


def parameter_fingerprints(module):
    result = {}
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest = hashlib.sha256(memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast('B')).hexdigest()
        result[name] = {'sha256': digest, 'shape': list(value.shape),
                        'dtype': str(value.dtype), 'numel': value.numel()}
    if not result:
        raise AssertionError('No model parameters to audit')
    return result


def assert_fresh(trainer, module, expected_world_size, expected_seed):
    if trainer.world_size != expected_world_size or trainer.global_step != 0 or trainer.current_epoch != 0:
        raise AssertionError('Expected a fresh optimization run at step/epoch zero')
    if trainer.ckpt_path is not None:
        raise AssertionError('A checkpoint restoration path is forbidden')
    if (module.load_only_weights is not False or module._manual_ckpt_path is not None
            or getattr(module, '_needs_trainer_reset', False)):
        raise AssertionError('Teacher loading/reset is forbidden for random initialization')
    if module.training_task != 'click' or module.rerank_objective != 'listwise':
        raise AssertionError('Expected Joint-listwise training')
    if len(trainer.optimizers) != 1 or trainer.optimizers[0].state:
        raise AssertionError('Expected one optimizer with empty initial state')
    if os.environ.get('PL_GLOBAL_SEED') != str(expected_seed) or torch.initial_seed() != expected_seed:
        raise AssertionError('Unexpected actual training seed')
    if not all(parameter.requires_grad for parameter in module.parameters()):
        raise AssertionError('All Joint parameters must be trainable')


def update_evidence(module, optimizer, initial, expected_step=2):
    current = parameter_fingerprints(module)
    if current.keys() != initial.keys():
        raise AssertionError('Parameter structure changed during training')
    changed = {group: [] for group in GROUPS}
    steps = {}
    for name, parameter in module.named_parameters():
        before, after = initial[name], current[name]
        if any(before[key] != after[key] for key in ('shape', 'dtype', 'numel')):
            raise AssertionError(f'Parameter identity changed: {name}')
        if before['sha256'] != after['sha256']:
            changed[parameter_group(name)].append(name)
        state = optimizer.state.get(parameter)
        if state:
            step = float(state.get('step', -1))
            if not math.isfinite(step) or step != expected_step:
                raise AssertionError(f'Unexpected optimizer step for {name}: {step}')
            steps[name] = int(step)
    if not all(changed.values()) or not steps:
        raise AssertionError(f'Missing actual parameter update group: {changed}')
    for names in changed.values():
        if not all(name in steps for name in names):
            raise AssertionError('Changed parameter has no matching optimizer state')
    return {'changed_parameters': changed, 'optimizer_parameter_steps': steps,
            'parameter_fingerprints_after': current}


class RandomInitAudit(Callback):
    def __init__(self, report_dir, smoke=False, expected_world_size=3, expected_seed=42):
        self.report_dir = Path(report_dir)
        self.smoke = bool(smoke)
        self.expected_world_size = int(expected_world_size)
        self.expected_seed = int(expected_seed)
        self.initial = None
        self.fit_start_checked = False
        self.train_start_checked = False
        self.gradient_checks = []
        self.updated = False
        self.validation_batches = 0

    def write(self, trainer, phase, payload):
        report = dict(schema_version=1, status='passed', phase=phase,
                      method='joint-listwise-random-init', smoke=self.smoke,
                      global_rank=trainer.global_rank, world_size=trainer.world_size,
                      seed=self.expected_seed, global_step=trainer.global_step, **payload)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / f'rank_{trainer.global_rank}_{phase}.json'
        with path.open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
        print(f'[random-init-audit] passed: {path}', flush=True)

    def setup(self, trainer, pl_module, stage):
        if stage == 'fit':
            if self.initial is not None:
                raise AssertionError('Random initialization setup may run only once')
            if (pl_module.load_only_weights is not False or pl_module._manual_ckpt_path is not None
                    or trainer.ckpt_path is not None):
                raise AssertionError('Random initialization cannot load a checkpoint')
            self.initial = parameter_fingerprints(pl_module)

    def fresh_check(self, trainer, pl_module):
        assert_fresh(trainer, pl_module, self.expected_world_size, self.expected_seed)
        if self.initial is None or parameter_fingerprints(pl_module) != self.initial:
            raise AssertionError('Constructed random parameters changed before training')
        allocation = check_affinity(rank=True)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise AssertionError('PyTorch thread pools are not bounded')
        return allocation

    def on_fit_start(self, trainer, pl_module):
        allocation = self.fresh_check(trainer, pl_module)
        self.fit_start_checked = True
        self.write(trainer, 'initialization', {
            'constructed_parameters_unchanged': True, 'optimizer_initially_empty': True,
            'initialization_checkpoint': None, 'resume_checkpoint': None,
            'parameter_fingerprints': self.initial, 'allocation': allocation,
            'cpu': process_tree_snapshot(),
        })

    def on_train_start(self, trainer, pl_module):
        if not self.fit_start_checked:
            raise AssertionError('Missing fit-start initialization audit')
        allocation = self.fresh_check(trainer, pl_module)
        self.train_start_checked = True
        self.write(trainer, 'train_start', {
            'constructed_parameters_unchanged': True, 'optimizer_initially_empty': True,
            'checked_after_training_state_restore': True,
            'initialization_checkpoint': None, 'resume_checkpoint': None,
            'parameter_fingerprints': self.initial, 'allocation': allocation,
            'cpu': process_tree_snapshot(),
            'gpu': torch.cuda.get_device_name(torch.cuda.current_device()),
            'model_config': {name: getattr(pl_module, name) for name in
                             ('training_task', 'rerank_objective', 'load_only_weights',
                              'top_k_for_generation', 'top_k_for_score', 'active_top_k_for_score',
                              'rerank_loss_weight', 'no_hit_aux_loss_weight',
                              'rerank_logit_weight', 'beam_log_prob_weight')},
            'trainer_config': {'max_steps': trainer.max_steps,
                               'accumulate_grad_batches': trainer.accumulate_grad_batches,
                               'precision': str(trainer.precision)},
        })

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if len(self.gradient_checks) >= 2:
            return
        if not self.train_start_checked or trainer.global_step != len(self.gradient_checks):
            raise AssertionError('Unexpected first optimizer-step sequence')
        self.gradient_checks.append(check_gradient_groups(pl_module, 'listwise'))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.updated or trainer.global_step < 2:
            return
        if trainer.global_step != 2 or len(self.gradient_checks) != 2:
            raise AssertionError('Missed the first two optimizer updates')
        evidence = update_evidence(pl_module, trainer.optimizers[0], self.initial)
        self.write(trainer, 'updates', dict(evidence, gradient_checks=self.gradient_checks,
                                          cpu=process_tree_snapshot()))
        self.updated = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            self.validation_batches += 1

    def on_fit_end(self, trainer, pl_module):
        if (not self.updated or not self.train_start_checked or not self.validation_batches
                or (self.smoke and trainer.global_step != 2)):
            raise AssertionError('Need audited updates and real validation; smoke must end at step two')
        metrics = {}
        for name in ('val/loss', 'val/recall@5', 'val/ndcg@5'):
            value = trainer.callback_metrics.get(name)
            if value is None or not math.isfinite(float(value)):
                raise AssertionError(f'Missing/nonfinite validation metric: {name}')
            metrics[name] = float(value)
        self.write(trainer, 'finished', {
            'validation_batches': self.validation_batches, 'validation_metrics': metrics,
            'first_two_updates_verified': True, 'cpu': process_tree_snapshot(),
            'scope': 'two-step smoke only; not a quality result' if self.smoke else 'formal random-init training',
        })


def verify_reports(report_dir, smoke=False, expected_world_size=3, expected_seed=42):
    directory = Path(report_dir)
    wanted = {f'rank_{rank}_{phase}.json' for rank in range(expected_world_size) for phase in PHASES}
    if {path.name for path in directory.glob('rank_*.json')} != wanted:
        raise AssertionError('Missing, duplicate, or unexpected rank audit reports')
    reference = None
    for rank in range(expected_world_size):
        reports = {}
        for phase in PHASES:
            report = json.loads((directory / f'rank_{rank}_{phase}.json').read_text())
            expected = dict(schema_version=1, status='passed', phase=phase,
                            method='joint-listwise-random-init', smoke=bool(smoke),
                            global_rank=rank, world_size=expected_world_size, seed=expected_seed)
            if any(type(report.get(key)) is not type(value) or report[key] != value for key, value in expected.items()):
                raise AssertionError(f'Wrong audit identity: rank {rank}, {phase}')
            reports[phase] = report
        for phase in ('initialization', 'train_start'):
            report = reports[phase]
            if (type(report['global_step']) is not int or report['global_step'] != 0
                    or report['constructed_parameters_unchanged'] is not True
                    or report['optimizer_initially_empty'] is not True
                    or report['initialization_checkpoint'] is not None or report['resume_checkpoint'] is not None):
                raise AssertionError('Invalid fresh initialization evidence')
            hashes = report['parameter_fingerprints']
            if not hashes or (reference is not None and hashes != reference):
                raise AssertionError('Initial parameters differ across phases/ranks')
            reference = hashes
        if reports['train_start']['checked_after_training_state_restore'] is not True:
            raise AssertionError('Missing post-restoration audit')
        update = reports['updates']
        if (type(update['global_step']) is not int or update['global_step'] != 2
                or len(update['gradient_checks']) != 2
                or any(set(g) != set(GROUPS) or any(type(v) is not int or v <= 0 for v in g.values())
                       for g in update['gradient_checks'])
                or set(update['changed_parameters']) != set(GROUPS)
                or not all(update['changed_parameters'].values())
                or not update['optimizer_parameter_steps']
                or any(type(v) is not int or v != 2 for v in update['optimizer_parameter_steps'].values())):
            raise AssertionError('Invalid first-two-update evidence')
        after = update['parameter_fingerprints_after']
        if after.keys() != reference.keys():
            raise AssertionError('Updated parameter identity does not match initialization')
        expected_changed = {group: [] for group in GROUPS}
        for name, before in reference.items():
            if any(after[name][key] != before[key] for key in ('shape', 'dtype', 'numel')):
                raise AssertionError('Parameter metadata changed in update report')
            if before['sha256'] != after[name]['sha256']:
                expected_changed[parameter_group(name)].append(name)
        if (update['changed_parameters'] != expected_changed
                or any(name not in update['optimizer_parameter_steps']
                       for names in expected_changed.values() for name in names)):
            raise AssertionError('Changed-parameter claims do not match full hashes/optimizer')
        final = reports['finished']
        metrics = final['validation_metrics']
        if (type(final['global_step']) is not int or final['global_step'] < 2
                or (smoke and final['global_step'] != 2)
                or type(final['validation_batches']) is not int or final['validation_batches'] <= 0
                or final['first_two_updates_verified'] is not True
                or set(metrics) != {'val/loss', 'val/recall@5', 'val/ndcg@5'}
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values())):
            raise AssertionError('Incomplete fit/validation evidence')
    return {'status': 'passed', 'world_size': expected_world_size, 'seed': expected_seed,
            'smoke': bool(smoke), 'report_dir': str(directory.resolve()),
            'initial_parameters_identical_across_ranks': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_reports(args.report_dir, args.smoke)))
