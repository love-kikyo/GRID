#!/usr/bin/env python3
"""Prepare a full-state resume copy, inside an allocated CPU-bounded Slurm job."""

import argparse
import hashlib
import json
from pathlib import Path

from src.checkpoint_resume import relocate_best_checkpoint
from src.runtime_cpu import check_affinity, configure_environment, configure_torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    args = parser.parse_args()
    configure_environment()
    check_affinity(rank=False)
    import torch
    import yaml
    configure_torch(torch)

    root = Path(__file__).resolve().parents[1]
    source, run_dir = args.source.resolve(), args.run_dir.resolve()
    if not source.is_relative_to(root / 'logs/train/runs') or source.is_symlink():
        raise ValueError('Resume input must be a local formal run checkpoint')
    if not run_dir.is_relative_to(root / 'logs/train/runs') or run_dir == source.parent.parent:
        raise ValueError('New run must have an independent local output directory')
    parent = source.parent.parent
    config_path = parent / '.hydra/config.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    if cfg['seed'] != args.seed or cfg['training_task'] != 'click' or cfg['model']['rerank_objective'] != 'listwise':
        raise ValueError('Resume seed/method do not match the requested listwise run')
    if Path(cfg['model']['manual_ckpt_path']).resolve() != args.teacher.resolve():
        raise ValueError('Parent run used a different SID teacher')
    parent_preflight = root / f"logs/training_preflight/listwise_seed{args.seed}_{parent.name.rsplit('-', 1)[-1]}.json"
    old = json.loads(parent_preflight.read_text())
    new = json.loads(args.preflight.read_text())
    if old['status'] != 'passed' or new['status'] != 'passed' or old['data_root'] != new['data_root']:
        raise ValueError('Parent/current data preflights do not agree')
    if Path(cfg['data_dir']).resolve() != Path(old['data_root']).resolve():
        raise ValueError('Parent run config and audited data disagree')
    if cfg['trainer']['max_steps'] != 120000 or cfg['optim']['scheduler']['num_training_steps'] != 120000:
        raise ValueError('Parent training budget differs from the supported 120000-step schedule')
    if old['counts'] != new['counts'] or old['source_manifests'] != new['source_manifests']:
        raise ValueError('Frozen data changed since parent training')
    if old['initialization_checkpoint']['sha256'] != new['initialization_checkpoint']['sha256']:
        raise ValueError('SID teacher identity changed')
    before = source.stat()
    source_hash = sha256(source)
    checkpoint = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    step = int(checkpoint['global_step'])
    if not 0 < step < 120000:
        raise ValueError('Resume step is outside the unchanged training budget')
    destination = run_dir / 'checkpoints' / source.name
    prepared = relocate_best_checkpoint(checkpoint, source, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        torch.save(prepared, handle)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Parent checkpoint changed during preparation')
    report = {
        'status': 'prepared', 'source_checkpoint': str(source), 'source_sha256': source_hash,
        'prepared_checkpoint': str(destination), 'prepared_sha256': sha256(destination),
        'parent_config': str(config_path), 'parent_config_sha256': sha256(config_path),
        'parent_preflight': str(parent_preflight), 'parent_preflight_sha256': sha256(parent_preflight),
        'current_preflight': str(args.preflight.resolve()), 'current_preflight_sha256': sha256(args.preflight),
        'global_step': step, 'teacher_checkpoint': str(args.teacher.resolve()), 'seed': args.seed,
        'changes': 'Only ModelCheckpoint output paths rebased; tensors, optimizer, scheduler, loops and early stopping retained',
        'data_cursor_restored': False,
    }
    with (run_dir / 'resume_preparation.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(step)


if __name__ == '__main__':
    main()
