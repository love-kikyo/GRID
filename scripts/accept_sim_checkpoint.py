"""Audit a completed SIM run on a compute node before any quality evaluation."""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path

import torch
import yaml

from scripts.export_paired_evaluation import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def identity(path):
    path = Path(path).resolve()
    before = path.stat()
    sha = file_sha256(path)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return dict(path=str(path), sha256=sha, size_bytes=after.st_size, mtime_ns=after.st_mtime_ns)


def finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all().item(), 'Nonfinite checkpoint tensor'
        return 1
    if isinstance(value, dict):
        return sum(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(finite(v) for v in value)
    if isinstance(value, float):
        assert math.isfinite(value)
    return 0


def frozen_teacher_checks(state, teacher):
    checks = []
    for prefix in ('teacher_model.', 'teacher.sid_model.'):
        actual = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        assert actual.keys() == teacher.keys(), f'Missing teacher state: {prefix}'
        for key, value in teacher.items():
            assert actual[key].shape == value.shape and actual[key].dtype == value.dtype
            assert torch.equal(actual[key], value), prefix + key
        checks.append(dict(prefix=prefix, tensors=len(actual), values=sum(v.numel() for v in actual.values()), exact=True))
    return checks


def accept(launch_path):
    launch = json.loads(launch_path.read_text())
    run = ROOT / launch['run_dir']
    cfg_path = run / '.hydra/config.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    pf_path = ROOT / launch['preflight']
    pf = json.loads(pf_path.read_text())
    assert launch['mode'] == 'formal_training' and launch['source_dirty_at_submit'] is False
    assert cfg['seed'] == launch['seed'] and cfg['test'] is False
    assert cfg['model']['_target_'].endswith('.SIMRerankModule')
    assert pf['status'] == 'passed' and pf['protocol'] == 'history-last-user-v1'
    assert Path(cfg['data_dir']).resolve() == Path(pf['data_root']).resolve()
    assert Path(cfg['model']['sid_teacher_ckpt_path']).resolve() == Path(pf['initialization_checkpoint']['path']).resolve()
    for entry in pf['source_files'] + list(pf['source_manifests'].values()) + [pf['semantic_id'], pf['item2row_mapping']]:
        assert file_sha256(entry['path']) == entry['sha256'], entry['path']
    teacher_record = identity(pf['initialization_checkpoint']['path'])
    assert teacher_record['sha256'] == pf['initialization_checkpoint']['sha256']
    checkpoints = list((run / 'checkpoints').glob('*.ckpt'))
    assert len(checkpoints) == 1, 'Expected save_top_k=1 and exactly one checkpoint'
    checkpoint = checkpoints[0]
    record = identity(checkpoint)
    rows = list(csv.DictReader((run / 'csv/metrics.csv').open()))
    validation = [(int(row['step']) + 1, float(row['val/recall@5']))
                  for row in rows if row.get('val/recall@5')]
    assert validation and all(math.isfinite(score) for _, score in validation)
    best_step, best_score = max(validation, key=lambda row: row[1])
    assert cfg['callbacks']['model_checkpoint']['monitor'] == 'val/recall@5'
    assert cfg['callbacks']['early_stopping']['patience'] == 5
    assert cfg['model']['rerank_logit_weight'] == .1
    stderr = ROOT / f"logs/slurm/GRID-SIM-HLastV1-{launch['job_id']}.err"
    error_text = stderr.read_text()
    assert 'Signaling Trainer to stop' in error_text and 'Traceback (most recent call last)' not in error_text
    assert len(validation) - next(i for i, v in enumerate(validation) if v[0] == best_step) - 1 == 5
    restart = json.loads((run / 'metadata/restart_metadata.json').read_text())
    assert restart['current_run'] == 0 and not restart['restarts']
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert payload['global_step'] == best_step and payload['epoch'] == 0
    assert len(payload['optimizer_states']) == len(payload['lr_schedulers']) == 1
    assert payload['lr_schedulers'][0]['last_epoch'] == best_step
    callbacks = [v for k, v in payload['callbacks'].items() if k.startswith('ModelCheckpoint')]
    assert len(callbacks) == 1
    assert Path(callbacks[0]['best_model_path']).resolve() == checkpoint.resolve()
    assert abs(float(callbacks[0]['best_model_score']) - best_score) < 1e-8
    model_count, optim_count = finite(payload['state_dict']), finite(payload['optimizer_states'])
    teacher = torch.load(teacher_record['path'], map_location='cpu', mmap=True, weights_only=False)['state_dict']
    frozen = frozen_teacher_checks(payload['state_dict'], teacher)
    return dict(status='passed', checked_at=datetime.now().astimezone().isoformat(),
                training_job_id=launch['job_id'], audit_job_id=os.environ.get('SLURM_JOB_ID'),
                source_git_commit=os.environ['SOURCE_GIT_COMMIT'], training_source_commit=launch['source_git_commit'],
                scope='training artifact and best-checkpoint acceptance; no full validation or test result',
                run_dir=str(run), seed=cfg['seed'], checkpoint=record, configuration=identity(cfg_path),
                training_preflight=identity(pf_path), teacher_checkpoint=teacher_record,
                launch=identity(launch_path), metrics_csv=identity(run / 'csv/metrics.csv'),
                stderr=identity(stderr), restart_metadata=identity(run / 'metadata/restart_metadata.json'),
                best_step=best_step, stopped_step=validation[-1][0], best_runtime_val_recall5=best_score,
                validation_curve=validation, stopping_reason='patience_5', frozen_teacher_checks=frozen,
                model_tensors_finite=model_count, optimizer_tensors_finite=optim_count,
                scheduler_terminal_status='unavailable: accounting database refused connection; controller expired',
                cpu_affinity=sorted(os.sched_getaffinity(0)), audit_script=identity(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert len(os.sched_getaffinity(0)) == 1, 'Checkpoint audit must stay on one CPU'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not args.output.exists()
    result = accept(args.launch)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('[sim-checkpoint] ' + json.dumps(result), flush=True)
