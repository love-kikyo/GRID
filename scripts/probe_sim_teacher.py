"""Read-only, same-batch probe of SIM, Separate and original SID generation.

This is diagnostic evidence, never a validation acceptance or metric selection.
No model, checkpoint, or existing export is changed.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path

import torch  # Initialize before the data module imports TensorFlow.
import h5py
import numpy as np

from scripts import export_paired_evaluation as export
from src.runtime_export_cpu import configure_export_cpu, export_cpu_snapshot

RUNS = {
    'sim': ('2026-10-12/sim-soft-train-seed42-139478', '010000'),
    'separate': ('2026-09-09/rerank-small-seed42-132051', '010000'),
    'sid': ('2026-09-05/sid-seed42-131908', '090000'),
}
OLD = {
    'sim_smoke': 'sim_soft_seed42_139478_val_smoke_v2',
    'separate_full': 'separate_seed42_132051_val_full_v1',
    'sid_smoke': 'sid_seed42_131908_val_smoke_v1',
}


def compare(a, b):
    assert a.shape == b.shape
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(shape=list(a.shape), equal_values=int((a == b).sum()),
                total_values=int(a.size), equal_rows=int((a == b).reshape(len(a), -1).all(1).sum()),
                max_abs=float(diff.max()), mean_abs=float(diff.mean()))


def tensor_record(t):
    t = t.detach().cpu().contiguous()
    return dict(shape=list(t.shape), dtype=str(t.dtype),
                sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kernel-sweep', action='store_true', help='SID-only numerical backend isolation')
    args = p.parse_args()
    assert not args.output.exists()
    budget = configure_export_cpu(torch, 1)
    assert budget is not None and torch.cuda.is_available()
    device = torch.device('cuda')
    cfg = export.load_run_config(Path('logs/train/runs') / RUNS['sim'][0])
    export.configure_dataloader(cfg, 'val', 32, 1)
    dm = export.instantiate_datamodule(cfg)
    loader = export.get_loader(dm, 'val')
    batches = [export.unwrap_batch(b) for b in itertools.islice(loader, 2)]
    ids = np.concatenate([export.extract_user_ids(b) for b in batches])
    hashes = np.concatenate([export.example_hashes(b) for b in batches])
    assert len(ids) == 64
    references = {}
    for label, name in OLD.items():
        path = Path('logs/analysis/paired_evaluation') / (name + '.h5')
        with h5py.File(path, 'r') as f:
            ref_ids = f['user_id'][:]
            order = np.argsort(ref_ids)
            pos = np.searchsorted(ref_ids[order], ids)
            assert (pos < len(order)).all()
            aligned = order[pos]
            np.testing.assert_array_equal(ids, ref_ids[aligned])
            np.testing.assert_array_equal(hashes, f['example_hash'][:][aligned])
            # Read only the diagnostic rows, not the full prediction arrays.
            references[label] = {k: np.stack([f[k][i] for i in aligned])
                                 for k in ('beam_sid', 'beam_score')}
    report = dict(scope='64 validation rows; diagnostic only; no acceptance',
                  source_git_commit=os.environ['SOURCE_GIT_COMMIT'], job_id=os.environ['SLURM_JOB_ID'],
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                  matmul_precision=torch.get_float32_matmul_precision(),
                  cpu_budget=budget, input_identity_alignment=True, models={}, comparisons={})
    results = {}
    teacher_path = Path('logs/train/runs') / RUNS['sid'][0] / 'checkpoints/checkpoint_000_090000.ckpt'
    teacher_state = torch.load(teacher_path, map_location='cpu', weights_only=False, mmap=True)['state_dict']
    selected_runs = {'sid': RUNS['sid']} if args.kernel_sweep else RUNS
    for label, (run, step) in selected_runs.items():
        print('[probe] loading ' + label, flush=True)
        run = Path('logs/train/runs') / run
        model = export.instantiate_model(export.load_run_config(run),
                                         run / f'checkpoints/checkpoint_000_{step}.ckpt', torch.device('cpu'))
        teacher = model if label == 'sid' else model.teacher.sid_model
        state = teacher.state_dict()
        assert set(state) == set(teacher_state)
        different = [k for k in state if not torch.equal(state[k], teacher_state[k])]
        report['models'][label] = dict(
            unequal_teacher_parameters=different,
            training_modules=[n for n, m in teacher.named_modules() if m.training],
            buffers={n: tensor_record(t) for n, t in teacher.named_buffers()},
            decoder_config=teacher.decoder.decoder.config.to_dict(),
            attention_classes={n: type(m).__name__ for n, m in teacher.named_modules() if 'self_attn' in n},
        )
        assert not different, different
        model.to(device)
        variants = [('bf16', None), ('none', None)]
        if args.kernel_sweep:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            variants += [('bf16', b) for b in (SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
                                              SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION)]
        for amp, backend in variants:
            key = label + '/' + amp + (('/' + backend.name) if backend is not None else '')
            chunks = {k: [] for k in ('beam_sid', 'beam_score', 'embeds', 'initial_logits')}
            for batch in batches:
                b = export.move_batch_to_device(batch, device)
                sid = b.transformed_sequences['sequence_data'][:, :-4].long().contiguous()
                mask = b.mask[:, :-4].contiguous()
                try:
                    with torch.inference_mode(), export.autocast_context(device, amp), \
                            (sdpa_kernel(backend) if backend is not None else nullcontext()):
                        emb, _ = teacher.build_sid_block_embeds(sid, mask, add_bos_token=True)
                        initial, _, _ = teacher._forward_history_for_next_sid(sid, mask)
                        beam, score, _, _ = teacher._beam_search_semantic_ids(sid, mask)
                except RuntimeError as error:
                    if backend is None:
                        raise
                    report['comparisons'][key] = {'backend_error': str(error)}
                    print('[probe] ' + key + ' unavailable: ' + str(error), flush=True)
                    break
                chunks['beam_sid'].append(beam.reshape(32, 20, 4).cpu().numpy())
                chunks['beam_score'].append(score.float().cpu().numpy())
                chunks['embeds'].append(emb.float().cpu().numpy())
                chunks['initial_logits'].append(initial.float().cpu().numpy())
            if len(chunks['beam_sid']) != len(batches):
                continue
            results[key] = {k: np.concatenate(v) for k, v in chunks.items()}
            report['comparisons'][key + '_vs_saved'] = {
                ref: {k: compare(results[key][k], data[k]) for k in data}
                for ref, data in references.items()}
            print('[probe] ' + key + ' ' + json.dumps(report['comparisons'][key + '_vs_saved']), flush=True)
        del state, teacher, model
        gc.collect()
        torch.cuda.empty_cache()
    for left, right in itertools.combinations(results, 2):
        report['comparisons'][left + '_vs_' + right] = {
            k: compare(results[left][k], results[right][k]) for k in results[left]}
    report['cpu_process_tree'] = export_cpu_snapshot(budget['allowed_cpus'])
    report['checked_at'] = datetime.now().astimezone().isoformat()
    report['probe_sha256'] = export.file_sha256(Path(__file__))
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print('[probe] diagnostic completed: ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
