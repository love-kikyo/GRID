"""Run the frozen prefix diagnostic; all interventions share natural beam20."""
import argparse
import json
import os
from pathlib import Path
import time

from scripts.prepare_retrieval_diagnostic import ROOT, POLICIES, load_manifest, sha, write_new


def require_gate(m, directory):
    if directory is None:
        raise ValueError('Diagnostic requires accepted smoke')
    path = directory / 'acceptance.json'
    acceptance = json.loads(path.read_text())
    run = json.loads((directory / 'run.json').read_text())
    if (acceptance['status'] != 'passed' or acceptance['seed'] != 42 or
            run['phase'] != 'smoke' or run['seed'] != 42 or run['processed'] != 65 or
            run['manifest_sha256'] != m['_manifest_sha256'] or run['smoke_gate']['status'] != 'passed'):
        raise ValueError('Wrong smoke acceptance identity')
    from scripts.audit_retrieval_diagnostic import audit
    fresh = audit(directory, m, 'smoke')
    # Only the timestamp may differ after a read-only repeat of the same audit.
    # Bind the receipt to all current artifacts, metadata and computed metrics.
    if ({key: value for key, value in acceptance.items() if key != 'checked_at'} !=
            {key: value for key, value in fresh.items() if key != 'checked_at'}):
        raise ValueError('Smoke acceptance receipt differs from current artifacts')
    return {'path': str(path), 'sha256': sha(path)}, run['smoke_gate']


def run(args):
    started = time.monotonic()
    m = load_manifest(args.manifest)
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing existing directory: {args.output_dir}')
    from src.runtime_cpu import THREAD_VARIABLES
    if (os.environ.get('GRID_EXPORT_CPU_BUDGET') != '4' or
            any(os.environ.get(key) != '1' for key in THREAD_VARIABLES)):
        raise RuntimeError('Require four-core allocation and one-thread libraries before import')
    import h5py
    import numpy as np
    import torch
    from src.runtime_export_cpu import configure_export_cpu, export_cpu_snapshot
    from scripts import export_paired_evaluation as e
    from scripts.profile_efficiency_cpu4 import gpu_info, instantiate
    from scripts.retrieval_diagnostic_core import infer_diagnostics, smoke_gate
    if e.ROOT.resolve() != ROOT:
        raise RuntimeError('Wrong execution worktree imported')
    budget = configure_export_cpu(torch, 1)
    cpu = [export_cpu_snapshot(budget['allowed_cpus'])]
    prior_gate, gate = None, None
    if args.phase == 'diagnostic':
        prior_gate, gate = require_gate(m, args.smoke_dir)
    elif args.smoke_dir is not None:
        raise ValueError('Smoke cannot reuse a prior gate')
    for field in ('checkpoint', 'reference_test'):
        if sha(m[field]['path']) != m[field]['sha256']:
            raise ValueError(f'Changed accepted {field}')
    cfg = e.load_run_config(Path(m['run_dir']))
    if (cfg.seed != 42 or cfg.sequence_length != 32 or cfg.hist_itemkey_length != 100 or
            cfg.model.top_k_for_generation != 20 or cfg.model.top_k_for_score != 10 or
            cfg.model.active_top_k_for_score is not None or cfg.model.rerank_objective != 'listwise' or
            cfg.model.rerank_logit_weight != .1 or cfg.model.beam_log_prob_weight != 1.):
        raise ValueError('Unexpected trained model configuration')
    provenance, expected = e.validate_provenance(
        cfg, 'test', Path(m['data_root']), m['protocol'], Path(m['training_preflight']),
        preflight_source_root=Path(m['asset_root']))
    if expected != 2061101:
        raise ValueError('Original test population changed')
    e.configure_dataloader(cfg, 'test', 32, 1)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device('cuda')
    hardware = gpu_info(torch)
    model, loading = instantiate(cfg, {'model': 'listwise', 'checkpoint': m['checkpoint']}, device, torch)
    model.requires_grad_(False)
    limit = m['smoke_samples'] if args.phase == 'smoke' else m['diagnostic_samples']
    dm = e.instantiate_datamodule(cfg)
    iterator = iter(e.get_loader(dm, 'test'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    marker = args.output_dir / 'INCOMPLETE'
    marker.touch(exist_ok=False)
    writers = {name: e.H5StreamWriter(args.output_dir / f'{name}.h5') for name in POLICIES}
    buffers = {name: None for name in POLICIES}
    processed = batches = buffered = 0
    forward_seconds = 0.
    torch.cuda.reset_peak_memory_stats()
    try:
        with h5py.File(m['reference_test']['path'], 'r') as reference:
            for wrapped in iterator:
                batch = e.unwrap_batch(wrapped)
                size = len(batch.transformed_sequences['sequence_data'])
                if size > limit - processed:
                    batch = e.slice_batch(batch, limit - processed)
                count = len(batch.transformed_sequences['sequence_data'])
                stop = processed + count
                identity = {'user_id': e.extract_user_ids(batch), 'example_hash': e.example_hashes(batch),
                            'target_sid': batch.transformed_sequences['sequence_data'][:, -4:].numpy()}
                for field, value in identity.items():
                    if not np.array_equal(value, reference[field][processed:stop]):
                        raise ValueError(f'Frozen test prefix mismatch: {field}, {processed}')
                if args.phase == 'smoke' and batches == 0:
                    gate = smoke_gate(model, batch, device)
                    if gate.get('status') != 'passed':
                        raise ValueError('Model diagnostic gate failed')
                torch.cuda.synchronize()
                tick = time.monotonic()
                outputs = infer_diagnostics(model, batch, device)
                torch.cuda.synchronize()
                forward_seconds += time.monotonic() - tick
                if set(outputs) != set(POLICIES):
                    raise ValueError('Unexpected diagnostic policies')
                for name, arrays in outputs.items():
                    for field, value in identity.items():
                        if not np.array_equal(arrays[field], value):
                            raise ValueError(f'Prediction identity mismatch: {name}/{field}')
                    if buffers[name] is None:
                        buffers[name] = {key: [] for key in arrays}
                    for key, value in arrays.items():
                        buffers[name][key].append(value)
                processed, batches, buffered = stop, batches + 1, buffered + count
                if buffered >= 2048:
                    for name in POLICIES:
                        e.flush_buffer(writers[name], buffers[name])
                    buffered = 0
                if batches == 1 or batches % 100 == 0:
                    print(json.dumps({'phase': args.phase, 'batches': batches, 'samples': processed,
                                      'elapsed_seconds': time.monotonic() - started}), flush=True)
                if batches == 1 or batches % 128 == 0:
                    cpu.append(export_cpu_snapshot(budget['allowed_cpus']))
                if processed == limit:
                    break
            if processed != limit:
                raise ValueError(f'Wrong diagnostic prefix count: {processed}/{limit}')
            for name in POLICIES:
                e.flush_buffer(writers[name], buffers[name])
    finally:
        for writer in writers.values():
            writer.close()
        cpu.append(export_cpu_snapshot(budget['allowed_cpus']))
        if hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
    hardware_end = gpu_info(torch)
    if hardware_end['uuid'] != hardware['uuid']:
        raise RuntimeError('GPU changed within diagnostic allocation')
    write_new(args.output_dir / 'run.json', {
        'status': 'complete', 'phase': args.phase, 'seed': 42, 'processed': processed,
        'expected_samples': expected, 'policies': POLICIES, 'batches': batches,
        'manifest_sha256': m['_manifest_sha256'], 'source_git_commit': m['source_git_commit'],
        'job_id': os.environ.get('SLURM_JOB_ID'), 'hardware': hardware, 'hardware_end': hardware_end,
        'cpu_budget': budget, 'cpu_process_tree_samples': cpu, 'smoke_gate': gate,
        'prior_smoke_acceptance': prior_gate, 'checkpoint_loading': loading,
        'checkpoint': m['checkpoint'], 'provenance': provenance, 'beta': .5, 'runtime_beta': .1,
        'sampling': m['sampling'], 'selection': m['selection'], 'layout_control': m['layout_control'],
        'batch_size': 32, 'amp': 'bf16', 'training_performed': False,
        'group_forward_wall_seconds': forward_seconds,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'elapsed_seconds_before_audit': time.monotonic() - started,
    })
    marker.unlink()  # Remove only this run's own empty completion marker.
    print(json.dumps({'status': 'export_complete', 'phase': args.phase, 'samples': processed}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--phase', choices=['smoke', 'diagnostic'], required=True)
    p.add_argument('--smoke-dir', type=Path)
    run(p.parse_args())
