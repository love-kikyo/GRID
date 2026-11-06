"""Stream one authorized seed42 sensitivity group, then independently audit it."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

from scripts.prepare_inference_sensitivity import GROUPS, sha, write_new


def load_bound_manifest(path):
    if sha(path) != os.environ.get('SENSITIVITY_MANIFEST_SHA256'):
        raise ValueError('Manifest hash binding mismatch')
    m = json.loads(path.read_text())
    if (m['seed'] != 42 or m['groups'] != GROUPS or m['beta'] != .5 or
            m['beam_weight'] != 1. or m['runtime_beta'] != .1 or
            m['batch_size'] != 32 or m['num_workers'] != 1 or m['amp'] != 'bf16' or
            m['sequence_length'] != 32 or m['hist_itemkey_length'] != 100 or
            m['expected_test_samples'] != 2061101 or m['smoke_samples'] != 65 or
            m['protocol'] != 'history-last-user-v1' or m['dirty_at_submission']):
        raise ValueError('Manifest exceeds authorized scope')
    if m['source_git_commit'] != os.environ.get('SOURCE_GIT_COMMIT'):
        raise ValueError('Execution commit binding mismatch')
    for entry in m['execution_sources'] + m['input_metadata']:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError(f"Source/input changed: {entry['path']}")
    m['_manifest_sha256'] = sha(path)
    return m


def require_gate(m, group, smoke_dir):
    if smoke_dir is None:
        raise ValueError('Full inference requires an accepted smoke directory')
    acceptance = smoke_dir / 'acceptance.json'
    report = json.loads(acceptance.read_text())
    run = json.loads((smoke_dir / 'run.json').read_text())
    if (report['status'] != 'passed' or run['phase'] != 'smoke' or
            run['group'] != group or run['processed'] != 65 or
            run['manifest_sha256'] != m['_manifest_sha256'] or
            run['smoke_gate']['status'] != 'passed'):
        raise ValueError('Smoke acceptance identity mismatch')
    # Re-audit the small saved fixture, not merely a status flag.
    from scripts.audit_inference_sensitivity import audit_group
    audit_group(smoke_dir, m, 'smoke', group)
    return {'path': str(acceptance), 'sha256': sha(acceptance)}, run['smoke_gate']


def run(args):
    started = time.monotonic()
    m = load_bound_manifest(args.manifest)
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing existing output directory: {args.output_dir}')
    from src.runtime_cpu import THREAD_VARIABLES
    if (os.environ.get('GRID_EXPORT_CPU_BUDGET') != '4' or
            any(os.environ.get(key) != '1' for key in THREAD_VARIABLES)):
        raise RuntimeError('Four-core budget and single-thread libraries required before imports')
    import h5py
    import numpy as np
    import torch
    from src.runtime_export_cpu import configure_export_cpu, export_cpu_snapshot
    from scripts import export_paired_evaluation as e
    from scripts.profile_efficiency_cpu4 import gpu_info, instantiate
    from scripts.sensitivity_core import infer_settings, smoke_gate
    budget = configure_export_cpu(torch, 1)
    cpu = [export_cpu_snapshot(budget['allowed_cpus'])]
    gate_record, gate = None, None
    if args.phase == 'full':
        gate_record, gate = require_gate(m, args.group, args.smoke_dir)
    elif args.smoke_dir is not None:
        raise ValueError('Smoke cannot consume a prior gate')
    for key in ['checkpoint', 'reference_test']:
        if sha(m[key]['path']) != m[key]['sha256']:
            raise ValueError(f'Accepted {key} bytes changed')
    cfg = e.load_run_config(Path(m['run_dir']))
    if (cfg.seed != 42 or cfg.sequence_length != 32 or cfg.hist_itemkey_length != 100 or
            cfg.model.top_k_for_generation != 20 or cfg.model.top_k_for_score != 10 or
            cfg.model.active_top_k_for_score is not None or
            cfg.model.rerank_objective != 'listwise' or
            cfg.model.rerank_logit_weight != .1 or cfg.model.beam_log_prob_weight != 1.):
        raise ValueError('Unexpected original run configuration')
    provenance, expected = e.validate_provenance(
        cfg, 'test', Path(m['data_root']), m['protocol'], Path(m['training_preflight']))
    if expected != m['expected_test_samples']:
        raise ValueError('Test count changed')
    e.configure_dataloader(cfg, 'test', 32, 1)
    device = torch.device('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    hardware = gpu_info(torch)
    model, loading = instantiate(cfg, {'model': 'listwise', 'checkpoint': m['checkpoint']}, device, torch)
    model.requires_grad_(False)
    limit = 65 if args.phase == 'smoke' else expected
    dm = e.instantiate_datamodule(cfg)
    iterator = iter(e.get_loader(dm, 'test'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    marker = args.output_dir / 'INCOMPLETE'
    marker.touch(exist_ok=False)
    names = [f'c{c}_m{length}' for c, length in GROUPS[args.group]]
    writers = {name: e.H5StreamWriter(args.output_dir / f'{name}.h5') for name in names}
    buffers = {name: None for name in names}
    processed, buffered, batches = 0, 0, 0
    timings = []
    torch.cuda.reset_peak_memory_stats()
    try:
        with h5py.File(m['reference_test']['path'], 'r') as reference:
            for wrapped in iterator:
                batch = e.unwrap_batch(wrapped)
                size = batch.transformed_sequences['sequence_data'].shape[0]
                if size > limit - processed:
                    batch = e.slice_batch(batch, limit - processed)
                count = batch.transformed_sequences['sequence_data'].shape[0]
                stop = processed + count
                # The entire input/GT is frozen, not only the user set.
                identity = {'user_id': e.extract_user_ids(batch), 'example_hash': e.example_hashes(batch),
                            'target_sid': batch.transformed_sequences['sequence_data'][:, -4:].numpy()}
                for key, values in identity.items():
                    if not np.array_equal(values, reference[key][processed:stop]):
                        raise ValueError(f'Input/target differs from frozen test: {key} at {processed}')
                if args.phase == 'smoke' and batches == 0:
                    gate = smoke_gate(model, batch, device)
                    if gate.get('status') != 'passed':
                        raise ValueError('Model equivalence gate did not pass')
                torch.cuda.synchronize()
                tick = time.monotonic()
                outputs = infer_settings(model, batch, device, args.group)
                torch.cuda.synchronize()
                timings.append(time.monotonic() - tick)
                if set(outputs) != set(names):
                    raise ValueError('Unexpected setting outputs')
                for name, arrays in outputs.items():
                    for key, values in identity.items():
                        if not np.array_equal(arrays[key], values):
                            raise ValueError(f'Output identity changed: {name}/{key}')
                    if buffers[name] is None:
                        buffers[name] = {key: [] for key in arrays}
                    for key, values in arrays.items():
                        buffers[name][key].append(values)
                processed, batches, buffered = stop, batches + 1, buffered + count
                if buffered >= 8192:
                    for name in names:
                        e.flush_buffer(writers[name], buffers[name])
                    buffered = 0
                if batches == 1 or batches % 1000 == 0:
                    print(json.dumps({'group': args.group, 'batches': batches, 'samples': processed,
                                      'elapsed_seconds': time.monotonic() - started}), flush=True)
                if batches == 1 or batches % 5000 == 0:
                    cpu.append(export_cpu_snapshot(budget['allowed_cpus']))
                if processed == limit:
                    break
            if processed != limit:
                raise ValueError(f'Wrong record count: {processed} != {limit}')
            for name in names:
                e.flush_buffer(writers[name], buffers[name])
    finally:
        for writer in writers.values():
            writer.close()
        cpu.append(export_cpu_snapshot(budget['allowed_cpus']))
        if hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
    hardware_end = gpu_info(torch)
    if hardware_end['uuid'] != hardware['uuid']:
        raise RuntimeError('GPU changed within allocation')
    report = {
        'status': 'complete', 'phase': args.phase, 'group': args.group, 'seed': 42,
        'processed': processed, 'expected_samples': expected, 'batches': batches,
        'manifest_sha256': m['_manifest_sha256'], 'source_git_commit': m['source_git_commit'],
        'job_id': os.environ.get('SLURM_JOB_ID'), 'hardware': hardware, 'hardware_end': hardware_end,
        'cpu_budget': budget, 'cpu_process_tree_samples': cpu, 'smoke_gate': gate,
        'prior_smoke_acceptance': gate_record, 'checkpoint_loading': loading,
        'checkpoint': m['checkpoint'], 'provenance': provenance,
        'group_forward_wall_seconds': sum(timings),
        'per_batch_forward_wall_seconds': timings if args.phase == 'smoke' else None,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'elapsed_seconds_before_audit': time.monotonic() - started,
        'timing_scope': 'Group forward includes transfer/extraction and shared generation; not per-setting latency',
        'candidate_policy': m['candidate_policy'], 'retrieval_policy': m['retrieval_policy'],
        'beta': .5, 'runtime_beta': .1, 'amp': 'bf16', 'batch_size': 32,
        'training_performed': False,
    }
    write_new(args.output_dir / 'run.json', report)
    marker.unlink()  # Only our own empty incomplete marker; predictions are never removed.
    del model, iterator, dm, buffers
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({'status': 'export_complete', 'records': processed, 'group': args.group}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'full'], required=True)
    parser.add_argument('--group', choices=list(GROUPS), required=True)
    parser.add_argument('--smoke-dir', type=Path)
    run(parser.parse_args())
