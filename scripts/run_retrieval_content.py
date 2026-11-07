"""Frozen seed42 full-test retrieval-content comparison, with smoke gating."""
import argparse
import json
import os
from pathlib import Path
import time

from scripts.prepare_retrieval_content import ROOT, POLICIES, load_manifest, sha, write_new


def require_gate(manifest, directory):
    if directory is None:
        raise ValueError('Full export requires accepted smoke')
    path = directory / 'acceptance.json'
    acceptance = json.loads(path.read_text())
    run = json.loads((directory / 'run.json').read_text())
    if (acceptance.get('status') != 'passed' or acceptance.get('seed') != 42 or
            acceptance.get('phase') != 'smoke' or acceptance.get('num_examples') != 65 or
            run.get('phase') != 'smoke' or run.get('seed') != 42 or run.get('processed') != 65 or
            run.get('manifest_sha256') != manifest['_manifest_sha256'] or
            run.get('smoke_gate', {}).get('status') != 'passed'):
        raise ValueError('Wrong smoke acceptance identity')
    from scripts.audit_retrieval_content import audit
    fresh = audit(directory, manifest, 'smoke')
    if ({k: v for k, v in acceptance.items() if k != 'checked_at'} !=
            {k: v for k, v in fresh.items() if k != 'checked_at'}):
        raise ValueError('Smoke acceptance receipt differs from current artifacts')
    paired_path = directory / 'paired.json'
    paired = json.loads(paired_path.read_text())
    from scripts.analyze_retrieval_content import analyze
    fresh_paired = analyze(directory, manifest, 'smoke')
    # The review process can have a different CPU affinity; all scientific
    # fields and artifact bindings must reproduce exactly.
    transient = {'checked_at', 'analysis_cpu_affinity'}
    if ({k: v for k, v in paired.items() if k not in transient} !=
            {k: v for k, v in fresh_paired.items() if k not in transient}):
        raise ValueError('Smoke paired analysis differs from current artifacts')
    return {'path': str(path), 'sha256': sha(path),
            'paired_analysis': {'path': str(paired_path), 'sha256': sha(paired_path)}}, run['smoke_gate']


def check_config(cfg):
    if (cfg.seed != 42 or cfg.sequence_length != 32 or cfg.hist_itemkey_length != 100 or
            cfg.model.top_k_for_generation != 20 or cfg.model.top_k_for_score != 10 or
            cfg.model.active_top_k_for_score is not None or cfg.model.rerank_objective != 'listwise' or
            cfg.model.rerank_logit_weight != .1 or cfg.model.beam_log_prob_weight != 1.):
        raise ValueError('Unexpected trained model configuration')


def run(args):
    started = time.monotonic()
    manifest = load_manifest(args.manifest)
    if args.phase not in ('smoke', 'full'):
        raise ValueError('Unsupported content phase')
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
    from scripts import export_paired_evaluation as export
    from scripts.profile_efficiency_cpu4 import gpu_info, instantiate
    from scripts.retrieval_content_core import infer_content, smoke_gate
    if export.ROOT.resolve() != ROOT:
        raise RuntimeError('Wrong execution worktree imported')
    budget = configure_export_cpu(torch, 1)
    cpu = [export_cpu_snapshot(budget['allowed_cpus'])]
    prior_gate, gate = None, None
    if args.phase == 'full':
        prior_gate, gate = require_gate(manifest, args.smoke_dir)
    elif args.smoke_dir is not None:
        raise ValueError('Smoke cannot reuse a prior gate')
    for field in ('checkpoint', 'reference_test'):
        if sha(manifest[field]['path']) != manifest[field]['sha256']:
            raise ValueError(f'Changed accepted {field}')
    cfg = export.load_run_config(Path(manifest['run_dir']))
    check_config(cfg)
    provenance, expected = export.validate_provenance(
        cfg, 'test', Path(manifest['data_root']), manifest['protocol'], Path(manifest['training_preflight']),
        preflight_source_root=Path(manifest['asset_root']))
    if expected != 2061101:
        raise ValueError('Original test population changed')
    export.configure_dataloader(cfg, 'test', 32, 1)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device('cuda')
    hardware = gpu_info(torch)
    model, loading = instantiate(cfg, {'model': 'listwise', 'checkpoint': manifest['checkpoint']}, device, torch)
    model.requires_grad_(False)
    limit = manifest['smoke_samples'] if args.phase == 'smoke' else manifest['full_samples']
    dm = export.instantiate_datamodule(cfg)
    iterator = iter(export.get_loader(dm, 'test'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    marker = args.output_dir / 'INCOMPLETE'
    marker.touch(exist_ok=False)
    writers = {name: export.H5StreamWriter(args.output_dir / f'{name}.h5') for name in POLICIES}
    buffers = {name: None for name in POLICIES}
    processed = batches = buffered = 0
    forward_seconds = 0.
    torch.cuda.reset_peak_memory_stats()
    try:
        with h5py.File(manifest['reference_test']['path'], 'r') as reference:
            for wrapped in iterator:
                batch = export.unwrap_batch(wrapped)
                size = len(batch.transformed_sequences['sequence_data'])
                if size > limit - processed:
                    batch = export.slice_batch(batch, limit - processed)
                count = len(batch.transformed_sequences['sequence_data'])
                stop = processed + count
                identity = {'user_id': export.extract_user_ids(batch), 'example_hash': export.example_hashes(batch),
                            'target_sid': batch.transformed_sequences['sequence_data'][:, -4:].numpy()}
                for field, value in identity.items():
                    if not np.array_equal(value, reference[field][processed:stop]):
                        raise ValueError(f'Frozen test identity mismatch: {field}, {processed}')
                if args.phase == 'smoke' and batches == 0:
                    gate = smoke_gate(model, batch, device)
                    if gate.get('status') != 'passed':
                        raise ValueError('Model content gate failed')
                torch.cuda.synchronize()
                tick = time.monotonic()
                outputs = infer_content(model, batch, device)
                torch.cuda.synchronize()
                forward_seconds += time.monotonic() - tick
                if list(outputs) != POLICIES:
                    raise ValueError('Unexpected content policies')
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
                        export.flush_buffer(writers[name], buffers[name])
                    buffered = 0
                if batches == 1 or batches % 1000 == 0:
                    print(json.dumps({'phase': args.phase, 'batches': batches, 'samples': processed,
                                      'elapsed_seconds': time.monotonic() - started}), flush=True)
                if batches == 1 or batches % 1000 == 0:
                    cpu.append(export_cpu_snapshot(budget['allowed_cpus']))
                if processed == limit:
                    break
            if processed != limit:
                raise ValueError(f'Wrong export count: {processed}/{limit}')
            for name in POLICIES:
                export.flush_buffer(writers[name], buffers[name])
    finally:
        for writer in writers.values():
            writer.close()
        cpu.append(export_cpu_snapshot(budget['allowed_cpus']))
        if hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
    hardware_end = gpu_info(torch)
    if hardware_end['uuid'] != hardware['uuid']:
        raise RuntimeError('GPU changed within export allocation')
    write_new(args.output_dir / 'run.json', {
        'status': 'complete', 'phase': args.phase, 'seed': 42, 'processed': processed,
        'expected_samples': expected, 'policies': POLICIES, 'batches': batches,
        'manifest_sha256': manifest['_manifest_sha256'], 'source_git_commit': manifest['source_git_commit'],
        'job_id': os.environ.get('SLURM_JOB_ID'), 'hardware': hardware, 'hardware_end': hardware_end,
        'cpu_budget': budget, 'cpu_process_tree_samples': cpu, 'smoke_gate': gate,
        'prior_smoke_acceptance': prior_gate, 'checkpoint_loading': loading,
        'checkpoint': manifest['checkpoint'], 'provenance': provenance, 'beta': .5, 'runtime_beta': .1,
        'sampling': manifest['sampling'], 'selection': manifest['selection'], 'layout_control': manifest['layout_control'],
        'batch_size': 32, 'amp': 'bf16', 'training_performed': False,
        'group_forward_wall_seconds': forward_seconds,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'elapsed_seconds_before_audit': time.monotonic() - started,
    })
    marker.unlink()  # Only this run's own empty completion marker.
    print(json.dumps({'status': 'export_complete', 'phase': args.phase, 'samples': processed}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'full'], required=True)
    parser.add_argument('--smoke-dir', type=Path)
    run(parser.parse_args())
