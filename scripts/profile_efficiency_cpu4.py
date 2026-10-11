"""Fixed-checkpoint, same-GPU efficiency suite. No training or metric selection.

The parent uses only stdlib; each model/repetition runs in a fresh child process.
CUDA timing excludes loading/H2D/validation. Peak memory holds one input batch,
not the entire fixture. Separate's unused inherited wrapper stays on the CPU.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from src.runtime_cpu import THREAD_VARIABLES, configure_torch
from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {'sid': ['beam_only'], 'listwise': ['beam_only', 'rerank_cache', 'rerank_cache_fp32', 'rerank_no_cache_fp32'],
            'pointwise': ['beam_only', 'rerank_cache', 'rerank_cache_fp32', 'rerank_no_cache_fp32'],
            'separate': ['beam_only', 'separate_rerank']}
COUNTS = {'smoke': (2, 3, 1), 'formal': (10, 100, 3)}


def budget_guard():
    if os.environ.get('GRID_PROFILE_CPU_BUDGET') != '4':
        raise RuntimeError('GRID_PROFILE_CPU_BUDGET=4 is required')
    if any(os.environ.get(k) != '1' for k in THREAD_VARIABLES):
        raise RuntimeError('All library thread limits must be 1 before Python starts')
    return check_export_allocation()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(record):
    if sha(record['path']) != record['sha256']:
        raise ValueError(f"Fingerprint changed: {record['path']}")


def write_new(path, payload):
    with Path(path).open('x') as f:
        f.write(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def summary(rows):
    if not rows:
        raise ValueError('Empty measurement')
    fields = ('gen_cuda_ms', 'rerank_cuda_ms', 'total_cuda_ms', 'wall_ms',
              'baseline_allocated_gib', 'peak_allocated_gib', 'peak_increment_gib', 'peak_reserved_gib')
    result = {}
    for key in fields:
        values = [r[key] for r in rows]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError(f'Invalid measurement: {key}')
        result[key] = {'mean': statistics.mean(values), 'p50': statistics.median(values),
                       'p95': sorted(values)[math.ceil(.95 * len(values)) - 1], 'max': max(values)}
    if result['total_cuda_ms']['mean'] <= 0 or result['wall_ms']['mean'] <= 0:
        raise ValueError('Nonpositive forward duration')
    result['users_per_second_wall'] = sum(r['batch_size'] for r in rows) / (sum(r['wall_ms'] for r in rows) / 1000)
    result['batches'] = len(rows)
    return result


def load_manifest(path):
    if sha(path) != os.environ['PROFILE_MANIFEST_SHA256']:
        raise ValueError('Manifest binding mismatch')
    m = json.loads(path.read_text())
    if m['source_git_commit'] != os.environ['SOURCE_GIT_COMMIT'] or m['dirty_at_submission']:
        raise ValueError('Source commit binding mismatch')
    assert set(m['models']) == set(VARIANTS) and m['seed'] == 44
    for entry in m['execution_sources']:
        verify(entry)
    return m


def gpu_info(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one allocated CUDA device is required')
    prop = torch.cuda.get_device_properties(0)
    uuid = str(prop.uuid)
    if not uuid.startswith('GPU-'):
        uuid = 'GPU-' + uuid
    query = subprocess.check_output([
        'nvidia-smi', '-i', uuid,
        '--query-gpu=uuid,name,driver_version,pci.bus_id,memory.total,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    foreign = [int(line) for line in apps.splitlines() if line.strip().isdigit() and int(line) != os.getpid()]
    if foreign:
        raise RuntimeError('Other compute processes occupy the allocated GPU')
    if '4090' not in prop.name:
        raise RuntimeError(f'Expected RTX 4090, found {prop.name}')
    return {'uuid': uuid, 'name': prop.name, 'total_memory_bytes': prop.total_memory,
            'hostname': os.uname().nodename, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'nvidia_smi': query,
            'other_compute_processes_on_device': 0}


def instantiate(cfg, entry, device, torch):
    import hydra
    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(entry['checkpoint']['path'], map_location='cpu', mmap=True, weights_only=False)
    missing, extra = model.load_state_dict(state['state_dict'], strict=False)
    assert not missing, missing
    if entry['model'] == 'separate':
        assert len(extra) == 62 and all(k.startswith(('teacher.', 'teacher_model.')) for k in extra)
    else:
        assert not extra, extra
    del state
    gc.collect()
    if entry['model'] == 'separate':
        model.setup(stage='validate')
        # Only the actual teacher + reranker pipeline is resident on the GPU.
        # This does not prune/rewrite weights or modify the saved checkpoint.
        model.teacher.to(device)
        model.reranker.to(device)
    else:
        model.to(device)
    model.eval()
    return model, {'missing': len(missing), 'unexpected': len(extra)}


def fixture(cfg, entry, count, budget):
    import h5py
    import numpy as np
    from scripts import export_paired_evaluation as e
    e.configure_dataloader(cfg, 'test', 32, 1)
    dm = e.instantiate_datamodule(cfg)
    loader = e.get_loader(dm, 'test')
    iterator = iter(loader)
    batches, digest = [], hashlib.sha256()
    try:
        with h5py.File(entry['reference_h5'], 'r') as reference:
            for index in range(count):
                batch = e.unwrap_batch(next(iterator))
                users, fingerprints = e.extract_user_ids(batch), e.example_hashes(batch)
                assert len(users) == 32
                start = index * 32
                assert np.array_equal(reference['user_id'][start:start+32], users)
                assert np.array_equal(reference['example_hash'][start:start+32], fingerprints)
                digest.update(users.tobytes())
                digest.update(fingerprints.tobytes())
                batches.append(batch)
        cpu = export_cpu_snapshot(budget['allowed_cpus'])
    finally:
        if hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
    del iterator, loader, dm
    gc.collect()
    return batches, digest.hexdigest(), cpu


def equivalence(model, batch, device, torch, p, e):
    """Same fixed natural candidates; FP32 semantic gate plus BF16 error report."""
    results = {}
    with torch.inference_mode():
        for amp, atol, rtol in [('none', 2e-4, 2e-4), ('bf16', .1, .01)]:
            ids, scores, mask, cache = p.run_beam(model, batch, device, amp)
            a = p.rerank_with_cache(model, batch, ids, scores, mask, cache, device, amp)
            b = p.rerank_without_cache(model, batch, ids, scores, device, amp)
            beam = ids.reshape(32, 20, 4)
            av = e.align_rerank_logits_to_beam(beam, a[0], a[2])
            bv = e.align_rerank_logits_to_beam(beam, b[0], b[2])
            if amp == 'none':
                torch.testing.assert_close(av, bv, atol=atol, rtol=rtol)
            results[amp] = {'atol': atol, 'rtol': rtol,
                            'within_tolerance': bool(torch.isclose(av, bv, atol=atol, rtol=rtol).all()),
                            'outside_tolerance_count': int((~torch.isclose(av, bv, atol=atol, rtol=rtol)).sum()),
                            'num_candidate_logits': av.numel(),
                            'scope': 'strict FP32 gate' if amp == 'none' else 'diagnostic only; BF16 no-cache excluded from timing comparison',
                            'max_abs_logit_error': float((av-bv).abs().max()),
                            'ranking_row_exact_fraction': float((a[0] == b[0]).all(-1).all(-1).float().mean())}
    return results


def measure_one(model, batch, variant, torch, p, device):
    amp = 'none' if variant.endswith('_fp32') else 'bf16'
    path_variant = variant.removesuffix('_fp32')
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    events[0].record()
    if variant == 'separate_rerank' or model.__class__.__name__ == 'SeparateRerankModule':
        beam = p.run_separate_beam(model, batch, device, amp)
        events[1].record()
        out = p.run_separate_rerank(model, beam, batch, device, amp) if variant == 'separate_rerank' else None
        candidate_ids, candidate_scores = beam['candidate_sid'], beam['beam_scores']
    else:
        ids, scores, mask, cache = p.run_beam(model, batch, device, amp)
        events[1].record()
        if path_variant == 'rerank_cache':
            out = p.rerank_with_cache(model, batch, ids, scores, mask, cache, device, amp)
        elif path_variant == 'rerank_no_cache':
            del cache  # no artificial retention of the discarded generation KV cache
            out = p.rerank_without_cache(model, batch, ids, scores, device, amp)
            # Match the two lightweight accumulator updates in the cached implementation.
            model.click_log_prob_accumulator(torch.log_softmax(out[1], dim=-1).mean())
            model.beam_log_prob_accumulator(scores.mean())
        else:
            assert variant == 'beam_only'
            out = None
        candidate_ids, candidate_scores = ids.reshape(32, 20, 4), scores
    events[2].record()
    events[2].synchronize()
    wall_ms = (time.perf_counter() - start) * 1000
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    # Validation / copies happen after timing and after sampling memory peaks.
    assert torch.isfinite(candidate_scores).all()
    if out is not None:
        assert torch.isfinite(out[1]).all() and torch.isfinite(out[2]).all()
    digest = hashlib.sha256(candidate_ids.cpu().numpy().tobytes())
    digest.update(candidate_scores.float().cpu().numpy().tobytes())
    return {'batch_size': 32, 'amp': amp, 'gen_cuda_ms': events[0].elapsed_time(events[1]),
            'rerank_cuda_ms': 0.0 if variant == 'beam_only' else events[1].elapsed_time(events[2]),
            'total_cuda_ms': events[0].elapsed_time(events[2]), 'wall_ms': wall_ms,
            'baseline_allocated_gib': baseline / 1024**3, 'peak_allocated_gib': peak / 1024**3,
            'peak_increment_gib': (peak-baseline) / 1024**3, 'peak_reserved_gib': reserved / 1024**3,
            'natural_beam_sha256': digest.hexdigest()}


def child(args, manifest, budget):
    import torch
    configure_torch(torch)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2027)
    from scripts import export_paired_evaluation as e, profile_rerank_efficiency as p
    entry = manifest['models'][args.model]
    for key in ('checkpoint', 'configuration', 'training_preflight', 'accepted_artifact', 'sidecar'):
        verify(entry[key])
    accepted = json.loads(Path(entry['accepted_artifact']['path']).read_text())
    assert accepted['status'] == 'passed' and accepted['checkpoint_sha256'] == entry['checkpoint']['sha256']
    cfg = e.load_run_config(Path(entry['run_dir']))
    provenance, n = e.validate_provenance(cfg, 'test', Path(manifest['data_root']),
                                         'history-last-user-v1', Path(entry['training_preflight']['path']))
    assert n == 2061101 and cfg.seed == 44
    assert cfg.sequence_length == 32 and cfg.hist_itemkey_length == 100
    assert cfg.model.top_k_for_generation == 20 and cfg.model.top_k_for_score == 10
    assert cfg.model.active_top_k_for_score is None
    assert accepted['output'] == entry['reference_h5'] and accepted['sidecar_sha256'] == entry['sidecar']['sha256']
    if args.model != 'sid':
        verify(entry['selection'])
        selected = json.loads(Path(entry['selection']['path']).read_text())
        assert selected['selection_scope'] == 'validation_only' and selected['input_checkpoint_sha256'] == entry['checkpoint']['sha256']
        assert selected['selected']['rerank_weight'] == .5
        cfg.model.rerank_logit_weight = .5
    device = torch.device('cuda:0')
    hardware_start = gpu_info(torch)
    model, loading = instantiate(cfg, entry, device, torch)
    warmup, measured, _ = COUNTS[args.phase]
    batches, input_digest, first_cpu = fixture(cfg, entry, warmup+measured, budget)
    checks = {}
    if args.phase == 'smoke' and args.model in ('listwise', 'pointwise'):
        batch = p.split_batch(model, e.move_batch_to_device(batches[0], device))
        checks = equivalence(model, batch, device, torch, p, e)
        del batch
    if args.phase == 'smoke' and args.model == 'separate':
        batch = p.split_batch(model, e.move_batch_to_device(batches[0], device))
        with torch.inference_mode(), p.autocast_context(device, 'bf16'):
            ref = model._run_rerank(batch['hist_sid'], batch['hist_mask'], batch['hist_itemkey'], ensure_gt_in_candidates=False)
            got = p.run_separate_rerank(model, p.run_separate_beam(model, batch, device, 'bf16'), batch, device, 'bf16')
            assert torch.equal(ref['rerank_ids'], got[0])
            torch.testing.assert_close(ref['rerank_scores'], got[1], atol=0, rtol=0)
        checks = {'active_pipeline_matches_model_rerank': True}
        del batch, ref, got
    variants = VARIANTS[args.model]
    offset = args.repeat % len(variants)
    variants = variants[offset:] + variants[:offset]
    reports = {}
    with torch.inference_mode():
        for variant in variants:
            gc.collect()
            torch.cuda.empty_cache()
            rows = []
            for index, cpu_batch in enumerate(batches):
                gpu_batch = e.move_batch_to_device(cpu_batch, device)
                batch = p.split_batch(model, gpu_batch)
                row = measure_one(model, batch, variant, torch, p, device)
                del batch, gpu_batch
                if index >= warmup:
                    rows.append(dict(batch_index=index-warmup, **row))
            reports[variant] = {'records': rows, 'summary': summary(rows)}
            print(f"[profile] {args.model} repeat={args.repeat} {variant}: {reports[variant]['summary']['wall_ms']['mean']:.3f} ms/batch", flush=True)
    result = {'status': 'passed', 'model': args.model, 'repeat': args.repeat, 'phase': args.phase,
              'warmup_batches': warmup, 'measure_batches': measured, 'input_sha256': input_digest,
              'hardware_start': hardware_start, 'hardware_end': gpu_info(torch), 'checkpoint_loading': loading,
              'parameters': p.summarize_parameters(model, 'separate_rerank' if args.model == 'separate' else 'recochain'),
              'gpu_resident_parameter_count': sum(x.numel() for x in model.parameters() if x.device.type == 'cuda'),
              'equivalence_checks': checks, 'variants': reports, 'variant_execution_order': variants,
              'cpu_budget': budget, 'cpu_process_tree_samples': [first_cpu, export_cpu_snapshot(budget['allowed_cpus'])],
              'inference_settings': {'batch_size': 32, 'beam': 20, 'amp': 'BF16 main comparison; FP32 cache ablation', 'tf32': False,
                                     'sequence_length': 32, 'hist_itemkey_length': 100,
                                     'top_k_for_score': 10, 'active_top_k_for_score': None,
                                     'fusion_beta': .5 if args.model != 'sid' else None},
              'dataset_provenance': provenance, 'manifest_sha256': sha(args.manifest)}
    write_new(args.output / f'{args.model}_repeat{args.repeat}.json', result)


def validate_cases(cases, phase):
    warm, measured, repeats = COUNTS[phase]
    assert len(cases) == 4 * repeats
    assert {(c['model'], c['repeat']) for c in cases} == {(m, r) for m in VARIANTS for r in range(repeats)}
    assert len({c['input_sha256'] for c in cases}) == 1
    assert len({c[k]['uuid'] for c in cases for k in ('hardware_start', 'hardware_end')}) == 1
    assert len({(c[k]['hostname'], c[k]['name'], c[k]['torch'], c[k]['cuda']) for c in cases for k in ('hardware_start', 'hardware_end')}) == 1
    beams = {}
    for case in cases:
        assert case['status'] == 'passed' and case['phase'] == phase
        assert case['warmup_batches'] == warm and case['measure_batches'] == measured
        assert set(case['variants']) == set(VARIANTS[case['model']])
        for variant, result in case['variants'].items():
            rows = result['records']
            assert len(rows) == measured and [r['batch_index'] for r in rows] == list(range(measured))
            assert all(r['batch_size'] == 32 for r in rows)
            amp = 'none' if variant.endswith('_fp32') else 'bf16'
            assert all(r['amp'] == amp for r in rows)
            assert result['summary'] == summary(rows)
            fingerprints = [r['natural_beam_sha256'] for r in rows]
            identity = ('teacher' if case['model'] in ('sid', 'separate') else case['model'], amp)
            assert beams.setdefault(identity, fingerprints) == fingerprints, 'Natural candidates/scores changed'
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=COUNTS, required=True)
    parser.add_argument('--smoke-report', type=Path)
    parser.add_argument('--model', choices=VARIANTS)
    parser.add_argument('--repeat', type=int, default=0)
    args = parser.parse_args()
    budget = budget_guard()
    manifest = load_manifest(args.manifest)
    if args.model:
        child(args, manifest, budget)
        return
    if args.phase == 'formal':
        if args.smoke_report is None:
            raise ValueError('Formal profiling requires an accepted smoke report')
        smoke = json.loads(args.smoke_report.read_text())
        assert smoke['status'] == 'passed' and smoke['phase'] == 'smoke'
        assert smoke['manifest_sha256'] == sha(args.manifest)
        smoke_cases = []
        for record in smoke['case_files']:
            verify(record)
            smoke_cases.append(json.loads(Path(record['path']).read_text()))
        validate_cases(smoke_cases, 'smoke')
        assert all(c['equivalence_checks'] for c in smoke_cases if c['model'] != 'sid')
        assert all(c['equivalence_checks']['none']['within_tolerance'] for c in smoke_cases if c['model'] in ('listwise', 'pointwise'))
    args.output.mkdir(parents=True, exist_ok=False)
    cases, records = [], []
    _, _, repeats = COUNTS[args.phase]
    models = list(VARIANTS)
    for repeat in range(repeats):
        order = models[repeat % 4:] + models[:repeat % 4]
        for model in order:
            subprocess.run([sys.executable, '-m', 'scripts.profile_efficiency_cpu4', '--manifest', str(args.manifest),
                            '--output', str(args.output), '--phase', args.phase, '--model', model, '--repeat', str(repeat)], check=True)
            path = args.output / f'{model}_repeat{repeat}.json'
            cases.append(json.loads(path.read_text()))
            records.append({'path': str(path.resolve()), 'sha256': sha(path)})
    validate_cases(cases, args.phase)
    load_manifest(args.manifest)  # execution sources must still match at completion
    grouped = {}
    for model, variants in VARIANTS.items():
        for variant in variants:
            selected = [c['variants'][variant] for c in cases if c['model'] == model]
            means = [c['summary']['wall_ms']['mean'] for c in selected]
            grouped[f'{model}:{variant}'] = {
                'pooled_batches': summary([r for c in selected for r in c['records']]),
                'repetition_wall_mean_ms': means,
                'repetition_wall_mean_std_ms': statistics.stdev(means) if repeats > 1 else None}
    result = {'status': 'passed', 'phase': args.phase, 'seed': 44, 'repetitions': repeats,
              'manifest_sha256': sha(args.manifest), 'source_git_commit': manifest['source_git_commit'],
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'case_files': records,
              'hardware': cases[0]['hardware_start'], 'input_sha256': cases[0]['input_sha256'],
              'cpu_budget': budget, 'summary': grouped,
              'smoke_report': {'path': str(args.smoke_report), 'sha256': sha(args.smoke_report)} if args.smoke_report else None,
              'timing_scope': 'One preloaded batch; model forward only. CUDA events span generation and reranking without intermediate synchronization; synchronized wall time includes Python dispatch. Excludes data/H2D/loading/hash checks.',
              'memory_scope': 'One input batch plus GPU-resident active pipeline; separate inherited inactive wrapper stays on CPU. Allocated and reserved peaks reported separately; no artificial retention of discarded KV cache.',
              'limitations': ['Fixed first test batches, batch32/beam20/BF16, not online single-request latency',
                              'One training seed; three timing repetitions are not training-seed replicates',
                              'Same allocated physical GPU throughout suite; node may host other GPUs/jobs; clocks not locked',
                              'Main model comparison is BF16; strict cache/no-cache timing comparison is separately FP32. No cross-precision speedup ratios.',
                              'BF16 cache/no-cache differences retained as diagnostic only; BF16 no-cache is not timed. FP32 tolerance gate is not bitwise equivalence.',
                              'Frozen validation beta=0.50; model topk implementation timed, not offline full-test stable-sort analysis']}
    write_new(args.output / 'acceptance.json', result)
    print('[passed] efficiency suite', args.output, flush=True)


if __name__ == '__main__':
    main()
