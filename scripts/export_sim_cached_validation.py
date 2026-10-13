"""Full seed42 validation with the accepted SIM cached-candidate scorer.

The 64-row smoke stays immutable. This entry never trains, generates beam
candidates, selects weights, or accesses test data.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import h5py
import numpy as np

from scripts import export_paired_evaluation as e
from scripts import smoke_sim_cached_rerank as cached
from scripts.accept_sim_validation import accept, check_candidates
from src.runtime_export_cpu import configure_export_cpu, export_cpu_snapshot

SMOKE = cached.REPORT / 'cached_smoke_v1.json'
SMOKE_SHA256 = '40ab3a813edc226e19cc5b33dc241092baf1ed2377904a00b7a3f223ffdf8c2b'
N = 347416


def validate_smoke():
    assert e.file_sha256(SMOKE) == SMOKE_SHA256
    a = json.loads(SMOKE.read_text())
    assert a['status'] == 'passed' and a['num_samples'] == 64
    assert a['cached_beam_exact'] and a['teacher_regenerated'] is False
    assert a['runtime_fusion_recomputed_exact']
    path = Path(a['output'])
    assert e.file_sha256(path) == a['h5_sha256']
    assert e.file_sha256(path.with_suffix('.json')) == a['sidecar_sha256']
    assert e.file_sha256(cached.CHECKPOINT_ACCEPTANCE) == a['checkpoint_acceptance_sha256']
    assert e.file_sha256(cached.REFERENCE_SELECTION) == a['reference_selection_sha256']
    m = json.loads(path.with_suffix('.json').read_text())
    for entry in m['source_files']:
        assert cached.record(entry['path']) == entry, entry['path']
    return a, m


def export_full(output):
    assert not output.exists() and not output.with_suffix('.json').exists()
    cpu = configure_export_cpu(torch, 1)
    assert cpu is not None
    smoke, previous = validate_smoke()
    checkpoint = json.loads(cached.CHECKPOINT_ACCEPTANCE.read_text())
    run, ckpt = Path(checkpoint['run_dir']), Path(checkpoint['checkpoint']['path'])
    assert e.file_sha256(ckpt) == checkpoint['checkpoint']['sha256'] == smoke['checkpoint_sha256']
    assert e.file_sha256(run / '.hydra/config.yaml') == checkpoint['configuration']['sha256']
    cfg = e.load_run_config(run)
    assert cfg.seed == 42 and str(cfg.model._target_).endswith('.SIMRerankModule')
    provenance, count = e.validate_provenance(
        cfg, 'val', e.ROOT / 'data/taobao_history_last_user_v1', 'history-last-user-v1',
        Path(checkpoint['training_preflight']['path']))
    assert count == N
    source = previous['candidate_source']
    ref_path = Path(source['h5'])
    assert e.file_sha256(ref_path) == source['h5_sha256']
    assert e.file_sha256(ref_path.with_suffix('.json')) == source['sidecar_sha256']
    e.configure_dataloader(cfg, 'val', 32, 1)
    dm = e.instantiate_datamodule(cfg)
    loader = e.get_loader(dm, 'val')
    device = torch.device('cuda')
    model = e.instantiate_model(cfg, ckpt, device)
    assert not model.training and model.beam_log_prob_weight == 1. and model.rerank_logit_weight == .1
    temporary = output.with_name(output.name + '.incomplete.' + os.environ['SLURM_JOB_ID'])
    writer = e.H5StreamWriter(temporary)
    arrays, samples, processed, buffered = {}, [], 0, 0
    try:
        with h5py.File(ref_path, 'r', rdcc_nbytes=16 * 1024**2) as handle, \
                patch.object(model.teacher, 'generate_candidates', side_effect=AssertionError('Regenerated beam')) as guard:
            cache = cached.CandidateCache(handle)
            for index, wrapped in enumerate(loader):
                batch = e.unwrap_batch(wrapped)
                ids, scores = cache.aligned(e.extract_user_ids(batch), e.example_hashes(batch),
                                           batch.transformed_sequences['sequence_data'][:, -4:].numpy())
                result = cached.score_cached_batch(model, batch, ids, scores, device)
                current = e.extract_batch_arrays(model, batch, result)
                check_candidates(current)
                for key, value in current.items():
                    arrays.setdefault(key, []).append(value)
                processed += len(ids)
                buffered += len(ids)
                if buffered >= 8192:
                    e.flush_buffer(writer, arrays)
                    buffered = 0
                if index == 0:
                    samples.append(export_cpu_snapshot(cpu['allowed_cpus']))
                if (index + 1) % 1000 == 0:
                    print(f'[sim-cached-validation] batches={index + 1} samples={processed}/{N}', flush=True)
            assert guard.call_count == 0
        e.flush_buffer(writer, arrays)
        assert processed == writer.num_samples == N
        samples.append(export_cpu_snapshot(cpu['allowed_cpus']))
    finally:
        writer.close()
    temporary.replace(output)
    sources = previous['source_files'] + [cached.record(p) for p in (
        Path(__file__), e.ROOT / 'run_sim_cached_validation_cpu4.sh',
        e.ROOT / 'src/data/loading/components/iterators.py')]
    metadata = {**previous, 'limits': dict(max_batches=None, max_samples=None),
                'dataset_provenance': provenance, 'cpu_budget': cpu, 'cpu_process_tree_samples': samples,
                'git_commit': os.environ['SOURCE_GIT_COMMIT'], 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                'source_files': sources, 'exporter_sha256': e.file_sha256(Path(__file__)),
                'smoke_acceptance': cached.record(SMOKE), 'no_candidate_regeneration': True,
                'flush_samples': 8192, 'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
                'timing_scope': 'cached reranking; not end-to-end inference'}
    metadata.pop('inference_checks')  # Checked in the bound smoke, not re-run here.
    e.write_metadata(output, metadata, processed)


def verify_runtime_fusion(arrays):
    raw = torch.from_numpy(arrays['raw_rerank_logit']).to(torch.bfloat16)
    np.testing.assert_array_equal(raw.float().numpy(), arrays['raw_rerank_logit'])
    combined = arrays['beam_score'] + (raw * .1).float().numpy()
    matches = (arrays['rerank_sid'][:, :, None, :] == arrays['beam_sid'][:, None, :, :]).all(-1)
    assert (matches.sum(1) == 1).all() and (matches.sum(2) == 1).all()
    expected = np.take_along_axis(combined, matches.argmax(-1), axis=1)
    np.testing.assert_array_equal(expected, arrays['rerank_score'])


def accept_full(output, receipt):
    assert len(os.sched_getaffinity(0)) == 1 and not receipt.exists()
    smoke, previous = validate_smoke()
    m = json.loads(output.with_suffix('.json').read_text())
    assert m['candidate_source'] == previous['candidate_source']
    assert m['smoke_acceptance'] == cached.record(SMOKE) and m['no_candidate_regeneration']
    result = accept(output, cached.CHECKPOINT_ACCEPTANCE, cached.REFERENCE_SELECTION, smoke=False)
    with h5py.File(output, 'r') as f, h5py.File(smoke['output'], 'r') as short:
        for key in short.keys():
            np.testing.assert_array_equal(f[key][:64], short[key][:], err_msg='Accepted prefix ' + key)
        for start in range(0, N, 8192):
            verify_runtime_fusion({k: f[k][start:start + 8192] for k in (
                'raw_rerank_logit', 'beam_score', 'beam_sid', 'rerank_sid', 'rerank_score')})
    result.update(scope='full cached-candidate SIM validation; not test', candidate_source=m['candidate_source'],
                  cached_beam_exact=True, teacher_regenerated=False, runtime_fusion_recomputed_exact=True,
                  accepted_smoke_prefix_exact=True, smoke_acceptance=cached.record(SMOKE),
                  cached_audit_script_sha256=e.file_sha256(Path(__file__)),
                  checked_at=datetime.now().astimezone().isoformat())
    with receipt.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('[sim-cached-validation-acceptance] ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--acceptance', type=Path)
    args = p.parse_args()
    if args.acceptance:
        accept_full(args.output, args.acceptance)
    else:
        export_full(args.output)
