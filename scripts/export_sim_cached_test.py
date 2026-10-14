"""Seed42 full SIM test on accepted SID-only natural candidates.

Reuse the validated scorer unchanged. Freeze validation selection before opening
test data; never search on test, train, or regenerate teacher candidates.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch  # Before the data module initializes TensorFlow.
import h5py
import numpy as np

from scripts import export_paired_evaluation as e
from scripts import smoke_sim_cached_rerank as cached
from scripts.accept_sim_validation import check_candidates
from scripts.export_sim_cached_validation import verify_runtime_fusion
from src.runtime_export_cpu import configure_export_cpu, export_cpu_snapshot

N = 2061101
ANALYSIS = e.ROOT / 'logs/analysis/paired_evaluation'
FUSION = cached.REPORT / 'fusion_v1.json'
FUSION_SHA = 'c30d11e9a27fcb59eefa8fc4ad7614b04550e705d92cf38fe44f3e22e7bfbe19'
TEST_ACCEPTANCE = ANALYSIS / 'test_acceptance_seed42_full_v1.json'
TEST_ACCEPTANCE_SHA = 'ace4fcaa5112f44f4b01853e00ed218c800249c20af4e6e2de53e221f9f92564'
CACHE = ANALYSIS / 'sid_seed42_131908_test_full_v1.h5'


def check_selection(fusion, selection, acceptance):
    """Fail closed on a different split, checkpoint, input, or selected beta."""
    assert fusion['status'] == acceptance['status'] == 'passed'
    assert fusion['num_samples'] == acceptance['num_samples'] == selection['num_examples'] == 347416
    assert selection['input_split'] == 'val'
    assert fusion['candidates_checked'] == 51 and fusion['metric_values_checked'] == 204
    assert fusion['grid_expanded'] is False
    assert fusion['checkpoint_sha256'] == acceptance['checkpoint_sha256']
    assert fusion['input_sha256'] == acceptance['h5_sha256'] == selection['input_sha256']
    assert selection['input_sidecar_sha256'] == acceptance['sidecar_sha256']
    assert fusion['selected'] == selection['selected']
    assert selection['selected']['beam_weight'] == 1.
    assert selection['selected']['rerank_weight'] == .09
    assert selection['selected']['variant'] == 'fusion@1@0.09'


def validate_frozen_validation():
    assert e.file_sha256(FUSION) == FUSION_SHA
    fusion = json.loads(FUSION.read_text())
    selection_path, accepted_path = Path(fusion['selection']), Path(fusion['validation_acceptance'])
    assert e.file_sha256(selection_path) == fusion['selection_sha256']
    assert e.file_sha256(selection_path.with_suffix('.md')) == fusion['markdown_sha256']
    assert e.file_sha256(accepted_path) == fusion['validation_acceptance_sha256']
    assert e.file_sha256(e.ROOT / 'scripts/accept_sim_fusion.py') == fusion['audit_script_sha256']
    selection, accepted = json.loads(selection_path.read_text()), json.loads(accepted_path.read_text())
    check_selection(fusion, selection, accepted)
    val = Path(accepted['output'])
    assert val == Path(selection['input'])
    assert e.file_sha256(val) == accepted['h5_sha256']
    assert e.file_sha256(val.with_suffix('.json')) == accepted['sidecar_sha256']
    previous = json.loads(val.with_suffix('.json').read_text())
    assert previous['split'] == 'val' and previous['num_samples'] == 347416
    assert accepted['cached_beam_exact'] and accepted['teacher_regenerated'] is False
    assert accepted['runtime_fusion_recomputed_exact'] and accepted['accepted_smoke_prefix_exact']
    assert e.file_sha256(cached.CHECKPOINT_ACCEPTANCE) == accepted['checkpoint_acceptance_sha256']
    for entry in previous['source_files']:
        assert cached.record(entry['path']) == entry, entry['path']
    frozen = dict(selection=cached.record(selection_path), acceptance=cached.record(FUSION),
                  validation_acceptance=cached.record(accepted_path),
                  beam_weight=1., rerank_weight=.09, variant='fusion@1@0.09',
                  score_dtype='float32', tie_break='earlier original beam position',
                  scope='validation-only; no test search')
    return frozen, previous


def validate_test_cache(checkpoint):
    assert e.file_sha256(TEST_ACCEPTANCE) == TEST_ACCEPTANCE_SHA
    accepted = json.loads(TEST_ACCEPTANCE.read_text())
    assert accepted['status'] == 'passed'
    assert accepted['num_samples_per_model'] == accepted['unique_users_per_model'] == N
    assert accepted['cross_model_user_hash_target_equal']
    assert accepted['separate_and_sid_beam_sid_and_scores_exactly_equal']
    source = accepted['models']['sid']
    assert e.file_sha256(CACHE) == source['h5_sha256']
    assert e.file_sha256(CACHE.with_suffix('.json')) == source['sidecar_sha256']
    meta = json.loads(CACHE.with_suffix('.json').read_text())
    assert meta['split'] == 'test' and meta['num_samples'] == N and meta['amp'] == 'bf16'
    assert meta['checkpoint_sha256'] == source['checkpoint_sha256'] == checkpoint['teacher_checkpoint']['sha256']
    assert meta['dataset_provenance']['protocol'] == 'history-last-user-v1'
    for entry in meta['dataset_provenance']['manifests'].values():
        assert cached.record(entry['path']) == entry
    return dict(kind='cached_frozen_teacher_natural_beam', h5=str(CACHE),
                h5_sha256=source['h5_sha256'], sidecar_sha256=source['sidecar_sha256'],
                reference_acceptance=cached.record(TEST_ACCEPTANCE),
                separate_beam_exact_in_reference=True)


def export_full(output):
    assert not output.exists() and not output.with_suffix('.json').exists()
    cpu = configure_export_cpu(torch, 1)
    assert cpu is not None
    frozen, previous = validate_frozen_validation()
    print('[sim-cached-test] validation acceptance and beta=0.09 locked', flush=True)
    checkpoint = json.loads(cached.CHECKPOINT_ACCEPTANCE.read_text())
    run, ckpt = Path(checkpoint['run_dir']), Path(checkpoint['checkpoint']['path'])
    assert e.file_sha256(ckpt) == checkpoint['checkpoint']['sha256'] == previous['checkpoint_sha256']
    assert e.file_sha256(run / '.hydra/config.yaml') == checkpoint['configuration']['sha256']
    cfg = e.load_run_config(run)
    assert cfg.seed == 42 and str(cfg.model._target_).endswith('.SIMRerankModule')
    provenance, count = e.validate_provenance(
        cfg, 'test', e.ROOT / 'data/taobao_history_last_user_v1', 'history-last-user-v1',
        Path(checkpoint['training_preflight']['path']))
    assert count == N
    source = validate_test_cache(checkpoint)
    print('[sim-cached-test] checkpoint, data, source and SID-only cache hashes passed', flush=True)
    e.configure_dataloader(cfg, 'test', 32, 1)
    dm = e.instantiate_datamodule(cfg)
    loader = e.get_loader(dm, 'test')
    device = torch.device('cuda')
    model = e.instantiate_model(cfg, ckpt, device)
    assert not model.training and model.beam_log_prob_weight == 1. and model.rerank_logit_weight == .1
    temporary = output.with_name(output.name + '.incomplete.' + os.environ['SLURM_JOB_ID'])
    writer = e.H5StreamWriter(temporary)
    arrays, samples, processed, buffered = {}, [], 0, 0
    try:
        with h5py.File(CACHE, 'r', rdcc_nbytes=16 * 1024**2) as handle, \
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
                if index == 0 or (index + 1) % 10000 == 0:
                    samples.append(export_cpu_snapshot(cpu['allowed_cpus']))
                if index == 0 or (index + 1) % 1000 == 0:
                    print(f'[sim-cached-test] batches={index + 1} samples={processed}/{N}', flush=True)
            assert guard.call_count == 0
        e.flush_buffer(writer, arrays)
        assert processed == writer.num_samples == N
        samples.append(export_cpu_snapshot(cpu['allowed_cpus']))
    finally:
        writer.close()
    temporary.replace(output)
    metadata = {**previous, 'split': 'test', 'limits': dict(max_batches=None, max_samples=None),
                'candidate_source': source, 'frozen_selection': frozen, 'dataset_provenance': provenance,
                'cpu_budget': cpu, 'cpu_process_tree_samples': samples,
                'git_commit': os.environ['SOURCE_GIT_COMMIT'], 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                'source_files': previous['source_files'] + [cached.record(p) for p in (
                    Path(__file__), e.ROOT / 'run_sim_cached_test_cpu4.sh')],
                'exporter_sha256': e.file_sha256(Path(__file__)), 'no_candidate_regeneration': True,
                'validation_acceptance': frozen['validation_acceptance'],
                'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
                'timing_scope': 'cached reranking; not end-to-end inference'}
    e.write_metadata(output, metadata, processed)


def audit_predictions(handle, reference, expected):
    """Chunked independent identity, candidate, finite-score and fusion checks."""
    shapes = dict(user_id=(expected,), example_hash=(expected, 2), target_sid=(expected, 4),
                  beam_sid=(expected, 20, 4), rerank_sid=(expected, 20, 4),
                  beam_score=(expected, 20), raw_rerank_logit=(expected, 20), rerank_score=(expected, 20))
    assert set(handle.keys()) == set(shapes)
    assert all(handle[k].shape == shape for k, shape in shapes.items())
    assert handle['user_id'].dtype == np.int64 and handle['example_hash'].dtype == np.uint64
    assert len(np.unique(handle['user_id'][:])) == expected
    assert len(np.unique(handle['example_hash'][:], axis=0)) == expected
    cache = cached.CandidateCache(reference)
    assert len(cache.ids) == expected
    hits = 0
    for start in range(0, expected, 8192):
        arrays = {key: handle[key][start:start + 8192] for key in shapes}
        ids, scores = cache.aligned(arrays['user_id'], arrays['example_hash'], arrays['target_sid'])
        np.testing.assert_array_equal(arrays['beam_sid'], ids)
        np.testing.assert_array_equal(arrays['beam_score'], scores)
        hits += check_candidates(arrays)
        verify_runtime_fusion(arrays)
    return hits


def accept_full(output, receipt):
    assert len(os.sched_getaffinity(0)) == 1 and not receipt.exists()
    frozen, previous = validate_frozen_validation()
    checkpoint = json.loads(cached.CHECKPOINT_ACCEPTANCE.read_text())
    source = validate_test_cache(checkpoint)
    m = json.loads(output.with_suffix('.json').read_text())
    assert m['num_samples'] == N and m['split'] == 'test' and m['schema_version'] == 3
    assert m['limits'] == dict(max_batches=None, max_samples=None)
    assert m['amp'] == 'bf16' and m['batch_size'] == 32 and m['num_workers'] == 1
    assert m['checkpoint_sha256'] == checkpoint['checkpoint']['sha256'] == previous['checkpoint_sha256']
    assert m['checkpoint'] == checkpoint['checkpoint']['path']
    assert m['hydra_config_sha256'] == checkpoint['configuration']['sha256']
    assert m['checkpoint_acceptance'] == cached.record(cached.CHECKPOINT_ACCEPTANCE)
    assert m['candidate_source'] == source and m['frozen_selection'] == frozen
    assert m['no_candidate_regeneration'] and m['validation_acceptance'] == frozen['validation_acceptance']
    assert m['beam_log_prob_weight'] == 1. and m['rerank_logit_weight'] == .1
    assert m['git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and m['slurm_job_id'] == os.environ['SLURM_JOB_ID']
    provenance = m['dataset_provenance']
    assert provenance['protocol'] == 'history-last-user-v1' and provenance['expected_samples'] == N
    assert provenance['source_compatibility'] is None
    assert provenance['training_preflight']['sha256'] == checkpoint['training_preflight']['sha256']
    assert provenance['initialization_checkpoint']['sha256'] == checkpoint['teacher_checkpoint']['sha256']
    for entry in m['source_files'] + list(provenance['manifests'].values()):
        assert cached.record(entry['path']) == entry, entry['path']
    for path, digest in provenance['verified_training_sources'].items():
        assert e.file_sha256(path) == digest, path
    assert m['cpu_budget']['allocated_cpus'] == 4 and m['cpu_budget']['library_threads'] == 1
    cpus = m['cpu_budget']['allowed_cpus']
    assert 0 < len(cpus) <= 4 and len(m['cpu_process_tree_samples']) >= 2
    for sample in m['cpu_process_tree_samples']:
        assert sample['allowed_cpus'] == cpus and sample['processes']
        assert all(p['threads_checked'] > 0 for p in sample['processes'])
    with h5py.File(output, 'r') as f, h5py.File(CACHE, 'r') as reference:
        hits = audit_predictions(f, reference, N)
    original = json.loads(TEST_ACCEPTANCE.read_text())['models']['sid']
    assert hits == original['coverage_count_at_20']
    result = dict(status='passed', scope='full cached-candidate SIM test integrity; paired statistics pending',
                  checked_at=datetime.now().astimezone().isoformat(), num_samples=N, unique_users=N,
                  job_id=os.environ['SLURM_JOB_ID'], source_git_commit=m['git_commit'],
                  natural_beam_hits=hits, checkpoint_sha256=m['checkpoint_sha256'], output=str(output),
                  h5_sha256=e.file_sha256(output), sidecar_sha256=e.file_sha256(output.with_suffix('.json')),
                  checkpoint_acceptance=cached.record(cached.CHECKPOINT_ACCEPTANCE),
                  candidate_source=source, frozen_selection=frozen, input_identity_alignment=True,
                  cached_beam_exact=True, teacher_regenerated=False, runtime_fusion_recomputed_exact=True,
                  cpu_budget=m['cpu_budget'], audit_script_sha256=e.file_sha256(Path(__file__)))
    with receipt.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('[sim-cached-test-acceptance] ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--acceptance', type=Path)
    args = p.parse_args()
    if args.acceptance:
        accept_full(args.output, args.acceptance)
    else:
        export_full(args.output)
