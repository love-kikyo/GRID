"""64-row SIM validation smoke on accepted frozen-teacher candidate caches.

Never trains, selects beta, runs test, or regenerates candidates. The cache
supplies only natural beam IDs/scores; previous reranker predictions are unused.
"""
import argparse
import copy
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch  # Before data-module / TensorFlow initialization.
import h5py
import numpy as np

from scripts import export_paired_evaluation as e
from scripts.accept_sim_validation import accept, check_candidates
from src.runtime_export_cpu import configure_export_cpu, export_cpu_snapshot

REPORT = e.ROOT / 'logs/training_launch/sim_seed42_validation_2026-10-13'
CHECKPOINT_ACCEPTANCE = REPORT / 'checkpoint_acceptance_v2.json'
REFERENCE_SELECTION = e.ROOT / 'logs/analysis/paired_evaluation/separate_seed42_132051_fusion_val.json'


class CandidateCache:
    """Look up by user ID, then require the complete input hash and target match."""
    def __init__(self, handle):
        self.handle = handle
        ids = handle['user_id'][:]
        self.order = np.argsort(ids)
        self.ids = ids[self.order]
        assert len(np.unique(ids)) == len(ids), 'Duplicate cache users'

    def aligned(self, ids, hashes, targets):
        assert len(np.unique(ids)) == len(ids), 'Duplicate batch users'
        positions = np.searchsorted(self.ids, ids)
        assert (positions < len(self.ids)).all(), 'User absent from cache'
        np.testing.assert_array_equal(self.ids[positions], ids, err_msg='User absent from cache')
        rows = self.order[positions]
        # h5py fancy indices must be increasing; restore the requested batch order.
        sorting = np.argsort(rows)
        inverse = np.argsort(sorting)
        def take(key):
            return self.handle[key][rows[sorting]][inverse]
        np.testing.assert_array_equal(take('example_hash'), hashes, err_msg='Input mismatch')
        np.testing.assert_array_equal(take('target_sid'), targets, err_msg='Target mismatch')
        return take('beam_sid'), take('beam_score')


def score_cached_batch(model, batch, beam_sid, beam_score, device, amp='bf16'):
    """Only pre-GT history item keys and candidate IDs enter SIM's scorer."""
    history = batch.transformed_sequences['hist_itemkey'][:, :-1].long().to(device)
    ids = torch.as_tensor(beam_sid, dtype=torch.long, device=device)
    scores = torch.as_tensor(beam_score, dtype=torch.float32, device=device)
    assert ids.shape == (len(history), model.top_k_for_generation, model.num_hierarchies)
    assert scores.shape == ids.shape[:2]
    keys = (ids * model.powers).sum(-1)
    with torch.inference_mode(), e.autocast_context(device, amp):
        logits = model.reranker.score(history, keys, auxiliary=False)['esu_logits']
        final = model._combine_rerank_scores(scores, logits)
        top = final.topk(model.top_k_for_generation, dim=-1)
    return {'beam': {'ids': ids, 'scores': scores},
            'rerank': {'ids': ids.gather(1, top.indices[..., None].expand_as(ids)),
                       'scores': top.values, 'click_logits': logits.gather(1, top.indices)}}


def compare_outputs(a, b):
    for section in ('beam', 'rerank'):
        for key in a[section]:
            torch.testing.assert_close(a[section][key], b[section][key], rtol=0, atol=0)


def verify_inference(model, batch, beam_sid, beam_score, device):
    # Any accidental generation in the cached path is a hard error.
    with patch.object(model.teacher, 'generate_candidates', side_effect=AssertionError('Regenerated beam')) as guard:
        result = score_cached_batch(model, batch, beam_sid, beam_score, device)
        changed = copy.deepcopy(batch)
        changed.transformed_sequences['sequence_data'][:, -4:] = (
            changed.transformed_sequences['sequence_data'][:, -4:] + 1) % 4096
        changed.transformed_sequences['hist_itemkey'][:, -1] = 123456789
        compare_outputs(result, score_cached_batch(model, changed, beam_sid, beam_score, device))
        assert guard.call_count == 0
    # Independent integration route: use the original SIM INFER wrapper, with
    # only its candidate provider replaced by the same immutable natural cache.
    from src.models.modules.huggingface.transformer_base_module import ModelMode
    def provider(*args, ensure_gt_in_candidates, **kwargs):
        assert ensure_gt_in_candidates is False
        ids, scores = result['beam']['ids'].clone(), result['beam']['scores'].clone()
        keys = (ids * model.powers).sum(-1)
        return dict(candidate_sid=ids, candidate_itemkey=keys, beam_scores=scores,
                    labels=torch.zeros_like(scores))
    with patch.object(model.teacher, 'generate_candidates', side_effect=provider) as supplied:
        with torch.inference_mode(), e.autocast_context(device, 'bf16'):
            direct = model.model_step(e.move_batch_to_device(batch, device), ModelMode.INFER)
        assert supplied.call_count == 1
        compare_outputs(result, direct)
    return result, dict(rows=len(beam_sid), no_candidate_regeneration=True,
                        gt_mutation_invariant=True, original_sim_infer_exact=True)


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=e.file_sha256(path))


def smoke(output):
    assert not output.exists() and not output.with_suffix('.json').exists()
    cpu = configure_export_cpu(torch, 1)
    assert cpu is not None
    accepted = json.loads(CHECKPOINT_ACCEPTANCE.read_text())
    ref = json.loads(REFERENCE_SELECTION.read_text())
    assert accepted['status'] == 'passed'
    run = Path(accepted['run_dir'])
    ckpt = Path(accepted['checkpoint']['path'])
    assert e.file_sha256(ckpt) == accepted['checkpoint']['sha256']
    assert e.file_sha256(run / '.hydra/config.yaml') == accepted['configuration']['sha256']
    cfg = e.load_run_config(run)
    assert cfg.seed == 42 and str(cfg.model._target_).endswith('.SIMRerankModule')
    provenance, count = e.validate_provenance(
        cfg, 'val', e.ROOT / 'data/taobao_history_last_user_v1', 'history-last-user-v1',
        Path(accepted['training_preflight']['path']))
    assert count == 347416 and ref['input_split'] == 'val' and ref['num_examples'] == count
    ref_path = Path(ref['input'])
    assert e.file_sha256(ref_path) == ref['input_sha256']
    assert e.file_sha256(ref_path.with_suffix('.json')) == ref['input_sidecar_sha256']
    ref_meta = json.loads(ref_path.with_suffix('.json').read_text())
    assert ref_meta['dataset_provenance']['initialization_checkpoint']['sha256'] == accepted['teacher_checkpoint']['sha256']
    assert ref_meta['split'] == 'val' and ref_meta['amp'] == 'bf16'
    e.configure_dataloader(cfg, 'val', 32, 1)
    dm = e.instantiate_datamodule(cfg)
    loader = e.get_loader(dm, 'val')
    device = torch.device('cuda')
    model = e.instantiate_model(cfg, ckpt, device)
    assert not model.training and model.beam_log_prob_weight == 1. and model.rerank_logit_weight == .1
    incomplete = output.with_name(output.name + '.incomplete.' + os.environ['SLURM_JOB_ID'])
    checks, samples = [], []
    writer = e.H5StreamWriter(incomplete)
    try:
        with h5py.File(ref_path, 'r') as f:
            cache = CandidateCache(f)
            for wrapped in itertools.islice(loader, 2):
                batch = e.unwrap_batch(wrapped)
                ids, scores = cache.aligned(e.extract_user_ids(batch), e.example_hashes(batch),
                                           batch.transformed_sequences['sequence_data'][:, -4:].numpy())
                result, check = verify_inference(model, batch, ids, scores, device)
                arrays = e.extract_batch_arrays(model, batch, result)
                check_candidates(arrays)
                writer.append(arrays)
                checks.append(check)
                samples.append(export_cpu_snapshot(cpu['allowed_cpus']))
        assert writer.num_samples == 64 and len(checks) == 2
    finally:
        writer.close()
    incomplete.replace(output)
    sources = [Path(__file__), e.ROOT / 'scripts/export_paired_evaluation.py',
               e.ROOT / 'scripts/accept_sim_validation.py', e.ROOT / 'src/runtime_cpu.py',
               e.ROOT / 'src/runtime_export_cpu.py', e.ROOT / 'run_sim_cached_smoke_cpu4.sh']
    sources += e.frozen_reranker_sources(str(cfg.model._target_))
    metadata = dict(
        schema_version=3, split='val', limits=dict(max_batches=2, max_samples=None),
        amp='bf16', batch_size=32, num_workers=1, checkpoint=str(ckpt),
        checkpoint_sha256=accepted['checkpoint']['sha256'],
        hydra_config_sha256=accepted['configuration']['sha256'],
        git_commit=os.environ['SOURCE_GIT_COMMIT'], slurm_job_id=os.environ['SLURM_JOB_ID'],
        dataset_provenance=provenance, source_files=[record(p) for p in sources],
        cpu_budget=cpu, cpu_process_tree_samples=samples, run_dir=str(run),
        model_target=str(cfg.model._target_), top_k_for_generation=20, num_hierarchies=4,
        num_embeddings_per_hierarchy=4096, beam_log_prob_weight=1., rerank_logit_weight=.1,
        metric_identity='exact_semantic_id_tuple', sample_unit='unique_user_history',
        candidate_source=dict(kind='cached_frozen_teacher_natural_beam',
                              h5=str(ref_path), h5_sha256=ref['input_sha256'],
                              sidecar_sha256=ref['input_sidecar_sha256'],
                              reference_selection=record(REFERENCE_SELECTION)),
        inference_checks=checks, checkpoint_acceptance=record(CHECKPOINT_ACCEPTANCE),
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        timing_scope='smoke only; cached reranking is not end-to-end inference')
    e.write_metadata(output, metadata, 64)


def independent_accept(output, receipt):
    assert len(os.sched_getaffinity(0)) == 1 and not receipt.exists()
    m = json.loads(output.with_suffix('.json').read_text())
    ref = json.loads(REFERENCE_SELECTION.read_text())
    source = m['candidate_source']
    assert source['kind'] == 'cached_frozen_teacher_natural_beam'
    assert source['h5'] == ref['input'] and source['h5_sha256'] == ref['input_sha256']
    assert source['sidecar_sha256'] == ref['input_sidecar_sha256']
    assert source['reference_selection'] == record(REFERENCE_SELECTION)
    assert m['checkpoint_acceptance'] == record(CHECKPOINT_ACCEPTANCE)
    assert m['inference_checks'] == [dict(rows=32, no_candidate_regeneration=True,
        gt_mutation_invariant=True, original_sim_infer_exact=True)] * 2
    result = accept(output, CHECKPOINT_ACCEPTANCE, REFERENCE_SELECTION, smoke=True)
    # Recompute BF16 runtime residual multiplication independently of the model.
    # Future selected-beta metrics will use the unchanged FP32 offline protocol.
    with h5py.File(output, 'r') as f:
        raw = torch.from_numpy(f['raw_rerank_logit'][:]).to(torch.bfloat16)
        assert np.array_equal(raw.float().numpy(), f['raw_rerank_logit'][:])
        combined = f['beam_score'][:] + (raw * .1).float().numpy()
        beam, rerank = f['beam_sid'][:], f['rerank_sid'][:]
        matches = (rerank[:, :, None, :] == beam[:, None, :, :]).all(-1)
        assert (matches.sum(1) == 1).all() and (matches.sum(2) == 1).all()
        expected = np.take_along_axis(combined, matches.argmax(-1), axis=1)
        np.testing.assert_array_equal(expected, f['rerank_score'][:])
    result.update(scope='64-row cached-candidate SIM validation smoke; not full validation or test',
                  candidate_source=source, cached_beam_exact=True, teacher_regenerated=False,
                  runtime_fusion_recomputed_exact=True, inference_checks=m['inference_checks'],
                  cached_audit_script_sha256=e.file_sha256(Path(__file__)),
                  checked_at=datetime.now().astimezone().isoformat())
    with receipt.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('[cached-smoke-acceptance] ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--acceptance', type=Path)
    args = p.parse_args()
    if args.acceptance:
        independent_accept(args.output, args.acceptance)
    else:
        smoke(args.output)
