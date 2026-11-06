"""Fixed-checkpoint inference budgets; no training, tuning or GT-conditioned search.

Each retrieval group shares one *new* natural beam and its scores. Every rerank
branch receives an independent KV clone; changing C always reruns beam search.
The existing runtime fusion is preserved. The reporting layer applies the
separately frozen validation beta to exported FP32 raw scores.
"""
from contextlib import contextmanager
import copy

import numpy as np
import torch

from scripts import export_paired_evaluation as export


SETTINGS = {
    "retrieval": [(20, 0), (20, 5), (20, 10), (20, 20)],
    "beam": [(10, 10), (40, 10)],
}
PREDICTION_FIELDS = (
    "beam_sid", "beam_score", "raw_rerank_logit", "rerank_sid", "rerank_score",
)


def setting_name(c, m):
    return f"c{c}_m{m}"


@contextmanager
def restored_settings(model):
    names = ("top_k_for_generation", "top_k_for_score", "active_top_k_for_score")
    original = {name: getattr(model, name) for name in names}
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(model, name, value)


def _infer_sequence(model, batch, device, settings):
    """List result deliberately supports repeated settings for isolation tests."""
    if model.training:
        raise ValueError("Sensitivity inference requires model.eval()")
    device = torch.device(device)
    model_batch = export.move_batch_to_device(batch, device)
    hierarchies = model.num_hierarchies
    sid = model_batch.transformed_sequences["sequence_data"].long()
    hist_sid = sid[:, :-hierarchies].contiguous()
    hist_mask = model_batch.mask[:, :-hierarchies].long().contiguous()
    # The last item is the target, not eligible history. No target is passed
    # into either search or reranking, including auxiliary loss calculations.
    history = model_batch.transformed_sequences["hist_itemkey"][:, :-1].long().contiguous()
    batch_size = sid.shape[0]
    results, previous_c, beam = [], None, None
    with restored_settings(model), torch.inference_mode(), export.autocast_context(device, "bf16"):
        for c, m in settings:
            if c not in (10, 20, 40) or m not in (0, 5, 10, 20):
                raise ValueError(f"Unsupported sensitivity setting: {(c, m)}")
            model.top_k_for_generation = c
            model.top_k_for_score = m
            model.active_top_k_for_score = None
            if c != previous_c:
                beam = None  # release a previous width before generating a larger beam
                beam = model._beam_search_semantic_ids(sid=hist_sid, attention_mask=hist_mask)
                previous_c = c
            generated, scores, input_mask, source_cache = beam
            branch_cache = model.clone_kv_cache(source_cache)
            rerank_ids, rerank_scores, logits = model._rerank_with_click_prediction(
                generated_ids=generated,
                beam_log_prob=scores,
                input_mask=input_mask,
                past_key_values=branch_cache,
                hist_itemkey=history,
                batch_size=batch_size,
            )
            result = {
                "beam": {"ids": generated.reshape(batch_size, c, hierarchies), "scores": scores},
                "rerank": {"ids": rerank_ids, "scores": rerank_scores, "click_logits": logits},
            }
            arrays = export.extract_batch_arrays(model, batch, result)
            results.append((setting_name(c, m), arrays))
            del branch_cache, result, rerank_ids, rerank_scores, logits
            del generated, scores, input_mask, source_cache
    return results


def infer_settings(model, batch, device, group):
    if group not in SETTINGS:
        raise ValueError(f"Unknown sensitivity group: {group}")
    return dict(_infer_sequence(model, batch, device, SETTINGS[group]))


def _assert_prediction_equal(left, right, label):
    for field in PREDICTION_FIELDS:
        if not np.array_equal(left[field], right[field]):
            raise AssertionError(f"{label}: {field} differs")


def smoke_gate(model, batch, device):
    """Exact BF16 gates on one real batch; no tolerance relaxation or selection."""
    parameter_versions = {name: parameter._version for name, parameter in model.named_parameters()}
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Inference started with existing parameter gradients")
    original_settings = (model.top_k_for_generation, model.top_k_for_score, model.active_top_k_for_score)
    original_weights = (model.beam_log_prob_weight, model.rerank_logit_weight)
    if original_settings != (20, 10, None) or original_weights != (1.0, 0.1):
        raise AssertionError("Expected frozen C20/M10 training configuration and runtime fusion")

    retrieval = infer_settings(model, batch, device, "retrieval")
    reference = export.extract_batch_arrays(
        model, batch, export.infer_batch(model, batch, torch.device(device), "bf16")
    )
    _assert_prediction_equal(retrieval["c20_m10"], reference, "original inference equivalence")
    for arrays in retrieval.values():
        for field in ("beam_sid", "beam_score"):
            if not np.array_equal(arrays[field], reference[field]):
                raise AssertionError(f"Retrieval settings changed {field}")

    repeated = _infer_sequence(model, batch, device, [(20, 10), (20, 20), (20, 10)])
    _assert_prediction_equal(repeated[0][1], repeated[-1][1], "cache branch isolation")
    _assert_prediction_equal(repeated[0][1], reference, "independent repeat")

    altered = copy.copy(batch)
    altered.transformed_sequences = {
        key: value.clone() for key, value in batch.transformed_sequences.items()
    }
    target = altered.transformed_sequences["sequence_data"][:, -model.num_hierarchies:]
    target.copy_((target % model.num_embeddings_per_hierarchy) + 1)
    altered.transformed_sequences["hist_itemkey"][:, -1] = (
        target.long() * model.powers.to(target.device).long()
    ).sum(-1)
    changed = infer_settings(model, altered, device, "retrieval")
    for name in retrieval:
        _assert_prediction_equal(retrieval[name], changed[name], f"GT independence {name}")
    beam = infer_settings(model, batch, device, "beam")
    changed_beam = infer_settings(model, altered, device, "beam")
    for name in beam:
        _assert_prediction_equal(beam[name], changed_beam[name], f"GT independence {name}")
    if parameter_versions != {name: parameter._version for name, parameter in model.named_parameters()}:
        raise AssertionError("Model parameters mutated during sensitivity inference")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Inference created parameter gradients")
    if (model.top_k_for_generation, model.top_k_for_score, model.active_top_k_for_score) != original_settings:
        raise AssertionError("Inference settings not restored")
    if (model.beam_log_prob_weight, model.rerank_logit_weight) != original_weights:
        raise AssertionError("Runtime fusion weights changed")
    return {
        "status": "passed", "samples": len(reference["user_id"]),
        "precision": "bf16", "comparison": "exact_array_equality",
        "original_inference_equivalence": True, "retrieval_candidates_and_scores_identical": True,
        "cache_order_independence": True, "gt_independence_all_settings": True,
        "parameter_versions_unchanged": True, "parameter_gradients_absent": True,
        "all_settings": list(retrieval) + list(beam),
        "note": "Current-hardware equivalence only; no assertion of equality to old cross-hardware exports",
    }
