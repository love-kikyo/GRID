"""Controlled retrieval-content/layout interventions on one shared natural beam.

No model source or weights are changed. Random selection depends only on user ID,
fixed seed and eligible history positions; it is shared by a user's candidates.
All chosen sets are ordered by the unchanged candidate-similarity implementation.
"""
from contextlib import contextmanager
import copy
import hashlib

import numpy as np
import torch

from scripts import export_paired_evaluation as export
from scripts.sensitivity_core import _assert_prediction_equal, restored_settings


POLICIES = ('sim10', 'mask0', 'mask5', 'recent10', 'random10', 'short0', 'short5')
RANDOM_SEED = 42
RANDOM_NAMESPACE = 'grid-retrieval-diagnostic-v1-history-positions'
EXTRA_FIELDS = ('context_itemkey', 'history_valid_count', 'candidate_position', 'rerank_input_length')


def select_history(history, user_ids, policy, masking_token=-1):
    """CPU position selection, preserving duplicate interactions and chronology.

    Returns ten slots per user; padding is on the right. Random positions are
    sorted chronologically before the original candidate-dependent ordering.
    """
    if policy not in ('recent10', 'random10'):
        raise ValueError(f'Unknown history selection: {policy}')
    history = np.asarray(history, dtype=np.int64)
    users = np.asarray(user_ids, dtype=np.int64)
    if history.ndim != 2 or users.shape != (history.shape[0],):
        raise ValueError('History/user shape mismatch')
    chosen = np.full((len(users), 10), masking_token, dtype=np.int64)
    for row, user in enumerate(users):
        valid = np.flatnonzero(history[row] != masking_token)
        if policy == 'recent10':
            indices = valid[-10:]
        else:
            key = f'{RANDOM_NAMESPACE}:{RANDOM_SEED}:{int(user)}'.encode('ascii')
            seed = int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), 'little')
            generator = np.random.Generator(np.random.PCG64(seed))
            indices = np.sort(generator.choice(valid, min(10, len(valid)), replace=False))
        chosen[row, :len(indices)] = history[row, indices]
    return chosen


def _check_context(context, target, history, masking_token, expected_slots):
    if context.shape != (len(target), expected_slots + 1):
        raise AssertionError('Unexpected retrieval context shape')
    if not torch.equal(context[:, -1], target):
        raise AssertionError('Final scoring candidate changed')
    retrieved = context[:, :-1]
    eligible = ((retrieved.unsqueeze(-1) == history.unsqueeze(1)).any(-1)
                | retrieved.eq(masking_token))
    if not bool(eligible.all()):
        raise AssertionError('Retrieval context contains an item outside eligible history')
    wanted = history.ne(masking_token).sum(-1).clamp(max=expected_slots)
    if not torch.equal(retrieved.ne(masking_token).sum(-1), wanted):
        raise AssertionError('Retrieved valid-item count differs from available history')


def _contexts(model, history, users, target):
    mask = model.masking_token
    expanded = history.repeat_interleave(20, dim=0)
    original = model.get_similar_itemkey
    baseline = original(target_itemkey=target, hist_itemkey=expanded)
    _check_context(baseline, target, expanded, mask, 10)
    contexts = {'sim10': baseline.clone()}
    for active in (0, 5):
        value = baseline.clone()
        value[:, active:10] = mask
        contexts[f'mask{active}'] = value
        contexts[f'short{active}'] = torch.cat([baseline[:, :active], target[:, None]], -1)
    host_history = history.detach().cpu().numpy()
    for name in ('recent10', 'random10'):
        selected = torch.as_tensor(select_history(host_history, users, name, mask), device=history.device)
        selected = selected.repeat_interleave(20, dim=0)
        value = original(target_itemkey=target, hist_itemkey=selected)
        _check_context(value, target, expanded, mask, 10)
        contexts[name] = value
    return contexts, expanded


@contextmanager
def _fixed_context(model, context, target, history):
    """Exception-safe instance override; never leave a shadowed class method."""
    missing = object()
    previous = model.__dict__.get('get_similar_itemkey', missing)

    def retrieve(target_itemkey, hist_itemkey):
        if not torch.equal(target_itemkey, target) or not torch.equal(hist_itemkey, history):
            raise AssertionError('Reranking retrieval inputs differ from frozen candidate/history')
        return context.clone()

    model.get_similar_itemkey = retrieve
    try:
        yield
    finally:
        if previous is missing:
            del model.get_similar_itemkey
        else:
            model.get_similar_itemkey = previous


@contextmanager
def _observe_positions(model, context):
    """Observe actual decoder payload and actual RoPE position_ids, not a formula."""
    try:
        decoder = model.decoder
        rotary = decoder.decoder.rotary_emb
    except AttributeError as exc:
        raise RuntimeError('Actual decoder rotary position hook unavailable') from exc
    observed = {'decoder_calls': 0, 'rotary_calls': 0}

    def decoder_hook(module, args, kwargs):
        embedding = kwargs.get('sequence_embedding')
        attention = kwargs.get('attention_mask')
        if embedding is None or attention is None:
            raise AssertionError('Cannot observe actual reranking decoder inputs')
        observed['decoder_calls'] += 1
        observed['input_length'] = embedding.shape[1]
        tail = attention[:, -context.shape[1]:].bool()
        if not torch.equal(tail, context.ne(model.masking_token)):
            raise AssertionError('Actual attention mask differs from controlled context')

    def rotary_hook(module, args, kwargs):
        position = kwargs.get('position_ids')
        if position is None and len(args) >= 2:
            position = args[1]
        if position is None or position.ndim != 2:
            raise AssertionError('Cannot observe actual rotary position_ids')
        observed['rotary_calls'] += 1
        observed['positions'] = position.detach().cpu().numpy().copy()

    decoder_handle = decoder.register_forward_pre_hook(decoder_hook, with_kwargs=True)
    try:
        rotary_handle = rotary.register_forward_pre_hook(rotary_hook, with_kwargs=True)
        try:
            yield observed
        finally:
            rotary_handle.remove()
    finally:
        decoder_handle.remove()


def _position_arrays(observed, batch_size, expected_length):
    if (observed['decoder_calls'] != 1 or observed['rotary_calls'] != 1
            or observed['input_length'] != expected_length):
        raise AssertionError('Unexpected actual decoder/rotary call count or length')
    positions = observed['positions']
    if not np.issubdtype(positions.dtype, np.integer):
        raise AssertionError('Actual rotary positions must use integer indices')
    if positions.shape[1] != expected_length or positions.shape[0] not in (1, batch_size * 20):
        raise AssertionError('Unexpected actual rotary positions shape')
    if positions.shape[0] == 1:
        positions = np.broadcast_to(positions, (batch_size * 20, expected_length))
    positions = positions.reshape(batch_size, 20, expected_length)
    if not np.array_equal(positions, np.broadcast_to(positions[:, :1], positions.shape)):
        raise AssertionError('Candidate branches use different position numbering')
    return positions[:, 0, -1].astype(np.int32), np.full(batch_size, expected_length, dtype=np.int16)


def _infer_sequence(model, batch, device, policies):
    if model.training:
        raise ValueError('Diagnostic inference requires model.eval()')
    if not policies or any(name not in POLICIES for name in policies):
        raise ValueError('Unsupported diagnostic policy')
    if model.masking_token != -1:
        raise ValueError('Frozen protocol requires masking_token=-1')
    device = torch.device(device)
    moved = export.move_batch_to_device(batch, device)
    sid = moved.transformed_sequences['sequence_data'].long()
    hist_sid = sid[:, :-model.num_hierarchies].contiguous()
    hist_mask = moved.mask[:, :-model.num_hierarchies].long().contiguous()
    history = moved.transformed_sequences['hist_itemkey'][:, :-1].long().contiguous()
    users = export.extract_user_ids(batch)
    batch_size = len(users)
    result = []
    with restored_settings(model), torch.inference_mode(), export.autocast_context(device, 'bf16'):
        model.top_k_for_generation, model.top_k_for_score, model.active_top_k_for_score = 20, 10, None
        generated, beam_scores, input_mask, source_cache = model._beam_search_semantic_ids(
            sid=hist_sid, attention_mask=hist_mask)
        target = (generated * model.powers).sum(-1)
        contexts, expanded_history = _contexts(model, history, users, target)
        counts = history.ne(-1).sum(-1).cpu().numpy().astype(np.int32)
        for policy in policies:
            context = contexts[policy]
            slots = context.shape[1] - 1
            model.top_k_for_score = slots
            model.active_top_k_for_score = int(policy[-1]) if policy.startswith('mask') else None
            cache = model.clone_kv_cache(source_cache)
            with _fixed_context(model, context, target, expanded_history), _observe_positions(model, context) as observed:
                rids, rscores, logits = model._rerank_with_click_prediction(
                    generated_ids=generated, beam_log_prob=beam_scores, input_mask=input_mask,
                    past_key_values=cache, hist_itemkey=history, batch_size=batch_size)
            prediction = {'beam': {'ids': generated.reshape(batch_size, 20, model.num_hierarchies),
                                   'scores': beam_scores},
                          'rerank': {'ids': rids, 'scores': rscores, 'click_logits': logits}}
            arrays = export.extract_batch_arrays(model, batch, prediction)
            stored_context = np.full((batch_size, 20, 10), -1, dtype=np.int64)
            stored_context[:, :, :slots] = context[:, :-1].reshape(batch_size, 20, slots).cpu().numpy()
            arrays['context_itemkey'] = stored_context
            arrays['history_valid_count'] = counts.copy()
            arrays['candidate_position'], arrays['rerank_input_length'] = _position_arrays(observed, batch_size, slots + 3)
            result.append((policy, arrays))
            del cache, prediction, rids, rscores, logits
    return result


def _check_controls(outputs):
    baseline = outputs['sim10']
    for name in POLICIES:
        for field in ('beam_sid', 'beam_score', 'history_valid_count'):
            if not np.array_equal(outputs[name][field], baseline[field]):
                raise AssertionError(f'Shared candidate/history changed: {name}/{field}')
        offset = 10 if name == 'short0' else 5 if name == 'short5' else 0
        if not np.array_equal(outputs[name]['candidate_position'], baseline['candidate_position'] - offset):
            raise AssertionError(f'Actual candidate positions violate layout control: {name}')
    for active in (0, 5):
        left, right = outputs[f'mask{active}'], outputs[f'short{active}']
        if not np.array_equal(left['context_itemkey'], right['context_itemkey']):
            raise AssertionError('Masked and shortened contexts differ in content')
        wanted = baseline['context_itemkey'].copy()
        wanted[:, :, active:] = -1
        if not np.array_equal(left['context_itemkey'], wanted):
            raise AssertionError('Mask intervention changed retained content')


def infer_diagnostics(model, batch, device):
    outputs = dict(_infer_sequence(model, batch, device, POLICIES))
    _check_controls(outputs)
    return outputs


def smoke_gate(model, batch, device):
    versions = {name: p._version for name, p in model.named_parameters()}
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError('Existing gradients before inference')
    settings = (model.top_k_for_generation, model.top_k_for_score, model.active_top_k_for_score)
    weights = (model.beam_log_prob_weight, model.rerank_logit_weight)
    if settings != (20, 10, None) or weights != (1., .1):
        raise AssertionError('Expected frozen C20/M10 and runtime beta0.1')
    results = infer_diagnostics(model, batch, device)
    native = export.extract_batch_arrays(model, batch, export.infer_batch(model, batch, torch.device(device), 'bf16'))
    _assert_prediction_equal(results['sim10'], native, 'native sim10 equivalence')
    for active in (0, 5):
        with restored_settings(model):
            model.active_top_k_for_score = active
            native_mask = export.extract_batch_arrays(model, batch, export.infer_batch(model, batch, torch.device(device), 'bf16'))
        _assert_prediction_equal(results[f'mask{active}'], native_mask, f'native active mask{active}')
    altered = copy.copy(batch)
    altered.transformed_sequences = {key: value.clone() for key, value in batch.transformed_sequences.items()}
    target = altered.transformed_sequences['sequence_data'][:, -model.num_hierarchies:]
    target.copy_((target % model.num_embeddings_per_hierarchy) + 1)
    altered.transformed_sequences['hist_itemkey'][:, -1] = (target.long() * model.powers.to(target.device).long()).sum(-1)
    changed = infer_diagnostics(model, altered, device)
    for name in POLICIES:
        _assert_prediction_equal(results[name], changed[name], f'GT independence {name}')
        for field in EXTRA_FIELDS:
            if not np.array_equal(results[name][field], changed[name][field]):
                raise AssertionError(f'GT changed context/layout: {name}/{field}')
    repeated = _infer_sequence(model, batch, device, ('random10', 'sim10', 'mask0', 'random10'))
    for name, arrays in repeated:
        _assert_prediction_equal(results[name], arrays, f'cache/order independence {name}')
        for field in EXTRA_FIELDS:
            if not np.array_equal(results[name][field], arrays[field]):
                raise AssertionError(f'Order changed context/layout: {name}/{field}')
    if versions != {name: p._version for name, p in model.named_parameters()} or any(p.grad is not None for p in model.parameters()):
        raise AssertionError('Parameters or gradients changed')
    if settings != (model.top_k_for_generation, model.top_k_for_score, model.active_top_k_for_score) or weights != (model.beam_log_prob_weight, model.rerank_logit_weight):
        raise AssertionError('Original inference configuration was not restored')
    return {'status': 'passed', 'samples': len(native['user_id']), 'policies': list(POLICIES),
            'native_sim10_exact': True, 'native_active_masks_exact': True,
            'shared_candidates_and_scores': True, 'gt_independence_all_policies': True,
            'cache_order_independence': True, 'actual_rotary_positions_observed': True,
            'fixed_slots_candidate_positions_equal': True, 'paired_short_mask_content_equal': True,
            'parameters_unchanged_and_gradients_absent': True,
            'random_seed': RANDOM_SEED, 'random_namespace': RANDOM_NAMESPACE,
            'candidate_positions': {n: sorted(set(map(int, a['candidate_position']))) for n, a in results.items()},
            'input_lengths': {n: sorted(set(map(int, a['rerank_input_length']))) for n, a in results.items()}}
