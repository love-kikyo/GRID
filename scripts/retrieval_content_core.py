"""Five fixed-layout content policies on one shared, fresh natural beam20.

The frozen diagnostic implementation remains unchanged. This narrow adapter
selects its already-tested policies and adds five-policy consistency checks.
"""
import numpy as np

from scripts import retrieval_diagnostic_core as frozen


POLICIES = ('sim10', 'recent10', 'random10', 'mask5', 'mask0')


def check_controls(outputs):
    if tuple(outputs) != POLICIES:
        raise AssertionError('Unexpected content policies or order')
    center = outputs['sim10']
    for name, arrays in outputs.items():
        for field in ('user_id', 'example_hash', 'target_sid', 'beam_sid', 'beam_score',
                      'history_valid_count', 'candidate_position'):
            if not np.array_equal(arrays[field], center[field]):
                raise AssertionError(f'Frozen identity/candidate/layout changed: {name}/{field}')
        if not bool((arrays['rerank_input_length'] == 13).all()):
            raise AssertionError(f'Physical ten-slot layout changed: {name}')
    for active in (0, 5):
        expected = center['context_itemkey'].copy()
        expected[:, :, active:] = -1
        if not np.array_equal(outputs[f'mask{active}']['context_itemkey'], expected):
            raise AssertionError(f'Masked context changed: mask{active}')


def infer_content(model, batch, device):
    outputs = dict(frozen._infer_sequence(model, batch, device, POLICIES))
    check_controls(outputs)
    return outputs


def smoke_gate(model, batch, device):
    # Keep the previously accepted native/GT/cache/parameter/actual-position
    # checks, then independently prove the new five-policy public entry point.
    versions = {name: value._version for name, value in model.named_parameters()}
    gate = frozen.smoke_gate(model, batch, device)
    reference = frozen.infer_diagnostics(model, batch, device)
    current = infer_content(model, batch, device)
    for name in POLICIES:
        if set(current[name]) != set(reference[name]):
            raise AssertionError(f'Frozen diagnostic output schema changed: {name}')
        for field in reference[name]:
            if not np.array_equal(current[name][field], reference[name][field]):
                raise AssertionError(f'Frozen diagnostic differs: {name}/{field}')
    if (versions != {name: value._version for name, value in model.named_parameters()}
            or any(value.grad is not None for value in model.parameters())):
        raise AssertionError('Content entry changed parameters or gradients')
    gate = dict(gate, policies=list(POLICIES), subset_matches_frozen_diagnostic=True)
    for key in ('candidate_positions', 'input_lengths'):
        gate[key] = {name: gate[key][name] for name in POLICIES}
    return gate
