from dataclasses import replace

import numpy as np
import pytest

from src.baselines.liger_data import Record, pack_batch
from src.baselines.liger_supervision import expand_training_records, pack_supervised_batch


def record(uid=11, sequence=(0, 1, 2), digest='a' * 64):
    return Record(uid, sequence, digest)


def assets():
    return np.arange(80).reshape(20, 4) % 8, np.arange(100).reshape(20, 5).astype(np.int8)


def test_minimal_parent_has_one_final_prefix_and_preserves_row_zero():
    examples = expand_training_records([record(sequence=(4, 0))])
    assert len(examples) == 1
    example = examples[0]
    assert example.sequence == (4, 0)
    assert example.prefix_target_index == example.parent_target_index == 1
    assert example.window_start_index == 0 and example.parent_prefix_count == 1
    assert example.is_final
    _, metadata = pack_supervised_batch(examples, *assets(), weighting='parent')
    assert metadata['target_rows'].tolist() == [0]
    assert metadata['loss_weights'].tolist() == [1.0]


def test_long_parent_expands_only_final_eight_items_and_keeps_original_gt_once():
    parent = record(sequence=tuple(range(20)))
    examples = expand_training_records([parent])
    assert len(examples) == 7
    assert [e.prefix_target_index for e in examples] == list(range(1, 8))
    assert [e.parent_target_index for e in examples] == list(range(13, 20))
    assert all(e.window_start_index == 12 and e.parent_sequence_length == 20 for e in examples)
    assert [e.is_final for e in examples] == [False] * 6 + [True]
    assert all(e.sequence == tuple(range(12, e.parent_target_index + 1)) for e in examples)
    assert parent.sequence == tuple(range(20))


def test_repeated_items_are_not_deduplicated_and_parent_order_is_stable():
    parents = [record(8, (2, 4, 2, 4)), record(-3, (0, 1), 'b' * 64)]
    examples = expand_training_records(iter(parents))
    assert examples == expand_training_records(parents)
    assert [e.parent_user_id for e in examples] == [8, 8, 8, -3]
    assert [e.sequence for e in examples] == [(2, 4), (2, 4, 2), (2, 4, 2, 4), (0, 1)]
    assert [e.parent_serialized_sha256 for e in examples] == ['a' * 64] * 3 + ['b' * 64]


def test_final_only_is_identical_to_existing_packing_including_padding():
    parents = [record(7, tuple(range(20))), record(8, (2, 0), 'b' * 64)]
    examples = expand_training_records(parents, mode='final_only')
    original_inputs, original_metadata = pack_batch(parents, *assets())
    inputs, metadata = pack_supervised_batch(examples, *assets())
    for key in original_inputs:
        np.testing.assert_array_equal(inputs[key], original_inputs[key])
    np.testing.assert_array_equal(metadata['target_rows'], original_metadata['target_rows'])
    assert metadata['history_rows'] == original_metadata['history_rows']
    assert metadata['is_final'].all()
    assert metadata['parent_prefix_counts'].tolist() == [1, 1]


def test_packed_prefixes_have_same_padding_and_metadata_never_enters_model():
    examples = expand_training_records([record(sequence=(1, 2, 3, 4))])
    inputs, metadata = pack_supervised_batch(examples, *assets(), codebook_size=8)
    assert set(inputs) == {'input_sids', 'attention_mask_sids', 'labels_sids', 'input_embeddings'}
    assert not set(inputs) & set(metadata)
    assert inputs['input_sids'].shape == (3, 29)
    assert inputs['attention_mask_sids'].sum(axis=1).tolist() == [4, 8, 12]
    assert (inputs['input_sids'][:, -1] == 0).all()
    assert (inputs['input_embeddings'][0, 1:] == 0).all()
    assert metadata['parent_user_ids'].tolist() == [11, 11, 11]
    assert metadata['history_rows'] == [[1], [1, 2], [1, 2, 3]]
    assert metadata['target_rows'].tolist() == [2, 3, 4]
    assert metadata['loss_weighting'] == 'prefix'
    assert metadata['loss_weights'].tolist() == [1., 1., 1.]


def test_parent_uniform_weights_sum_to_one_per_parent_and_subset_keeps_weights():
    examples = expand_training_records([record(1, (0, 1, 2, 3)), record(2, (4, 5))])
    _, metadata = pack_supervised_batch(examples, *assets(), weighting='parent')
    for uid in (1, 2):
        selected = metadata['parent_user_ids'] == uid
        np.testing.assert_allclose(metadata['loss_weights'][selected].sum(), 1.0)
    _, subset = pack_supervised_batch(examples[:1], *assets(), weighting='parent')
    np.testing.assert_allclose(subset['loss_weights'], [1 / 3])


def test_changing_final_gt_does_not_change_its_encoder_input():
    a = expand_training_records([record(sequence=(1, 2, 3))])[-1:]
    b = expand_training_records([record(sequence=(1, 2, 4))])[-1:]
    pa, _ = pack_supervised_batch(a, *assets())
    pb, _ = pack_supervised_batch(b, *assets())
    for key in ('input_sids', 'attention_mask_sids', 'input_embeddings'):
        np.testing.assert_array_equal(pa[key], pb[key])
    assert not np.array_equal(pa['labels_sids'], pb['labels_sids'])


def test_custom_history_budget_sets_window_and_prefix_count():
    examples = expand_training_records([record(sequence=tuple(range(10)))], history_items=2)
    assert [e.sequence for e in examples] == [(7, 8), (7, 8, 9)]
    inputs, _ = pack_supervised_batch(examples, *assets(), history_items=2)
    assert inputs['input_sids'].shape == (2, 9)
    with pytest.raises(ValueError, match='window'):
        pack_supervised_batch(examples, *assets(), history_items=7)


@pytest.mark.parametrize('sequence', [(), (1,), (0, -1), (0, 1.5), (0, True),
                                     (0, 2 ** 63), tuple(range(1001))])
def test_invalid_sequences_rejected_before_expansion(sequence):
    with pytest.raises(ValueError, match='sequence'):
        expand_training_records([record(sequence=sequence)])


@pytest.mark.parametrize('uid,digest', [(True, 'a' * 64), (1.5, 'a' * 64),
                                      (2 ** 63, 'a' * 64), (1, 'not-a-sha')])
def test_invalid_parent_identity_is_rejected(uid, digest):
    with pytest.raises(ValueError):
        expand_training_records([record(uid, digest=digest)])


@pytest.mark.parametrize('mode', ['test', 'evaluation', 'all', None])
def test_only_explicit_training_supervision_modes_are_supported(mode):
    with pytest.raises(ValueError, match='mode'):
        expand_training_records([record()], mode=mode)


@pytest.mark.parametrize('history_items', [0, -1, True, 1.5])
def test_invalid_history_budget_rejected(history_items):
    with pytest.raises(ValueError, match='history_items'):
        expand_training_records([record()], history_items=history_items)


def test_duplicate_parent_users_are_rejected_but_distinct_prefixes_are_allowed():
    with pytest.raises(ValueError, match='Duplicate parent'):
        expand_training_records([record(), record(sequence=(1, 2), digest='b' * 64)])
    examples = expand_training_records([record()])
    pack_supervised_batch(examples, *assets())
    with pytest.raises(ValueError, match='Duplicate supervised'):
        pack_supervised_batch([examples[0], examples[0]], *assets())


@pytest.mark.parametrize('changes', [dict(parent_serialized_sha256='b' * 64),
                                     dict(sequence=(2, 1, 2))])
def test_conflicting_same_parent_provenance_or_content_is_rejected(changes):
    examples = expand_training_records([record()])
    examples[-1] = replace(examples[-1], **changes)
    with pytest.raises(ValueError, match='Conflicting'):
        pack_supervised_batch(examples, *assets())


@pytest.mark.parametrize('changes', [dict(is_final=False), dict(prefix_target_index=1),
                                     dict(window_start_index=1), dict(parent_prefix_count=3)])
def test_tampered_prefix_metadata_is_rejected(changes):
    example = expand_training_records([record()])[-1]
    with pytest.raises(ValueError):
        pack_supervised_batch([replace(example, **changes)], *assets())


def test_empty_inputs_unknown_weighting_and_asset_rows_are_rejected():
    with pytest.raises(ValueError, match='Nonempty'):
        expand_training_records([])
    with pytest.raises(ValueError, match='Nonempty'):
        pack_supervised_batch([], *assets())
    examples = expand_training_records([record()])
    with pytest.raises(ValueError, match='weighting'):
        pack_supervised_batch(examples, *assets(), weighting='unknown')
    with pytest.raises(ValueError, match='sequence'):
        pack_supervised_batch(expand_training_records([record(sequence=(0, 20))]), *assets())
