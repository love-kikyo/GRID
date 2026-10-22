from collections import Counter
from dataclasses import replace
import hashlib
import json

import pytest

from src.baselines.liger_data import Record, epoch_batches
from src.baselines.liger_prefix_stream import census_records, prefix_epoch_batches
from src.baselines.liger_supervision import expand_training_records
from src.baselines import liger_prefix_stream as prefix_stream


def record(uid, sequence):
    return Record(uid, tuple(sequence), f'{uid:064x}')


def fixture():
    return [record(1, (2, 0)), record(2, (1, 2, 3)),
            record(3, range(20)), record(4, (2, 4, 2, 4))]


def stream(entries):
    for records in entries:
        yield from records


def batches(**kwargs):
    records = fixture()
    options = dict(batch_size=5, seed=42, epoch=0, shuffle_buffer=3,
                   prefix_shuffle_buffer=4)
    options.update(kwargs)
    return list(prefix_epoch_batches([records[:2], records[2:]], stream, **options))


def identity(example):
    return (example.parent_user_id, example.parent_serialized_sha256,
            example.prefix_target_index, example.sequence)


def test_every_prefix_once_with_short_long_repeated_items_and_partial_tail():
    result = batches()
    actual = [e for _, batch in result for e in batch]
    expected = expand_training_records(fixture())
    assert Counter(map(identity, actual)) == Counter(map(identity, expected))
    assert [index for index, _ in result] == [0, 1, 2]
    assert [len(batch) for _, batch in result] == [5, 5, 3]
    assert sum(e.is_final for e in actual) == 4
    assert all(len(e.sequence) <= 8 for e in actual)
    assert all(e.sequence[0] == 12 for e in actual if e.parent_user_id == 3)
    assert (2, 4, 2, 4) in [e.sequence for e in actual if e.parent_user_id == 4]
    for parent in fixture():
        final = [e for e in actual if e.parent_user_id == parent.user_id and e.is_final]
        assert len(final) == 1 and final[0].sequence[-1] == parent.sequence[-1]


def test_same_seed_epoch_is_exact_and_new_epoch_or_seed_reorders_without_omission():
    original = batches()
    assert original == batches()
    for altered in (batches(epoch=1), batches(seed=43)):
        assert altered != original
        assert Counter(identity(e) for _, b in altered for e in b) == Counter(
            identity(e) for _, b in original for e in b)


def test_prefix_buffer_one_preserves_the_existing_parent_shuffle_order():
    records = fixture()
    entries = [records[:2], records[2:]]
    parent_batches = epoch_batches(entries, stream, batch_size=1, seed=42,
                                   epoch=2, training=True, shuffle_buffer=3)
    expected = [e for _, parents in parent_batches for e in expand_training_records(parents)]
    actual = list(prefix_epoch_batches(entries, stream, batch_size=5, seed=42,
                 epoch=2, shuffle_buffer=3, prefix_shuffle_buffer=1))
    assert [e for _, batch in actual for e in batch] == expected


def test_batch_can_cross_parents_and_resume_exactly_inside_one_parent():
    records = [record(1, range(8)), record(2, (8, 9))]
    options = dict(batch_size=3, seed=42, epoch=0, shuffle_buffer=1,
                   prefix_shuffle_buffer=1)
    complete = list(prefix_epoch_batches([records], stream, **options))
    assert [e.parent_user_id for e in complete[0][1]] == [1, 1, 1]
    assert [e.prefix_target_index for e in complete[1][1]] == [4, 5, 6]
    assert [e.parent_user_id for e in complete[2][1]] == [1, 2]
    resumed = list(prefix_epoch_batches([records], stream, start_batch=1, **options))
    assert resumed == complete[1:]


def test_every_resume_cursor_including_tail_end_and_skipped_amp_batches():
    original = batches()
    for consumed in range(len(original) + 1):
        assert batches(start_batch=consumed) == original[consumed:]
    # An AMP skip still consumes the second input batch: resume uses 2, not 1.
    successful_updates, consumed_batches = 1, 2
    assert successful_updates != consumed_batches
    assert batches(start_batch=consumed_batches) == original[2:]
    with pytest.raises(ValueError, match='cursor'):
        batches(start_batch=len(original) + 1)


def test_custom_history_budget_limits_expansion_and_census():
    result = batches(history_items=2)
    examples = [e for _, batch in result for e in batch]
    assert len(examples) == 7 and all(len(e.sequence) <= 3 for e in examples)
    assert sum(e.is_final for e in examples) == 4
    counts = census_records(fixture(), history_items=2)
    assert counts['window_length_histogram'] == {'2': 1, '3': 3}
    assert counts['prefixes'] == len(examples)


def test_early_close_is_lazy_bounded_and_closes_source():
    consumed, closed = [], []
    def factory(entries):
        try:
            for uid in range(100000):
                consumed.append(uid)
                yield record(uid, (0, uid + 1))
        finally:
            closed.append(True)
    iterator = prefix_epoch_batches(['one-shard'], factory, batch_size=3, seed=42,
                  epoch=0, shuffle_buffer=4, prefix_shuffle_buffer=5)
    next(iterator)
    assert 3 <= len(consumed) <= 4 + 5 + 3 + 3
    assert not closed
    iterator.close()
    assert closed == [True]


def test_failure_during_expansion_closes_source_and_never_silently_drops_parent():
    closed = []
    def factory(entries):
        try:
            yield record(1, (0, 1))
            yield record(2, (0,))
        finally:
            closed.append(True)
    with pytest.raises(ValueError, match='sequence'):
        list(prefix_epoch_batches(['one-shard'], factory, batch_size=1, seed=42,
             epoch=0, shuffle_buffer=1, prefix_shuffle_buffer=1))
    assert closed == [True]


def test_census_counts_match_expanded_stream_and_exact_identity_chain():
    records = fixture()
    counts = census_records(iter(records))
    assert counts['parents'] == counts['final_targets'] == 4
    assert counts['prefixes'] == 13
    assert counts['original_sequence_items'] == 29
    assert counts['retained_window_items'] == 17
    assert counts['window_length_histogram'] == {'2': 1, '3': 1, '4': 1, '5': 0, '6': 0, '7': 0, '8': 1}
    assert counts['prefix_history_length_histogram'] == {'1': 4, '2': 3, '3': 2, '4': 1, '5': 1, '6': 1, '7': 1}
    assert sum(counts['prefix_history_length_histogram'].values()) == counts['prefixes']
    chain = hashlib.sha256(b'liger-prefix-census-v1\x007').digest()
    for parent in records:
        payload = [parent.user_id, parent.serialized_sha256, len(parent.sequence),
                   max(0, len(parent.sequence) - 8), list(parent.sequence[-8:])]
        chain = hashlib.sha256(chain + json.dumps(payload, separators=(',', ':')).encode('ascii')).digest()
    assert counts['ordered_parent_chain_sha256'] == chain.hex()
    assert counts == census_records(records)
    assert counts['ordered_parent_chain_sha256'] != census_records(list(reversed(records)))['ordered_parent_chain_sha256']
    changed = records.copy()
    changed[0] = replace(changed[0], serialized_sha256='f' * 64)
    assert counts['ordered_parent_chain_sha256'] != census_records(changed)['ordered_parent_chain_sha256']


def test_census_closes_source_on_success_and_error():
    for invalid in (False, True):
        closed = []
        def source():
            try:
                yield record(1, (0,) if invalid else (0, 1))
            finally:
                closed.append(True)
        if invalid:
            with pytest.raises(ValueError, match='sequence'):
                census_records(source())
        else:
            assert census_records(source())['parents'] == 1
        assert closed == [True]


def test_census_never_iterates_full_history_and_only_packs_one_bounded_final(monkeypatch):
    class PrevalidatedLongSequence:
        def __len__(self):
            return 1000

        def __iter__(self):
            raise AssertionError('Full-history iteration is forbidden during census')

        def __getitem__(self, key):
            assert key == slice(992, None, None)
            return tuple(range(992, 1000))

    observed = []
    expand = prefix_stream.expand_training_records
    def observed_expand(records, **kwargs):
        observed.append((len(records), len(records[0].sequence), kwargs['mode']))
        return expand(records, **kwargs)
    monkeypatch.setattr(prefix_stream, 'expand_training_records', observed_expand)
    parent = Record(1, PrevalidatedLongSequence(), 'a' * 64)
    counts = census_records([parent])
    assert observed == [(1, 8, 'final_only')]
    assert counts['original_sequence_items'] == 1000 and counts['retained_window_items'] == 8
    assert counts['prefixes'] == 7 and counts['final_targets'] == 1


def test_empty_stream_is_counted_and_resume_beyond_empty_epoch_is_rejected():
    assert list(prefix_epoch_batches([], stream, batch_size=3, seed=42, epoch=0)) == []
    with pytest.raises(ValueError, match='cursor'):
        list(prefix_epoch_batches([], stream, batch_size=3, seed=42, epoch=0, start_batch=1))
    counts = census_records([])
    assert counts['parents'] == counts['prefixes'] == counts['final_targets'] == 0
    assert sum(counts['window_length_histogram'].values()) == 0


@pytest.mark.parametrize('name,value', [('batch_size', 0), ('batch_size', True),
    ('shuffle_buffer', 0), ('prefix_shuffle_buffer', 0), ('history_items', 0),
    ('start_batch', -1), ('start_batch', 1.5), ('epoch', -1), ('seed', -1)])
def test_invalid_stream_arguments_rejected(name, value):
    with pytest.raises(ValueError):
        batches(**{name: value})


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_census_history_budget_rejected(value):
    with pytest.raises(ValueError):
        census_records(fixture(), history_items=value)
