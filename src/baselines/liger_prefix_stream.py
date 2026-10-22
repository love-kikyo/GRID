"""Bounded-memory, train-only suffix-prefix streams over frozen parent records.

The caller supplies the existing ``record_stream(tf, train_entries, asset_rows)``
factory, after verifying the frozen training split. No TensorFlow is imported
here and validation/test adapters are unchanged. Global parent-user uniqueness
is an upstream data-audit invariant, not an in-memory set maintained here.
"""
import hashlib
import json
from numbers import Integral
import random

from src.baselines.liger_data import Record, epoch_batches, shuffled_records
from src.baselines.liger_supervision import expand_training_records


def _integer(value, name, minimum=1):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _close(iterator):
    if hasattr(iterator, 'close'):
        iterator.close()


def prefix_epoch_batches(entries, stream_factory, *, batch_size, seed, epoch,
                         shuffle_buffer=8192, start_batch=0, history_items=7,
                         prefix_shuffle_buffer=8192):
    """Yield ``(prefix_batch_index, list[PrefixExample])``; keep the final tail.

    Shards and parents use ``epoch_batches``' existing deterministic seed/epoch
    shuffle. Each parent is expanded only over its final history_items + 1
    items. A separate bounded prefix buffer uses seed ``seed:epoch:prefixes``;
    setting prefix_shuffle_buffer=1 preserves the parent-then-prefix order.

    Every input parent's selected prefixes appear once per complete epoch.
    Memory holds at most the two configured buffers, one expanded parent and
    one output batch, never the full corpus. Prefix identities retain parent
    UID/SHA, target index and is_final for exposure accounting outside the model.

    start_batch counts consumed *prefix* batches, including AMP-skipped updates.
    Resume replays from the beginning and drops that many complete batches, even
    if the boundary falls inside one parent. It restores no model/optimizer/RNG
    state; the trainer must restore those separately. Data and buffer settings
    must match the original run. A cursor at epoch end yields no batches; one
    past the end is rejected rather than silently starting another epoch.
    """
    for name, value in (('batch_size', batch_size), ('shuffle_buffer', shuffle_buffer),
                        ('history_items', history_items), ('prefix_shuffle_buffer', prefix_shuffle_buffer)):
        _integer(value, name)
    for name, value in (('seed', seed), ('epoch', epoch), ('start_batch', start_batch)):
        _integer(value, name, minimum=0)
    parents = epoch_batches(entries, stream_factory, batch_size=1, seed=seed,
                            epoch=epoch, training=True, shuffle_buffer=shuffle_buffer,
                            start_batch=0)

    def expanded_parents():
        try:
            for _, records in parents:
                yield from expand_training_records(records, history_items=history_items,
                                                   mode='all_prefix')
        finally:
            _close(parents)

    expanded = expanded_parents()
    selected = shuffled_records(expanded, random.Random(f'{seed}:{epoch}:prefixes'),
                                prefix_shuffle_buffer)
    batch, index = [], 0
    try:
        for example in selected:
            batch.append(example)
            if len(batch) == batch_size:
                if index >= start_batch:
                    yield index, batch
                index += 1
                batch = []
        if batch:
            if index >= start_batch:
                yield index, batch
            index += 1
        if start_batch > index:
            raise ValueError('Resume cursor exceeds the completed prefix epoch')
    finally:
        for iterator in (selected, expanded, parents):
            _close(iterator)


def census_records(records, history_items=7):
    """Count parent/prefix/final-target exposures with constant corpus memory.

    Counts refer to input records, not globally distinct UIDs. Supply prevalidated
    records from the existing record_stream in canonical shard/record order:
    that reader already checks the full record's schema and row bounds. Here
    only sequence length and the bounded suffix are checked again; no traversal
    of the full historical sequence or construction of all prefixes is performed.
    The histogram keys
    are retained-window item counts (including GT), while the prefix-history
    histogram counts supervised examples with that many preceding input items.

    The ordered SHA256 chain starts at SHA256("liger-prefix-census-v1\\0" +
    ASCII(history_items)). Each record extends it with canonical compact JSON
    [parent_uid, serialized_sha256, original_length, window_start, window_rows].
    The chain binds record order, provenance and the actual retained content;
    it is not a replacement for file hashes or the upstream uniqueness audit.
    """
    _integer(history_items, 'history_items')
    counts = dict(schema='liger.train_prefix_census.v1', history_items=int(history_items),
                  parents=0, prefixes=0, final_targets=0, original_sequence_items=0,
                  retained_window_items=0,
                  window_length_histogram={str(n): 0 for n in range(2, history_items + 2)},
                  prefix_history_length_histogram={str(n): 0 for n in range(1, history_items + 1)})
    chain = hashlib.sha256(b'liger-prefix-census-v1\0' + str(history_items).encode('ascii')).digest()
    iterator = iter(records)
    try:
        for record in iterator:
            original_length = len(record.sequence)
            if not 2 <= original_length <= 1000:
                raise ValueError('Invalid history-last row sequence length')
            window_start = max(0, original_length - history_items - 1)
            bounded = Record(record.user_id, tuple(record.sequence[window_start:]),
                             record.serialized_sha256)
            final = expand_training_records([bounded], history_items=history_items,
                                            mode='final_only')[0]
            length = len(final.sequence)
            counts['parents'] += 1
            counts['prefixes'] += length - 1
            counts['final_targets'] += 1
            counts['original_sequence_items'] += original_length
            counts['retained_window_items'] += length
            counts['window_length_histogram'][str(length)] += 1
            for prefix_history in range(1, length):
                counts['prefix_history_length_histogram'][str(prefix_history)] += 1
            payload = [final.parent_user_id, final.parent_serialized_sha256,
                       original_length, window_start, list(final.sequence)]
            chain = hashlib.sha256(chain + json.dumps(payload, separators=(',', ':'),
                                                      ensure_ascii=True).encode('ascii')).digest()
    finally:
        _close(iterator)
    counts['ordered_parent_chain_sha256'] = chain.hex()
    return counts
