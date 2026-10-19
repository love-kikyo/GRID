"""Single-process LIGER streams over the frozen history-last train/validation.

No TensorFlow import or worker pool here. The compute-node caller must initialize
Torch before its CPU-only TensorFlow reader and bind the whole process to CPU4.
"""
from dataclasses import dataclass
import hashlib
import random

import numpy as np

from src.baselines.liger_inputs import pack_sequences


@dataclass(frozen=True)
class Record:
    user_id: int
    sequence: tuple
    serialized_sha256: str


def split_files(manifest, data_root, split):
    """Training entry points cannot open test by accidentally changing a glob."""
    from scripts.verify_taobao_training_data import _assert_protocol_manifest, _assert_file_identity
    if split not in {'train', 'evaluation'}:
        raise ValueError('Only train/evaluation are allowed in the training adapter')
    _assert_protocol_manifest(manifest, 'train', 'history-last-user-v1')
    output = manifest['outputs'][split]
    files = output['files']
    if not files or len(files) != output['shards'] or len({e['path'] for e in files}) != len(files):
        raise ValueError('Missing or duplicate frozen split files')
    directory = 'training' if split == 'train' else 'evaluation'
    for entry in files:
        if _assert_file_identity(entry).parent != (data_root / directory).resolve():
            raise ValueError('File is outside the selected split directory')
    return tuple(files)


def record_stream(tf, entries, asset_rows):
    if asset_rows < 1:
        raise ValueError('Positive asset row count required')
    for entry in entries:
        iterator = tf.compat.v1.io.tf_record_iterator(entry['path'], options='GZIP')
        try:
            for raw in iterator:
                fields = tf.train.Example.FromString(raw).features.feature
                if set(fields) != {'user_id', 'sequence_data'}:
                    raise ValueError('Unexpected TFRecord fields')
                if any(v.WhichOneof('kind') != 'int64_list' for v in fields.values()):
                    raise ValueError('Expected int64 fields')
                users = fields['user_id'].int64_list.value
                seq = tuple(fields['sequence_data'].int64_list.value)
                if len(users) != 1 or not 2 <= len(seq) <= 1000:
                    raise ValueError('Invalid user/history-last sequence')
                if min(seq) < 0 or max(seq) >= asset_rows:
                    raise ValueError('Sequence contains an unknown asset row')
                yield Record(int(users[0]), seq, hashlib.sha256(raw).hexdigest())
        finally:
            del iterator


def shuffled_records(records, rng, buffer_size):
    """Bounded-memory permutation: every input is emitted once, including tail."""
    if buffer_size < 1:
        raise ValueError('Positive shuffle buffer required')
    buffer = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
        else:
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = record
    rng.shuffle(buffer)
    yield from buffer


def epoch_batches(entries, stream_factory, *, batch_size, seed, epoch, training,
                  shuffle_buffer=8192, start_batch=0):
    """Deterministic replay resume, not random access into compressed shards.

    start_batch counts already consumed batches, including AMP-skipped updates.
    Resume re-reads/discards that prefix; checkpoints must also restore model,
    optimizer, scaler, scheduler and Torch RNG. This helper restores data only.
    """
    if batch_size < 1 or epoch < 0 or start_batch < 0 or seed < 0:
        raise ValueError('Invalid batch/seed/epoch/cursor')
    ordered = list(entries)
    if training:
        random.Random(f'{seed}:{epoch}:shards').shuffle(ordered)
    records = stream_factory(ordered)
    selected = shuffled_records(records, random.Random(f'{seed}:{epoch}:records'), shuffle_buffer) if training else records
    batch, index = [], 0
    try:
        for record in selected:
            batch.append(record)
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
            raise ValueError('Resume cursor exceeds the completed epoch')
    finally:
        for iterator in (selected, records):
            if hasattr(iterator, 'close'):
                iterator.close()


def pack_batch(records, semantic_ids, content, history_items=7):
    """Keep identity/GT row metadata out of the model's keyword dictionary."""
    if not records or len({r.user_id for r in records}) != len(records):
        raise ValueError('Empty batch or duplicate users')
    inputs = pack_sequences([r.sequence for r in records], semantic_ids, content,
                            history_items=history_items)
    metadata = dict(user_ids=np.asarray([r.user_id for r in records], dtype=np.int64),
                    serialized_sha256=[r.serialized_sha256 for r in records],
                    target_rows=inputs.pop('target_rows'), history_rows=inputs.pop('history_rows'))
    return inputs, metadata
