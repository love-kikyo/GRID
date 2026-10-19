import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.liger_data import Record, epoch_batches, pack_batch, record_stream, split_files


def record(uid):
    return Record(uid, (0, uid % 12), 'a' * 64)


def stream(entries):
    for entry in entries:
        yield from [record(i) for i in entry]


def batches(**kwargs):
    return list(epoch_batches([range(10), range(10, 17)], stream, batch_size=4,
                seed=42, epoch=kwargs.pop('epoch', 0), training=kwargs.pop('training', True),
                shuffle_buffer=5, **kwargs))


def test_epoch_has_every_record_once_with_partial_tail_and_reproducible_shuffle():
    first = batches()
    users = [r.user_id for _, batch in first for r in batch]
    assert sorted(users) == list(range(17)) and len(first[-1][1]) == 1
    assert first == batches() and first != batches(epoch=1)


def test_replay_resume_is_exact_suffix_and_checks_cursor():
    assert batches(start_batch=2) == batches()[2:]
    assert batches(start_batch=5) == []
    with pytest.raises(ValueError, match='cursor'):
        batches(start_batch=6)


def test_validation_never_shuffles_and_is_epoch_independent():
    assert batches(training=False) == batches(training=False, epoch=4)
    assert [r.user_id for _, batch in batches(training=False) for r in batch] == list(range(17))


def test_closing_batches_closes_the_source():
    closed = []
    def factory(entries):
        try:
            yield from stream(entries)
        finally:
            closed.append(True)
    iterator = epoch_batches([range(20)], factory, batch_size=2, seed=1, epoch=0,
                             training=True, shuffle_buffer=3)
    next(iterator)
    iterator.close()
    assert closed == [True]


def test_pack_keeps_user_ids_out_of_model_and_preserves_row_zero_and_target():
    sids = np.arange(48).reshape(12, 4)
    content = np.arange(60).reshape(12, 5).astype(np.int8)
    a, meta = pack_batch([Record(-3, tuple(range(12)), 'b' * 64), Record(8, (2, 0), 'c' * 64)], sids, content)
    assert set(a) == {'input_sids', 'attention_mask_sids', 'labels_sids', 'input_embeddings'}
    assert meta['target_rows'].tolist() == [11, 0] and meta['user_ids'].tolist() == [-3, 8]
    assert meta['history_rows'] == [list(range(4, 11)), [2]]
    with pytest.raises(ValueError, match='duplicate'):
        pack_batch([record(1), record(1)], sids, content)


def test_changing_gt_does_not_change_encoder_inputs():
    sids = np.arange(48).reshape(12, 4)
    content = np.arange(60).reshape(12, 5).astype(np.int8)
    a, _ = pack_batch([Record(7, (0, 2, 3), 'a' * 64)], sids, content)
    b, _ = pack_batch([Record(7, (0, 2, 4), 'b' * 64)], sids, content)
    for key in ('input_sids', 'attention_mask_sids', 'input_embeddings'):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a['labels_sids'], b['labels_sids'])


def fake_tf(raw):
    def feature(values):
        return SimpleNamespace(int64_list=SimpleNamespace(value=values), WhichOneof=lambda _: 'int64_list')
    def parse(data):
        fields = {key: feature(value) for key, value in json.loads(data).items()}
        return SimpleNamespace(features=SimpleNamespace(feature=fields))
    return SimpleNamespace(compat=SimpleNamespace(v1=SimpleNamespace(io=SimpleNamespace(
        tf_record_iterator=lambda path, options: iter([raw])))),
        train=SimpleNamespace(Example=SimpleNamespace(FromString=parse)))


@pytest.mark.parametrize('uid,seq', [([-1], [2, 0]), ([], [0, 1]), ([1, 2], [0, 1]),
                                     ([1], [0]), ([1], [0, 12]), ([1], [-1, 3])])
def test_reader_checks_exact_schema_identity_and_last_target(uid, seq):
    raw = json.dumps(dict(user_id=uid, sequence_data=seq)).encode()
    result = record_stream(fake_tf(raw), [dict(path='train')], 12)
    if uid == [-1]:
        assert list(result) == [Record(-1, (2, 0), hashlib.sha256(raw).hexdigest())]
    else:
        with pytest.raises(ValueError):
            list(result)


@pytest.mark.parametrize('split', ['train', 'evaluation', 'test'])
def test_split_selection_is_explicit_and_never_opens_test(tmp_path, monkeypatch, split):
    from scripts import verify_taobao_training_data as verify
    monkeypatch.setattr(verify, '_assert_protocol_manifest', lambda *args: None)
    monkeypatch.setattr(verify, '_assert_file_identity', lambda e: Path(e['path']))
    train = dict(path=str(tmp_path / 'training/a'))
    evaluation = dict(path=str(tmp_path / 'evaluation/a'))
    m = dict(outputs=dict(train=dict(files=[train], shards=1), evaluation=dict(files=[evaluation], shards=1)))
    if split == 'test':
        with pytest.raises(ValueError, match='Only'):
            split_files(m, tmp_path, split)
    else:
        assert split_files(m, tmp_path, split) == (train if split == 'train' else evaluation,)
        m['outputs'][split]['files'][0] = dict(path=str(tmp_path / 'testing/wrong'))
        with pytest.raises(ValueError, match='outside'):
            split_files(m, tmp_path, split)
