import numpy as np
import pytest

from scripts.audit_taobao_split_boundary import compare_sequences, measure_boundaries


def mapping():
    return np.arange(80, dtype=np.int32).reshape(20, 4)


def test_exact_duplicate_and_changed_target_are_distinct():
    sid = mapping()
    same = compare_sequences(np.array([1, 2, 3]), np.array([1, 2, 3]), sid, 3)
    assert same['full_sequence_equal'] and same['generator_sample_equal_sid']
    changed = compare_sequences(np.array([1, 2, 3]), np.array([1, 2, 4]), sid, 3)
    assert changed['full_input_history_equal'] and changed['generator_input_equal_sid']
    assert not changed['target_item_equal'] and not changed['generator_sample_equal_sid']


def test_cropped_duplicate_is_detected_without_full_duplicate():
    result = compare_sequences(np.array([1, 2, 3, 4]), np.array([5, 2, 3, 4]), mapping(), 3)
    assert not result['full_sequence_equal']
    assert result['generator_sample_equal_item'] and result['generator_sample_equal_sid']


def test_sid_collision_and_short_sequence_padding_boundaries():
    sid = mapping()
    sid[5] = sid[1]
    result = compare_sequences(np.array([1, 2]), np.array([5, 2]), sid, 3)
    assert not result['generator_sample_equal_item'] and result['generator_sample_equal_sid']
    result = compare_sequences(np.array([1, 2]), np.array([3, 1, 2]), sid, 3)
    assert not result['generator_sample_equal_sid']


def test_overlap_counts_distinguish_validation_and_test_new_users():
    report = measure_boundaries([(1, [1, 2]), (2, [2, 3]), (3, [3, 4])],
                                [(1, [1, 2]), (4, [3, 4])], [(2, [0, 2, 3])], mapping(), 2)
    assert report['counts'] == {'test': 3, 'train': 2, 'evaluation': 1}
    assert report['train_validation_overlap_users'] == 0
    train, val = report['pairs']['train_test'], report['pairs']['evaluation_test']
    assert train['overlap_users'] == val['overlap_users'] == 1
    assert train['comparisons_within_overlapping_users']['full_sequence_equal'] == 1
    assert val['comparisons_within_overlapping_users']['full_sequence_equal'] == 0
    assert val['comparisons_within_overlapping_users']['generator_sample_equal_sid'] == 1


@pytest.mark.parametrize('test,train,val', [
    ([(1, [1, 2]), (1, [1, 2])], [], []),
    ([], [(1, [1, 2]), (1, [1, 2])], []),
    ([], [(1, [1, 2])], [(1, [1, 2])]),
    ([], [], [(1, [1, 2]), (1, [1, 2])]),
    ([(1, [-1, 2])], [], []), ([(1, [1, 999])], [], []), ([(1, [1])], [], []),
])
def test_invalid_or_duplicate_records_are_rejected(test, train, val):
    with pytest.raises(ValueError):
        measure_boundaries(test, train, val, mapping())


def test_empty_overlap_has_explicit_zero_counts():
    result = measure_boundaries([(1, [1, 2])], [(2, [1, 2])], [], mapping())
    metrics = result['pairs']['train_test']['comparisons_within_overlapping_users']
    assert result['pairs']['train_test']['overlap_users'] == 0
    assert all(value == 0 for value in metrics.values())


@pytest.mark.parametrize('limit,expected', [(None, 2), (1, 1)])
def test_tfrecord_reader_preserves_ragged_sequences(tmp_path, limit, expected):
    import tensorflow as tf
    from scripts.audit_taobao_split_boundary import iter_records
    path = tmp_path / 'tiny.tfrecord.gz'
    original = [(0, [0, 1]), (2**40, [2, 3, 4])]
    with tf.io.TFRecordWriter(str(path), options='GZIP') as writer:
        for user, sequence in original:
            example = tf.train.Example(features=tf.train.Features(feature={
                'user_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[user])),
                'sequence_data': tf.train.Feature(int64_list=tf.train.Int64List(value=sequence)),
            }))
            writer.write(example.SerializeToString())
    actual = list(iter_records([path], 'fixture', batch_size=2, max_records=limit))
    assert [(user, seq.tolist()) for user, seq in actual] == original[:expected]
