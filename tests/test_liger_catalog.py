import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest

from src.baselines.liger_catalog import TrainCatalog, validate_row_sets, sid_identity_summary
from scripts import build_liger_train_catalog as builder


def test_sets_match_independent_python_union_and_do_not_treat_row_zero_as_padding():
    sequences = [list(range(13)), [0, 4], [2, 2, 2], [1, 6, 0]]
    c = TrainCatalog(20)
    for seq in sequences:
        c.update(seq)
    a = c.arrays()
    assert a['all_record_rows'].tolist() == sorted({x for seq in sequences for x in seq})
    assert a['input_target_rows'].tolist() == sorted({x for seq in sequences for x in seq[-8:]})
    assert a['target_rows'].tolist() == [0, 2, 4, 12]
    assert c.sequence_items == sum(map(len, sequences)) and c.records == len(sequences)
    assert c.consumed_items == sum(min(len(seq), 8) for seq in sequences)
    validate_row_sets(a, 20)


@pytest.mark.parametrize('seq', [[1], [], [-1, 3], [0, 20], [0.] * 2, [0] * 1001])
def test_invalid_rows_rejected(seq):
    with pytest.raises(ValueError):
        TrainCatalog(20).update(seq)


@pytest.mark.parametrize('name', ['target_rows', 'input_target_rows', 'all_record_rows'])
def test_invalid_row_sets_rejected(name):
    a = dict(target_rows=np.array([2], np.int32), input_target_rows=np.array([1, 2], np.int32),
             all_record_rows=np.array([0, 1, 2], np.int32))
    a[name] = np.array([10, 10], np.int32)
    with pytest.raises(AssertionError):
        validate_row_sets(a, 20)


def test_sid_alias_counts_keep_item_rows_distinct():
    sid = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 2]])
    assert sid_identity_summary(np.array([0, 1, 2]), sid) == dict(
        item_rows=3, unique_sid_tuples=2, duplicate_sid_groups=1,
        rows_in_duplicate_sid_groups=2, extra_rows_sharing_sid=1, maximum_rows_per_sid=2)


def test_training_iterator_never_uses_held_out_files(tmp_path, monkeypatch):
    from scripts import verify_taobao_training_data as verifier
    monkeypatch.setattr(verifier, '_assert_protocol_manifest', lambda m, s, p: None)
    (tmp_path / 'training').mkdir()
    path = tmp_path / 'training/train.tfrecord.gz'
    path.write_bytes(b'fixture')
    stat = path.stat()
    entry = dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    manifest = dict(outputs=dict(train=dict(files=[entry], shards=1),
        evaluation=dict(files=[dict(path='must-not-read')]), test=dict(files=[dict(path='must-not-read')])) )
    assert builder.training_files(manifest, tmp_path) == [entry]
    bad = copy.deepcopy(manifest)
    bad['outputs']['train']['files'][0]['path'] = str(tmp_path / 'nonexistent')
    with pytest.raises(AssertionError):
        builder.training_files(bad, tmp_path)


def test_launcher_has_one_cpu_no_gpu_and_no_training_or_successor():
    path = builder.ROOT / 'run_liger_train_catalog_cpu1.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    assert '#SBATCH --cpus-per-task=1' in text and '#SBATCH --gres=gpu:0' in text
    assert '--cpu-bind=verbose,threads' in text and 'LIGER_CATALOG_MANIFEST_SHA256' in text
    assert '-m src.train' not in text and 'sbatch ' not in text
    assert text.index('catalog run') < text.index('catalog accept')
