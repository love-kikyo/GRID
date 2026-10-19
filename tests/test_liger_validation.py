import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.probe_liger_cpu4 import write_new
from src.baselines.liger_candidates import SidRowIndex
from src.baselines.liger_retrieval import ranking_metrics
from src.baselines.liger_validation import PredictionWriter, add_metrics, audit


def fixture(tmp_path):
    ids = np.zeros((31, 4), dtype=np.int32)
    ids[:, -1] = np.arange(31)
    index = SidRowIndex.build(ids)
    rows = np.tile(np.arange(20), (3, 1)).astype(np.int32)
    scores = np.tile(np.arange(20, 0, -1), (3, 1)).astype(float)
    a = dict(user_ids=np.arange(3), target_rows=np.array([0, 7, 25]),
             record_sha256=np.array(['a' * 64, 'b' * 64, 'c' * 64], dtype='S64'),
             beam_tokens=ids[rows] + np.array([1, 4097, 8193, 12289]),
             raw_beam_rows=rows, valid_beam_rows=rows, hybrid_rows=rows, hybrid_scores=scores,
             beam_only_rows=rows, beam_only_scores=scores)
    path = tmp_path / 'validation'
    writer = PredictionWriter(path, shard_rows=2)
    writer.append({k: v[:2] for k, v in a.items()})
    writer.append({k: v[2:] for k, v in a.items()})
    writer.flush()
    metrics = {mode: ranking_metrics(rows, a['target_rows']) for mode in ('hybrid', 'beam_only')}
    report = dict(status='passed', records=3, cold_rows=30, checkpoint={'sha256': 'fixed'},
                  gt_independence_verified=True, test_records_read=False, cache_seconds=1., elapsed_seconds=2.,
                  artifacts=writer.artifacts, record_identity_sha256=hashlib.sha256(a['record_sha256'].tobytes()).hexdigest(),
                  valid_generated_slots=60, metrics=metrics)
    return path, report, index


def test_streamed_tail_independent_metrics_and_identity_audit(tmp_path):
    path, report, index = fixture(tmp_path)
    write_new(path / 'report.json', report)
    r = audit(path, np.array([0]), index, 3, 30, {'sha256': 'fixed'})
    assert r['status'] == 'passed' and r['records'] == 3
    assert [x['rows'] for x in report['artifacts']] == [2, 1]
    with pytest.raises(FileExistsError): PredictionWriter(path)


@pytest.mark.parametrize('corruption', ['metric', 'hash', 'count', 'test', 'checkpoint', 'identity'])
def test_audit_rejects_damaged_results(tmp_path, corruption):
    path, report, index = fixture(tmp_path)
    if corruption == 'metric': report['metrics']['hybrid']['Recall@5'] += .1
    if corruption == 'hash': report['artifacts'][0]['sha256'] = 'wrong'
    if corruption == 'count': report['records'] = 2
    if corruption == 'test': report['test_records_read'] = True
    if corruption == 'checkpoint': report['checkpoint'] = {}
    if corruption == 'identity': report['record_identity_sha256'] = 'wrong'
    write_new(path / 'report.json', report)
    with pytest.raises(AssertionError): audit(path, np.array([0]), index, 3, 30, {'sha256': 'fixed'})
    assert not (path / 'acceptance.json').exists()
