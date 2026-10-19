"""Build a global item-identity index, without choosing an inference universe."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from scripts.probe_liger_cpu4 import ROOT, sha, git, write_new, verify_hashes
from scripts.build_liger_train_catalog import cpu_guard
from src.baselines.liger_candidates import SidRowIndex

PRIOR = ROOT / 'logs/training_launch/liger_batch_2026-10-19_v1'
PRIOR_SHA = 'cc176d687e3dacace6681dddd1af17f0c8208541a88d5be3e6e980b0191a956f'
SOURCES = ['scripts/build_liger_sid_index.py', 'src/baselines/liger_candidates.py',
           'tests/test_liger_candidates.py', 'tests/test_liger_sid_index.py',
           'run_liger_sid_index_cpu1.sh', 'scripts/probe_liger_cpu4.py',
           'scripts/build_liger_train_catalog.py', 'src/runtime_cpu.py']


def prepare(path):
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    receipt = PRIOR / 'batch_acceptance.json'
    assert sha(receipt) == PRIOR_SHA
    a = json.loads(receipt.read_text())
    assert a['status'] == 'passed' and sha(a['report']['path']) == a['report']['sha256']
    r = json.loads(Path(a['report']['path']).read_text())
    assert sha(r['manifest']['path']) == r['manifest']['sha256']
    previous = json.loads(Path(r['manifest']['path']).read_text())
    inputs = {str(p): sha(p) for p in (receipt, Path(a['report']['path']), Path(r['manifest']['path']))}
    m = dict(phase='global_sid_identity_only', source_git_commit=git(ROOT, 'rev-parse', 'HEAD'),
        dirty_at_submission=False, input_metadata=inputs, asset=previous['assets']['frozen_sid'],
        asset_rows=previous['asset_rows'], chunk_size=262144, candidate_universe_finalized=False,
        formal_training_ready=False, held_out_records_read=False,
        execution_sources={str(ROOT / name): sha(ROOT / name) for name in SOURCES})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])), flush=True)


def binding(path):
    assert sha(path) == os.environ['LIGER_INDEX_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and not m['dirty_at_submission']
    for key in ('input_metadata', 'execution_sources'):
        verify_hashes(m[key])
    stat = Path(m['asset']['path']).stat()
    assert stat.st_size == m['asset']['size_bytes'] and stat.st_mtime_ns == m['asset']['mtime_ns']
    return m


def verify_index(index, semantic_ids, chunk_size):
    """Independently re-encode every sorted row using base-4096 arithmetic."""
    assert len(index.keys) == len(semantic_ids)
    for start in range(0, len(index.keys), chunk_size):
        stop = min(start + chunk_size, len(index.keys))
        codes = semantic_ids[index.rows[start:stop]].astype(np.uint64)
        expected = (codes * np.array([4096**3, 4096**2, 4096, 1], dtype=np.uint64)).sum(axis=1)
        np.testing.assert_array_equal(index.keys[start:stop], expected)
    assert index.lookup_codes(semantic_ids[0:1]).tolist() == [0]
    return dict(rows_checked=len(index.keys), sid_aliases=0, row_zero_preserved=True)


def run(path):
    initial = cpu_guard()
    m = binding(path)
    output = path.parent / 'index_report.json'
    assert not any((path.parent / name).exists() for name in ('index_report.json', 'sid_keys.npy', 'asset_rows.npy'))
    report = dict(status='failed', phase=m['phase'], job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=m['source_git_commit'], manifest=dict(path=str(path), sha256=sha(path)),
        candidate_universe_finalized=False, formal_training_ready=False, held_out_records_read=False,
        training_records_read=False, cpu_snapshots=[initial])
    started = time.monotonic()
    try:
        assert sha(m['asset']['path']) == m['asset']['sha256']
        sids = np.load(m['asset']['path'], mmap_mode='r', allow_pickle=False)
        assert sids.shape == (m['asset_rows'], 4)
        print(f'[liger-index] sorting {len(sids)} asset identities on one CPU', flush=True)
        index = SidRowIndex.build(sids, chunk_size=m['chunk_size'])
        report['identity'] = verify_index(index, sids, m['chunk_size'])
        artifacts = {}
        for name, array in (('sid_keys', index.keys), ('asset_rows', index.rows)):
            target = path.parent / f'{name}.npy'
            with target.open('xb') as handle:
                np.save(handle, array, allow_pickle=False)
            artifacts[name] = dict(path=str(target), sha256=sha(target), dtype=str(array.dtype),
                                   rows=len(array), size_bytes=target.stat().st_size)
        report['artifacts'] = artifacts
        report['cpu_snapshots'].append(cpu_guard())
        binding(path)
        report['status'] = 'passed'
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report.update(checked_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started)
        write_new(output, report)
        print('[liger-index] ' + json.dumps(dict(status=report['status'], report=str(output))), flush=True)


def accept(path):
    cpu_guard()
    m = binding(path)
    report_path = path.parent / 'index_report.json'
    r = json.loads(report_path.read_text())
    assert r['status'] == 'passed' and r['phase'] == 'global_sid_identity_only'
    assert r['source_git_commit'] == m['source_git_commit'] and r['job_id'] == os.environ['SLURM_JOB_ID']
    assert r['manifest'] == dict(path=str(path), sha256=sha(path))
    assert all(r[k] is False for k in ('candidate_universe_finalized', 'formal_training_ready',
                                      'held_out_records_read', 'training_records_read'))
    assert set(r['artifacts']) == {'sid_keys', 'asset_rows'}
    arrays = {}
    for name, a in r['artifacts'].items():
        assert Path(a['path']) == path.parent / f'{name}.npy' and sha(a['path']) == a['sha256']
        assert Path(a['path']).stat().st_size == a['size_bytes']
        arrays[name] = np.load(a['path'], mmap_mode='r', allow_pickle=False)
        assert len(arrays[name]) == a['rows'] == m['asset_rows'] and str(arrays[name].dtype) == a['dtype']
    index = SidRowIndex(arrays['sid_keys'], arrays['asset_rows'])
    # Rehash the source in this independent process before all-row revalidation.
    assert sha(m['asset']['path']) == m['asset']['sha256']
    sids = np.load(m['asset']['path'], mmap_mode='r', allow_pickle=False)
    identity = verify_index(index, sids, m['chunk_size'])
    assert identity == r['identity']
    assert len(r['cpu_snapshots']) == 2
    allowed = r['cpu_snapshots'][0]['allowed_cpus']
    assert len(allowed) == 1
    for snapshot in r['cpu_snapshots']:
        assert snapshot['allowed_cpus'] == allowed and snapshot['processes']
        assert all(p['threads'] > 0 for p in snapshot['processes'])
    snapshot = cpu_guard()
    binding(path)
    write_new(path.parent / 'index_acceptance.json', dict(status='passed', job_id=r['job_id'],
        checked_at=datetime.now().astimezone().isoformat(), source_git_commit=m['source_git_commit'],
        report=dict(path=str(report_path), sha256=sha(report_path)), identity=identity,
        cpu_snapshot=snapshot, candidate_universe_finalized=False, formal_training_ready=False))
    print('[liger-index] independent all-row acceptance passed ' + json.dumps(identity), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    globals()[args.mode](args.manifest.resolve())
