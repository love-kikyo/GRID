"""Inventory only frozen training records for LIGER catalog/loss planning.

Does not read held-out examples, select cold candidates, train, or evaluate.
Three named row sets keep full stored history distinct from consumed inputs.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.probe_liger_cpu4 import ROOT, sha, git, write_new, verify_hashes

PRIOR = ROOT / 'logs/training_launch/liger_probe_2026-10-12_v2'
PRIOR_ACCEPTANCE_SHA = 'c00bd6e50016e8c5d87fbc6191d28c2c43c8bfbbca3c42084a45a3180e28fcd6'
SOURCES = ['scripts/build_liger_train_catalog.py', 'src/baselines/liger_catalog.py',
           'tests/test_liger_catalog.py', 'run_liger_train_catalog_cpu1.sh',
           'scripts/probe_liger_cpu4.py', 'scripts/verify_taobao_training_data.py',
           'src/baselines/liger_inputs.py', 'src/runtime_cpu.py']


def training_files(manifest, data_root):
    """Never enumerate evaluation/test shards from the train-source manifest."""
    from scripts.verify_taobao_training_data import _assert_protocol_manifest, _assert_file_identity
    _assert_protocol_manifest(manifest, 'train', 'history-last-user-v1')
    entries = manifest['outputs']['train']['files']
    assert len(entries) == manifest['outputs']['train']['shards']
    for entry in entries:
        path = _assert_file_identity(entry)
        assert path.parent == (data_root / 'training').resolve(), path
    assert len({e['path'] for e in entries}) == len(entries)
    return entries


def prepare(path):
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    assert sha(PRIOR / 'acceptance.json') == PRIOR_ACCEPTANCE_SHA
    accepted = json.loads((PRIOR / 'acceptance.json').read_text())
    assert accepted['status'] == 'passed'
    assert sha(PRIOR / 'manifest.json') == accepted['manifest_sha256']
    assert sha(PRIOR / 'probe.json') == accepted['report_sha256']
    previous = json.loads((PRIOR / 'manifest.json').read_text())
    verify_hashes(previous['upstream_files'])
    verify_hashes(previous['input_metadata'])
    root = Path(previous['data_root'])
    train = json.loads((root / 'protocol_manifest_train.json').read_text())
    entries = training_files(train, root)
    audit = json.loads((root / 'protocol_audit.json').read_text())
    assert audit['status'] == 'passed' and not audit['partial_run']
    expected = audit['destinations']['train']
    assert train['outputs']['train']['samples'] == expected['records'] == 6573674
    manifest = dict(scope='train-only catalog inventory, not formal training',
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        prior_acceptance=dict(path=str(PRIOR / 'acceptance.json'), sha256=PRIOR_ACCEPTANCE_SHA),
        upstream_commit=previous['upstream_commit'], upstream_files=previous['upstream_files'],
        assets=previous['assets'], input_metadata=previous['input_metadata'], data_root=str(root),
        train_files=entries, expected_records=expected['records'],
        expected_sequence_items=expected['sequence_items'], history_items=7,
        execution_sources={str(ROOT / p): sha(ROOT / p) for p in SOURCES},
        held_out_records_read=False, formal_training_ready=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, manifest)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=manifest['source_git_commit'])), flush=True)


def binding(path):
    assert sha(path) == os.environ['LIGER_CATALOG_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and not m['dirty_at_submission']
    for key in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[key])
    assert sha(m['prior_acceptance']['path']) == m['prior_acceptance']['sha256'] == PRIOR_ACCEPTANCE_SHA
    train = json.loads((Path(m['data_root']) / 'protocol_manifest_train.json').read_text())
    assert training_files(train, Path(m['data_root'])) == m['train_files']
    for entry in m['assets'].values():
        stat = Path(entry['path']).stat()
        assert stat.st_size == entry['size_bytes'] and stat.st_mtime_ns == entry['mtime_ns']
    return m


def cpu_guard():
    from src.runtime_cpu import THREAD_VARIABLES, process_tree_snapshot
    assert os.environ['SLURM_NTASKS'] == os.environ['SLURM_CPUS_PER_TASK'] == '1'
    assert os.environ['SLURM_JOB_CPUS_PER_NODE'] in {'1', '1(x1)'}
    assert len(os.sched_getaffinity(0)) == 1
    assert all(os.environ.get(key) == '1' for key in THREAD_VARIABLES)
    return process_tree_snapshot()


def run(path):
    cpu_guard()
    m = binding(path)
    output = path.parent
    assert not any((output / name).exists() for name in (
        'catalog_report.json', 'catalog_acceptance.json', 'all_record_rows.npy',
        'input_target_rows.npy', 'target_rows.npy'))
    # Match the accepted GRID probe's runtime initialization order.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import numpy as np
    from src.baselines.liger_catalog import TrainCatalog, validate_row_sets, sid_identity_summary
    from src.baselines.liger_inputs import catalog_memory_lower_bounds
    sid_asset, content_asset = m['assets']['frozen_sid'], m['assets']['model_embedding_copy']
    assert sha(sid_asset['path']) == sid_asset['sha256']
    sids = np.load(sid_asset['path'], mmap_mode='r', allow_pickle=False)
    content = np.load(content_asset['path'], mmap_mode='r', allow_pickle=False)
    assert sids.shape == (35458467, 4) and content.shape == (35458467, 128)
    catalog = TrainCatalog(len(sids), history_items=m['history_items'])
    users = np.empty(m['expected_records'], dtype=np.int64)
    files, snapshots = [], [cpu_guard()]
    started = time.monotonic()
    print('[liger-catalog] start 33 frozen training shards; no held-out records', flush=True)
    for index, entry in enumerate(m['train_files']):
        digest, before = hashlib.sha256(), catalog.records
        iterator = tf.compat.v1.io.tf_record_iterator(entry['path'], options='GZIP')
        for raw in iterator:
            digest.update(raw)
            features = tf.train.Example.FromString(raw).features.feature
            assert set(features) == {'user_id', 'sequence_data'}
            assert features['sequence_data'].WhichOneof('kind') == 'int64_list'
            user = features['user_id'].int64_list.value
            assert len(user) == 1 and catalog.records < len(users)
            users[catalog.records] = user[0]
            rows = np.fromiter(features['sequence_data'].int64_list.value, dtype=np.int64)
            catalog.update(rows)
            if catalog.records % 100000 == 0:
                elapsed = time.monotonic() - started
                print(f'[liger-catalog] records={catalog.records}/{len(users)} elapsed_s={elapsed:.1f}', flush=True)
        del iterator
        files.append(dict(path=entry['path'], records=catalog.records - before,
                          serialized_records_sha256=digest.hexdigest()))
        snapshots.append(cpu_guard())
        print(f'[liger-catalog] shard={index + 1}/{len(m["train_files"])} complete', flush=True)
    assert catalog.records == len(users) == len(np.unique(users))
    assert catalog.sequence_items == m['expected_sequence_items']
    del users
    arrays = catalog.arrays()
    validate_row_sets(arrays, len(sids))
    identity = sid_identity_summary(arrays['input_target_rows'], sids)
    artifacts = {}
    for name, rows in arrays.items():
        target = output / (name + '.npy')
        with target.open('xb') as f:
            np.save(f, rows, allow_pickle=False)
        artifacts[name] = dict(path=str(target), sha256=sha(target), rows=len(rows), dtype=str(rows.dtype))
    binding(path)  # Detect source/file identity changes while scanning.
    snapshots.append(cpu_guard())
    report = dict(status='passed', phase='train_catalog_inventory_only',
        checked_at=datetime.now().astimezone().isoformat(), job_id=os.environ['SLURM_JOB_ID'],
        source_git_commit=m['source_git_commit'], manifest=dict(path=str(path), sha256=sha(path)),
        records=catalog.records, unique_users=catalog.records, sequence_items=catalog.sequence_items,
        consumed_input_plus_target_occurrences=catalog.consumed_items, history_items=7,
        asset_rows=len(sids), artifacts=artifacts, input_target_sid_identity=identity,
        definitions=dict(all_record_rows='All items in stored training sequences, including last GT',
            input_target_rows='Union of last seven pre-GT history items and last GT from training only',
            target_rows='Last GT item rows from training only'),
        full_record_rows_outside_consumed_set=len(arrays['all_record_rows']) - len(arrays['input_target_rows']),
        asset_rows_outside_consumed_train_set=len(sids) - len(arrays['input_target_rows']),
        held_out_records_read=False, candidate_universe_finalized=False, formal_training_ready=False,
        per_shard=files, cpu_snapshots=snapshots, elapsed_seconds=time.monotonic() - started,
        catalog_buffer_lower_bounds=catalog_memory_lower_bounds(len(arrays['input_target_rows'])),
        asset_verification='SID fully rehashed; content header/stat checked against prior audited asset (content values unused)',
        remaining_gates=['seen/cold candidate policy', 'bounded exact full-catalog loss with gradients',
                         'formal adapter, item identity and real-scale smoke'])
    write_new(output / 'catalog_report.json', report)
    print('[liger-catalog] ' + json.dumps({k: report[k] for k in (
        'status', 'records', 'artifacts', 'input_target_sid_identity', 'elapsed_seconds')}), flush=True)


def accept(path):
    cpu_guard()
    m = binding(path)
    import numpy as np
    from src.baselines.liger_catalog import validate_row_sets, sid_identity_summary
    report_path = path.parent / 'catalog_report.json'
    r = json.loads(report_path.read_text())
    assert r['status'] == 'passed' and r['phase'] == 'train_catalog_inventory_only'
    assert r['source_git_commit'] == m['source_git_commit'] and r['job_id'] == os.environ['SLURM_JOB_ID']
    assert r['records'] == r['unique_users'] == m['expected_records']
    assert r['sequence_items'] == m['expected_sequence_items']
    assert r['history_items'] == m['history_items'] == 7
    assert r['manifest'] == dict(path=str(path), sha256=sha(path))
    assert r['held_out_records_read'] is False and r['formal_training_ready'] is False
    assert r['candidate_universe_finalized'] is False
    assert [f['path'] for f in r['per_shard']] == [f['path'] for f in m['train_files']]
    assert sum(f['records'] for f in r['per_shard']) == r['records']
    arrays = {}
    for name, entry in r['artifacts'].items():
        assert sha(entry['path']) == entry['sha256']
        arrays[name] = np.load(entry['path'], mmap_mode='r', allow_pickle=False)
        assert len(arrays[name]) == entry['rows']
    validate_row_sets(arrays, r['asset_rows'])
    sids = np.load(m['assets']['frozen_sid']['path'], mmap_mode='r', allow_pickle=False)
    assert sid_identity_summary(arrays['input_target_rows'], sids) == r['input_target_sid_identity']
    assert all(len(s['allowed_cpus']) == 1 and s['processes'] for s in r['cpu_snapshots'])
    acceptance = dict(status='passed', checked_at=datetime.now().astimezone().isoformat(),
        job_id=r['job_id'], source_git_commit=m['source_git_commit'], records=r['records'],
        report=dict(path=str(report_path), sha256=sha(report_path)), artifacts=r['artifacts'],
        sorted_unique_row_sets_and_inclusions=True, sid_identity_recomputed=True,
        held_out_records_read=False, formal_training_ready=False,
        audit_scope='Output/hash/inclusion/SID identity audit; does not rescan TFRecords',
        audit_script_sha256=sha(Path(__file__)))
    write_new(path.parent / 'catalog_acceptance.json', acceptance)
    print('[liger-catalog-acceptance] ' + json.dumps(acceptance), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'run', 'accept'])
    p.add_argument('manifest', type=Path)
    args = p.parse_args()
    {'prepare': prepare, 'run': run, 'accept': accept}[args.mode](args.manifest.resolve())
