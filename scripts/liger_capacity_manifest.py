"""Create-only authority for 384-wide resource probes, never formal training."""
import argparse
import copy
import json
import os
from pathlib import Path

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new

PRIOR = ROOT / 'logs/training_launch/liger_sampled_train_seed42_2026-11-12_v1/manifest.json'
PRIOR_SHA = '48729be8c73243f126dd72c99270fa47bb50499f9fd72c46d47272313e7ff811'
RECEIPT = ROOT / 'logs/train/runs/liger_sampled_train_seed42_2026-11-12_v1/acceptance.json'
CONFIG = ROOT / 'configs/baselines/liger_capacity384_probe_v1.json'
SOURCES = [
    'src/baselines/liger_capacity.py', 'src/baselines/liger_logical_batch.py',
    'src/baselines/liger_sharded_retrieval.py', 'scripts/liger_capacity_manifest.py',
    'scripts/liger_capacity_runtime.py', 'scripts/probe_liger_capacity.py',
    'scripts/probe_liger_capacity_validation.py',
    'tests/test_liger_capacity.py', 'tests/test_liger_logical_batch.py',
    'tests/test_liger_sharded_retrieval.py', 'tests/test_liger_capacity_probe.py',
    'tests/test_liger_capacity_runtime.py', 'tests/test_liger_capacity_validation_probe.py',
    'run_liger_capacity_train_cpu4.sh', 'run_liger_capacity_val_cpu4.sh',
]
REMOVED_KEYS = frozenset({'smoke_acceptance', 'expected_initial_state_sha256',
                          'run_directory', 'stage'})
CHANGED_KEYS = frozenset({'phase', 'formal_training_ready', 'automatic_successor',
    'test_records_read', 'initialization', 'source_git_commit', 'dirty_at_submission',
    'predecessor', 'predecessor_acceptance', 'capacity_config', 'execution_sources',
    'input_metadata'})


def verify(m):
    from src.baselines.liger_capacity import validate_probe_config
    assert sha(PRIOR) == PRIOR_SHA
    old = json.loads(PRIOR.read_text())
    assert set(m) == (set(old) - REMOVED_KEYS) | CHANGED_KEYS
    assert m['phase'] == 'capacity384_resource_only'
    assert all(m[k] is False for k in ('formal_training_ready', 'automatic_successor',
                                       'test_records_read', 'dirty_at_submission'))
    assert m['initialization'] == 'fresh seed42 capacity384 diagnostic weights only'
    assert isinstance(m['source_git_commit'], str) and len(m['source_git_commit']) == 40
    assert all(char in '0123456789abcdef' for char in m['source_git_commit'])
    for key in set(old) - REMOVED_KEYS - CHANGED_KEYS:
        assert m[key] == old[key], key
    assert m['predecessor'] == dict(path=str(PRIOR), sha256=PRIOR_SHA)
    receipt = m['predecessor_acceptance']
    assert receipt['path'] == str(RECEIPT) and sha(RECEIPT) == receipt['sha256']
    accepted = json.loads(RECEIPT.read_text())
    assert accepted['status'] == 'passed' and accepted['job_id'] == '159959'
    assert accepted['manifest_sha256'] == PRIOR_SHA
    assert sha(accepted['report']['path']) == accepted['report']['sha256']
    assert m['capacity_config'] == dict(path=str(CONFIG), sha256=sha(CONFIG))
    expected_metadata = dict(old['input_metadata'])
    expected_metadata.update({str(CONFIG): sha(CONFIG), str(PRIOR): PRIOR_SHA,
                              str(RECEIPT): sha(RECEIPT)})
    assert m['input_metadata'] == expected_metadata
    config = json.loads(CONFIG.read_text())
    validate_probe_config(config)
    added = {str(ROOT / p) for p in SOURCES}
    assert set(m['execution_sources']) == set(old['execution_sources']) | added
    assert all(m['execution_sources'][p] == h for p, h in old['execution_sources'].items())
    for group in ('execution_sources', 'upstream_files', 'input_metadata'):
        verify_hashes(m[group])
    for entry in (*m['assets'].values(),):
        stat = Path(entry['path']).stat()
        assert stat.st_size == entry['size_bytes'] and stat.st_mtime_ns == entry['mtime_ns']
    for entry in (m['catalog'], m['standardization'], *m['index_artifacts'].values()):
        assert sha(entry['path']) == entry['sha256']
    return config


def prepare(path):
    path = Path(path).resolve()
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    assert sha(PRIOR) == PRIOR_SHA
    m = copy.deepcopy(json.loads(PRIOR.read_text()))
    for key in REMOVED_KEYS:
        m.pop(key, None)
    m.update(phase='capacity384_resource_only', formal_training_ready=False,
        automatic_successor=False, test_records_read=False,
        initialization='fresh seed42 capacity384 diagnostic weights only',
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        predecessor=dict(path=str(PRIOR), sha256=PRIOR_SHA),
        predecessor_acceptance=dict(path=str(RECEIPT), sha256=sha(RECEIPT)),
        capacity_config=dict(path=str(CONFIG), sha256=sha(CONFIG)))
    m['execution_sources'].update({str(ROOT / p): sha(ROOT / p) for p in SOURCES})
    m['input_metadata'].update({str(CONFIG): sha(CONFIG), str(PRIOR): PRIOR_SHA,
                                str(RECEIPT): sha(RECEIPT)})
    verify(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])))


def binding(path):
    path = Path(path).resolve()
    assert sha(path) == os.environ['LIGER_CAPACITY_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT']
    return m, verify(m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    prepare(parser.parse_args().manifest)
