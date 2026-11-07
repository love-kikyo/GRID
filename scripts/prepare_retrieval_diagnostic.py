"""Freeze a seed42 prefix-only diagnostic, without changing old scan sources."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from scripts.prepare_inference_sensitivity import ASSET_ROOT, ROOT, SOURCES, record, sha, write_new

POLICIES = ['sim10', 'mask0', 'mask5', 'recent10', 'random10', 'short0', 'short5']
OLD_MANIFEST = ASSET_ROOT / 'logs/training_launch/sensitivity_seed42_2026-11-07_v1/manifest.json'
OLD_SHA = '6b7a007b294607cd4cce423bc64609b4726abd6809d1af82e134190832bac36f'
NEW_SOURCES = [
    'scripts/prepare_retrieval_diagnostic.py', 'scripts/run_retrieval_diagnostic.py',
    'scripts/retrieval_diagnostic_core.py', 'scripts/audit_retrieval_diagnostic.py',
    'tests/test_retrieval_diagnostic_core.py', 'tests/test_retrieval_diagnostic_audit.py',
    'tests/test_retrieval_diagnostic_runner.py', 'run_retrieval_diagnostic_cpu4.sh',
]


def prepare(path):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit sources before freezing the diagnostic')
    record(OLD_MANIFEST, OLD_SHA)
    old = json.loads(OLD_MANIFEST.read_text())
    for entry in old['input_metadata']:
        record(entry['path'], entry['sha256'])
    prior = ASSET_ROOT / 'logs/analysis/inference_sensitivity_seed42_2026-11-07_v1/full_retrieval/acceptance.json'
    previous = json.loads(prior.read_text())
    if (previous['status'] != 'passed' or previous['seed'] != 42 or
            previous['manifest_sha256'] != OLD_SHA or previous['num_examples'] != 2061101):
        raise ValueError('Prior scan acceptance identity mismatch')
    m = {k: old[k] for k in ('run_dir', 'data_root', 'training_preflight', 'checkpoint',
                             'reference_test', 'frozen_beta', 'input_metadata')}
    m.update({
        'schema_version': 1, 'experiment': 'retrieval_structure_content_diagnostic_v1',
        'seed': 42, 'C': 20, 'physical_slots': 10, 'policies': POLICIES,
        'protocol': 'history-last-user-v1', 'asset_root': str(ASSET_ROOT),
        'execution_root': str(ROOT), 'dirty_at_submission': False,
        'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'beta': .5, 'beam_weight': 1., 'runtime_beta': .1,
        'batch_size': 32, 'num_workers': 1, 'amp': 'bf16',
        'sequence_length': 32, 'hist_itemkey_length': 100,
        'expected_test_samples': 2061101, 'smoke_samples': 65, 'diagnostic_samples': 16384,
        'sampling': 'Fixed first 16384 records in frozen test order; exploratory, no tuning or full-test claim',
        'selection': 'Recent/random select valid history positions without replacement, retaining repeated items; then native candidate-similarity ordering',
        'random_seed': 42,
        'layout_control': 'short0/5 use exactly the same selected content as mask0/5 but physically remove empty slots; short5 takes sim10 prefix to avoid top-k tie changes',
        'candidate_policy': 'One fresh natural beam20 per batch, shared candidates and generation scores across all seven policies',
        'hardware_requirement': 'One RTX4090 and four CPUs for entire job; one worker; all library threads one',
        'scope': 'Two diagnostic contrasts only; no retraining, other seeds, beta selection or model-source edits',
        'execution_sources': [record(ROOT / p) for p in dict.fromkeys(SOURCES + NEW_SOURCES)],
    })
    m['input_metadata'] += [record(OLD_MANIFEST, OLD_SHA), record(prior)]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps({'manifest': str(path), 'sha256': sha(path), 'source_git_commit': m['source_git_commit']}))


def load_manifest(path):
    if sha(path) != os.environ.get('DIAGNOSTIC_MANIFEST_SHA256'):
        raise ValueError('Diagnostic manifest hash mismatch')
    m = json.loads(path.read_text())
    expected = dict(schema_version=1, experiment='retrieval_structure_content_diagnostic_v1',
                    seed=42, C=20, physical_slots=10, policies=POLICIES,
                    protocol='history-last-user-v1', asset_root=str(ASSET_ROOT), execution_root=str(ROOT),
                    dirty_at_submission=False, beta=.5, beam_weight=1., runtime_beta=.1,
                    batch_size=32, num_workers=1, amp='bf16', sequence_length=32,
                    hist_itemkey_length=100, expected_test_samples=2061101, smoke_samples=65,
                    diagnostic_samples=16384, random_seed=42)
    if any(m.get(k) != value for k, value in expected.items()) or type(m['seed']) is not int:
        raise ValueError('Diagnostic exceeds authorized scope')
    if m['source_git_commit'] != os.environ.get('SOURCE_GIT_COMMIT'):
        raise ValueError('Execution commit binding mismatch')
    for entry in m['execution_sources'] + m['input_metadata']:
        record(entry['path'], entry['sha256'])
    m['_manifest_sha256'] = sha(path)
    return m


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    prepare(p.parse_args().manifest.resolve())
