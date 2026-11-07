"""Freeze the authorized seed42 five-policy full-test retrieval comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from scripts.prepare_inference_sensitivity import ASSET_ROOT, ROOT, record, sha, write_new

POLICIES = ['sim10', 'recent10', 'random10', 'mask5', 'mask0']
STATISTICS = dict(bootstrap_samples=5000, bootstrap_batch_size=50, bootstrap_seed=2027,
                  confidence=.95, ks=[5, 10, 20],
                  primary_comparison='sim10_minus_recent10:fused', primary_metric='ndcg@10')
PRIOR_MANIFEST = ASSET_ROOT / 'logs/training_launch/retrieval_diagnostic_seed42_2026-11-07_v1/manifest.json'
PRIOR_MANIFEST_SHA = '160a1a3efa4f9a7c37c2e3effc16c5282eaf71495c25c4249ba15836b98a1e93'
PRIOR_ACCEPTANCE = ASSET_ROOT / 'logs/analysis/retrieval_diagnostic_seed42_2026-11-07_v1/diagnostic/acceptance.json'
PRIOR_ACCEPTANCE_SHA = 'f8571d7b279e3adf357ebebf63c88ec0567182e451ce63e953c662f19dd7d66e'
NEW_SOURCES = [
    'scripts/prepare_retrieval_content.py', 'scripts/run_retrieval_content.py',
    'scripts/retrieval_content_core.py', 'scripts/audit_retrieval_content.py',
    'scripts/analyze_retrieval_content.py', 'scripts/analyze_paired_evaluation.py',
    'tests/test_retrieval_content_manifest.py', 'tests/test_retrieval_content_core.py',
    'tests/test_retrieval_content_runner.py', 'tests/test_retrieval_content_audit.py',
    'tests/test_retrieval_content_paired.py', 'run_retrieval_content_cpu4.sh',
]


def scope_fields():
    return dict(schema_version=1, experiment='retrieval_content_seed42_full_v1',
                seed=42, C=20, physical_slots=10, policies=list(POLICIES),
                protocol='history-last-user-v1', asset_root=str(ASSET_ROOT), execution_root=str(ROOT),
                dirty_at_submission=False, beta=.5, beam_weight=1., runtime_beta=.1,
                batch_size=32, num_workers=1, amp='bf16', sequence_length=32,
                hist_itemkey_length=100, expected_test_samples=2061101, smoke_samples=65,
                full_samples=2061101, random_seed=42, statistics=dict(STATISTICS))


def prepare(path):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit sources before freezing the full-test comparison')
    old_record = record(PRIOR_MANIFEST, PRIOR_MANIFEST_SHA)
    acceptance_record = record(PRIOR_ACCEPTANCE, PRIOR_ACCEPTANCE_SHA)
    old = json.loads(PRIOR_MANIFEST.read_text())
    prior = json.loads(PRIOR_ACCEPTANCE.read_text())
    if (prior['status'] != 'passed' or prior['phase'] != 'diagnostic' or prior['seed'] != 42
            or prior['num_examples'] != 16384 or prior['manifest_sha256'] != PRIOR_MANIFEST_SHA):
        raise ValueError('Prior diagnostic identity mismatch')
    for entry in old['input_metadata'] + old['execution_sources']:
        record(entry['path'], entry['sha256'])
    m = {k: old[k] for k in ('run_dir', 'data_root', 'training_preflight', 'checkpoint',
                             'reference_test', 'frozen_beta', 'selection')}
    m.update(scope_fields())
    m.update({
        'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sampling': 'All 2061101 original test records in frozen order; smoke uses only the first 65',
        'layout_control': 'All five policies retain ten physical context slots and the native candidate position; actual positions and masks observed',
        'candidate_policy': 'One fresh natural beam20 per batch, shared candidates and generation scores; recompute sim10 center, never mix old predictions',
        'hardware_requirement': 'One RTX4090 and four CPUs for entire job; one worker; all library threads one',
        'scope': 'Five fixed-slot full-test policies, seed42 only; no retraining, beta tuning, checkpoint selection or model-source edits',
        'statistics_interpretation': 'Primary: sim10 minus recent10 fused NDCG@10. Other contrasts/metrics exploratory; user-paired intervals do not measure training-seed uncertainty or establish practical importance',
        'input_metadata': list(old['input_metadata']) + [old_record, acceptance_record],
        'execution_sources': list(old['execution_sources']) + [record(ROOT / p) for p in NEW_SOURCES],
    })
    paths = [x['path'] for x in m['execution_sources']]
    if len(paths) != len(set(paths)):
        raise ValueError('Repeated source binding')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps({'manifest': str(path), 'sha256': sha(path), 'source_git_commit': m['source_git_commit']}))


def load_manifest(path):
    if sha(path) != os.environ.get('CONTENT_MANIFEST_SHA256'):
        raise ValueError('Content manifest hash mismatch')
    m = json.loads(Path(path).read_text())
    if any(m.get(k) != v for k, v in scope_fields().items()):
        raise ValueError('Content comparison exceeds authorized scope')
    for key in ('seed', 'C', 'physical_slots', 'batch_size', 'num_workers', 'expected_test_samples',
                'smoke_samples', 'full_samples', 'random_seed'):
        if type(m[key]) is not int:
            raise ValueError(f'Expected integer scope field: {key}')
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
