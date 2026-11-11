"""Bind the fixed random-init 90k model to its own full-validation selection.

This creates no predictions and never selects on test. The old test export is
an input-identity reference only; its candidates and scores are not consumed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from scripts.prepare_inference_sensitivity import ROOT, record, sha, write_new

RUN = ROOT / 'logs/train/runs/2026-11-10/click-listwise-random-seed42-158562'
CHECKPOINT = RUN / 'checkpoints/checkpoint_000_090000.ckpt'
PREFLIGHT = ROOT / 'logs/training_preflight/listwise_random_seed42_158562.json'
DATA = ROOT / 'data/taobao_history_last_user_v1'
REFERENCE = ROOT / 'logs/analysis/paired_evaluation/listwise_seed42_132049_test_full_v1.h5'
REFERENCE_SHA = 'fec1d315af68a972778c38b98c86bf4a6a7a721b743b782775cd8a34600ebed7'
POLICIES = ['sim10', 'recent10', 'random10', 'mask5', 'mask0']
STATISTICS = dict(bootstrap_samples=5000, bootstrap_batch_size=50, bootstrap_seed=2027,
                  confidence=.95, ks=[5, 10, 20],
                  primary_comparison='sim10_minus_recent10:fused', primary_metric='ndcg@10')
BOUND_PATHS = ['src', 'scripts', 'tests', 'configs', 'run_random_retrieval_content_cpu4.sh']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_sources(commit):
    require(isinstance(commit, str) and len(commit) == 40
            and all(c in '0123456789abcdef' for c in commit), 'Explicit execution commit required')
    subprocess.run(['git', 'diff', '--exit-code', commit, '--', *BOUND_PATHS], cwd=ROOT, check=True)
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--', *BOUND_PATHS],
                                       cwd=ROOT, text=True).strip(), 'Uncommitted execution sources')


def scope():
    return dict(schema_version=1, experiment='random_init_seed42_retrieval_content_v1',
                seed=42, C=20, physical_slots=10, policies=list(POLICIES),
                protocol='history-last-user-v1', asset_root=str(ROOT), execution_root=str(ROOT),
                run_dir=str(RUN), data_root=str(DATA), training_preflight=str(PREFLIGHT),
                beam_weight=1., runtime_beta=.1, batch_size=32, num_workers=1,
                amp='bf16', sequence_length=32, hist_itemkey_length=100,
                expected_test_samples=2061101, smoke_samples=65, full_samples=2061101,
                random_seed=42, statistics=dict(STATISTICS), training_performed=False,
                reference_test={'path': str(REFERENCE), 'sha256': REFERENCE_SHA},
                reference_role='Only user_id, example_hash, target_sid; never old candidates/scores',
                training_selection_caveat='Fixed author-requested 90k checkpoint; training validation windows drifted. Not a retrospectively reselected full-validation best.')


def validate_selection(selection, checkpoint_sha):
    require(selection.get('input_split') == 'val' and selection.get('selection_scope') == 'validation_only'
            and selection.get('num_examples') == 347416, 'Require own complete validation selection')
    require(selection.get('input_checkpoint_sha256') == checkpoint_sha, 'Selection checkpoint mismatch')
    protocol = selection.get('selection_protocol', {})
    require(protocol.get('metric_order') == ['ndcg@10', 'ndcg@5', 'recall@10', 'recall@5']
            and protocol.get('tie_breaker') == 'smaller_rerank_weight'
            and protocol.get('beam_weight') == 1.
            and protocol.get('hash_sample') == {'word': 0, 'modulus': 1, 'remainder': 0},
            'Selection rule changed')
    grid = protocol.get('rerank_weight_grid', {})
    require(set(grid) == {'start', 'stop', 'step'}
            and [float(grid[k]) for k in ('start', 'stop', 'step')] == [0., .5, .01], 'Selection grid changed')
    candidates = selection.get('candidates', [])
    require(len(candidates) == 51 and [r.get('rerank_weight') for r in candidates] == [i / 100 for i in range(51)]
            and all(r.get('beam_weight') == 1. for r in candidates), 'Incomplete selection grid')
    metrics = protocol['metric_order']
    require(all(type(r.get('metrics', {}).get(k)) in (int, float)
                and math.isfinite(r['metrics'][k]) and 0 <= r['metrics'][k] <= 1
                for r in candidates for k in metrics), 'Invalid validation metric')
    best = max(candidates, key=lambda r: tuple(r['metrics'][k] for k in metrics) + (-r['rerank_weight'],))
    require(selection.get('selected') == best, 'Selection does not follow the frozen rule')
    return float(best['rerank_weight'])


def verify_validation(directory, commit):
    directory = Path(directory).resolve()
    receipt_path = directory / 'acceptance.json'
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get('status') == 'passed' and receipt.get('num_examples') == 347416
            and receipt.get('fresh_single_pass') is True
            and receipt.get('checkpoint_global_step') == 90000
            and receipt.get('checkpoint_loading') == {'missing': 0, 'unexpected': 0}
            and receipt.get('source_git_commit') == commit, 'Incomplete own validation acceptance')
    checkpoint = receipt['checkpoint']
    require(checkpoint.get('path') == str(CHECKPOINT), 'Wrong fixed checkpoint')
    records = [record(receipt_path)]
    for key, expected in [('checkpoint', CHECKPOINT), ('validation', directory / 'validation.h5'),
                          ('sidecar', directory / 'validation.json'), ('selection', directory / 'fusion.json')]:
        bound = receipt[key]
        require(bound['path'] == str(expected), f'Wrong validation binding: {key}')
        records.append(record(expected, bound['sha256']))
    selection = json.loads((directory / 'fusion.json').read_text())
    beta = validate_selection(selection, checkpoint['sha256'])
    require(selection['input'] == receipt['validation']['path']
            and selection['input_sha256'] == receipt['validation']['sha256']
            and selection['input_sidecar_sha256'] == receipt['sidecar']['sha256'], 'Selection input identity mismatch')
    return receipt, beta, records


def prepare(path, validation_dir, commit):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT / 'logs/training_launch') and not path.exists(), 'Refusing manifest target')
    verify_sources(commit)
    accepted, beta, metadata = verify_validation(validation_dir, commit)
    sources = subprocess.check_output(['git', 'ls-files', '--', *BOUND_PATHS], cwd=ROOT, text=True).splitlines()
    require(bool(sources), 'No bound execution sources')
    metadata += [record(RUN / '.hydra/config.yaml'), record(PREFLIGHT)]
    for name in ('protocol_manifest_train.json', 'protocol_manifest_test.json', 'protocol_audit.json'):
        metadata.append(record(DATA / name))
    for rank in range(3):
        for phase in ('initialization', 'train_start', 'updates', 'finished'):
            audit = ROOT / f'logs/random_init_audit/seed42_158562/formal/rank_{rank}_{phase}.json'
            entry = json.loads(audit.read_text())
            require(entry.get('status') == 'passed', 'Formal random-init audit failed')
            if phase == 'finished':
                require(entry.get('global_step') == 115000, 'Unexpected training stop step')
            metadata.append(record(audit))
    m = scope()
    m.update(source_git_commit=commit, dirty_at_submission=False, beta=beta,
             checkpoint=accepted['checkpoint'], frozen_beta=accepted['selection'],
             validation_acceptance=record(Path(validation_dir) / 'acceptance.json'),
             sampling='All 2061101 original test records; smoke uses first 65 only',
             selection='Recent/random select valid history positions without replacement, retaining repeated items; then native candidate-similarity ordering',
             layout_control='All five policies retain ten physical slots and the native candidate position',
             candidate_policy='Fresh beam20 per batch shared by five policies; no old candidate reuse',
             execution_sources=[record(ROOT / p) for p in sources], input_metadata=metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps({'manifest': str(path), 'sha256': sha(path), 'beta': beta, 'source_git_commit': commit}))


def load_manifest(path):
    path = Path(path).resolve()
    require(sha(path) == os.environ.get('CONTENT_MANIFEST_SHA256'), 'Manifest hash mismatch')
    m = json.loads(path.read_text())
    require(all(m.get(k) == value for k, value in scope().items()), 'Random content scope changed')
    require(m.get('source_git_commit') == os.environ.get('SOURCE_GIT_COMMIT'), 'Source commit mismatch')
    require(m.get('checkpoint', {}).get('path') == str(CHECKPOINT), 'Wrong checkpoint')
    for entry in m['execution_sources'] + m['input_metadata']:
        record(entry['path'], entry['sha256'])
    require(m['frozen_beta'] in m['input_metadata'] and m['validation_acceptance'] in m['input_metadata'],
            'Missing validation binding')
    selected = json.loads(Path(m['frozen_beta']['path']).read_text())
    require(m['beta'] == validate_selection(selected, m['checkpoint']['sha256']), 'Frozen beta mismatch')
    m['_manifest_sha256'] = sha(path)
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--validation-dir', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.validation_dir, args.expected_commit)
