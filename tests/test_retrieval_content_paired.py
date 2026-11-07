import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import pytest

from scripts.analyze_paired_evaluation import paired_bootstrap_ci
from scripts.analyze_retrieval_content import (
    POLICIES, STATISTICS, _contributions, analyze, comparisons, main, sha,
)


def fixture(tmp_path, n=9, phase='smoke'):
    from scripts.audit_retrieval_content import SMOKE_FLAGS, audit
    root = tmp_path / ('prior' if phase == 'full' else 'out')
    root.mkdir()
    full = n if phase == 'full' else n + 5
    uid = np.arange(full, dtype=np.int64)
    hashes = np.stack([uid, uid + 900], axis=1).astype(np.uint64)
    target = np.ones((full, 4), dtype=np.int16)
    target[:, -1] = np.resize([1, 1, 2, 5, 10, 15, 20, 100, 100], full)
    reference = tmp_path / 'reference.h5'
    with h5py.File(reference, 'w') as f:
        f['user_id'], f['example_hash'], f['target_sid'] = uid, hashes, target
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'input_split': 'val', 'selection_scope': 'validation_only',
                                    'input_checkpoint_sha256': 'checkpoint',
                                    'selected': {'beam_weight': 1., 'rerank_weight': .5}}))
    m = dict(seed=42, C=20, physical_slots=10, policies=POLICIES,
             protocol='history-last-user-v1', beta=.5, beam_weight=1., runtime_beta=.1,
             source_git_commit='fixture-commit', batch_size=32, amp='bf16', random_seed=42,
             expected_test_samples=full, smoke_samples=n, full_samples=full, statistics=STATISTICS,
             checkpoint={'path': '/unused.ckpt', 'sha256': 'checkpoint'},
             reference_test={'path': str(reference), 'sha256': sha(reference)},
             frozen_beta={'path': str(selection), 'sha256': sha(selection)})
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(m))
    m['_manifest_sha256'] = sha(manifest_path)
    gate = dict(status='passed', samples=min(32, n), policies=POLICIES, random_seed=42,
                random_namespace='grid-retrieval-diagnostic-v1-history-positions',
                candidate_positions={p: [58] for p in POLICIES},
                input_lengths={p: [13] for p in POLICIES}, **{key: True for key in SMOKE_FLAGS})
    run = dict(status='complete', seed=42, phase='smoke', processed=n, expected_samples=full,
               manifest_sha256=m['_manifest_sha256'], smoke_gate=gate, prior_smoke_acceptance=None,
               **{k: m[k] for k in ('source_git_commit', 'checkpoint', 'policies', 'beta',
                                   'runtime_beta', 'batch_size', 'amp')}, training_performed=False,
               checkpoint_loading={'missing': 0, 'unexpected': 0},
               hardware={'name': 'RTX4090', 'uuid': 'GPU-test'}, hardware_end={'uuid': 'GPU-test'},
               cpu_budget={'allowed_cpus': [1, 2, 3, 4], 'allocated_cpus': 4, 'library_threads': 1},
               cpu_process_tree_samples=[{'allowed_cpus': [1, 2, 3, 4], 'processes': [{'pid': 1}]}] * 2)
    (root / 'run.json').write_text(json.dumps(run))
    history = np.resize(np.array([0, 3, 5, 9, 10, 15, 20], dtype=np.int32), n)
    context = np.where(np.arange(10)[None, None, :] < history[:, None, None],
                       np.broadcast_to(np.arange(10), (n, 20, 10)), -1).astype(np.int64)
    scales = dict(sim10=.8, mask0=0., mask5=1., recent10=4., random10=1.5)
    for policy in POLICIES:
        contexts = context.copy()
        if policy == 'mask0':
            contexts[:] = -1
        elif policy == 'mask5':
            contexts[:, :, 5:] = -1
        ids = np.ones((n, 20, 4), dtype=np.int16)
        ids[:, :, -1] = np.arange(1, 21)
        beam = np.broadcast_to(-np.arange(20, dtype=np.float32), (n, 20)).copy()
        raw = -beam * np.float32(scales[policy])
        runtime = beam + np.float32(.1) * raw
        order = np.argsort(-runtime, axis=1, kind='stable')
        with h5py.File(root / f'{policy}.h5', 'w') as f:
            f.attrs['num_samples'] = n
            f['user_id'], f['example_hash'], f['target_sid'] = uid[:n], hashes[:n], target[:n]
            f['beam_sid'], f['beam_score'], f['raw_rerank_logit'] = ids, beam, raw
            f['rerank_sid'] = np.take_along_axis(ids, order[:, :, None], axis=1)
            f['rerank_score'] = np.take_along_axis(runtime, order, axis=1)
            f['context_itemkey'], f['history_valid_count'] = contexts, history
            f['candidate_position'] = np.full(n, 58, dtype=np.int32)
            f['rerank_input_length'] = np.full(n, 13, dtype=np.int16)
    (root / 'acceptance.json').write_text(json.dumps(audit(root, m, 'smoke')))
    if phase == 'full':
        (root / 'paired.json').write_text(json.dumps(analyze(root, m, 'smoke')))
        destination = tmp_path / 'out'
        destination.mkdir()
        for policy in POLICIES:
            shutil.copyfile(root / f'{policy}.h5', destination / f'{policy}.h5')
        run['phase'] = 'full'
        run['prior_smoke_acceptance'] = {
            'path': str(root / 'acceptance.json'), 'sha256': sha(root / 'acceptance.json'),
            'paired_analysis': {'path': str(root / 'paired.json'), 'sha256': sha(root / 'paired.json')},
        }
        root = destination
        (root / 'run.json').write_text(json.dumps(run))
        (root / 'acceptance.json').write_text(json.dumps(audit(root, m, 'full')))
    return root, m


def update_receipt(root, callback):
    path = root / 'acceptance.json'
    receipt = json.loads(path.read_text())
    callback(receipt)
    path.write_text(json.dumps(receipt))


@pytest.mark.parametrize('phase', ['smoke', 'full'])
def test_full_contract_and_primary_direction(tmp_path, phase):
    root, m = fixture(tmp_path, phase=phase)
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.iterdir()}
    report = analyze(root, m, phase)
    assert report['status'] == 'passed' and report['num_examples'] == 9
    assert report['primary'] == {'comparison': 'sim10_minus_recent10:fused', 'metric': 'ndcg@10'}
    assert len(report['comparisons']) == 13
    assert len([entry for comp in report['comparisons'].values() for entry in comp['metrics'].values()
                if entry['role'] == 'primary']) == 1
    assert report['comparisons']['sim10_minus_recent10:fused']['metrics']['ndcg@10']['delta'] > 0
    assert report['all_point_estimates_match_acceptance']
    assert before == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.iterdir()}
    assert not (root / 'paired.json').exists()


def test_streamed_histograms_match_direct_bootstrap(tmp_path, monkeypatch):
    from scripts import analyze_retrieval_content as module
    root, m = fixture(tmp_path)
    monkeypatch.setattr(module, 'CHUNK_SIZE', 2)
    result = analyze(root, m, 'smoke')
    values = {}
    for policy in POLICIES:
        with h5py.File(root / f'{policy}.h5', 'r') as f:
            values[policy] = _contributions({k:f[k][:] for k in module.FIELDS})
    for name, (left, right) in comparisons().items():
        for metric, entry in result['comparisons'][name]['metrics'].items():
            delta = values[left[0]][left[1]][metric] - values[right[0]][right[1]][metric]
            expected = paired_bootstrap_ci(delta, 5000, 50, .95, np.random.default_rng(2027))
            assert entry['paired_bootstrap_ci'] == list(expected)
            assert entry['delta'] == pytest.approx(delta.mean(), abs=1e-14)
    empty_gain = result['comparisons']['mask0:fusion_minus_beam']['metrics']['ndcg@10']
    assert empty_gain['delta'] == 0 and empty_gain['paired_bootstrap_ci'] == [0., 0.]


@pytest.mark.parametrize('field', ['user_id', 'example_hash', 'target_sid', 'beam_sid', 'beam_score'])
def test_alignment_and_candidate_checks_even_with_rebound_hash(tmp_path, field):
    root, m = fixture(tmp_path)
    path = root / 'recent10.h5'
    with h5py.File(path, 'r+') as f:
        value = f[field][0]
        f[field][0] = value + 1
    update_receipt(root, lambda r: next(x for x in r['artifacts'] if x['path'] == str(path)).update(sha256=sha(path)))
    with pytest.raises(ValueError, match='Paired identity|Paired candidates'):
        analyze(root, m, 'smoke')


@pytest.mark.parametrize('kind', ['artifact', 'run', 'manifest', 'source', 'metrics', 'old_phase', 'old_policies'])
def test_rejects_changed_or_mixed_inputs(tmp_path, kind):
    root, m = fixture(tmp_path)
    if kind == 'artifact':
        with h5py.File(root / 'mask0.h5', 'r+') as f:
            f['raw_rerank_logit'][0, 0] = 1
    elif kind == 'run':
        (root / 'run.json').write_text((root / 'run.json').read_text() + ' ')
    elif kind == 'manifest':
        m['_manifest_sha256'] = 'other'
    else:
        def change(r):
            if kind == 'source':
                r['source_git_commit'] = 'other'
            elif kind == 'metrics':
                r['settings']['sim10']['metrics']['fused']['ndcg@10'] += .01
            elif kind == 'old_phase':
                r['phase'] = 'diagnostic'
            else:
                r['policies'] = ['sim10', 'mask0', 'mask5', 'recent10', 'random10', 'short0', 'short5']
        update_receipt(root, change)
    with pytest.raises(ValueError):
        analyze(root, m, 'smoke')


def test_statistics_scope_and_incomplete_guard(tmp_path):
    root, m = fixture(tmp_path)
    changed = dict(m, statistics=dict(STATISTICS, bootstrap_seed=42))
    with pytest.raises(ValueError, match='statistics'):
        analyze(root, changed, 'smoke')
    (root / 'INCOMPLETE').touch()
    with pytest.raises(ValueError, match='Incomplete'):
        analyze(root, m, 'smoke')


def test_detects_input_mutation_during_analysis(tmp_path, monkeypatch):
    from scripts import analyze_retrieval_content as module
    root, m = fixture(tmp_path)
    original = module.paired_bootstrap_ci_from_counts
    def mutate(*args, **kwargs):
        path = root / 'run.json'
        path.write_text(path.read_text() + ' ')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'paired_bootstrap_ci_from_counts', mutate)
    with pytest.raises(ValueError, match='changed during analysis'):
        analyze(root, m, 'smoke')


def test_cli_exclusive_output(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['paired', '--manifest', str(tmp_path / 'manifest.json'),
                                    '--output-dir', str(root), '--phase', 'smoke'])
    main()
    saved = (root / 'paired.json').read_bytes()
    assert json.loads(saved)['status'] == 'passed'
    with pytest.raises(ValueError, match='already exists'):
        main()
    assert (root / 'paired.json').read_bytes() == saved
