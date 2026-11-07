import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.audit_retrieval_content import MAX_ITEMKEY, POLICIES, SMOKE_FLAGS, audit, main, sha
from scripts.prepare_retrieval_content import STATISTICS
from scripts.compare_retrieval_paired_reports import compare_paired_reports


def fixture(tmp_path, n=7, phase='smoke', seed=42):
    root = tmp_path / 'out'
    root.mkdir()
    full = n if phase == 'full' else n + 5
    smoke = min(n, 65)
    uid = np.arange(full, dtype=np.int64)
    hashes = np.stack([uid, uid + 900], axis=1).astype(np.uint64)
    target = np.ones((full, 4), dtype=np.int16)
    target[:, -1] = [1, 5, 10, 15, 20] + [100] * (full - 5)
    reference = tmp_path / 'reference.h5'
    with h5py.File(reference, 'w') as f:
        f['user_id'], f['example_hash'], f['target_sid'] = uid, hashes, target
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'input_split': 'val', 'selection_scope': 'validation_only',
                                    'input_checkpoint_sha256': 'checkpoint',
                                    'selected': {'beam_weight': 1., 'rerank_weight': .5}}))
    m = dict(seed=seed, C=20, physical_slots=10, policies=POLICIES,
             protocol='history-last-user-v1', beta=.5, beam_weight=1., runtime_beta=.1,
             source_git_commit='fixture-commit', batch_size=32, amp='bf16', random_seed=42,
             statistics=STATISTICS,
             expected_test_samples=full, smoke_samples=smoke, full_samples=full,
             checkpoint={'path': '/unused.ckpt', 'sha256': 'checkpoint'},
             reference_test={'path': str(reference), 'sha256': sha(reference)},
             frozen_beta={'path': str(selection), 'sha256': sha(selection)}, _manifest_sha256='manifest')
    gate = dict(status='passed', samples=min(32, smoke), policies=POLICIES, random_seed=42,
                random_namespace='grid-retrieval-diagnostic-v1-history-positions',
                candidate_positions={p: [50] for p in POLICIES},
                input_lengths={p: [13] for p in POLICIES}, **{k: True for k in SMOKE_FLAGS})
    run = dict(status='complete', seed=seed, phase=phase, processed=n, expected_samples=full,
               manifest_sha256='manifest', smoke_gate=gate, prior_smoke_acceptance=None,
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
    scales = dict(sim10=3., mask0=0., mask5=1., recent10=2.1, random10=1.5)
    for policy in POLICIES:
        contexts = context.copy()
        if policy == 'mask0':
            contexts[:] = -1
        elif policy == 'mask5':
            contexts[:, :, 5:] = -1
        elif policy in {'recent10', 'random10'}:
            contexts[contexts >= 0] += 100 if policy == 'recent10' else 200
            contexts[:, ::2, :] = contexts[:, ::2, ::-1]
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
            f['candidate_position'] = np.full(n, 50, dtype=np.int32)
            f['rerank_input_length'] = np.full(n, 13, dtype=np.int16)
    if phase == 'full':
        prior_dir = tmp_path / 'prior_smoke'
        prior_dir.mkdir()
        for policy in POLICIES:
            with h5py.File(root / f'{policy}.h5', 'r') as source, h5py.File(prior_dir / f'{policy}.h5', 'w') as dest:
                dest.attrs['num_samples'] = smoke
                for key in source:
                    dest[key] = source[key][:smoke]
        (prior_dir / 'run.json').write_text(json.dumps(dict(run, phase='smoke', processed=smoke)))
        receipt = prior_dir / 'acceptance.json'
        receipt.write_text(json.dumps(audit(prior_dir, m, 'smoke')))
        from scripts.analyze_retrieval_content import analyze
        paired_path = prior_dir / 'paired.json'
        paired = analyze(prior_dir, m, 'smoke')
        paired_path.write_text(json.dumps(paired))
        run['prior_smoke_acceptance'] = {'path': str(receipt), 'sha256': sha(receipt),
                                         'paired_analysis': {'path': str(paired_path), 'sha256': sha(paired_path)},
                                         'paired_comparison': compare_paired_reports(paired, paired)}
        (root / 'run.json').write_text(json.dumps(run))
    return root, m


def update_run(root, **values):
    path = root / 'run.json'
    run = json.loads(path.read_text())
    run.update(values)
    path.write_text(json.dumps(run))
    return run


@pytest.mark.parametrize('phase', ['smoke', 'full'])
def test_metrics_contract_and_readonly_reaudit(tmp_path, phase):
    root, m = fixture(tmp_path, phase=phase)
    r = audit(root, m, phase)
    assert r['status'] == 'passed' and r['policies'] == POLICIES and r['num_examples'] == 7
    assert not (root / 'acceptance.json').exists()
    assert r['settings']['sim10']['coverage_hits'] == 5
    assert r['settings']['sim10']['metrics']['beam']['recall@5'] == 2 / 7
    assert r['settings']['sim10']['metrics']['fused']['recall@5'] == 1 / 7
    assert r['settings']['sim10']['mean_within_user_raw_logit_std'] == pytest.approx(np.arange(20).std() * 3)
    assert r['settings']['mask0']['mean_within_user_raw_logit_std'] == 0
    assert r['fixed_slot_candidate_positions_verified'] and r['controlled_mask_contents_verified']
    assert r['context_multiset_equal_to_sim10_fraction_per_user_candidate']['recent10'] == 1 / 7
    assert r['history_fewer_than_10_fraction'] == 4 / 7
    assert ('Complete' in r['sample_scope']) == (phase == 'full')
    for policy in POLICIES[1:]:
        for variant, values in r['minus_sim10'][policy].items():
            for metric, delta in values.items():
                assert delta == r['settings'][policy]['metrics'][variant][metric] - r['settings']['sim10']['metrics'][variant][metric]
    receipt = root / 'acceptance.json'
    receipt.write_text(json.dumps(r))
    previous = receipt.read_bytes()
    fresh = audit(root, m, phase)
    assert {k:v for k,v in fresh.items() if k != 'checked_at'} == {k:v for k,v in r.items() if k != 'checked_at'}
    assert receipt.read_bytes() == previous


@pytest.mark.parametrize('policy,field', [(p, 'candidate_position') for p in POLICIES[1:]]
                         + [('sim10', 'rerank_input_length'), ('mask5', 'rerank_input_length')])
def test_layout_changes_rejected(tmp_path, policy, field):
    root, m = fixture(tmp_path)
    with h5py.File(root / f'{policy}.h5', 'r+') as f:
        f[field][0] += 1
    with pytest.raises(ValueError, match='position mismatch|length mismatch'):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('policy,kind', [('mask0', 'count'), ('mask5', 'controlled'),
    ('recent10', 'candidate_set'), ('random10', 'candidate_set'), ('sim10', 'range'), ('mask5', 'history')])
def test_context_changes_rejected(tmp_path, policy, kind):
    root, m = fixture(tmp_path)
    with h5py.File(root / f'{policy}.h5', 'r+') as f:
        if kind == 'count':
            f['context_itemkey'][1, 0, 0] = 99
        elif kind in {'controlled', 'candidate_set'}:
            f['context_itemkey'][4, 0, 0] += 999
        elif kind == 'range':
            f['context_itemkey'][4, 0, 0] = MAX_ITEMKEY + 1
        else:
            f['history_valid_count'][4] = 9
    with pytest.raises(ValueError):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('field,value', [('seed', 43), ('phase', 'full'), ('processed', 6),
    ('manifest_sha256', 'wrong'), ('hardware_end', {'uuid': 'GPU-other'}),
    ('cpu_process_tree_samples', []), ('cpu_budget', {'allowed_cpus': list(range(5))}),
    ('source_git_commit', 'other'), ('checkpoint', {}), ('policies', ['sim10']),
    ('beta', .1), ('runtime_beta', .5), ('batch_size', 64), ('amp', 'fp32'),
    ('training_performed', True), ('checkpoint_loading', {'missing': 1, 'unexpected': 0})])
def test_run_contract(tmp_path, field, value):
    root, m = fixture(tmp_path)
    update_run(root, **{field: value})
    with pytest.raises(ValueError):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('flag', SMOKE_FLAGS)
def test_all_smoke_flags_are_required(tmp_path, flag):
    root, m = fixture(tmp_path)
    run = json.loads((root / 'run.json').read_text())
    run['smoke_gate'][flag] = 1  # Truthy is insufficient: the gate must record literal True.
    update_run(root, smoke_gate=run['smoke_gate'])
    with pytest.raises(ValueError, match='complete real smoke'):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('field,value', [('samples', 1), ('policies', ['sim10']), ('random_seed', 43),
    ('random_namespace', 'other'), ('candidate_positions', {'sim10': [50]}),
    ('input_lengths', {p: [8] for p in POLICIES})])
def test_smoke_metadata_layout(tmp_path, field, value):
    root, m = fixture(tmp_path)
    run = json.loads((root / 'run.json').read_text())
    run['smoke_gate'][field] = value
    update_run(root, smoke_gate=run['smoke_gate'])
    with pytest.raises(ValueError, match='Smoke gate'):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('field,value', [('seed', 42.), ('C', 40), ('physical_slots', 5),
    ('policies', ['sim10']), ('beta', .1), ('runtime_beta', .5), ('protocol', 'old'),
    ('smoke_samples', 0), ('full_samples', 7)])
def test_manifest_scope(tmp_path, field, value):
    root, m = fixture(tmp_path)
    m[field] = value
    with pytest.raises(ValueError):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('kind', ['target', 'shared', 'nan', 'duplicate', 'permutation', 'order', 'dtype', 'extra_file'])
def test_prediction_and_schema_checks(tmp_path, kind):
    root, m = fixture(tmp_path)
    with h5py.File(root / 'random10.h5', 'r+') as f:
        if kind == 'target':
            f['target_sid'][0, 0] = 2
        elif kind == 'shared':
            f['beam_score'][0, 0] = .5
        elif kind == 'nan':
            f['raw_rerank_logit'][0, 0] = np.nan
        elif kind == 'duplicate':
            f['beam_sid'][0, 1] = f['beam_sid'][0, 0]
        elif kind == 'permutation':
            f['rerank_sid'][0, 0, -1] = 500
        elif kind == 'order':
            f['rerank_score'][0, 1] = 999
        elif kind == 'dtype':
            data = f['context_itemkey'][:]
            del f['context_itemkey']
            f['context_itemkey'] = data.astype(np.float64)
        else:
            (root / 'short5.h5').touch()
    with pytest.raises(ValueError):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('kind', ['missing', 'hash', 'identity', 'artifact', 'gate', 'paired_hash', 'paired_binding'])
def test_full_reaudits_prior_smoke(tmp_path, kind):
    root, m = fixture(tmp_path, phase='full')
    run = json.loads((root / 'run.json').read_text())
    record = run['prior_smoke_acceptance']
    if kind == 'missing':
        run['prior_smoke_acceptance'] = None
    elif kind == 'hash':
        record['sha256'] = 'wrong'
    elif kind == 'identity':
        p = Path(record['path'])
        receipt = json.loads(p.read_text())
        receipt['seed'] = 43
        p.write_text(json.dumps(receipt))
        record['sha256'] = sha(p)
    elif kind == 'artifact':
        with h5py.File(Path(record['path']).parent / 'mask0.h5', 'r+') as f:
            f['raw_rerank_logit'][0, 0] = .001
    elif kind == 'gate':
        run['smoke_gate']['candidate_positions'] = {p: [51] for p in POLICIES}
    elif kind == 'paired_hash':
        record['paired_analysis']['sha256'] = 'wrong'
    else:
        p = Path(record['paired_analysis']['path'])
        paired = json.loads(p.read_text())
        paired['acceptance']['sha256'] = 'other'
        p.write_text(json.dumps(paired))
        record['paired_analysis']['sha256'] = sha(p)
    update_run(root, **run)
    with pytest.raises(ValueError, match='prior smoke|Prior smoke'):
        audit(root, m, 'full')


def test_full_chunk_boundary_and_duplicate_users(tmp_path):
    root, m = fixture(tmp_path, n=1025, phase='full')
    assert audit(root, m, 'full')['unique_users'] == 1025
    for path in [Path(m['reference_test']['path'])] + [root / f'{p}.h5' for p in POLICIES]:
        with h5py.File(path, 'r+') as f:
            f['user_id'][-1] = f['user_id'][0]
    with pytest.raises(ValueError, match='Repeated evaluation user'):
        audit(root, m, 'full')


def test_source_binding_and_incomplete_marker(tmp_path):
    root, m = fixture(tmp_path)
    m['execution_sources'] = [{'path': m['frozen_beta']['path'], 'sha256': 'wrong'}]
    with pytest.raises(ValueError, match='Source/input hash'):
        audit(root, m, 'smoke')
    m['execution_sources'] = []
    m['checkpoint']['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='Selection checkpoint'):
        audit(root, m, 'smoke')
    m['checkpoint']['sha256'] = 'checkpoint'
    (root / 'INCOMPLETE').touch()
    with pytest.raises(ValueError, match='Incomplete'):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_cli_exclusive_and_real_65_row_runner_gate(tmp_path, monkeypatch, seed):
    from scripts.run_retrieval_content import require_gate
    from scripts.analyze_retrieval_content import analyze

    root, m = fixture(tmp_path, n=65, seed=seed)
    path = tmp_path / 'manifest.json'
    del m['_manifest_sha256']
    path.write_text(json.dumps(m))
    m['_manifest_sha256'] = sha(path)
    update_run(root, manifest_sha256=m['_manifest_sha256'])
    monkeypatch.setattr('sys.argv', ['audit_retrieval_content', '--manifest', str(path),
                                   '--output-dir', str(root), '--phase', 'smoke'])
    main()
    receipt = root / 'acceptance.json'
    previous = receipt.read_bytes()
    paired_path = root / 'paired.json'
    paired_path.write_text(json.dumps(analyze(root, m, 'smoke')))
    record, gate = require_gate(m, root)
    assert record['path'] == str(receipt) and record['sha256'] == sha(receipt)
    assert record['paired_analysis'] == {'path': str(paired_path), 'sha256': sha(paired_path)}
    assert record['paired_comparison']['status'] == 'passed'
    assert json.loads(receipt.read_text())['seed'] == seed
    assert gate['samples'] == 32 and all(gate[k] is True for k in SMOKE_FLAGS)
    with pytest.raises(ValueError, match='Acceptance already exists'):
        main()
    assert receipt.read_bytes() == previous


@pytest.mark.parametrize('seed', [43, 44])
@pytest.mark.parametrize('phase', ['smoke', 'full'])
def test_additional_seed_small_h5_audit_and_mixed_seed_rejection(tmp_path, seed, phase):
    root, manifest = fixture(tmp_path, seed=seed, phase=phase)
    report = audit(root, manifest, phase)
    assert report['seed'] == seed and report['num_examples'] == 7
    assert report['settings']['sim10']['metrics']['beam']['recall@5'] == 2 / 7
    update_run(root, seed=42)
    with pytest.raises(ValueError, match='Run seed mismatch'):
        audit(root, manifest, phase)


@pytest.mark.parametrize('change', [
    lambda p: p.pop('paired_comparison'),
    lambda p: p['paired_comparison'].update(tolerance={'abs_tol': 1e-6, 'rel_tol': 0.}),
    lambda p: p['paired_comparison'].update(status='failed'),
    lambda p: p['paired_comparison'].update(ignored_runtime_fields=['seed']),
    lambda p: p['paired_comparison'].update(max_abs_difference=float('nan')),
    lambda p: p['paired_comparison'].update(exact_equal_excluding_runtime_fields=False),
    lambda p: p['paired_comparison'].update(tolerated_differences=[{
        'path': '/seed', 'recorded': 43., 'fresh': 43. + 1e-13, 'absolute_difference': 1e-13}]),
])
def test_full_requires_strict_numeric_smoke_review(tmp_path, change):
    root, manifest = fixture(tmp_path, seed=43, phase='full')
    run = json.loads((root / 'run.json').read_text())
    change(run['prior_smoke_acceptance'])
    update_run(root, **run)
    with pytest.raises(ValueError, match='Prior smoke numeric'):
        audit(root, manifest, 'full')
