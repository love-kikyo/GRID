import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from scripts.audit_retrieval_diagnostic import MAX_ITEMKEY, POLICIES, SMOKE_FLAGS, audit, main, sha


def fixture(tmp_path, n=7, phase='smoke'):
    root = tmp_path / 'out'
    root.mkdir()
    full = n + 5
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
    m = dict(seed=42, C=20, physical_slots=10, policies=POLICIES,
             protocol='history-last-user-v1', beta=.5, beam_weight=1., runtime_beta=.1,
             source_git_commit='fixture-commit', batch_size=32, amp='bf16', random_seed=42,
             expected_test_samples=full, smoke_samples=n, diagnostic_samples=n,
             checkpoint={'path': '/unused.ckpt', 'sha256': 'checkpoint'},
             reference_test={'path': str(reference), 'sha256': sha(reference)},
             frozen_beta={'path': str(selection), 'sha256': sha(selection)}, _manifest_sha256='manifest')
    gate = dict(status='passed', samples=min(32, n), policies=POLICIES, random_seed=42,
                random_namespace='grid-retrieval-diagnostic-v1-history-positions',
                **{key: True for key in SMOKE_FLAGS})
    run = dict(status='complete', seed=42, phase=phase, processed=n, expected_samples=full,
               manifest_sha256='manifest', smoke_gate=gate, prior_smoke_acceptance=None,
               **{k: m[k] for k in ('source_git_commit', 'checkpoint', 'policies', 'beta',
                                   'runtime_beta', 'batch_size', 'amp')}, training_performed=False,
               hardware={'name': 'RTX4090', 'uuid': 'GPU-test'}, hardware_end={'uuid': 'GPU-test'},
               cpu_budget={'allowed_cpus': [1, 2, 3, 4], 'allocated_cpus': 4, 'library_threads': 1},
               cpu_process_tree_samples=[{'allowed_cpus': [1, 2, 3, 4], 'processes': [{'pid': 1}]}] * 2)
    (root / 'run.json').write_text(json.dumps(run))
    history_count = np.resize(np.array([0, 3, 5, 9, 10, 15, 20], dtype=np.int32), n)
    context = np.where(np.arange(10)[None, None, :] < history_count[:, None, None],
                       np.broadcast_to(np.arange(10), (n, 20, 10)), -1).astype(np.int64)
    scales = dict(sim10=3., mask0=0., mask5=1., recent10=2.1, random10=1.5, short0=1., short5=.2)
    for policy in POLICIES:
        contexts = context.copy()
        if policy in {'mask0', 'short0'}:
            contexts[:] = -1
        elif policy in {'mask5', 'short5'}:
            contexts[:, :, 5:] = -1
        elif policy in {'recent10', 'random10'}:
            contexts[contexts >= 0] += 100 if policy == 'recent10' else 200
            # Same selected multiset but potentially different candidate order.
            contexts[:, ::2, :] = contexts[:, ::2, ::-1]
        short = 10 if policy == 'short0' else (5 if policy == 'short5' else 0)
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
            f['context_itemkey'], f['history_valid_count'] = contexts, history_count
            f['candidate_position'] = np.full(n, 50 - short, dtype=np.int32)
            f['rerank_input_length'] = np.full(n, 13 - short, dtype=np.int16)
    if phase == 'diagnostic':
        prior_dir = tmp_path / 'prior_smoke'
        prior_dir.mkdir()
        for policy in POLICIES:
            shutil.copyfile(root / f'{policy}.h5', prior_dir / f'{policy}.h5')
        (prior_dir / 'run.json').write_text(json.dumps(dict(run, phase='smoke')))
        receipt = prior_dir / 'acceptance.json'
        receipt.write_text(json.dumps(audit(prior_dir, m, 'smoke')))
        run['prior_smoke_acceptance'] = {'path': str(receipt), 'sha256': sha(receipt)}
        (root / 'run.json').write_text(json.dumps(run))
    return root, m


@pytest.mark.parametrize('phase', ['smoke', 'diagnostic'])
def test_valid_report_is_readonly_and_metrics_are_correct(tmp_path, phase):
    root, m = fixture(tmp_path, phase=phase)
    r = audit(root, m, phase)
    assert r['status'] == 'passed' and r['seed'] == 42 and r['num_examples'] == 7
    assert not (root / 'acceptance.json').exists()
    assert r['settings']['sim10']['coverage_hits'] == 5
    assert r['settings']['sim10']['metrics']['beam']['recall@5'] == 2 / 7
    assert r['settings']['sim10']['metrics']['fused']['recall@5'] == 1 / 7
    assert r['settings']['sim10']['mean_within_user_raw_logit_std'] == pytest.approx(np.arange(20).std() * 3)
    assert r['settings']['mask0']['mean_within_user_raw_logit_std'] == 0
    assert r['fixed_slot_candidate_positions_verified'] and r['matched_short_control_contents_verified']
    assert r['context_multiset_equal_to_sim10_fraction_per_user_candidate']['recent10'] == 1 / 7
    assert r['history_fewer_than_10_fraction'] == 4 / 7
    assert 'exploratory' in r['sample_scope']
    for k in (0, 5):
        delta = r['fixed_minus_short'][f'mask{k}_minus_short{k}']
        for variant in delta:
            for metric, value in delta[variant].items():
                expected = (r['settings'][f'mask{k}']['metrics'][variant][metric]
                            - r['settings'][f'short{k}']['metrics'][variant][metric])
                assert value == expected


@pytest.mark.parametrize('policy,field', [('mask0', 'candidate_position'), ('mask5', 'candidate_position'),
    ('recent10', 'candidate_position'), ('random10', 'candidate_position'), ('short0', 'candidate_position'),
    ('short5', 'candidate_position'), ('sim10', 'rerank_input_length'), ('short5', 'rerank_input_length')])
def test_positions_and_physical_lengths_are_checked(tmp_path, policy, field):
    root, m = fixture(tmp_path)
    with h5py.File(root / f'{policy}.h5', 'r+') as f:
        f[field][0] += 1
    with pytest.raises(ValueError, match='position mismatch|length mismatch'):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('policy,kind', [('mask0', 'count'), ('mask5', 'controlled'),
    ('short0', 'count'), ('short5', 'controlled'), ('recent10', 'candidate_set'),
    ('random10', 'candidate_set'), ('sim10', 'range'), ('mask5', 'history')])
def test_corrupt_contexts_are_rejected(tmp_path, policy, kind):
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
    ('manifest_sha256', 'wrong'), ('smoke_gate', {}), ('hardware_end', {'uuid': 'GPU-other'}),
    ('cpu_process_tree_samples', []), ('cpu_budget', {'allowed_cpus': list(range(5))})])
def test_bad_run_metadata_is_rejected(tmp_path, field, value):
    root, m = fixture(tmp_path)
    path = root / 'run.json'
    run = json.loads(path.read_text())
    run[field] = value
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('kind', ['target', 'shared', 'nan', 'duplicate', 'permutation', 'order'])
def test_prediction_checks(tmp_path, kind):
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
        else:
            f['rerank_score'][0, 1] = 999
    with pytest.raises(ValueError):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('field,value', [('seed', 43), ('seed', 42.), ('C', 40),
    ('physical_slots', 5), ('policies', ['sim10']), ('beta', .1), ('runtime_beta', .5),
    ('protocol', 'old'), ('smoke_samples', 0)])
def test_manifest_scope(tmp_path, field, value):
    root, m = fixture(tmp_path)
    m[field] = value
    with pytest.raises(ValueError):
        audit(root, m, 'smoke')


def test_source_selection_and_checkpoint_binding(tmp_path):
    root, m = fixture(tmp_path)
    m['checkpoint']['sha256'] = 'other'
    with pytest.raises(ValueError, match='Selection checkpoint'):
        audit(root, m, 'smoke')
    m['checkpoint']['sha256'] = 'checkpoint'
    m['execution_sources'] = [{'path': m['frozen_beta']['path'], 'sha256': 'wrong'}]
    with pytest.raises(ValueError, match='Source/input hash'):
        audit(root, m, 'smoke')


def test_reaudit_preserves_existing_receipt_and_incomplete_blocks(tmp_path):
    root, m = fixture(tmp_path)
    r = audit(root, m, 'smoke')
    receipt = root / 'acceptance.json'
    receipt.write_text(json.dumps(r))
    previous = receipt.read_bytes()
    assert audit(root, m, 'smoke')['settings'] == r['settings']
    assert receipt.read_bytes() == previous
    (root / 'INCOMPLETE').touch()
    with pytest.raises(ValueError, match='Incomplete'):
        audit(root, m, 'smoke')


def test_cli_exclusive_receipt_creation(tmp_path, monkeypatch):
    root, m = fixture(tmp_path)
    path = tmp_path / 'manifest.json'
    del m['_manifest_sha256']
    path.write_text(json.dumps(m))
    run_path = root / 'run.json'
    run = json.loads(run_path.read_text())
    run['manifest_sha256'] = sha(path)
    run_path.write_text(json.dumps(run))
    monkeypatch.setattr('sys.argv', ['audit_retrieval_diagnostic', '--manifest', str(path),
                                   '--output-dir', str(root), '--phase', 'smoke'])
    main()
    receipt = root / 'acceptance.json'
    previous = receipt.read_bytes()
    assert json.loads(previous)['status'] == 'passed'
    with pytest.raises(ValueError, match='Acceptance already exists'):
        main()
    assert receipt.read_bytes() == previous


def test_chunked_audit_checks_every_row(tmp_path):
    root, m = fixture(tmp_path, n=1025, phase='diagnostic')
    r = audit(root, m, 'diagnostic')
    assert r['unique_users'] == 1025 and r['settings']['sim10']['coverage_hits'] == 5
    with h5py.File(root / 'short5.h5', 'r+') as f:
        f['candidate_position'][-1] += 1
    with pytest.raises(ValueError, match='position mismatch'):
        audit(root, m, 'diagnostic')


@pytest.mark.parametrize('kind', ['missing', 'extra', 'shape', 'count', 'dtype', 'sid_range'])
def test_hdf5_contract_is_enforced(tmp_path, kind):
    root, m = fixture(tmp_path)
    with h5py.File(root / 'sim10.h5', 'r+') as f:
        if kind == 'missing':
            del f['history_valid_count']
        elif kind == 'extra':
            f['unexpected'] = np.zeros(1)
        elif kind == 'count':
            f.attrs['num_samples'] = 6
        elif kind == 'sid_range':
            f['target_sid'][0, 0] = 4097
        else:
            values = f['context_itemkey'][:]
            del f['context_itemkey']
            f['context_itemkey'] = values[:, :, :9] if kind == 'shape' else values.astype(np.float64)
    with pytest.raises(ValueError):
        audit(root, m, 'smoke')


def test_true_itemkey_upper_bound_and_zero_are_accepted(tmp_path):
    root, m = fixture(tmp_path)
    assert MAX_ITEMKEY > 2**48
    # The production base-4096 itemkey packing permits digit 4096. Keep all
    # matched mask/short contexts in lockstep while exercising its real bound.
    for policy in ('sim10', 'mask5', 'short5'):
        with h5py.File(root / f'{policy}.h5', 'r+') as f:
            assert f['context_itemkey'][4, 0, 0] == 0
            f['context_itemkey'][4, 0, 0] = MAX_ITEMKEY
    assert audit(root, m, 'smoke')['status'] == 'passed'


def test_duplicate_users_are_rejected_even_with_aligned_exports(tmp_path):
    root, m = fixture(tmp_path)
    paths = [Path(m['reference_test']['path'])] + [root / f'{p}.h5' for p in POLICIES]
    for path in paths:
        with h5py.File(path, 'r+') as f:
            f['user_id'][1] = f['user_id'][0]
    with pytest.raises(ValueError, match='Repeated evaluation user'):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('flag', SMOKE_FLAGS)
@pytest.mark.parametrize('value', [False, None, 1])
def test_every_smoke_gate_flag_must_be_literal_true(tmp_path, flag, value):
    root, m = fixture(tmp_path)
    path = root / 'run.json'
    run = json.loads(path.read_text())
    if value is None:
        del run['smoke_gate'][flag]
    else:
        run['smoke_gate'][flag] = value
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='complete real smoke'):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('field,value', [('source_git_commit', 'other'),
    ('checkpoint', {'path': '/other', 'sha256': 'other'}), ('policies', ['sim10']),
    ('beta', .1), ('runtime_beta', .5), ('batch_size', 64), ('amp', 'fp32'),
    ('training_performed', True), ('training_performed', 0)])
def test_run_inference_metadata_must_match_manifest(tmp_path, field, value):
    root, m = fixture(tmp_path)
    path = root / 'run.json'
    run = json.loads(path.read_text())
    run[field] = value
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='Run metadata|must not perform training'):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('field,value', [('samples', 1), ('policies', ['sim10']),
    ('random_seed', 43), ('random_namespace', 'another-experiment')])
def test_smoke_metadata_must_match_manifest(tmp_path, field, value):
    root, m = fixture(tmp_path)
    path = root / 'run.json'
    run = json.loads(path.read_text())
    run['smoke_gate'][field] = value
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='Smoke gate metadata'):
        audit(root, m, 'smoke')


@pytest.mark.parametrize('kind', ['missing', 'hash', 'phase', 'seed', 'count', 'manifest',
                                 'checkpoint', 'commit', 'run_hash', 'gate'])
def test_diagnostic_prior_smoke_is_bound_to_same_experiment(tmp_path, kind):
    root, m = fixture(tmp_path, phase='diagnostic')
    run_path = root / 'run.json'
    run = json.loads(run_path.read_text())
    record = run['prior_smoke_acceptance']
    prior_path = Path(record['path'])
    prior = json.loads(prior_path.read_text())
    if kind == 'missing':
        run['prior_smoke_acceptance'] = None
    elif kind == 'hash':
        record['sha256'] = 'wrong'
    else:
        if kind in {'phase', 'seed', 'count', 'manifest', 'checkpoint', 'commit'}:
            field, value = {'phase': ('phase', 'diagnostic'), 'seed': ('seed', 43),
                            'count': ('num_examples', 6), 'manifest': ('manifest_sha256', 'other'),
                            'checkpoint': ('checkpoint', {}), 'commit': ('source_git_commit', 'other')}[kind]
            prior[field] = value
        elif kind == 'run_hash':
            prior['run']['sha256'] = 'wrong'
        else:
            # This descriptive field is not itself a mandatory bool. Altering
            # it still must not allow the diagnostic to claim a different gate.
            run['smoke_gate']['candidate_positions'] = {'sim10': [999]}
        prior_path.write_text(json.dumps(prior))
        record['sha256'] = sha(prior_path)
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='prior smoke|Prior smoke'):
        audit(root, m, 'diagnostic')


def test_real_65_row_smoke_receipt_passes_runner_gate(tmp_path):
    from scripts.run_retrieval_diagnostic import require_gate

    root, m = fixture(tmp_path, n=65)
    report = audit(root, m, 'smoke')
    receipt = root / 'acceptance.json'
    receipt.write_text(json.dumps(report))
    record, gate = require_gate(m, root)
    assert record == {'path': str(receipt), 'sha256': sha(receipt)}
    assert gate['samples'] == 32 and all(gate[k] is True for k in SMOKE_FLAGS)
    assert report['num_examples'] == 65
