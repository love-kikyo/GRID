import json
from types import SimpleNamespace

import pytest

from scripts.run_retrieval_content import check_config, require_gate
from scripts.prepare_retrieval_content import sha, write_new
from scripts.compare_retrieval_paired_reports import COMPARISONS, METRICS, POLICIES, VARIANTS


def config(seed=42):
    return SimpleNamespace(seed=seed, sequence_length=32, hist_itemkey_length=100,
        model=SimpleNamespace(top_k_for_generation=20, top_k_for_score=10,
            active_top_k_for_score=None, rerank_objective='listwise',
            rerank_logit_weight=.1, beam_log_prob_weight=1.))


def test_trained_config_is_fixed():
    check_config(config())


@pytest.mark.parametrize('field,value', [('seed', 43), ('sequence_length', 64), ('hist_itemkey_length', 50),
    ('model.top_k_for_generation', 40), ('model.top_k_for_score', 5),
    ('model.active_top_k_for_score', 5), ('model.rerank_objective', 'pointwise'),
    ('model.rerank_logit_weight', .5), ('model.beam_log_prob_weight', 0.)])
def test_config_drift_rejected(field, value):
    cfg = config()
    target = cfg.model if field.startswith('model.') else cfg
    setattr(target, field.split('.')[-1], value)
    with pytest.raises(ValueError, match='configuration'):
        check_config(cfg)


def test_missing_smoke_rejected():
    with pytest.raises(ValueError, match='accepted smoke'):
        require_gate({}, None)


def receipt_fixture(tmp_path, monkeypatch, seed=42):
    from scripts import audit_retrieval_content as auditor
    from scripts import analyze_retrieval_content as paired_module
    receipt = {'status': 'passed', 'seed': seed, 'phase': 'smoke', 'num_examples': 65,
        'manifest_sha256': 'manifest', 'checked_at': 'original', 'artifacts': [{'sha256': 'old'}]}
    run = {'phase': 'smoke', 'seed': seed, 'processed': 65, 'manifest_sha256': 'manifest',
           'smoke_gate': {'status': 'passed'}}
    write_new(tmp_path / 'acceptance.json', receipt)
    write_new(tmp_path / 'run.json', run)
    current = dict(receipt, checked_at='later')
    monkeypatch.setattr(auditor, 'audit', lambda *args: current)
    paired = {'status': 'passed', 'seed': seed, 'phase': 'smoke', 'num_examples': 65,
        'manifest_sha256': 'manifest', 'checked_at': 'original', 'analysis_cpu_affinity': [0],
        'acceptance': {'path': str(tmp_path / 'acceptance.json'), 'sha256': sha(tmp_path / 'acceptance.json')}}
    paired['point_estimates'] = {p: {v: {m: .25 for m in METRICS} for v in VARIANTS} for p in POLICIES}
    paired['comparisons'] = {c: dict(left=['sim10', 'fused'], right=['recent10', 'fused'],
        direction='left_minus_right', metrics={m: dict(left_mean=.3, right_mean=.2,
            delta=.1, relative_gain=.5, paired_bootstrap_ci=[.09, .11], confidence=.95,
            nonzero_user_deltas=17, role='secondary_exploratory') for m in METRICS}) for c in COMPARISONS}
    write_new(tmp_path / 'paired.json', paired)
    current_paired = dict(paired, checked_at='later', analysis_cpu_affinity=[1, 2, 3, 4])
    monkeypatch.setattr(paired_module, 'analyze', lambda *args: current_paired)
    return receipt, run, current, current_paired


def test_receipt_bound_to_full_reaudit_except_timestamp(tmp_path, monkeypatch):
    _, _, current, _ = receipt_fixture(tmp_path, monkeypatch)
    record, gate = require_gate({'seed': 42, '_manifest_sha256': 'manifest'}, tmp_path)
    assert record['sha256'] == sha(tmp_path / 'acceptance.json')
    assert record['paired_analysis']['sha256'] == sha(tmp_path / 'paired.json')
    assert gate['status'] == 'passed'
    current['artifacts'] = [{'sha256': 'changed'}]
    with pytest.raises(ValueError, match='receipt differs'):
        require_gate({'seed': 42, '_manifest_sha256': 'manifest'}, tmp_path)


@pytest.mark.parametrize('owner,field,value', [
    ('receipt', 'seed', 43), ('receipt', 'phase', 'full'), ('receipt', 'num_examples', 16384),
    ('receipt', 'status', 'failed'), ('run', 'seed', 43), ('run', 'phase', 'full'),
    ('run', 'processed', 64), ('run', 'manifest_sha256', 'other'),
    ('run', 'smoke_gate', {'status': 'failed'})])
def test_wrong_smoke_identity_rejected(tmp_path, monkeypatch, owner, field, value):
    receipt, run, _, _ = receipt_fixture(tmp_path, monkeypatch)
    payload = receipt if owner == 'receipt' else run
    payload[field] = value
    path = tmp_path / ('acceptance.json' if owner == 'receipt' else 'run.json')
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='acceptance identity'):
        require_gate({'seed': 42, '_manifest_sha256': 'manifest'}, tmp_path)


def test_paired_analysis_must_exist(tmp_path, monkeypatch):
    receipt_fixture(tmp_path, monkeypatch)
    (tmp_path / 'paired.json').unlink()
    with pytest.raises(FileNotFoundError):
        require_gate({'seed': 42, '_manifest_sha256': 'manifest'}, tmp_path)


@pytest.mark.parametrize('field,value', [('manifest_sha256', 'different'), ('seed', 43),
    ('acceptance', {'path': '/wrong', 'sha256': 'other'}), ('status', 'failed')])
def test_paired_analysis_binds_all_recomputed_fields(tmp_path, monkeypatch, field, value):
    _, _, _, current_paired = receipt_fixture(tmp_path, monkeypatch)
    current_paired[field] = value
    with pytest.raises(ValueError, match='paired analysis differs'):
        require_gate({'seed': 42, '_manifest_sha256': 'manifest'}, tmp_path)


@pytest.mark.parametrize('seed', [43, 44])
def test_additional_seed_requires_matching_training_configuration(seed):
    check_config(config(seed), seed)
    with pytest.raises(ValueError, match='configuration'):
        check_config(config(42), seed)
    with pytest.raises(ValueError, match='configuration'):
        check_config(config(seed), 45)


@pytest.mark.parametrize('seed', [43, 44])
def test_gate_is_seed_bound_and_records_statistical_roundoff(tmp_path, monkeypatch, seed):
    _, _, _, paired = receipt_fixture(tmp_path, monkeypatch, seed)
    metric = paired['comparisons']['sim10_minus_mask0:raw']['metrics']['ndcg@5']
    metric['paired_bootstrap_ci'][0] += 1e-16
    record, gate = require_gate({'seed': seed, '_manifest_sha256': 'manifest'}, tmp_path)
    review = record['paired_comparison']
    assert review['status'] == 'passed' and review['tolerance'] == {'abs_tol': 1e-12, 'rel_tol': 0.}
    assert len(review['tolerated_differences']) == 1
    assert gate['status'] == 'passed'
    with pytest.raises(ValueError, match='acceptance identity'):
        require_gate({'seed': 42, '_manifest_sha256': 'manifest'}, tmp_path)
    metric['paired_bootstrap_ci'][0] += 1e-8
    with pytest.raises(ValueError, match='paired analysis differs'):
        require_gate({'seed': seed, '_manifest_sha256': 'manifest'}, tmp_path)


@pytest.mark.parametrize('seed', [None, 42., True, 45])
def test_gate_rejects_missing_or_invalid_seed(tmp_path, monkeypatch, seed):
    receipt_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='acceptance identity'):
        require_gate({'seed': seed, '_manifest_sha256': 'manifest'}, tmp_path)
