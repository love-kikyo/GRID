import json
from types import SimpleNamespace

import pytest

from scripts.run_retrieval_content import check_config, require_gate
from scripts.prepare_retrieval_content import sha, write_new


def config():
    return SimpleNamespace(seed=42, sequence_length=32, hist_itemkey_length=100,
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


def receipt_fixture(tmp_path, monkeypatch):
    from scripts import audit_retrieval_content as auditor
    from scripts import analyze_retrieval_content as paired_module
    receipt = {'status': 'passed', 'seed': 42, 'phase': 'smoke', 'num_examples': 65,
        'manifest_sha256': 'manifest', 'checked_at': 'original', 'artifacts': [{'sha256': 'old'}]}
    run = {'phase': 'smoke', 'seed': 42, 'processed': 65, 'manifest_sha256': 'manifest',
           'smoke_gate': {'status': 'passed'}}
    write_new(tmp_path / 'acceptance.json', receipt)
    write_new(tmp_path / 'run.json', run)
    current = dict(receipt, checked_at='later')
    monkeypatch.setattr(auditor, 'audit', lambda *args: current)
    paired = {'status': 'passed', 'seed': 42, 'phase': 'smoke', 'num_examples': 65,
        'manifest_sha256': 'manifest', 'checked_at': 'original', 'analysis_cpu_affinity': [0],
        'acceptance': {'path': str(tmp_path / 'acceptance.json'), 'sha256': sha(tmp_path / 'acceptance.json')}}
    write_new(tmp_path / 'paired.json', paired)
    current_paired = dict(paired, checked_at='later', analysis_cpu_affinity=[1, 2, 3, 4])
    monkeypatch.setattr(paired_module, 'analyze', lambda *args: current_paired)
    return receipt, run, current, current_paired


def test_receipt_bound_to_full_reaudit_except_timestamp(tmp_path, monkeypatch):
    _, _, current, _ = receipt_fixture(tmp_path, monkeypatch)
    record, gate = require_gate({'_manifest_sha256': 'manifest'}, tmp_path)
    assert record['sha256'] == sha(tmp_path / 'acceptance.json')
    assert record['paired_analysis']['sha256'] == sha(tmp_path / 'paired.json')
    assert gate['status'] == 'passed'
    current['artifacts'] = [{'sha256': 'changed'}]
    with pytest.raises(ValueError, match='receipt differs'):
        require_gate({'_manifest_sha256': 'manifest'}, tmp_path)


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
        require_gate({'_manifest_sha256': 'manifest'}, tmp_path)


def test_paired_analysis_must_exist(tmp_path, monkeypatch):
    receipt_fixture(tmp_path, monkeypatch)
    (tmp_path / 'paired.json').unlink()
    with pytest.raises(FileNotFoundError):
        require_gate({'_manifest_sha256': 'manifest'}, tmp_path)


@pytest.mark.parametrize('field,value', [('manifest_sha256', 'different'), ('seed', 43),
    ('acceptance', {'path': '/wrong', 'sha256': 'other'}), ('status', 'failed')])
def test_paired_analysis_binds_all_recomputed_fields(tmp_path, monkeypatch, field, value):
    _, _, _, current_paired = receipt_fixture(tmp_path, monkeypatch)
    current_paired[field] = value
    with pytest.raises(ValueError, match='paired analysis differs'):
        require_gate({'_manifest_sha256': 'manifest'}, tmp_path)
