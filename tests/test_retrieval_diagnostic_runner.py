import json

import pytest

from scripts.prepare_retrieval_diagnostic import ASSET_ROOT, ROOT, POLICIES, load_manifest, sha, write_new
from scripts.run_retrieval_diagnostic import require_gate


def manifest_fixture(tmp_path, monkeypatch):
    m = dict(schema_version=1, experiment='retrieval_structure_content_diagnostic_v1',
             seed=42, C=20, physical_slots=10, policies=POLICIES, protocol='history-last-user-v1',
             asset_root=str(ASSET_ROOT), execution_root=str(ROOT), dirty_at_submission=False,
             source_git_commit='test', beta=.5, beam_weight=1., runtime_beta=.1,
             batch_size=32, num_workers=1, amp='bf16', sequence_length=32, hist_itemkey_length=100,
             expected_test_samples=2061101, smoke_samples=65, diagnostic_samples=16384,
             random_seed=42, execution_sources=[], input_metadata=[])
    path = tmp_path / 'manifest.json'
    write_new(path, m)
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'test')
    monkeypatch.setenv('DIAGNOSTIC_MANIFEST_SHA256', sha(path))
    return path, m


def test_frozen_manifest(tmp_path, monkeypatch):
    p, _ = manifest_fixture(tmp_path, monkeypatch)
    assert load_manifest(p)['seed'] == 42
    monkeypatch.setenv('DIAGNOSTIC_MANIFEST_SHA256', 'wrong')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_manifest(p)


@pytest.mark.parametrize('field,value', [('seed', 43), ('diagnostic_samples', 2061101),
    ('policies', ['sim10']), ('physical_slots', 20), ('C', 40), ('beta', .1),
    ('random_seed', 43), ('asset_root', '/wrong'), ('execution_root', '/wrong'),
    ('dirty_at_submission', True), ('amp', 'fp32'), ('num_workers', 4)])
def test_scope_boundaries(tmp_path, monkeypatch, field, value):
    p, m = manifest_fixture(tmp_path, monkeypatch)
    m[field] = value
    p.write_text(json.dumps(m))
    monkeypatch.setenv('DIAGNOSTIC_MANIFEST_SHA256', sha(p))
    with pytest.raises(ValueError, match='authorized scope'):
        load_manifest(p)


def test_source_and_commit_binding(tmp_path, monkeypatch):
    p, m = manifest_fixture(tmp_path, monkeypatch)
    source = tmp_path / 'source.py'
    source.write_text('original')
    m['execution_sources'] = [{'path': str(source), 'sha256': sha(source)}]
    p.write_text(json.dumps(m))
    monkeypatch.setenv('DIAGNOSTIC_MANIFEST_SHA256', sha(p))
    source.write_text('changed')
    with pytest.raises(ValueError, match='Changed accepted source'):
        load_manifest(p)
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'different')
    with pytest.raises(ValueError, match='commit binding'):
        load_manifest(p)


def test_missing_or_mixed_smoke(tmp_path):
    with pytest.raises(ValueError, match='accepted smoke'):
        require_gate({}, None)
    write_new(tmp_path / 'acceptance.json', {'status': 'passed', 'seed': 43})
    write_new(tmp_path / 'run.json', {})
    with pytest.raises(ValueError, match='acceptance identity'):
        require_gate({}, tmp_path)


def test_smoke_receipt_binds_reaudited_artifacts(tmp_path, monkeypatch):
    from scripts import audit_retrieval_diagnostic as auditor
    receipt = {'status': 'passed', 'seed': 42, 'phase': 'smoke',
               'manifest_sha256': 'manifest', 'num_examples': 65,
               'checked_at': 'original', 'artifacts': [{'sha256': 'original'}]}
    write_new(tmp_path / 'acceptance.json', receipt)
    write_new(tmp_path / 'run.json', {'phase': 'smoke', 'seed': 42, 'processed': 65,
              'manifest_sha256': 'manifest', 'smoke_gate': {'status': 'passed'}})
    current = dict(receipt, checked_at='later')
    monkeypatch.setattr(auditor, 'audit', lambda *args: current)
    record, gate = require_gate({'_manifest_sha256': 'manifest'}, tmp_path)
    assert record['sha256'] == sha(tmp_path / 'acceptance.json')
    assert gate['status'] == 'passed'
    current['artifacts'] = [{'sha256': 'changed'}]
    with pytest.raises(ValueError, match='receipt differs'):
        require_gate({'_manifest_sha256': 'manifest'}, tmp_path)
