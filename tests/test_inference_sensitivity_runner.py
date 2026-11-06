import json

import pytest

from scripts.prepare_inference_sensitivity import ASSET_ROOT, ROOT, GROUPS, sha, write_new
from scripts.run_inference_sensitivity import load_bound_manifest, require_gate


def fixture_manifest(tmp_path, monkeypatch):
    m = dict(schema_version=2, seed=42, asset_root=str(ASSET_ROOT), execution_root=str(ROOT),
             groups=GROUPS, beta=.5, beam_weight=1., runtime_beta=.1,
             batch_size=32, num_workers=1, amp='bf16', sequence_length=32,
             hist_itemkey_length=100, expected_test_samples=2061101, smoke_samples=65,
             protocol='history-last-user-v1', dirty_at_submission=False,
             source_git_commit='test-commit', execution_sources=[], input_metadata=[])
    p = tmp_path / 'manifest.json'
    write_new(p, m)
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'test-commit')
    monkeypatch.setenv('SENSITIVITY_MANIFEST_SHA256', sha(p))
    return p, m


def test_manifest_binding(tmp_path, monkeypatch):
    p, _ = fixture_manifest(tmp_path, monkeypatch)
    assert load_bound_manifest(p)['seed'] == 42
    monkeypatch.setenv('SENSITIVITY_MANIFEST_SHA256', 'wrong')
    with pytest.raises(ValueError, match='hash binding'):
        load_bound_manifest(p)


@pytest.mark.parametrize('key,value', [('seed', 45), ('seed', 43.0), ('beta', .4), ('amp', 'none'),
                                    ('asset_root', '/wrong'), ('execution_root', '/wrong'),
                                    ('schema_version', 1),
                                    ('batch_size', 16), ('sequence_length', 64),
                                    ('expected_test_samples', 100), ('smoke_samples', 64),
                                    ('groups', {'beam': [[40, 0]]}), ('dirty_at_submission', True)])
def test_reject_scope_drift(tmp_path, monkeypatch, key, value):
    p, m = fixture_manifest(tmp_path, monkeypatch)
    m[key] = value
    p.write_text(json.dumps(m))
    monkeypatch.setenv('SENSITIVITY_MANIFEST_SHA256', sha(p))
    with pytest.raises(ValueError, match='authorized scope'):
        load_bound_manifest(p)


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_authorized_seeds(tmp_path, monkeypatch, seed):
    p, m = fixture_manifest(tmp_path, monkeypatch)
    m['seed'] = seed
    p.write_text(json.dumps(m))
    monkeypatch.setenv('SENSITIVITY_MANIFEST_SHA256', sha(p))
    assert load_bound_manifest(p)['seed'] == seed


@pytest.mark.parametrize('field', ['report', 'run'])
def test_gate_rejects_wrong_seed_before_reaudit(tmp_path, field):
    run = dict(seed=43, phase='smoke', group='beam', processed=65,
               manifest_sha256='bound', smoke_gate={'status': 'passed'})
    report = dict(seed=43, status='passed')
    (run if field == 'run' else report)['seed'] = 42
    write_new(tmp_path / 'run.json', run)
    write_new(tmp_path / 'acceptance.json', report)
    with pytest.raises(ValueError, match='Smoke acceptance identity'):
        require_gate({'seed': 43, '_manifest_sha256': 'bound'}, 'beam', tmp_path)


def test_refuse_changed_source(tmp_path, monkeypatch):
    p, m = fixture_manifest(tmp_path, monkeypatch)
    source = tmp_path / 'source.py'
    source.write_text('old')
    m['execution_sources'] = [{'path': str(source), 'sha256': sha(source)}]
    p.write_text(json.dumps(m))
    monkeypatch.setenv('SENSITIVITY_MANIFEST_SHA256', sha(p))
    source.write_text('new')
    with pytest.raises(ValueError, match='Source/input changed'):
        load_bound_manifest(p)


def test_no_overwrite_and_no_missing_gate(tmp_path):
    p = tmp_path / 'one.json'
    write_new(p, {})
    with pytest.raises(FileExistsError):
        write_new(p, {})
    with pytest.raises(ValueError, match='accepted smoke'):
        require_gate({}, 'beam', None)
