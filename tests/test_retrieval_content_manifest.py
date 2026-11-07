import copy
import json

import pytest

from scripts.prepare_retrieval_content import load_manifest, scope_fields, sha, write_new


def fixture(tmp_path, monkeypatch):
    m = copy.deepcopy(scope_fields())
    m.update(source_git_commit='test', execution_sources=[], input_metadata=[])
    path = tmp_path / 'manifest.json'
    write_new(path, m)
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'test')
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(path))
    return path, m


def test_scope_and_manifest_hash(tmp_path, monkeypatch):
    p, _ = fixture(tmp_path, monkeypatch)
    assert load_manifest(p)['full_samples'] == 2061101
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', 'wrong')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_manifest(p)


@pytest.mark.parametrize('key,value', [('seed', 43), ('full_samples', 16384),
    ('smoke_samples', 64), ('policies', ['sim10']), ('physical_slots', 5),
    ('C', 40), ('beta', .1), ('runtime_beta', .5), ('num_workers', 4),
    ('asset_root', '/wrong'), ('execution_root', '/wrong'), ('dirty_at_submission', True)])
def test_scope_drift(tmp_path, monkeypatch, key, value):
    p, m = fixture(tmp_path, monkeypatch)
    m[key] = value
    p.write_text(json.dumps(m))
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(p))
    with pytest.raises(ValueError, match='authorized scope'):
        load_manifest(p)


@pytest.mark.parametrize('key,value', [('bootstrap_samples', 100), ('bootstrap_seed', 42),
    ('confidence', .90), ('primary_comparison', 'sim10_minus_mask0:fused'),
    ('primary_metric', 'recall@10'), ('ks', [10])])
def test_statistics_frozen(tmp_path, monkeypatch, key, value):
    p, m = fixture(tmp_path, monkeypatch)
    m['statistics'][key] = value
    p.write_text(json.dumps(m))
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(p))
    with pytest.raises(ValueError, match='authorized scope'):
        load_manifest(p)


def test_source_and_commit_hash(tmp_path, monkeypatch):
    p, m = fixture(tmp_path, monkeypatch)
    source = tmp_path / 'source.py'
    source.write_text('original')
    m['execution_sources'] = [{'path': str(source), 'sha256': sha(source)}]
    p.write_text(json.dumps(m))
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(p))
    source.write_text('changed')
    with pytest.raises(ValueError, match='Changed accepted source'):
        load_manifest(p)
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'wrong')
    with pytest.raises(ValueError, match='commit binding'):
        load_manifest(p)


def test_counts_require_integers(tmp_path, monkeypatch):
    p, m = fixture(tmp_path, monkeypatch)
    m['full_samples'] = float(m['full_samples'])
    p.write_text(json.dumps(m))
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(p))
    with pytest.raises(ValueError, match='integer scope'):
        load_manifest(p)
