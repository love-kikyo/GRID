import hashlib
import json
import os
import subprocess
import sys

import pytest

from scripts.audit_source_snapshot import audit_entry, audit_manifest, fingerprint, parse_metadata, resource_guard


def fixture_entry(tmp_path, git_blob=False):
    path = tmp_path / 'data.bin'
    path.write_bytes(b'example\x00data')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    etag = hashlib.sha1(b'blob 12\x00' + path.read_bytes()).hexdigest() if git_blob else expected
    metadata = tmp_path / 'data.bin.metadata'
    metadata.write_text('a' * 40 + '\n' + etag + '\n1769355346.6\n')
    return {'path': str(path), 'role': 'fixture', 'expected_sha256': expected,
            'build_stat': {'size_bytes': 12, 'mtime_ns': path.stat().st_mtime_ns},
            'hf_metadata': {'path': str(metadata), 'sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
                            'revision': 'a' * 40, 'etag': etag}}


@pytest.mark.parametrize('git_blob', [False, True])
def test_etag_byte_verification(tmp_path, git_blob):
    result = audit_entry(fixture_entry(tmp_path, git_blob))
    assert result['hf_metadata']['byte_match'] and result['frozen_sha256_match']
    assert result['build_stat_match'] and result['size_bytes'] == 12
    assert not result['hf_metadata']['remote_revision_verified']


@pytest.mark.parametrize('change', ['data', 'metadata', 'etag', 'revision', 'sha', 'stat'])
def test_drift_rejected(tmp_path, change):
    entry = fixture_entry(tmp_path)
    if change == 'data':
        (tmp_path / 'data.bin').write_bytes(b'bad')
    elif change == 'metadata':
        (tmp_path / 'data.bin.metadata').write_text('bad')
    elif change in ('etag', 'revision'):
        entry['hf_metadata'][change] = '0' * len(entry['hf_metadata'][change])
    elif change == 'sha':
        entry['expected_sha256'] = '0' * 64
    else:
        entry['build_stat']['mtime_ns'] += 1
    with pytest.raises(ValueError):
        audit_entry(entry)


@pytest.mark.parametrize('raw', [b'', b'a\nb\n1', ('a'*40+'\n'+'b'*64+'\nnan').encode(),
                                ('a'*40+'\n'+'b'*64+'\n-1').encode()])
def test_malformed_metadata(raw):
    with pytest.raises(ValueError):
        parse_metadata(raw)


def test_inventory_and_copy_failure(tmp_path):
    entry = fixture_entry(tmp_path)
    report = audit_manifest({'files': [entry]})
    assert report['status'] == 'passed' and report['verified_file_count'] == 1
    with pytest.raises(ValueError, match='Duplicate'):
        audit_manifest({'files': [entry, entry]})
    bad = audit_manifest({'files': [entry], 'equal_sha256_pairs': [[entry['path'], 'missing']]})
    assert bad['status'] == 'failed'
    os.unlink(entry['path'])
    missing = audit_manifest({'files': [entry]})
    assert missing['status'] == 'failed' and missing['verified_file_count'] == 0


def test_changed_stat_during_hash_rejected(tmp_path, monkeypatch):
    entry = fixture_entry(tmp_path)
    original = os.fstat
    calls = []
    def modified(fd):
        stat = original(fd)
        calls.append(1)
        if len(calls) == 2:
            (tmp_path / 'data.bin').write_bytes(b'changed')
        return stat
    monkeypatch.setattr(os, 'fstat', modified)
    with pytest.raises(ValueError, match='changed'):
        fingerprint(entry['path'])


def test_resource_guard(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    for key in ('SLURM_NTASKS', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_CPUS_PER_NODE'):
        monkeypatch.setenv(key, '1')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: {3})
    assert resource_guard()['affinity'] == [3]
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '24')
    with pytest.raises(ValueError):
        resource_guard()


def test_imports_are_stdlib_only():
    code = 'import scripts.audit_source_snapshot, sys; print(list(sys.modules))'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    for module in ('torch', 'numpy', 'tensorflow', 'pyarrow'):
        assert "'" + module + "'" not in result.stdout
