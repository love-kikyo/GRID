"""Read-only, streaming file identity audit; no dataset decoding or ML imports."""
import argparse
import hashlib
import json
import math
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_metadata(raw):
    lines = raw.decode('utf-8').splitlines()
    require(len(lines) == 3, 'Expected three Hugging Face metadata lines')
    revision, etag, timestamp = lines
    require(re.fullmatch('[0-9a-f]{40}', revision), 'Invalid commit hash')
    require(re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', etag), 'Invalid ETag')
    timestamp = float(timestamp)
    require(math.isfinite(timestamp) and timestamp > 0, 'Invalid download timestamp')
    return {'revision': revision, 'etag': etag, 'download_timestamp': timestamp,
            'etag_algorithm': 'sha256' if len(etag) == 64 else 'git_blob_sha1'}


def stat_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def fingerprint(path, git_blob=False):
    """Stream once, detecting ordinary in-place changes or replacement during reading."""
    path = Path(path)
    sha = hashlib.sha256()
    with path.open('rb') as source:
        before = os.fstat(source.fileno())
        blob = hashlib.sha1(f'blob {before.st_size}\0'.encode()) if git_blob else None
        total = 0
        while chunk := source.read(8 * 1024 * 1024):
            sha.update(chunk)
            if blob is not None:
                blob.update(chunk)
            total += len(chunk)
        after = os.fstat(source.fileno())
    require(stat_identity(before) == stat_identity(after) == stat_identity(path.stat()),
            f'File changed during hashing: {path}')
    require(total == before.st_size, f'Short read: {path}')
    return {'sha256': sha.hexdigest(), 'size_bytes': total, 'mtime_ns': before.st_mtime_ns,
            **({'git_blob_sha1': blob.hexdigest()} if blob is not None else {})}


def audit_entry(entry):
    metadata = None
    if 'hf_metadata' in entry:
        binding = entry['hf_metadata']
        raw = Path(binding['path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'Metadata binding changed')
        metadata = parse_metadata(raw)
        require(metadata['revision'] == binding['revision'], 'Metadata revision mismatch')
        require(metadata['etag'] == binding['etag'], 'Metadata ETag mismatch')
    result = {'path': entry['path'], 'role': entry['role'],
              **fingerprint(entry['path'], metadata is not None and len(metadata['etag']) == 40)}
    if metadata:
        require(result[metadata['etag_algorithm']] == metadata['etag'],
                f'Current bytes do not match cached ETag: {entry["path"]}')
        result['hf_metadata'] = {**metadata, 'metadata_sha256': binding['sha256'],
                                 'byte_match': True, 'remote_revision_verified': False}
    if 'expected_sha256' in entry:
        require(result['sha256'] == entry['expected_sha256'],
                f'Frozen SHA256 mismatch: {entry["path"]}')
        result['frozen_sha256_match'] = True
    if 'build_stat' in entry:
        require(all(result[k] == entry['build_stat'][k] for k in ('size_bytes', 'mtime_ns')),
                f'Original build manifest stat mismatch: {entry["path"]}')
        result['build_stat_match'] = True
    return result


def audit_manifest(manifest):
    entries = manifest['files']
    paths = [str(Path(e['path']).resolve()) for e in entries]
    require(len(paths) == len(set(paths)), 'Duplicate input path')
    require(entries, 'Empty inventory')
    results, errors = [], []
    for index, entry in enumerate(entries, 1):
        try:
            results.append(audit_entry(entry))
        except (OSError, ValueError) as exc:
            errors.append({'path': entry['path'], 'error': str(exc)})
        print(f'[{index}/{len(entries)}] {entry["path"]} errors={len(errors)}', flush=True)
    by_path = {r['path']: r for r in results}
    pairs = []
    for left, right in manifest.get('equal_sha256_pairs', []):
        equal = left in by_path and right in by_path and by_path[left]['sha256'] == by_path[right]['sha256']
        pairs.append({'left': left, 'right': right, 'equal': equal})
        if not equal:
            errors.append({'error': 'Asset copy mismatch or unavailable', 'left': left, 'right': right})
    return {'status': 'passed' if not errors else 'failed', 'files': results, 'errors': errors,
            'equal_sha256_pairs': pairs, 'expected_file_count': len(entries),
            'verified_file_count': len(results), 'bytes_read': sum(r['size_bytes'] for r in results)}


def resource_guard():
    """Only the one-CPU Slurm launcher is authorized for a full snapshot scan."""
    require(os.environ.get('SLURM_JOB_ID'), 'Run full audit through Slurm')
    for name in ('SLURM_NTASKS', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_CPUS_PER_NODE'):
        require(os.environ.get(name) == '1', f'{name} must be 1')
    affinity = sorted(os.sched_getaffinity(0))
    require(len(affinity) == 1, f'Expected one CPU affinity, got {affinity}')
    return {'affinity': affinity, 'allocated_cpus': 1, 'job_id': os.environ['SLURM_JOB_ID'],
            'hostname': socket.gethostname(), 'gpus_requested': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    resources = resource_guard()
    raw = args.manifest.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == os.environ.get('SOURCE_MANIFEST_SHA256'), 'Manifest SHA256 mismatch')
    manifest = json.loads(raw)
    require(manifest['source_git_commit'] == os.environ.get('SOURCE_GIT_COMMIT'), 'Source commit mismatch')
    for path, expected in manifest['execution_sources'].items():
        require(fingerprint(path)['sha256'] == expected, f'Execution source changed: {path}')
    require(not args.output.exists(), 'Refusing to overwrite an existing report')
    started = datetime.now(timezone.utc).isoformat()
    report = audit_manifest(manifest)
    report.update({'schema_version': 1, 'resources': resources, 'started_at': started,
                   'finished_at': datetime.now(timezone.utc).isoformat(),
                   'manifest_path': str(args.manifest.resolve()), 'manifest_sha256': digest,
                   'source_git_commit': manifest['source_git_commit'],
                   'scope': 'Current byte identity against local download metadata / frozen assets; '
                            'not remote authentication or proof of historical SID generation code'})
    # Exclusive creation preserves prior audit reports, including failed reports.
    with args.output.open('x') as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
        target.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'verified_file_count', 'expected_file_count', 'bytes_read')}))
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
