#!/usr/bin/env python3
"""Export a clean-HEAD allowlist into a new, repository-external draft tree.

No training, data discovery, Git history copy, overwrite or cleanup is performed.
All input bytes are validated and held in memory before creating the output.
A write failure leaves its partial output for inspection; nothing is deleted.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess


_ROOT_DENY = frozenset({
    '.git', '.env', '.aws', '.ssh', '.codex', '.agents', '.cache', '.venv',
    'venv', 'logs', 'data', 'outputs', 'checkpoints', 'lightning_logs', 'wandb',
    'tokens', 'credentials', 'secrets',
})
_ANY_DENY = frozenset({'.git', '.aws', '.ssh', '__pycache__', '.pytest_cache'})
_PRIVATE_STEMS = frozenset({'token', 'tokens', 'credential', 'credentials',
                            'secret', 'secrets', 'password', 'passwords',
                            'id_rsa', 'id_ed25519'})
_DATA_SUFFIXES = ('.ckpt', '.pt', '.pth', '.h5', '.hdf5', '.npy', '.npz',
                  '.tfrecord', '.tfrecords', '.safetensors', '.pkl', '.pickle',
                  '.parquet', '.arrow', '.csv', '.tsv', '.jsonl', '.bin',
                  '.pem', '.key', '.p12', '.pfx', '.gz', '.bz2', '.zst',
                  '.tar', '.zip', '.7z')
_NOTICE_FILES = ('LICENSE', 'notices.txt')


class ReleaseError(ValueError):
    """Invalid input or unsafe export request."""


@dataclass(frozen=True)
class ExportFile:
    source: str
    target: str
    data: bytes
    mode: int


@dataclass(frozen=True)
class ExportPlan:
    repository: Path
    source_commit: str
    manifest_path: str
    manifest_sha256: str
    files: tuple

    def metadata(self):
        # Only repository-relative allowlist names, never local absolute paths.
        return {
            'schema_version': 1, 'status': 'draft',
            'source_commit': self.source_commit,
            'source_manifest': {'path': self.manifest_path, 'sha256': self.manifest_sha256},
            'files': [dict(source=f.source, target=f.target, size=len(f.data),
                           sha256=hashlib.sha256(f.data).hexdigest(),
                           mode='100755' if f.mode == 0o755 else '100644')
                      for f in self.files],
        }


def _git(repository, *arguments):
    process = subprocess.run(
        ['git', '--no-optional-locks', '-C', str(repository), *arguments],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if process.returncode:
        raise ReleaseError('Cannot read the required clean Git HEAD/index state')
    return process.stdout


def _relative_path(value):
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        raise ReleaseError('Allowlist paths must be safe relative POSIX names')
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(p in ('', '.', '..') for p in path.parts):
        raise ReleaseError('Absolute, noncanonical or escaping allowlist path')
    if not path.parts:
        raise ReleaseError('Empty allowlist path')
    parts = tuple(part.casefold() for part in path.parts)
    if (parts[0] in _ROOT_DENY or any(part in _ANY_DENY for part in parts)
            or any(part == '.env' or part.startswith('.env.') for part in parts)
            or parts[-1].lstrip('.').split('.')[0] in _PRIVATE_STEMS
            or parts[-1].endswith(_DATA_SUFFIXES)):
        raise ReleaseError('Data, weights, secrets, Git internals or runtime artifacts are not releasable')
    return path


def _read_regular(repository, relative):
    """Read through no-follow directory FDs, rejecting every symlink component."""
    parts = _relative_path(relative).parts
    directory_fd = os.open(repository, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                          dir_fd=directory_fd)
        with os.fdopen(file_fd, 'rb') as stream:
            file_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ReleaseError('Allowlist source is not a regular file')
            return stream.read(), file_stat.st_mode
    except OSError as exc:
        raise ReleaseError('Missing source, symlink or non-directory source component') from exc
    finally:
        os.close(directory_fd)


def _clean_head_file(repository, relative, commit):
    _relative_path(relative)
    entry = _git(repository, 'ls-files', '--stage', '-z', '--', f':(literal){relative}').split(b'\0')
    if len(entry) != 2 or not entry[0] or entry[1]:
        raise ReleaseError(f'Source must be tracked with one clean index entry: {relative}')
    header, indexed_path = entry[0].split(b'\t', 1)
    mode, object_id, stage = header.split()
    if indexed_path.decode('utf-8') != relative or mode not in (b'100644', b'100755') or stage != b'0':
        raise ReleaseError(f'Source must be a tracked regular file: {relative}')
    head_entry = _git(repository, 'ls-tree', '-z', commit, '--', f':(literal){relative}').split(b'\0')
    if len(head_entry) != 2 or not head_entry[0]:
        raise ReleaseError(f'Source must already exist in HEAD: {relative}')
    head_header, head_path = head_entry[0].split(b'\t', 1)
    head_mode, kind, head_object_id = head_header.split()
    if (head_path != indexed_path or kind != b'blob' or head_mode != mode
            or object_id != head_object_id):
        raise ReleaseError(f'Source index differs from HEAD: {relative}')
    data, working_mode = _read_regular(repository, relative)
    original = _git(repository, 'cat-file', 'blob', head_object_id.decode('ascii'))
    if data != original or bool(working_mode & 0o111) != (mode == b'100755'):
        raise ReleaseError(f'Dirty source differs from HEAD: {relative}')
    return data, 0o755 if mode == b'100755' else 0o644


def _unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseError('Duplicate JSON manifest key')
        result[key] = value
    return result


def prepare_export(repository, manifest_path='release/manifest.json'):
    """Validate manifest plus every source; return an immutable in-memory plan."""
    repository = Path(repository).resolve(strict=True)
    declared_root = Path(_git(repository, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    if declared_root != repository:
        raise ReleaseError('--repo must name the repository root, not a subdirectory')
    commit = _git(repository, 'rev-parse', '--verify', 'HEAD').decode('ascii').strip()
    manifest_bytes, _ = _clean_head_file(repository, manifest_path, commit)
    try:
        manifest = json.loads(manifest_bytes, object_pairs_hook=_unique_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError('Manifest must be valid UTF-8 JSON') from exc
    if (not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int
            or manifest['schema_version'] != 1 or manifest.get('status') != 'draft'
            or not isinstance(manifest.get('files'), list) or not manifest['files']):
        raise ReleaseError('Expected schema_version=1, status=draft and nonempty files allowlist')
    entries, targets = [], set()
    for entry in manifest['files']:
        if not isinstance(entry, dict) or set(entry) != {'source', 'target'}:
            raise ReleaseError('Each allowlist entry must contain exactly source and target')
        source, target = entry['source'], entry['target']
        _relative_path(source)
        target_path = _relative_path(target)
        normalized = target_path.as_posix().casefold()
        if normalized == 'release_manifest.json' or normalized in targets:
            raise ReleaseError('Duplicate or reserved release target')
        targets.add(normalized)
        entries.append((source, target))
    for target in targets:
        if any(parent.as_posix() in targets for parent in PurePosixPath(target).parents
               if parent != PurePosixPath('.')):
            raise ReleaseError('A release file cannot also be another target directory')
    for notice in _NOTICE_FILES:
        if (notice, notice) not in entries:
            raise ReleaseError(f'{notice} must be explicitly listed with its unchanged root filename')
    files = tuple(ExportFile(source, target, *_clean_head_file(repository, source, commit))
                  for source, target in entries)
    return ExportPlan(repository, commit, manifest_path,
                      hashlib.sha256(manifest_bytes).hexdigest(), files)


def _output_directory(repository, output):
    output = Path(os.path.abspath(os.fspath(output)))
    if (output == repository or repository in output.parents or output in repository.parents
            or output == Path(output.anchor)):
        raise ReleaseError('Output must be a new directory outside the repository and its ancestors')
    # Existing parents only, no implicit mkdir -p and no symlink path aliases.
    for part in (*reversed(output.parents), output):
        try:
            metadata = part.lstat()
        except FileNotFoundError:
            if part != output:
                raise ReleaseError('Output parent must already exist')
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ReleaseError('Output path must not contain symlinks')
        if part == output:
            raise ReleaseError('Output already exists; refusing to overwrite')
        if not stat.S_ISDIR(metadata.st_mode):
            raise ReleaseError('Output parent is not a directory')
    return output


def export_release(plan, output):
    """Create the validated draft once; never overwrite or delete any path."""
    output = _output_directory(plan.repository, output)
    metadata = plan.metadata()
    metadata_bytes = (json.dumps(metadata, sort_keys=True, indent=2, ensure_ascii=True) + '\n').encode('utf-8')
    output.mkdir(mode=0o755)
    for file in plan.files:
        destination = output / file.target
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(file.data)
        destination.chmod(file.mode)
    with (output / 'RELEASE_MANIFEST.json').open('xb') as stream:
        stream.write(metadata_bytes)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', default='release/manifest.json', help='Repository-relative manifest')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true', help='Validate only; write nothing')
    action.add_argument('--output', type=Path, help='Explicit nonexistent repository-external directory')
    args = parser.parse_args(argv)
    try:
        plan = prepare_export(args.repo, args.manifest)
        if args.output is not None:
            export_release(plan, args.output)
    except (ReleaseError, OSError) as exc:
        parser.exit(2, f'Release preparation failed: {exc}\n')
    print(json.dumps(dict(status='draft', checked_only=args.check, files=len(plan.files),
                          source_commit=plan.source_commit, manifest_sha256=plan.manifest_sha256),
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
