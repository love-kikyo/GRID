import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import prepare_repro_release as release


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def write(root, relative, data):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def commit(root):
    git(root, 'add', '--all')
    git(root, '-c', 'user.name=Release Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-m', 'fixture')


def save_manifest(root, entries=None, **overrides):
    manifest = dict(schema_version=1, status='draft', files=entries if entries is not None else [
        dict(source='LICENSE', target='LICENSE'), dict(source='notices.txt', target='notices.txt'),
        dict(source='src/model.py', target='src/model.py'),
    ])
    manifest.update(overrides)
    write(root, 'release/manifest.json', json.dumps(manifest).encode())


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init')
    write(root, 'LICENSE', b'Original license\r\n\x00preserved bytes\n')
    write(root, 'notices.txt', b'Original third-party notices\n')
    write(root, 'src/model.py', b'print("fixture only")\n')
    write(root, 'unlisted.txt', b'never export this\n')
    save_manifest(root)
    commit(root)
    return root


def entries(root):
    return json.loads((root / 'release/manifest.json').read_bytes())['files']


def test_export_only_clean_allowlist_preserves_notices_and_has_no_absolute_metadata(repository, tmp_path):
    output = tmp_path / 'release-draft'
    plan = release.prepare_export(repository)
    metadata = release.export_release(plan, output)
    assert sorted(str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()) == [
        'LICENSE', 'RELEASE_MANIFEST.json', 'notices.txt', 'src/model.py']
    assert not (output / '.git').exists() and not (output / 'unlisted.txt').exists()
    for item in metadata['files']:
        exported = (output / item['target']).read_bytes()
        assert exported == (repository / item['source']).read_bytes()
        assert item['sha256'] == hashlib.sha256(exported).hexdigest()
        assert item['size'] == len(exported)
    assert metadata['source_commit'] == git(repository, 'rev-parse', 'HEAD').decode().strip()
    assert metadata['status'] == 'draft'
    assert metadata['source_manifest']['sha256'] == hashlib.sha256(
        (repository / 'release/manifest.json').read_bytes()).hexdigest()
    serialized = (output / 'RELEASE_MANIFEST.json').read_text()
    assert str(tmp_path) not in serialized and 'file://' not in serialized
    assert json.loads(serialized) == metadata


def test_check_only_writes_nothing_and_cli_requires_explicit_action(repository, tmp_path, capsys):
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*'))
    assert release.main(['--repo', str(repository), '--check']) == 0
    assert json.loads(capsys.readouterr().out)['checked_only'] is True
    assert before == sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*'))
    with pytest.raises(SystemExit) as error:
        release.main(['--repo', str(repository)])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        release.main(['--repo', str(repository), '--check', '--output', str(tmp_path / 'new')])


@pytest.mark.parametrize('target', [
    '../escaped', '/absolute', './file', 'a/../file', 'a//file', 'a\\file', 'C:/file',
    '.git/config', 'docs/.git/config', '.env', '.env.production', 'docs/.env',
    'logs/file.txt', 'data/file.txt', 'outputs/file.txt', 'checkpoints/file.txt',
    'file.ckpt', 'file.pt', 'file.pth', 'file.h5', 'file.npy', 'file.npz',
    'file.tfrecord.gz', 'file.safetensors', 'token.json', 'credentials.txt',
    'secrets.json', '.ssh/id_rsa', 'docs/__pycache__/cache.pyc', 'a\nfile',
])
def test_unsafe_target_rejected_before_mkdir(repository, tmp_path, target):
    selected = entries(repository)
    selected[-1]['target'] = target
    save_manifest(repository, selected)
    commit(repository)
    output = tmp_path / 'must-not-exist'
    with pytest.raises(SystemExit) as error:
        release.main(['--repo', str(repository), '--output', str(output)])
    assert error.value.code == 2 and not output.exists()


def test_src_data_is_allowed_but_private_source_cannot_be_renamed(repository):
    write(repository, 'src/data/loading.py', b'# source, not raw data\n')
    selected = entries(repository) + [dict(source='src/data/loading.py', target='src/data/loading.py')]
    save_manifest(repository, selected)
    commit(repository)
    assert len(release.prepare_export(repository).files) == 4
    write(repository, 'secret.ckpt', b'not a checkpoint, but disallowed by policy')
    selected.append(dict(source='secret.ckpt', target='safe.txt'))
    save_manifest(repository, selected)
    commit(repository)
    with pytest.raises(release.ReleaseError, match='not releasable'):
        release.prepare_export(repository)


@pytest.mark.parametrize('targets', [('src/model.py', 'src/model.py'),
    ('src/model.py', 'SRC/model.py'), ('thing', 'thing/nested.py'),
    ('RELEASE_MANIFEST.json', 'other.py')])
def test_target_collisions_reserved_metadata_and_file_directory_conflicts(repository, targets):
    selected = entries(repository)[:2] + [dict(source='src/model.py', target=t) for t in targets]
    save_manifest(repository, selected)
    commit(repository)
    with pytest.raises(release.ReleaseError):
        release.prepare_export(repository)


@pytest.mark.parametrize('change', ['working', 'index', 'untracked', 'new_tracked', 'mode'])
def test_only_tracked_clean_head_sources_are_allowed(repository, change):
    if change in ('working', 'index'):
        write(repository, 'src/model.py', b'changed\n')
        if change == 'index':
            git(repository, 'add', 'src/model.py')
            write(repository, 'src/model.py', b'print("fixture only")\n')
    elif change == 'mode':
        (repository / 'src/model.py').chmod(0o755)
    else:
        selected = entries(repository) + [dict(source='new.py', target='new.py')]
        save_manifest(repository, selected)
        commit(repository)
        write(repository, 'new.py', b'new source\n')
        if change == 'new_tracked':
            git(repository, 'add', 'new.py')
    with pytest.raises(release.ReleaseError):
        release.prepare_export(repository)


def test_unrelated_worktree_changes_do_not_leak_or_block_allowlist(repository):
    write(repository, 'unlisted.txt', b'dirty but not released\n')
    write(repository, 'local-private.txt', b'never released\n')
    assert len(release.prepare_export(repository).files) == 3


@pytest.mark.parametrize('kind', ['file', 'directory', 'fifo'])
def test_source_symlinks_and_nonregular_files_rejected(repository, tmp_path, kind):
    model = repository / 'src/model.py'
    if kind == 'directory':
        original = repository / 'src-original'
        (repository / 'src').rename(original)
        (repository / 'src').symlink_to(original, target_is_directory=True)
    else:
        model.unlink()
        if kind == 'file':
            external = tmp_path / 'external.py'
            external.write_bytes(b'print("fixture only")\n')
            model.symlink_to(external)
        else:
            os.mkfifo(model)
    with pytest.raises(release.ReleaseError):
        release.prepare_export(repository)


def test_tracked_symlink_is_rejected_even_if_git_clean(repository):
    (repository / 'src/model.py').unlink()
    (repository / 'src/model.py').symlink_to('../unlisted.txt')
    commit(repository)
    with pytest.raises(release.ReleaseError, match='regular'):
        release.prepare_export(repository)


@pytest.mark.parametrize('change', ['dirty', 'untracked', 'schema', 'status', 'missing_license',
    'renamed_notices', 'duplicate_key', 'unknown_entry_key'])
def test_manifest_is_tracked_clean_strict_and_preserves_both_notices(repository, change):
    manifest = repository / 'release/manifest.json'
    if change == 'dirty':
        manifest.write_bytes(manifest.read_bytes() + b'\n')
    elif change == 'untracked':
        git(repository, 'rm', '--cached', 'release/manifest.json')
    else:
        selected = entries(repository)
        if change == 'schema': save_manifest(repository, schema_version=True)
        elif change == 'status': save_manifest(repository, status='published')
        elif change == 'missing_license': save_manifest(repository, selected[1:])
        elif change == 'renamed_notices':
            selected[1]['target'] = 'NOTICE'
            save_manifest(repository, selected)
        elif change == 'unknown_entry_key':
            selected[0]['unexpected'] = True
            save_manifest(repository, selected)
        else:
            manifest.write_bytes(b'{"schema_version":1,"schema_version":1,"status":"draft","files":[]}')
        commit(repository)
    with pytest.raises(release.ReleaseError):
        release.prepare_export(repository)


def test_output_refuses_existing_repo_descendant_ancestor_root_and_symlink_paths(repository, tmp_path):
    plan = release.prepare_export(repository)
    existing = tmp_path / 'existing'
    existing.mkdir()
    sentinel = existing / 'keep.txt'
    sentinel.write_bytes(b'untouched')
    link = tmp_path / 'linked'
    link.symlink_to(existing, target_is_directory=True)
    for output in (existing, repository, repository / 'new', repository.parent, Path('/'),
                   link / 'new', tmp_path / 'missing-parent/new'):
        with pytest.raises(release.ReleaseError):
            release.export_release(plan, output)
    assert sentinel.read_bytes() == b'untouched'
    assert not (existing / 'new').exists() and not (repository / 'new').exists()


def test_executable_modes_are_copied_from_clean_head(repository, tmp_path):
    (repository / 'src/model.py').chmod(0o755)
    commit(repository)
    output = tmp_path / 'executable-release'
    release.export_release(release.prepare_export(repository), output)
    assert (output / 'src/model.py').stat().st_mode & 0o777 == 0o755
    assert (output / 'LICENSE').stat().st_mode & 0o777 == 0o644


def test_last_bad_source_prevents_any_output_creation(repository, tmp_path):
    selected = entries(repository) + [dict(source='missing.py', target='missing.py')]
    save_manifest(repository, selected)
    commit(repository)
    output = tmp_path / 'not-created'
    with pytest.raises(SystemExit):
        release.main(['--repo', str(repository), '--output', str(output)])
    assert not output.exists()


def test_write_failure_preserves_partial_directory_without_deleting_anything(repository, tmp_path, monkeypatch):
    plan = release.prepare_export(repository)
    output = tmp_path / 'partial'
    original_open = Path.open
    def fail_notices(path, *args, **kwargs):
        if path == output / 'notices.txt':
            raise OSError('simulated full disk')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_notices)
    with pytest.raises(OSError, match='full disk'):
        release.export_release(plan, output)
    assert output.is_dir() and (output / 'LICENSE').read_bytes() == (repository / 'LICENSE').read_bytes()
    assert not (output / 'RELEASE_MANIFEST.json').exists()


def test_manifest_path_cannot_escape_repository(repository):
    with pytest.raises(release.ReleaseError):
        release.prepare_export(repository, '../manifest.json')
