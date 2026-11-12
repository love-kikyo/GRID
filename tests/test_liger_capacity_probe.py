"""Pure authority/replay tests: no TensorFlow, GPU, or formal data access."""
import copy
import ast
import json
import shlex
from pathlib import Path

import pytest

from scripts import liger_capacity_manifest as contract
from scripts.probe_liger_capacity import check_replay, clean_step


@pytest.fixture
def fixture_manifest(tmp_path, monkeypatch):
    config_source = Path(contract.CONFIG).read_text()
    files = {}
    for name in ('old.py', 'new.py', 'upstream.py', 'meta.json', 'asset.npy',
                 'catalog.npy', 'stats.json', 'keys.npy', 'rows.npy', 'oldconfig.json',
                 'oldresult.json'):
        path = tmp_path / name
        path.write_text(name)
        files[name] = path
    config = tmp_path / 'config.json'
    config.write_text(config_source)
    prior, receipt = tmp_path / 'prior.json', tmp_path / 'acceptance.json'
    asset = files['asset.npy']
    entry = lambda name: dict(path=str(files[name]), sha256=contract.sha(files[name]))
    old = dict(
        phase='sampled_first_pass_formal', source_git_commit='a' * 40,
        dirty_at_submission=False, automatic_successor=False, initialization='old random',
        upstream_root=str(tmp_path), upstream_commit='c' * 40,
        upstream_files={str(files['upstream.py']): contract.sha(files['upstream.py'])},
        execution_sources={str(files['old.py']): contract.sha(files['old.py'])},
        input_metadata={str(files['meta.json']): contract.sha(files['meta.json'])},
        assets={'frozen_sid': dict(**entry('asset.npy'), size_bytes=asset.stat().st_size,
                                  mtime_ns=asset.stat().st_mtime_ns)},
        catalog=entry('catalog.npy'), standardization=entry('stats.json'),
        index_artifacts={'sid_keys': entry('keys.npy'), 'asset_rows': entry('rows.npy')},
        asset_rows=35458467, data_root=str(tmp_path / 'data'), train_files=['train'],
        validation_files=['evaluation'], seed=42, counts={'parents': 6573674},
        config=str(files['oldconfig.json']), config_sha256=contract.sha(files['oldconfig.json']),
        expected_feature_cache_sha256='d' * 64,
        smoke_acceptance={'old': True}, expected_initial_state_sha256='e' * 64,
        run_directory='old-run', stage='formal')
    prior.write_text(json.dumps(old))
    prior_sha = contract.sha(prior)
    receipt.write_text(json.dumps(dict(status='passed', job_id='159959',
        manifest_sha256=prior_sha, report=entry('oldresult.json'))))
    monkeypatch.setattr(contract, 'ROOT', tmp_path)
    monkeypatch.setattr(contract, 'CONFIG', config)
    monkeypatch.setattr(contract, 'PRIOR', prior)
    monkeypatch.setattr(contract, 'PRIOR_SHA', prior_sha)
    monkeypatch.setattr(contract, 'RECEIPT', receipt)
    monkeypatch.setattr(contract, 'SOURCES', ['new.py'])
    monkeypatch.setattr(contract, 'git', lambda root, *args: '' if args[0] == 'status' else 'b' * 40)
    path = tmp_path / 'launch/manifest.json'
    contract.prepare(path)
    return path, json.loads(path.read_text()), files


def test_manifest_preparation_preserves_all_old_bindings_without_old_weights(fixture_manifest):
    path, manifest, _ = fixture_manifest
    config = contract.verify(manifest)
    assert config['model']['d_model'] == 384
    assert not contract.REMOVED_KEYS.intersection(manifest)
    assert manifest['source_git_commit'] == 'b' * 40
    assert manifest['initialization'] == 'fresh seed42 capacity384 diagnostic weights only'
    with pytest.raises(AssertionError):
        contract.prepare(path)


@pytest.mark.parametrize('key,value', [
    ('phase', 'formal'), ('formal_training_ready', True), ('automatic_successor', True),
    ('test_records_read', True), ('dirty_at_submission', True),
    ('formal_training_ready', 0), ('automatic_successor', None),
    ('source_git_commit', 'x' * 40), ('initialization', 'load teacher'),
    ('seed', 43), ('config', 'replacement.json'), ('asset_rows', 12),
])
def test_manifest_rejects_scope_or_inherited_drift(fixture_manifest, key, value):
    _, manifest, _ = fixture_manifest
    manifest[key] = value
    with pytest.raises(AssertionError):
        contract.verify(manifest)


@pytest.mark.parametrize('key', ['resume_checkpoint', 'test_files', 'initial_checkpoint',
                               'formal_training', 'expected_initial_state_sha256'])
def test_manifest_rejects_unknown_or_old_weight_fields(fixture_manifest, key):
    _, manifest, _ = fixture_manifest
    manifest[key] = 'not-authorized'
    with pytest.raises(AssertionError):
        contract.verify(manifest)


@pytest.mark.parametrize('group', ['execution_sources', 'upstream_files', 'input_metadata'])
def test_manifest_rejects_deleted_source_metadata_groups(fixture_manifest, group):
    _, manifest, _ = fixture_manifest
    manifest[group] = {}
    with pytest.raises(AssertionError):
        contract.verify(manifest)


@pytest.mark.parametrize('name', ['old.py', 'new.py', 'upstream.py', 'meta.json',
                                'asset.npy', 'catalog.npy', 'stats.json', 'keys.npy'])
def test_manifest_rejects_mutated_bound_file(fixture_manifest, name):
    _, manifest, files = fixture_manifest
    files[name].write_text('changed bytes and size')
    with pytest.raises((AssertionError, ValueError)):
        contract.verify(manifest)


def test_manifest_binding_requires_exact_environment(fixture_manifest, monkeypatch):
    path, manifest, _ = fixture_manifest
    monkeypatch.setenv('LIGER_CAPACITY_MANIFEST_SHA256', contract.sha(path))
    monkeypatch.setenv('SOURCE_GIT_COMMIT', manifest['source_git_commit'])
    bound, config = contract.binding(path)
    assert bound == manifest and config['seed'] == 42
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'c' * 40)
    with pytest.raises(AssertionError):
        contract.binding(path)


def replay_reports():
    common = dict(status='passed', state_after_three='state', initial_state_sha256='fresh',
                  checkpoint_binding={'microbatch': 512}, gpus=[{'uuid': 'gpu'}])
    checkpoint = {'path': 'step01', 'sha256': 'bound'}
    steps = [dict(attempt=i, updates=i, total_loss=20 - i, rows=2048,
                  prefix_identity_sha256=f'input{i}', candidate_pool_sha256=f'pool{i}',
                  seconds=1.0, input_seconds=.4, training_seconds=.6,
                  peak_allocated_bytes=100, peak_reserved_bytes=200)
             for i in range(1, 4)]
    a = dict(**common, resumed=False, step_one_checkpoint=checkpoint, steps=steps)
    b = dict(**common, resumed=True, loaded_checkpoint=checkpoint, restore_exact=True,
             steps=copy.deepcopy(steps[1:]))
    return a, b


def test_replay_accepts_only_numerical_and_state_equality_not_timing():
    a, b = replay_reports()
    for step in b['steps']:
        step.update(seconds=4.0, input_seconds=1.0, training_seconds=3.0,
                    peak_allocated_bytes=200, peak_reserved_bytes=300)
    check_replay(a, b)
    assert 'seconds' not in clean_step(a['steps'][0])


@pytest.mark.parametrize('key,value', [('attempt', 9), ('updates', 0), ('total_loss', 1.0),
                                    ('rows', 81), ('prefix_identity_sha256', 'bad'),
                                    ('candidate_pool_sha256', 'wrong pool')])
def test_replay_rejects_learning_or_data_mismatch(key, value):
    a, b = replay_reports()
    b['steps'][0][key] = value
    with pytest.raises(AssertionError):
        check_replay(a, b)


@pytest.mark.parametrize('key,value', [('state_after_three', 'bad'),
    ('initial_state_sha256', 'bad'), ('checkpoint_binding', {}), ('gpus', []),
    ('loaded_checkpoint', {}), ('restore_exact', False), ('resumed', False),
    ('status', 'failed')])
def test_replay_rejects_restore_identity_mismatch(key, value):
    a, b = replay_reports()
    b[key] = value
    with pytest.raises(AssertionError):
        check_replay(a, b)


@pytest.mark.parametrize('kind,gpus', [('train', 1), ('val', 2)])
def test_launchers_keep_whole_job_resource_caps_and_no_successor(kind, gpus):
    root = Path(__file__).resolve().parents[1]
    source = (root / f'run_liger_capacity_{kind}_cpu4.sh').read_text()
    lines = source.splitlines()
    directives = [shlex.split(line.removeprefix('#SBATCH ')) for line in lines
                  if line.startswith('#SBATCH ')]
    assert all(len(parts) == 1 and '=' in parts[0] for parts in directives)
    entries = [parts[0].removeprefix('--').split('=', 1) for parts in directives]
    assert len(dict(entries)) == len(entries)
    options = dict(entries)
    for key, value in dict(partition='rtx4090', nodes='1', ntasks='1',
                           **{'cpus-per-task': '4', 'gres': f'gpu:{gpus}',
                              'mem': '24G', 'time': '00:45:00'}).items():
        assert options[key] == value
    assert 'set -euo pipefail' in lines
    assert 'srun --ntasks=1 --cpus-per-task=4 --cpu-bind=verbose,threads --kill-on-bad-exit=1' in source
    assert '${SOURCE_GIT_COMMIT:?' in source
    assert '${LIGER_CAPACITY_MANIFEST_SHA256:?' in source
    exports = {}
    for line in lines:
        if line.startswith('export '):
            for word in shlex.split(line)[1:]:
                name, value = word.split('=', 1)
                assert name not in exports
                exports[name] = value
    assert exports['GRID_CPU_BUDGET'] == '4'
    assert exports['NUMPY_MADVISE_HUGEPAGE'] == '0'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        assert exports[key] == '1'
    invocations = [shlex.split(line) for line in lines if line.startswith('run_bound ')]
    assert invocations[0][:4] == ['run_bound', '-m', 'pytest', '-q']
    if kind == 'train':
        assert 'tests/test_liger_capacity_runtime.py' in invocations[0]
        assert invocations[1:] == [
            ['run_bound', '-m', 'scripts.probe_liger_capacity', mode, '$1', '--microbatch', str(micro)]
            for micro in (512, 1024) for mode in ('continuous', 'resumed')
        ] + [['run_bound', '-m', 'scripts.probe_liger_capacity', 'accept', '$1']]
    else:
        assert 'tests/test_liger_capacity_validation_probe.py' in invocations[0]
        assert invocations[1:] == [['run_bound', '-m', 'scripts.probe_liger_capacity_validation', '$1']]
    shell_tokens = [token for line in lines if not line.startswith('#') for token in shlex.split(line)]
    assert not {'sbatch', 'torchrun', '--test', '--resume-checkpoint'}.intersection(shell_tokens)


@pytest.mark.parametrize('filename,expected', [
    ('probe_liger_capacity.py', {'mode', 'manifest', '--microbatch'}),
    ('probe_liger_capacity_validation.py', {'manifest'}),
])
def test_probe_cli_ast_exposes_no_formal_test_or_old_checkpoint_override(filename, expected):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'scripts' / filename).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument']
    assert {ast.literal_eval(node.args[0]) for node in calls} == expected
    if filename == 'probe_liger_capacity.py':
        choices = {ast.literal_eval(node.args[0]): ast.literal_eval(keyword.value)
                   for node in calls for keyword in node.keywords if keyword.arg == 'choices'}
        assert choices == {'mode': ['continuous', 'resumed', 'accept'], '--microbatch': [512, 1024]}


def test_manifest_binds_all_new_runtime_acceptance_tests():
    assert {'tests/test_liger_capacity_probe.py', 'tests/test_liger_capacity_runtime.py',
            'tests/test_liger_capacity_validation_probe.py'}.issubset(contract.SOURCES)
