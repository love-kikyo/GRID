"""Exercise the real shell pipeline with srun stubbed; no GPU or real outputs."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'run_retrieval_content_multiseed_cpu4.sh'
ASSET_ROOT = '/mnt/beegfs/home/MMReco2021/liuyu/GRID'
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')


@pytest.fixture
def invocation(tmp_path):
    binary = tmp_path / 'bin'
    binary.mkdir()
    stub = binary / 'srun'
    stub.write_text('''#!/home/MMReco2021/.conda/envs/GRID/bin/python
import json, os, sys
from pathlib import Path
p = Path(os.environ['LAUNCH_TEST_RECORD'])
records = json.loads(p.read_text()) if p.exists() else []
keys = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
        'NUMEXPR_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TF_NUM_INTRAOP_THREADS',
        'TF_NUM_INTEROP_THREADS', 'GRID_EXPORT_CPU_BUDGET', 'GRID_CPU_BUDGET', 'USE_TF',
        'SOURCE_GIT_COMMIT', 'CONTENT_MANIFEST_SHA256')
records.append({'args': sys.argv[1:], 'cwd': os.getcwd(),
                'environment': {k: os.environ.get(k) for k in keys}})
p.write_text(json.dumps(records))
sys.exit(7 if len(records) == int(os.environ.get('LAUNCH_TEST_FAIL_STEP', '0')) else 0)
''')
    stub.chmod(0o700)
    record = tmp_path / 'calls.json'
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
               LAUNCH_TEST_RECORD=str(record), SLURM_CPUS_PER_TASK='4', SLURM_NTASKS='1',
               SLURM_JOB_ID='999999999', SLURM_TMPDIR=str(tmp_path),
               SOURCE_GIT_COMMIT='fixture-commit', CONTENT_MANIFEST_SHA256='fixture-hash')

    def invoke(seed=43, failure=0, unset=None, manifest=None, overrides=None):
        current = dict(env, LAUNCH_TEST_FAIL_STEP=str(failure))
        if unset:
            current.pop(unset, None)
        current.update(overrides or {})
        name = f'retrieval_content_seed{seed}_full_launcher_test_{tmp_path.name}'
        path = manifest or f'{ASSET_ROOT}/logs/training_launch/{name}/manifest.json'
        result = subprocess.run(['bash', str(LAUNCHER), path], env=current,
                                capture_output=True, text=True)
        calls = json.loads(record.read_text()) if record.exists() else []
        return result, calls, path, f'{ASSET_ROOT}/logs/analysis/{name}'

    return invoke


@pytest.mark.parametrize('seed', [43, 44])
def test_independent_seed_pipeline_order_resources_and_paths(invocation, seed):
    result, calls, manifest, output = invocation(seed=seed)
    assert result.returncode == 0, result.stderr + result.stdout
    assert len(calls) == 7
    expected = [('pytest', None), ('scripts.run_retrieval_content', 'smoke'),
                ('scripts.audit_retrieval_content', 'smoke'),
                ('scripts.analyze_retrieval_content', 'smoke'),
                ('scripts.run_retrieval_content', 'full'),
                ('scripts.audit_retrieval_content', 'full'),
                ('scripts.analyze_retrieval_content', 'full')]
    for call, (module, phase) in zip(calls, expected):
        args, env = call['args'], call['environment']
        assert args[:4] == ['--ntasks=1', '--cpus-per-task=4', '--cpu-bind=verbose,cores',
                            '--kill-on-bad-exit=1']
        assert args[4:7] == ['/home/MMReco2021/.conda/envs/GRID/bin/python', '-m', module]
        assert call['cwd'] == str(ROOT)
        assert all(env[k] == '1' for k in THREADS)
        assert env['GRID_EXPORT_CPU_BUDGET'] == env['GRID_CPU_BUDGET'] == '4'
        assert env['USE_TF'] == '0'
        assert env['SOURCE_GIT_COMMIT'] == 'fixture-commit'
        assert env['CONTENT_MANIFEST_SHA256'] == 'fixture-hash'
        if phase:
            assert args[args.index('--phase') + 1] == phase
            assert args[args.index('--manifest') + 1] == manifest
            assert args[args.index('--output-dir') + 1] == f'{output}/{phase}'
        if module == 'scripts.run_retrieval_content' and phase == 'full':
            assert args[args.index('--smoke-dir') + 1] == output + '/smoke'
        else:
            assert '--smoke-dir' not in args
    assert not Path(output).exists()


@pytest.mark.parametrize('step', range(1, 8))
def test_any_stage_failure_stops_later_stages(invocation, step):
    result, calls, _, _ = invocation(failure=step)
    assert result.returncode == 7
    assert len(calls) == step


@pytest.mark.parametrize('unset', ['SLURM_CPUS_PER_TASK', 'SLURM_NTASKS', 'SLURM_JOB_ID',
                                  'SOURCE_GIT_COMMIT', 'CONTENT_MANIFEST_SHA256'])
def test_required_allocation_and_provenance(invocation, unset):
    result, calls, _, _ = invocation(unset=unset)
    assert result.returncode != 0 and calls == []


@pytest.mark.parametrize('manifest', ['relative.json', '/tmp/manifest.json',
    ASSET_ROOT + '/logs/training_launch/retrieval_content_seed42_full_test/manifest.json',
    ASSET_ROOT + '/logs/training_launch/retrieval_content_seed43_full_test/../manifest.json'])
def test_out_of_scope_manifest_path_rejected(invocation, manifest):
    result, calls, _, _ = invocation(manifest=manifest)
    assert result.returncode != 0 and calls == []


def test_scheduler_resources_allow_parallel_independent_dispatch():
    content = LAUNCHER.read_text()
    for option in ('--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=24G',
                   '--gres=gpu:4090:1', '--partition=rtx4090', '--time=06:00:00'):
        assert '#SBATCH ' + option in content
    assert '--dependency' not in content
    assert '#SBATCH --nodelist' not in content
