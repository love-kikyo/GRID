import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from scripts.select_score_fusion import main, selection_cpu_budget
from src.runtime_cpu import THREAD_VARIABLES


def bounded_environment(monkeypatch):
    for key, value in {
        'GRID_CPU_BUDGET': '4', 'GRID_FUSION_CPU_BUDGET': '4',
        'SLURM_NTASKS': '1', 'SLURM_CPUS_PER_TASK': '4', 'SLURM_JOB_CPUS_PER_NODE': '4',
        **{name: '1' for name in THREAD_VARIABLES},
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {2, 3, 8, 9})


def test_selection_budget_is_opt_in_and_rejects_unbounded_threads(monkeypatch):
    monkeypatch.delenv('GRID_FUSION_CPU_BUDGET', raising=False)
    assert selection_cpu_budget() is None
    bounded_environment(monkeypatch)
    assert selection_cpu_budget()['allocated_cpus'] == 4
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '24')
    with pytest.raises(RuntimeError, match='before Python starts'):
        selection_cpu_budget()


def test_launcher_preserves_full_validation_grid_and_binds_four_cores(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'val.h5'
    source.touch()
    source.with_suffix('.json').touch()
    mocks = '''
srun() { printf 'ARG=%s\\n' "$@"; printf 'THREADS=%s,%s\\n' "$OPENBLAS_NUM_THREADS" "$OMP_NUM_THREADS"; }
mkdir() { :; }
cd() { :; }
'''
    script = (root / 'run_select_score_fusion.sh').read_text()
    result = subprocess.run(['bash', '-c', mocks + script, 'fusion', 'method', str(source)],
                            env={'PATH': os.defpath}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '#SBATCH --nodes=1' in script and '#SBATCH --cpus-per-task=4' in script
    for arg in ['ARG=--ntasks=1', 'ARG=--cpu-bind=verbose,cores', 'ARG=--cpus-per-task=4',
                'ARG=-m\nARG=scripts.select_score_fusion', 'ARG=--rerank-weight-stop\nARG=0.50',
                'ARG=--rerank-weight-step\nARG=0.01', 'ARG=--sample-modulus\nARG=1', 'THREADS=1,1']:
        assert arg in result.stdout


def test_cpu_guard_keeps_selection_identical_and_refuses_test_or_limited_input(tmp_path, monkeypatch):
    source = tmp_path / 'val.h5'
    with h5py.File(source, 'w') as f:
        f['example_hash'] = np.array([[1, 2], [3, 4]], dtype=np.uint64)
        f['target_sid'] = np.array([[1], [3]], dtype=np.int16)
        f['beam_sid'] = np.array([[[2], [1]], [[3], [4]]], dtype=np.int16)
        f['beam_score'] = np.array([[1, 0], [1, 0]], dtype=np.float32)
        f['raw_rerank_logit'] = np.array([[0, 10], [0, 0]], dtype=np.float32)
    meta = dict(split='val', sample_unit='unique_user_history', num_samples=2,
                limits={'max_batches': None, 'max_samples': None}, checkpoint_sha256='fixture',
                dataset_provenance={'protocol': 'fixture', 'expected_samples': 2,
                                    'manifests': dict(train={}, test={}, audit={})})
    source.with_suffix('.json').write_text(json.dumps(meta))

    def invoke(tag):
        out = tmp_path / f'{tag}.json'
        monkeypatch.setattr(sys, 'argv', ['select', '--input', str(source), '--output-json', str(out),
                                        '--output-md', str(out.with_suffix('.md'))])
        main()
        return json.loads(out.read_text())

    monkeypatch.delenv('GRID_FUSION_CPU_BUDGET', raising=False)
    baseline = invoke('baseline')
    bounded_environment(monkeypatch)
    bounded = invoke('bounded')
    assert baseline['candidates'] == bounded['candidates'] and len(bounded['candidates']) == 51
    assert baseline['selected'] == bounded['selected']
    assert bounded['cpu_budget']['allocated_cpus'] == 4
    assert len(bounded['cpu_process_tree_samples']) == 2 and len(bounded['source_files']) == 5
    with pytest.raises(FileExistsError):
        invoke('bounded')
    meta['split'] = 'test'
    source.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='validation export'):
        invoke('test')
    meta['split'] = 'val'
    meta['limits']['max_batches'] = 2
    source.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='limited smoke'):
        invoke('limited')
