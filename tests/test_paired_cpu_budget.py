from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from scripts.analyze_paired_cpu4 import paired_cpu_budget, run
from scripts.analyze_paired_evaluation import analyze_h5
from src.runtime_cpu import THREAD_VARIABLES


def bounded_environment(monkeypatch):
    for key, value in {
        'GRID_CPU_BUDGET': '4', 'GRID_PAIRED_CPU_BUDGET': '4',
        'SLURM_NTASKS': '1', 'SLURM_CPUS_PER_TASK': '4', 'SLURM_JOB_CPUS_PER_NODE': '4',
        **{name: '1' for name in THREAD_VARIABLES},
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {2, 3, 8, 9})


def test_budget_rejects_missing_guard_threads_and_oversized_affinity(monkeypatch):
    monkeypatch.delenv('GRID_PAIRED_CPU_BUDGET', raising=False)
    with pytest.raises(RuntimeError, match='GRID_PAIRED_CPU_BUDGET'):
        paired_cpu_budget()
    bounded_environment(monkeypatch)
    assert paired_cpu_budget()['allocated_cpus'] == 4
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '24')
    with pytest.raises(RuntimeError, match='before Python starts'):
        paired_cpu_budget()
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '1')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: set(range(8)))
    with pytest.raises(RuntimeError, match='four-core ceiling'):
        paired_cpu_budget()


def test_launcher_preserves_seed42_comparison_order_and_binds_four_cores(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'test.h5'
    source.touch()
    selection = source.with_suffix('.json')
    selection.touch()
    mocks = '''
srun() { printf 'ARG=%s\\n' "$@"; printf 'THREADS=%s,%s\\n' "$OPENBLAS_NUM_THREADS" "$OMP_NUM_THREADS"; }
mkdir() { :; }
cd() { :; }
'''
    script = (root / 'run_paired_analysis.sh').read_text()
    result = subprocess.run(['bash', '-c', mocks + script, 'paired', 'fixture',
                             *[str(p) for p in [source, selection] * 3], str(source)],
                            env={'PATH': os.defpath}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '#SBATCH --nodes=1' in script and '#SBATCH --cpus-per-task=4' in script
    for arg in ['ARG=--ntasks=1', 'ARG=--cpu-bind=verbose,cores', 'ARG=--cpus-per-task=4',
                'ARG=-m\nARG=scripts.analyze_paired_cpu4', 'ARG=--bootstrap-samples\nARG=5000',
                'ARG=--bootstrap-batch-size\nARG=500', 'THREADS=1,1']:
        assert arg in result.stdout
    lines = result.stdout.splitlines()
    compares = [lines[i + 1][4:] for i, line in enumerate(lines) if line == 'ARG=--compare']
    assert compares == [
        'sid:beam,listwise:selected', 'sid:beam,pointwise:selected', 'sid:beam,separate:selected',
        'pointwise:selected,listwise:selected', 'separate:selected,listwise:selected',
        'listwise:beam,listwise:selected', 'pointwise:beam,pointwise:selected',
        'separate:beam,separate:selected', 'listwise:click,listwise:selected',
        'pointwise:click,pointwise:selected', 'separate:click,separate:selected',
    ]
    assert sum(line.startswith('ARG=sid=') for line in lines) == 1


def test_wrapper_preserves_metrics_and_intervals_and_rejects_bad_inputs(tmp_path, monkeypatch):
    bounded_environment(monkeypatch)
    source = tmp_path / 'test.h5'
    with h5py.File(source, 'w') as f:
        f['user_id'] = np.array([1, 2], dtype=np.int64)
        f['example_hash'] = np.array([[1, 2], [3, 4]], dtype=np.uint64)
        f['target_sid'] = np.array([[1], [3]], dtype=np.int16)
        f['beam_sid'] = np.array([[[2], [1]], [[3], [4]]], dtype=np.int16)
        f['beam_score'] = np.array([[1, 0], [1, 0]], dtype=np.float32)
        f['raw_rerank_logit'] = np.array([[0, 10], [0, 0]], dtype=np.float32)
        f['rerank_sid'] = np.array([[[1], [2]], [[3], [4]]], dtype=np.int16)
    meta = dict(split='test', sample_unit='unique_user_history', num_samples=2,
                limits={'max_batches': None, 'max_samples': None}, checkpoint_sha256='fixture',
                dataset_provenance={'protocol': 'fixture', 'expected_samples': 2,
                                    'manifests': dict(train={}, test={}, audit={})})
    sidecar = source.with_suffix('.json')
    sidecar.write_text(json.dumps(meta))
    args = Namespace(input=[f'model={source}'], selection=[], compare=['model:beam,model:click'],
                     ks=[1, 2], bootstrap_samples=500, bootstrap_batch_size=50,
                     confidence=.95, seed=2027, chunk_size=1, expected_split='test',
                     output_json=tmp_path / 'report.json', output_md=tmp_path / 'report.md')
    baseline = analyze_h5(args)
    bounded = run(args)
    assert baseline['comparisons'] == bounded['comparisons']
    assert baseline['analyzer_sha256'] == bounded['analyzer_sha256']
    assert len(bounded['cpu_process_tree_samples']) == 2 and len(bounded['source_files']) == 5
    assert json.loads(args.output_json.read_text()) == bounded
    with pytest.raises(FileExistsError):
        run(args)
    args.output_json, args.output_md = tmp_path / 'other.json', tmp_path / 'other.md'
    meta['limits']['max_samples'] = 2
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='limited smoke'):
        run(args)
    args.expected_split = 'val'
    with pytest.raises(ValueError, match='full test'):
        run(args)
    args.expected_split = 'test'
    args.input = ['model=fixture.npz']
    with pytest.raises(ValueError, match='HDF5'):
        run(args)
