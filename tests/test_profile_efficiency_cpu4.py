"""Resource, frozen-input and accounting checks; CUDA equivalence is a smoke gate."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.profile_efficiency_cpu4 import (
    VARIANTS, budget_guard, load_manifest, sha, summary, validate_cases, write_new,
)
from src.runtime_cpu import THREAD_VARIABLES


def test_resource_guard_before_numeric_imports(monkeypatch):
    monkeypatch.delenv('GRID_PROFILE_CPU_BUDGET', raising=False)
    with pytest.raises(RuntimeError, match='GRID_PROFILE_CPU_BUDGET'):
        budget_guard()
    for k, v in {'GRID_PROFILE_CPU_BUDGET': '4', 'GRID_CPU_BUDGET': '4', 'SLURM_NTASKS': '1',
                 'SLURM_CPUS_PER_TASK': '4', 'SLURM_JOB_CPUS_PER_NODE': '4',
                 **{name: '1' for name in THREAD_VARIABLES}}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: {0, 1, 28, 29})
    assert budget_guard()['allocated_cpus'] == 4
    monkeypatch.setenv('OMP_NUM_THREADS', '24')
    with pytest.raises(RuntimeError, match='before Python starts'):
        budget_guard()
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: set(range(8)))
    with pytest.raises(RuntimeError, match='four-core'):
        budget_guard()
    subprocess.run([sys.executable, '-c', 'import scripts.profile_efficiency_cpu4; import sys; '
                    'assert "torch" not in sys.modules and "numpy" not in sys.modules'], check=True)


def rows(amp='bf16'):
    return [dict(batch_index=i, batch_size=32, amp=amp, gen_cuda_ms=8., rerank_cuda_ms=2., total_cuda_ms=10.,
                 wall_ms=12., baseline_allocated_gib=1., peak_allocated_gib=3., peak_increment_gib=2.,
                 peak_reserved_gib=4., natural_beam_sha256=f'fixture-{i}') for i in range(3)]


def cases():
    hardware = dict(uuid='GPU-fixture', hostname='node', name='RTX 4090', torch='fixture', cuda='fixture')
    return [dict(status='passed', model=model, repeat=0, phase='smoke', warmup_batches=2,
                 measure_batches=3, input_sha256='same-input', hardware_start=hardware.copy(),
                 hardware_end=hardware.copy(),
                 variants={v: {'records': rows('none' if v.endswith('_fp32') else 'bf16'), 'summary': summary(rows())} for v in variants})
            for model, variants in VARIANTS.items()]


def test_summary_reports_batch_latency_and_correct_throughput():
    s = summary(rows())
    assert s['wall_ms']['mean'] == s['wall_ms']['p50'] == s['wall_ms']['p95'] == 12.
    assert math.isclose(s['users_per_second_wall'], 32/.012)
    assert s['peak_allocated_gib']['max'] == 3.
    with pytest.raises(ValueError, match='Empty'):
        summary([])
    for value in (float('nan'), float('inf'), -1):
        r = rows(); r[0]['wall_ms'] = value
        with pytest.raises(ValueError, match='Invalid'):
            summary(r)


def test_precision_groups_are_explicit_and_bf16_no_cache_is_not_timed():
    c = cases()
    assert 'rerank_no_cache' not in VARIANTS['listwise']
    for case in c:
        for variant, record in case['variants'].items():
            if variant.endswith('_fp32'):
                for row in record['records']:
                    row['natural_beam_sha256'] += '-fp32'
    assert validate_cases(c, 'smoke')
    c[1]['variants']['rerank_no_cache_fp32']['records'][0]['amp'] = 'bf16'
    with pytest.raises(AssertionError): validate_cases(c, 'smoke')


@pytest.mark.parametrize('damage', ['gpu', 'input', 'candidate', 'count', 'summary', 'missing'])
def test_suite_requires_same_gpu_inputs_candidates_and_complete_measurements(damage):
    c = cases()
    assert validate_cases(c, 'smoke')
    c = copy.deepcopy(c)
    if damage == 'gpu': c[1]['hardware_end']['uuid'] = 'GPU-other'
    if damage == 'input': c[1]['input_sha256'] = 'different-input'
    if damage == 'candidate': c[-1]['variants']['separate_rerank']['records'][0]['natural_beam_sha256'] = 'changed'
    if damage == 'count': c[0]['variants']['beam_only']['records'].pop()
    if damage == 'summary': c[0]['variants']['beam_only']['summary']['wall_ms']['mean'] = 1.
    if damage == 'missing': c.pop()
    with pytest.raises(AssertionError):
        validate_cases(c, 'smoke')


def test_frozen_manifest_and_no_overwrite(tmp_path, monkeypatch):
    source = tmp_path/'source'; source.write_text('frozen')
    path = tmp_path/'manifest.json'
    record = dict(source_git_commit='fixture', dirty_at_submission=False, models={k: {} for k in VARIANTS},
                  seed=44, execution_sources=[dict(path=str(source), sha256=sha(source))])
    write_new(path, record)
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'fixture')
    monkeypatch.setenv('PROFILE_MANIFEST_SHA256', sha(path))
    assert load_manifest(path) == record
    with pytest.raises(FileExistsError): write_new(path, record)
    source.write_text('changed')
    with pytest.raises(ValueError, match='Fingerprint changed'): load_manifest(path)


def test_launcher_is_serial_and_four_core_only():
    text = (Path(__file__).resolve().parents[1]/'run_profile_efficiency_cpu4.sh').read_text()
    for snippet in ['#SBATCH --cpus-per-task=4', '#SBATCH --gres=gpu:1', '--cpu-bind=verbose,threads',
                    'GRID_PROFILE_CPU_BUDGET=4', 'OPENBLAS_NUM_THREADS=1', 'TF_NUM_INTEROP_THREADS=1',
                    'set -euo pipefail', '-m scripts.profile_efficiency_cpu4']:
        assert snippet in text
    assert '--exclusive' not in text
