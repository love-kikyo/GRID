import math
import os
import sys
import subprocess

import pytest

from scripts.analyze_beam_cohorts_cpu4 import (
    GROUPS, summarize_matrix, guard, close, ranks, accumulate, check_training_seed, accepted_metrics,
)
from src.runtime_cpu import THREAD_VARIABLES


def matrix(*rows):
    m = [[0] * 21 for _ in range(21)]
    for i, j, count in rows:
        m[i][j] += count
    return m


def test_empty_cohort_is_not_zero_conditional_effect():
    result = summarize_matrix(matrix(), 10)
    assert result['num_examples'] == 0
    assert result['metrics']['ndcg@5']['conditional_delta'] is None
    assert result['metrics']['ndcg@5']['population_delta_contribution'] == 0


def test_disjoint_cohorts_reconstruct_population_not_unweighted_means():
    fixtures = [matrix((0, 0, 4)), matrix((1, 0, 2)), matrix((0, 1, 3)), matrix((10, 1, 1))]
    summaries = [summarize_matrix(m, 10) for m in fixtures]
    assert GROUPS == ('neither', 'baseline_only', 'challenger_only', 'both')
    assert sum(s['num_examples'] for s in summaries) == 10
    assert math.isclose(sum(s['metrics']['recall@5']['population_delta_contribution'] for s in summaries), .2)
    assert summaries[1]['metrics']['recall@5']['conditional_delta'] == -1
    assert summaries[1]['metrics']['recall@5']['population_delta_contribution'] == -.2


def test_rank_cutoffs_and_discounts():
    m = summarize_matrix(matrix((5, 6, 1), (10, 11, 1), (0, 20, 1)), 3)['metrics']
    close(m['recall@5']['conditional_delta'], -1/3)
    close(m['recall@10']['conditional_delta'], -1/3)
    close(m['ndcg@5']['baseline_mean'], 1/math.log2(6)/3)
    close(m['ndcg@10']['challenger_mean'], 1/math.log2(7)/3)


@pytest.mark.parametrize('m,n', [(matrix(), 0), ([[0]], 1), (matrix((1, 1, -1)), 1),
                                  (matrix((1, 1, 2)), 1), (matrix((1, 1, 1.5)), 2)])
def test_invalid_histogram_rejected(m, n):
    with pytest.raises(ValueError):
        summarize_matrix(m, n)


def test_guard_and_lazy_imports(monkeypatch):
    for k, v in {'GRID_CPU_BUDGET':'4', 'SLURM_NTASKS':'1', 'SLURM_CPUS_PER_TASK':'4',
                 'SLURM_JOB_CPUS_PER_NODE':'4', **{k:'1' for k in THREAD_VARIABLES}}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: {0, 1, 4, 5})
    assert guard()['allocated_cpus'] == 4
    monkeypatch.setenv('OMP_NUM_THREADS', '24')
    with pytest.raises(RuntimeError):
        guard()
    subprocess.run([sys.executable, '-c', 'import scripts.analyze_beam_cohorts_cpu4; import sys; '
                    'assert "torch" not in sys.modules and "numpy" not in sys.modules'], check=True)


def test_reconstruction_rejects_wrong_metric():
    with pytest.raises(ValueError):
        close(.1, .100001)


def test_numpy_rank_and_all_four_groups():
    import numpy as np
    candidates = np.broadcast_to(np.arange(20)[None, :, None], (4, 20, 1)).copy()
    baseline = ranks(np.array([[30], [0], [30], [19]]), candidates)
    challenger = ranks(np.array([[30], [30], [4], [5]]), candidates)
    h = np.zeros((4, 21, 21), dtype=np.int64)
    accumulate(h, baseline, challenger)
    assert baseline.tolist() == [0, 1, 0, 20]
    assert challenger.tolist() == [0, 0, 5, 6]
    assert h.sum(axis=(1, 2)).tolist() == [1, 1, 1, 1]
    assert h[0, 0, 0] == h[1, 1, 0] == h[2, 0, 5] == h[3, 20, 6] == 1
    candidates[1, 1] = candidates[1, 0]
    with pytest.raises(ValueError, match='Duplicate GT'):
        ranks(np.array([[30], [0], [30], [19]]), candidates)


def test_actual_stable_fusion_tie_and_absent_target():
    import numpy as np
    from scripts.analyze_paired_evaluation import ranked_candidates
    p = {'beam_sid':np.arange(20).reshape(1, 20, 1),
         'beam_score':np.zeros((1, 20), dtype=np.float32),
         'raw_rerank_logit':np.zeros((1, 20), dtype=np.float32)}
    assert ranks(np.array([[4]]), ranked_candidates(p, 'fusion@1@0.50')).tolist() == [5]
    p['raw_rerank_logit'][0, 19] = 2
    ranked = ranked_candidates(p, 'fusion@1@0.50')
    assert ranks(np.array([[4]]), ranked).tolist() == [6]
    assert ranks(np.array([[99]]), ranked).tolist() == [0]


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_seed_bound_to_top_level_frozen_configuration(seed):
    check_training_seed(seed, f'model:\n  seed: 0\nseed: {seed}\nother: 1\n')
    with pytest.raises(ValueError):
        check_training_seed(seed, 'seed: 0\n')
    with pytest.raises(ValueError):
        check_training_seed(seed, f'seed: {seed}\nseed: {seed}\n')
    with pytest.raises(ValueError):
        check_training_seed(str(seed), f'seed: {seed}\n')


def test_legacy_metric_receipt_is_only_accepted_for_seed42():
    fixture = {'verified_metrics': {'sid:beam': {'recall@5': .2}}}
    assert accepted_metrics(fixture, 42) == fixture['verified_metrics']
    with pytest.raises(ValueError):
        accepted_metrics(fixture, 43)
    for seed in (42, 43, 44):
        assert accepted_metrics({'independently_recomputed_metrics': fixture['verified_metrics']}, seed) == fixture['verified_metrics']
