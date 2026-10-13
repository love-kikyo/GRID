from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.accept_sim_fusion import contributions, ORDER
from scripts.analyze_paired_evaluation import ranked_candidates, metric_contributions
from scripts.export_sim_cached_validation import verify_runtime_fusion
from scripts.smoke_sim_cached_rerank import score_cached_batch


@pytest.mark.parametrize('tied', [False, True])
def test_independent_metrics_match_stable_sort_with_no_hits(tied):
    rng = np.random.default_rng(42)
    beam = np.arange(100 * 20 * 4).reshape(100, 20, 4)
    target = beam[np.arange(100), rng.integers(0, 20, 100)].copy()
    target[::7] = -1
    scores = rng.integers(-2, 3, (100, 20)).astype(np.float32) if tied else rng.normal(size=(100, 20)).astype(np.float32)
    result = contributions(target, beam, scores)
    ranked = ranked_candidates(dict(beam_sid=beam, raw_rerank_logit=scores), 'click')
    for i, key in enumerate(ORDER):
        name, k = key.split('@')
        np.testing.assert_array_equal(result[:, i], metric_contributions(target, ranked, int(k))[name])


def test_independent_rank_boundaries_and_absent_gt():
    beam = np.broadcast_to(np.arange(80).reshape(1, 20, 4), (6, 20, 4)).copy()
    target = beam[np.arange(6), [0, 4, 5, 9, 10, 0]].copy()
    target[-1] = -1
    result = contributions(target, beam, np.zeros((6, 20), dtype=np.float32))
    np.testing.assert_array_equal(result[:, 2], [1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(result[:, 3], [1, 1, 0, 0, 0, 0])
    assert not result[-1].any()


def test_runtime_fusion_handles_partial_batch_and_rejects_corruption():
    # Full validation ends with 24 rows. Exercise the same scorer and audit.
    n = 24
    beam = np.broadcast_to(np.arange(80).reshape(1, 20, 4), (n, 20, 4)).copy()
    scores = -np.tile(np.arange(20, dtype=np.float32), (n, 1))
    logits = torch.linspace(-3, 2, 20).to(torch.bfloat16).repeat(n, 1)
    model = SimpleNamespace(top_k_for_generation=20, num_hierarchies=4,
        powers=torch.tensor([4096**3, 4096**2, 4096, 1]),
        reranker=SimpleNamespace(score=lambda h, c, auxiliary: {'esu_logits': logits}),
        _combine_rerank_scores=lambda b, r: b + .1 * r)
    batch = SimpleNamespace(transformed_sequences={'hist_itemkey': torch.ones((n, 5), dtype=torch.long)})
    result = score_cached_batch(model, batch, beam, scores, torch.device('cpu'), 'none')
    arrays = dict(beam_sid=beam, beam_score=scores, raw_rerank_logit=logits.float().numpy(),
                  rerank_sid=result['rerank']['ids'].numpy(), rerank_score=result['rerank']['scores'].numpy())
    verify_runtime_fusion(arrays)
    arrays['rerank_score'][0, 0] += .01
    with pytest.raises(AssertionError):
        verify_runtime_fusion(arrays)


def test_full_launcher_orders_acceptance_before_selection_without_test_submission():
    root = Path(__file__).resolve().parents[1]
    path = root / 'run_sim_cached_validation_cpu4.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    s = path.read_text()
    assert '#SBATCH --cpus-per-task=4' in s and '#SBATCH --gres=gpu:1' in s
    assert s.index('--acceptance "$ACCEPTANCE"') < s.index('bash run_select_score_fusion.sh') < s.index('scripts.accept_sim_fusion')
    assert 'sbatch ' not in s and '_test_' not in s
    source = (root / 'scripts/export_sim_cached_validation.py').read_text()
    assert 'cached.score_cached_batch(' in source and 'cached.verify_inference(' not in source
    assert 'SMOKE_SHA256' in source and 'processed == writer.num_samples == N' in source
