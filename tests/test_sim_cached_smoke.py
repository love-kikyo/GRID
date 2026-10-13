from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from scripts.smoke_sim_cached_rerank import CandidateCache, score_cached_batch, compare_outputs


@pytest.fixture
def cache(tmp_path):
    with h5py.File(tmp_path / 'cache.h5', 'w') as f:
        f['user_id'] = [30, 10, 20]
        f['example_hash'] = np.arange(6, dtype=np.uint64).reshape(3, 2)
        f['target_sid'] = np.arange(12, dtype=np.int16).reshape(3, 4)
        f['beam_sid'] = np.arange(240, dtype=np.int16).reshape(3, 20, 4)
        f['beam_score'] = -np.tile(np.arange(20, dtype=np.float32), (3, 1))
        # No old rerank arrays: the reader must not require any of them.
        yield CandidateCache(f)


def test_alignment_restores_requested_order(cache):
    ids, scores = cache.aligned(np.array([20, 30]), np.array([[4, 5], [0, 1]], dtype=np.uint64),
                               np.array([[8, 9, 10, 11], [0, 1, 2, 3]]))
    np.testing.assert_array_equal(ids[:, 0, 0], [160, 0])
    assert scores.shape == (2, 20)


@pytest.mark.parametrize('problem', ['unknown_middle', 'unknown_end', 'duplicate', 'hash', 'target'])
def test_misaligned_cache_is_rejected(cache, problem):
    ids = np.array([30, 10])
    hashes = np.array([[0, 1], [2, 3]], dtype=np.uint64)
    targets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    if problem == 'unknown_middle': ids[0] = 15
    elif problem == 'unknown_end': ids[0] = 99
    elif problem == 'duplicate': ids[1] = 30
    elif problem == 'hash': hashes[0, 0] = 99
    else: targets[0, 0] = 99
    with pytest.raises(AssertionError):
        cache.aligned(ids, hashes, targets)


def test_cached_scoring_uses_only_history_and_candidate_ids():
    seen = []
    def score(history, candidates, auxiliary):
        assert auxiliary is False
        seen.append(history.clone())
        return {'esu_logits': candidates.float() + history.sum(-1, keepdim=True)}
    model = SimpleNamespace(top_k_for_generation=2, num_hierarchies=4,
                            powers=torch.tensor([4096**3, 4096**2, 4096, 1]),
                            reranker=SimpleNamespace(score=score),
                            _combine_rerank_scores=lambda beam, raw: beam + .1 * raw)
    batch = SimpleNamespace(transformed_sequences={'hist_itemkey': torch.tensor([[1, 2, 999]])})
    ids = np.array([[[0, 0, 0, 1], [0, 0, 0, 2]]])
    scores = np.array([[-1., -2.]], dtype=np.float32)
    first = score_cached_batch(model, batch, ids, scores, torch.device('cpu'), 'none')
    batch.transformed_sequences['hist_itemkey'][0, -1] = 123
    compare_outputs(first, score_cached_batch(model, batch, ids, scores, torch.device('cpu'), 'none'))
    assert all(t.tolist() == [[1, 2]] for t in seen)
    changed = score_cached_batch(model, batch, ids, scores - 7, torch.device('cpu'), 'none')
    torch.testing.assert_close(first['rerank']['click_logits'], changed['rerank']['click_logits'])
    torch.testing.assert_close(first['rerank']['scores'] - 7, changed['rerank']['scores'])


def test_launcher_is_bounded_smoke_only():
    p = Path(__file__).resolve().parents[1] / 'run_sim_cached_smoke_cpu4.sh'
    subprocess.run(['bash', '-n', str(p)], check=True)
    source = p.read_text()
    assert '#SBATCH --cpus-per-task=4' in source and '#SBATCH --gres=gpu:1' in source
    assert '--acceptance "$ACCEPTANCE"' in source and '--cpu-bind=verbose,cores' in source
    assert 'sbatch ' not in source and 'full' not in source
