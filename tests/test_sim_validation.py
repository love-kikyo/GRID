import numpy as np
import pytest

from scripts.accept_sim_validation import check_candidates


def candidates():
    beam = np.arange(80, dtype=np.int16).reshape(1, 20, 4)
    return dict(beam_sid=beam, target_sid=beam[:, 0], rerank_sid=beam[:, ::-1],
                beam_score=-np.arange(20, dtype=np.float32)[None, :],
                raw_rerank_logit=np.ones((1, 20), dtype=np.float32),
                rerank_score=-np.arange(20, dtype=np.float32)[None, :])


def test_exact_candidate_set_and_natural_hit():
    assert check_candidates(candidates()) == 1


@pytest.mark.parametrize('problem', ['duplicate', 'changed', 'nonfinite', 'unsorted', 'out_of_range'])
def test_malformed_candidates_fail(problem):
    a = candidates()
    a['rerank_sid'] = a['rerank_sid'].copy()
    if problem == 'duplicate':
        a['beam_sid'][:, 1] = a['beam_sid'][:, 0]
    elif problem == 'changed':
        a['rerank_sid'][:, 0] = 99
    elif problem == 'nonfinite':
        a['raw_rerank_logit'][0, 0] = np.nan
    elif problem == 'unsorted':
        a['beam_score'][0, 1] = 99
    else:
        a['target_sid'][:] = 4096
    with pytest.raises(AssertionError):
        check_candidates(a)
