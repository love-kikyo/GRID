import numpy as np
import pytest

from src.baselines.liger_eval_audit import audit_predictions


def fixture():
    beams = np.full((4, 20), -1, np.int32)
    beams[:, 0] = 0
    beam_scores = np.full((4, 20), -np.inf)
    beam_scores[:, 0] = .5
    return dict(user_ids=np.arange(4), target_rows=np.array([0, 4, 22, 63]), valid_beam_rows=beams,
        hybrid_rows=np.tile(np.r_[0, np.arange(3, 22)], (4, 1)), hybrid_scores=np.tile(np.arange(20., 0., -1.), (4, 1)),
        beam_only_rows=beams.copy(), beam_only_scores=beam_scores)


@pytest.mark.parametrize('corruption', [None, 'users', 'target', 'duplicate', 'seen_not_generated',
                                       'unsorted', 'tie', 'invalid_finite', 'nan', 'beam_not_generated'])
def test_persisted_rankings_are_valid_and_metrics_use_all_users(corruption):
    a = fixture()
    if corruption == 'users': a['user_ids'][0] = 1
    if corruption == 'target': a['target_rows'][0] = -1
    if corruption == 'duplicate': a['hybrid_rows'][0, 1] = 0
    if corruption == 'seen_not_generated': a['hybrid_rows'][0, 1] = 1
    if corruption == 'unsorted': a['hybrid_scores'][0, 1] = 99
    if corruption == 'tie':
        a['hybrid_scores'][0, :2] = 20
        a['hybrid_rows'][0, :2] = [3, 0]
    if corruption == 'invalid_finite': a['beam_only_scores'][0, 1] = .5
    if corruption == 'nan': a['hybrid_scores'][0, 0] = np.nan
    if corruption == 'beam_not_generated': a['beam_only_rows'][0, 0] = 4
    if corruption:
        with pytest.raises(AssertionError): audit_predictions(a, np.array([0, 1, 2]), 64, 4)
    else:
        metrics = audit_predictions(a, np.array([0, 1, 2]), 64, 4)
        assert metrics['hybrid']['Recall@5'] == .5 and metrics['hybrid']['NDCG@5'] == .375
        assert metrics['beam_only']['Recall@20'] == .25
