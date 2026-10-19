"""CPU-only audit of persisted LIGER rankings, independent metric arithmetic."""
import math
import numpy as np


def audit_predictions(arrays, seen, asset_rows, records):
    users, targets, beams = arrays['user_ids'], arrays['target_rows'], arrays['valid_beam_rows']
    assert users.shape == targets.shape == (records,) and len(np.unique(users)) == records
    assert np.issubdtype(targets.dtype, np.integer) and ((targets >= 0) & (targets < asset_rows)).all()
    assert beams.shape == (records, 20)
    metrics = {}
    for mode in ('hybrid', 'beam_only'):
        rows, scores = arrays[f'{mode}_rows'], arrays[f'{mode}_scores']
        assert rows.shape == scores.shape == (records, 20) and np.issubdtype(rows.dtype, np.integer)
        assert ((rows >= -1) & (rows < asset_rows)).all()
        assert np.isfinite(scores[rows >= 0]).all() and np.isneginf(scores[rows < 0]).all()
        ranks = []
        for values, logits, beam, target in zip(rows, scores, beams, targets):
            valid = values[values >= 0]
            assert len(valid) == len(np.unique(valid)) and (logits[1:] <= logits[:-1]).all()
            for j in range(1, len(valid)):
                assert logits[j] != logits[j - 1] or valid[j] > valid[j - 1]
            if mode == 'hybrid':
                assert len(valid) == 20
                positions = np.searchsorted(seen, valid)
                is_seen = (positions < len(seen)) & (seen[np.minimum(positions, len(seen) - 1)] == valid)
                assert np.isin(valid[is_seen], beam).all()
            else:
                assert np.isin(valid, beam).all()
            ranks.append(values.tolist().index(target) + 1 if target in values else 0)
        metrics[mode] = {}
        for k in (5, 10, 20):
            metrics[mode][f'Recall@{k}'] = sum(0 < rank <= k for rank in ranks) / records
            metrics[mode][f'NDCG@{k}'] = sum(1 / math.log2(rank + 1) for rank in ranks if 0 < rank <= k) / records
    return metrics
