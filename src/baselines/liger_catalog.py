"""Train-only catalog accounting; no split changes or recommendation metrics."""
import numpy as np


class TrainCatalog:
    def __init__(self, asset_rows, history_items=7):
        if asset_rows < 1 or asset_rows > np.iinfo(np.int32).max or history_items < 1:
            raise ValueError('Invalid catalog size/history budget')
        self.asset_rows, self.history_items = asset_rows, history_items
        self.all_record = np.zeros(asset_rows, dtype=bool)
        self.input_target = np.zeros(asset_rows, dtype=bool)
        self.target = np.zeros(asset_rows, dtype=bool)
        self.records = self.sequence_items = self.consumed_items = 0

    def update(self, sequence):
        rows = np.asarray(sequence)
        if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or
                not 2 <= len(rows) <= 1000 or (rows < 0).any() or (rows >= self.asset_rows).any()):
            raise ValueError('Invalid frozen history-last record')
        # Row zero is a real asset row, not a padding item at this stage.
        self.all_record[rows] = True
        consumed = rows[-(self.history_items + 1):]
        self.input_target[consumed] = True
        self.target[rows[-1]] = True
        self.records += 1
        self.sequence_items += len(rows)
        self.consumed_items += len(consumed)

    def arrays(self):
        return {name: np.flatnonzero(mask).astype(np.int32) for name, mask in (
            ('all_record_rows', self.all_record), ('input_target_rows', self.input_target),
            ('target_rows', self.target))}


def validate_row_sets(arrays, asset_rows):
    assert set(arrays) == {'all_record_rows', 'input_target_rows', 'target_rows'}
    for rows in arrays.values():
        assert rows.ndim == 1 and rows.dtype == np.int32 and len(rows) > 0
        assert 0 <= rows[0] <= rows[-1] < asset_rows and (np.diff(rows) > 0).all()
    for child, parent in [('target_rows', 'input_target_rows'), ('input_target_rows', 'all_record_rows')]:
        positions = np.searchsorted(arrays[parent], arrays[child])
        assert (positions < len(arrays[parent])).all()
        np.testing.assert_array_equal(arrays[parent][positions], arrays[child])


def sid_identity_summary(rows, semantic_ids):
    """Count SID aliases in actual model-input/target rows, not GT recall."""
    keys = np.empty(len(rows), dtype=np.uint64)
    for start in range(0, len(rows), 262144):
        codes = np.asarray(semantic_ids[rows[start:start + 262144]], dtype=np.int64)
        assert codes.shape[1] == 4 and (codes >= 0).all() and (codes < 4096).all()
        keys[start:start + len(codes)] = (codes * [4096**3, 4096**2, 4096, 1]).sum(-1)
    _, counts = np.unique(keys, return_counts=True)
    return dict(item_rows=len(rows), unique_sid_tuples=len(counts),
                duplicate_sid_groups=int((counts > 1).sum()),
                rows_in_duplicate_sid_groups=int(counts[counts > 1].sum()),
                extra_rows_sharing_sid=int(len(rows) - len(counts)),
                maximum_rows_per_sid=int(counts.max()) if len(counts) else 0)
