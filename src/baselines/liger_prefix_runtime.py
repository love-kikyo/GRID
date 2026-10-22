"""Separate formal runtime for bounded prefixes and train-seen standardization.

The original Runtime still owns model construction, optimizer/scheduler, TF
initialization and CPU/GPU restrictions. Only training supervision and the
shared frozen-content view change. Importing this module does not import TF.
"""
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from src.baselines.liger_data import record_stream
from src.baselines.liger_full_catalog import RowCatalog
from src.baselines.liger_runtime import Runtime as BaseRuntime
from src.baselines.liger_standardization import StandardizedContent, validate_standardization


SUPERVISION_TAG = 'bounded_last8_all_prefix_equal_weight_v1'
TRAIN_SEEN_ROWS = 11542755
HISTORY_ITEMS = 7


def _positive_integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value > 0


def training_budget(config):
    """Read explicit attempt/update limits; neither implies a fixed epoch cap.

    A short integration gate may have fewer attempts than scheduled updates.
    Its LR schedule is a test fixture, not authorization for the longer budget.
    """
    training = config['training']
    for key in ('max_optimizer_steps', 'max_attempts', 'warmup_steps', 'batch_size', 'shuffle_buffer'):
        if key not in training or not _positive_integer(training[key]):
            raise ValueError(f'Explicit positive training.{key} is required')
    if training['warmup_steps'] >= training['max_optimizer_steps']:
        raise ValueError('Warmup must be shorter than the fixed update schedule')
    if not _positive_integer(training.get('prefix_shuffle_buffer', 8192)):
        raise ValueError('Positive training.prefix_shuffle_buffer is required')
    if training.get('history_items', HISTORY_ITEMS) != HISTORY_ITEMS:
        raise ValueError('Prefix supervision is bound to the final eight item window')
    return {key: int(training[key]) for key in ('max_optimizer_steps', 'max_attempts')}


def seen_rows_digest(rows, *, chunk_size=65536):
    """Hash ordered row values as little-endian int64, NOT the .npy file bytes."""
    if (getattr(rows, 'ndim', None) != 1 or not len(rows)
            or rows.dtype.kind not in 'iu' or (rows < 0).any()
            or (rows > np.iinfo(np.int64).max).any()
            or not (rows[1:] > rows[:-1]).all() or not _positive_integer(chunk_size)):
        raise ValueError('Expected nonempty sorted unique int64-compatible seen rows')
    digest = hashlib.sha256()
    for start in range(0, len(rows), chunk_size):
        digest.update(np.asarray(rows[start:start + chunk_size], dtype='<i8').tobytes(order='C'))
    return digest.hexdigest()


def read_standardization(manifest, seen, *, content=None):
    """Validate bound statistics and return (statistics, checkpoint additions).

    Source asset bytes remain verified by the original compute-node Runtime.
    The statistics file binds that source SHA and the exact actual seen row
    values, independently of the existing .npy file fingerprint in the manifest.
    """
    entry = manifest.get('standardization')
    if not isinstance(entry, dict) or not {'path', 'sha256'} <= set(entry):
        raise ValueError('A standardization path/SHA256 binding is required')
    payload = Path(entry['path']).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != entry['sha256']:
        raise ValueError('Standardization file SHA256 mismatch')
    statistics = json.loads(payload)
    source_sha = manifest['assets']['model_embedding_copy']['sha256']
    validate_standardization(statistics, content=content, expected_content_sha256=source_sha)
    actual_seen_sha = seen_rows_digest(seen)
    if (statistics['seen_rows_sha256'] != actual_seen_sha
            or statistics['n_samples_seen'] != len(seen)
            or manifest['catalog'].get('rows', len(seen)) != len(seen)
            or statistics['content_shape'][0] != manifest['asset_rows']
            or int(seen[-1]) >= manifest['asset_rows']):
        raise ValueError('Standardization training-seen rows or asset dimensions changed')
    additions = dict(standardization_sha256=digest,
                     standardization_stats_sha256=statistics['stats_sha256'],
                     standardization_seen_rows_sha256=actual_seen_sha,
                     supervision_tag=SUPERVISION_TAG)
    return statistics, additions


class PrefixRuntime(BaseRuntime):
    """Use the same standardized content for history, train loss and retrieval."""

    def __init__(self, manifest, *, build_model=True):
        super().__init__(manifest, build_model=build_model)
        self.training_budget = training_budget(self.config)
        if len(self.seen) != TRAIN_SEEN_ROWS:
            raise ValueError('Formal prefix runtime requires the frozen train-seen catalog')
        self.standardization, additions = read_standardization(
            manifest, self.seen, content=self.content if build_model else None)
        self.binding = dict(self.binding, **additions)
        self.supervision_tag = SUPERVISION_TAG
        if build_model:
            self.raw_content = self.content
            self.content = StandardizedContent(
                self.raw_content, self.standardization,
                expected_content_sha256=manifest['assets']['model_embedding_copy']['sha256'])
            self.catalog = RowCatalog(self.content, self.seen)

    def batches(self, *, training, epoch=0, start_batch=0, validation_limit=None):
        if not training:
            # The original full validation stream keeps only each original GT.
            return super().batches(training=False, epoch=epoch, start_batch=start_batch,
                                   validation_limit=validation_limit)
        if validation_limit is not None:
            raise ValueError('Cannot cap train through validation argument')
        from src.baselines.liger_prefix_stream import prefix_epoch_batches

        def stream(files):
            records = record_stream(self.tf, files, self.m['asset_rows'])
            try:
                yield from records
            finally:
                records.close()

        return prefix_epoch_batches(
            self.m['train_files'], stream,
            batch_size=self.config['training']['batch_size'], seed=self.m['seed'], epoch=epoch,
            shuffle_buffer=self.config['training']['shuffle_buffer'], start_batch=start_batch,
            history_items=HISTORY_ITEMS,
            prefix_shuffle_buffer=self.config['training'].get('prefix_shuffle_buffer', 8192))


# New runners can retain the familiar name without importing the old runtime.
Runtime = PrefixRuntime
