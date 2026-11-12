"""Exact full-cold inference split across two devices, without shrinking its pool.

Only immutable eval projections are replicated; this is not distributed training.
The primary device owns queries and final rankings. GPU dispatch uses two Python
threads inside the caller's existing CPU affinity, never additional processes.
Call ``close()`` (or use a context manager) before releasing GPU allocations.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
from numbers import Integral

import numpy as np
import torch

from src.baselines.liger_candidates import unique_beam_rows
from src.baselines.liger_retrieval import ColdCatalogCache, _sort_small, full_cold_rank


def _device(value):
    value = torch.device(value)
    if value.type == 'cuda' and value.index is None:
        return torch.device('cuda', torch.cuda.current_device())
    if value.type == 'cpu':
        return torch.device('cpu')
    if value.type != 'cuda':
        raise ValueError('Only CPU test devices or CUDA inference devices are supported')
    return value


@torch.no_grad()
def _scan(cache, query, k, temperature):
    # Autograd, autocast and device selection are thread-local. Deliberately
    # recreate the no-grad/device context rather than inheriting caller state.
    if cache.device.type == 'cuda':
        with torch.cuda.device(cache.device), torch.autocast('cuda', enabled=False):
            return cache.topk(query, k, temperature)
    return cache.topk(query, k, temperature)


class ShardedColdCatalogCache:
    """Balanced contiguous catalog shards with exact global top-k merging.

    ``('cpu', 'cpu')`` is supported solely for small deterministic unit tests.
    Production requires two distinct CUDA devices, with ``projection`` already
    in FP32 eval mode on the first device. Each shard inherits the existing
    cache's one-GiB free-workspace guard and stale-weight checks.
    """

    def __init__(self, projection, content, cold_rows, *,
                 devices=('cuda:0', 'cuda:1'), chunk_size=65536):
        self._executor = None
        self._closed = False
        if not isinstance(devices, (tuple, list)) or len(devices) != 2:
            raise ValueError('Exactly two catalog-shard devices are required')
        self.devices = tuple(_device(value) for value in devices)
        if (self.devices[0].type != self.devices[1].type
                or (self.devices[0].type == 'cuda' and self.devices[0] == self.devices[1])):
            raise ValueError('Use two distinct CUDA devices, or two CPU test shards')
        rows = np.asarray(cold_rows)
        if (getattr(content, 'ndim', None) != 2 or rows.ndim != 1 or len(rows) < 2
                or not np.issubdtype(rows.dtype, np.integer)
                or rows[0] < 0 or rows[-1] >= len(content)
                or not (rows[1:] > rows[:-1]).all()
                or not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool)
                or chunk_size < 1):
            raise ValueError('Expected at least two sorted unique cold rows and a positive chunk size')
        # Own immutable identities: changing the caller's array cannot silently
        # change which items a cached projected vector represents.
        self.rows_np = np.array(rows, copy=True)
        self.rows_np.flags.writeable = False
        self.projection, self.content = projection, content
        self.device = self.devices[0]
        cut = (len(rows) + 1) // 2
        self.partition_offsets = ((0, cut), (cut, len(rows)))
        self.partition_rows = tuple(self.rows_np[start:stop]
                                    for start, stop in self.partition_offsets)
        self.shard_sizes = tuple(len(part) for part in self.partition_rows)
        self.shards = []
        for i, device in enumerate(self.devices):
            local_projection = projection if i == 0 else copy.deepcopy(projection).to(device)
            self.shards.append(ColdCatalogCache(local_projection, content,
                self.partition_rows[i], device=device, chunk_size=int(chunk_size)))
        self.shards = tuple(self.shards)
        self._shard_identities = tuple(id(cache) for cache in self.shards)
        if len({cache.vectors.shape[1] for cache in self.shards}) != 1:
            raise ValueError('Catalog shard projection dimensions disagree')
        self.last_scan_rows = 0
        self._check()
        if self.device.type == 'cuda':
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='liger-cold')

    def _check(self):
        if self._closed:
            raise RuntimeError('Catalog cache has been closed')
        if tuple(id(cache) for cache in self.shards) != self._shard_identities:
            raise RuntimeError('Catalog shard identity changed')
        for i, cache in enumerate(self.shards):
            if (cache.rows_np is not self.partition_rows[i]
                    or cache.rows_np.flags.writeable
                    or len(cache.rows_np) != self.shard_sizes[i]
                    or cache.device != self.devices[i]):
                raise RuntimeError('Catalog shard partition changed')
            cache._check()

    @property
    def cache_bytes(self):
        """Allocated vector + row-ID bytes, excluding model and scan workspace."""
        return tuple(cache.vectors.numel() * cache.vectors.element_size()
                     + cache.rows.numel() * cache.rows.element_size() for cache in self.shards)

    @torch.no_grad()
    def topk(self, query, k=20, temperature=.07):
        self._check()
        if (not isinstance(k, Integral) or isinstance(k, bool) or not 0 < k <= len(self.rows_np)
                or not np.isfinite(temperature) or temperature <= 0):
            raise ValueError('Invalid retrieval k/temperature')
        # Validate on the source device before creating either worker task.
        self.shards[0]._query(query)
        # Complete small query transfers before enqueuing either full scan.
        queries = tuple(query.to(device) for device in self.devices)
        arguments = [(cache, q, min(int(k), len(cache.rows_np)), temperature)
                     for cache, q in zip(self.shards, queries)]
        self.last_scan_rows = 0
        if self._executor is None:
            results = [_scan(*args) for args in arguments]
        else:
            futures = [self._executor.submit(_scan, *args) for args in arguments]
            try:
                results = [future.result() for future in futures]
            except BaseException:
                # Do not leave a background scan touching allocations after an
                # error is handed back to the owning evaluation routine.
                self.close()
                raise
        self.last_scan_rows = sum(cache.last_scan_rows for cache in self.shards)
        if self.last_scan_rows != len(self.rows_np):
            raise RuntimeError('Incomplete global cold scan')
        rows = torch.cat([result[0].to(self.device) for result in results], dim=1)
        scores = torch.cat([result[1].to(self.device) for result in results], dim=1)
        scores, rows = _sort_small(scores, rows, int(k))
        self._check()
        return rows, scores

    @torch.no_grad()
    def score_beams(self, query, beam_rows, temperature=.07):
        self._check()
        q = self.shards[0]._query(query)
        rows = unique_beam_rows(beam_rows)
        if (rows.shape[0] != len(q) or (rows >= len(self.content)).any()
                or not np.isfinite(temperature) or temperature <= 0):
            raise ValueError('Invalid beam rows/temperature')
        valid = rows >= 0
        flat = rows[valid]
        # Match the catalog's integer dtype, avoiding population-sized dtype
        # conversion inside NumPy searchsorted for every validation batch.
        positions = np.searchsorted(self.rows_np, flat.astype(self.rows_np.dtype, copy=False))
        safe = np.minimum(positions, len(self.rows_np) - 1)
        cold = (positions < len(self.rows_np)) & (self.rows_np[safe] == flat)
        vectors = q.new_empty((len(flat), q.shape[1]))
        for cache, (start, stop) in zip(self.shards, self.partition_offsets):
            selected = cold & (positions >= start) & (positions < stop)
            if selected.any():
                local = torch.as_tensor(positions[selected] - start, device=cache.device)
                vectors[torch.as_tensor(selected, device=self.device)] = cache.vectors[local].to(self.device)
        if (~cold).any():
            vectors[torch.as_tensor(~cold, device=self.device)] = self.shards[0]._project(flat[~cold])
        batch_indices = np.nonzero(valid)[0]
        scores = q.new_full(rows.shape, -torch.inf)
        scores[torch.as_tensor(valid, device=self.device)] = (
            q[torch.as_tensor(batch_indices, device=self.device)] * vectors).sum(-1) / temperature
        self._check()
        return rows, scores

    @torch.no_grad()
    def rank(self, query, beam_rows, k=20):
        # full_cold_rank uses our GLOBAL rows_np when removing redundant cold
        # beams, including those held on the second device.
        return full_cold_rank(self, query, beam_rows, k=k)

    def close(self):
        """Stop dispatcher threads; caller then releases cache GPU tensors."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()
