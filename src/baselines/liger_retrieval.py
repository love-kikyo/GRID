"""Exact full-cold retrieval with an eval-only, weight-bound FP32 cache."""
import numpy as np
import torch
import torch.nn.functional as F

from src.baselines.liger_candidates import rank_unique_rows, unique_beam_rows


def cold_rows_from_seen(asset_rows, seen_rows):
    seen = np.asarray(seen_rows)
    if (not 0 < asset_rows <= np.iinfo(np.int32).max or seen.ndim != 1
            or not np.issubdtype(seen.dtype, np.integer) or not len(seen)
            or seen[0] < 0 or seen[-1] >= asset_rows or not (seen[1:] > seen[:-1]).all()):
        raise ValueError('Expected sorted unique seen asset rows')
    mask = np.ones(asset_rows, dtype=bool)
    mask[seen] = False
    return np.flatnonzero(mask).astype(np.int32)


def _sort_small(scores, rows, k):
    """Sort at most a few top-k lists by score descending, then row ascending."""
    limit = torch.iinfo(torch.int64).max
    rows = torch.where(rows >= 0, rows, limit)
    by_row = torch.argsort(rows, dim=1, stable=True)
    rows, scores = rows.gather(1, by_row), scores.gather(1, by_row)
    by_score = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    rows, scores = rows.gather(1, by_score), scores.gather(1, by_score)
    return scores, torch.where(rows == limit, -1, rows)


def exact_topk(scores, rows, k):
    """Top-k without epsilon perturbation; resolve *all* boundary ties by row.

    torch.topk alone may discard a lower row ID tied at rank k. A second integer
    top-k retrieves the lowest row IDs at that score, before sorting <=2k values.
    Input rows must be distinct, nonnegative identities for these score columns.
    """
    if scores.ndim != 2 or rows.shape != (scores.shape[1],) or not 0 < k <= scores.shape[1]:
        raise ValueError('Invalid top-k shape or k')
    values, positions = scores.topk(k, dim=1, sorted=False)
    cutoff = values.min(dim=1, keepdim=True).values
    limit = torch.iinfo(torch.int64).max
    strong = values > cutoff
    strong_rows = torch.where(strong, rows[positions], limit)
    strong_scores = torch.where(strong, values, -torch.inf)
    tie_rows = torch.where(scores == cutoff, rows[None], limit)
    boundary_rows = tie_rows.topk(k, dim=1, largest=False, sorted=False).values
    boundary_scores = torch.where(boundary_rows != limit, cutoff.expand_as(boundary_rows), -torch.inf)
    return _sort_small(torch.cat([strong_scores, boundary_scores], dim=1),
                       torch.cat([strong_rows, boundary_rows], dim=1), k)


class ColdCatalogCache:
    """Projects every cold item once per checkpoint; never used in training.

    The cache must be discarded before updating projection weights. Identity
    and Tensor._version checks reject stale caches or replacement parameters.
    """
    def __init__(self, projection, content, cold_rows, *, device, chunk_size=65536):
        self.projection, self.content = projection, content
        self.rows_np = np.asarray(cold_rows)
        if (content.ndim != 2 or self.rows_np.ndim != 1 or not len(self.rows_np)
                or not np.issubdtype(self.rows_np.dtype, np.integer)
                or self.rows_np[0] < 0 or self.rows_np[-1] >= len(content)
                or not (self.rows_np[1:] > self.rows_np[:-1]).all() or chunk_size < 1):
            raise ValueError('Expected nonempty sorted cold rows within the asset')
        self.chunk_size, self.device = chunk_size, torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.state = self._state()
        self._check()
        with torch.no_grad():
            first = self._project(self.rows_np[:chunk_size])
            needed = len(self.rows_np) * (first.shape[1] * 4 + 8)
            if self.device.type == 'cuda':
                free, _ = torch.cuda.mem_get_info(self.device)
                if needed + 1024**3 > free:
                    raise RuntimeError('Insufficient free VRAM for complete FP32 cold cache plus workspace')
            self.vectors = torch.empty((len(self.rows_np), first.shape[1]), dtype=torch.float32, device=self.device)
            self.vectors[:len(first)].copy_(first)
            del first
            for start in range(chunk_size, len(self.rows_np), chunk_size):
                self.vectors[start:start + chunk_size].copy_(self._project(self.rows_np[start:start + chunk_size]))
            self.rows = torch.as_tensor(self.rows_np.astype(np.int64), device=self.device)
        self._check()

    def _state(self):
        return tuple((id(p), p._version) for p in self.projection.parameters())

    def _check(self):
        if (any(module.training for module in self.projection.modules())
                or self.state != self._state() or not self.state or any(True for _ in self.projection.buffers())
                or any(p.dtype != torch.float32 or p.device != self.device for p in self.projection.parameters())):
            raise RuntimeError('Cache requires unchanged FP32 eval projection on the same device')
        if torch.is_autocast_enabled(self.device.type):
            raise RuntimeError('Cold cache/scoring must run outside autocast')

    def _project(self, rows):
        raw = torch.as_tensor(self.content[rows], dtype=torch.float32, device=self.device)
        result = F.normalize(self.projection(raw), dim=-1)
        if not torch.isfinite(result).all().item():
            raise ValueError('Nonfinite catalog projection')
        return result

    def _query(self, query):
        self._check()
        if query.ndim != 2 or query.shape[1] != self.vectors.shape[1] or query.device != self.device:
            raise ValueError('Invalid query shape/device')
        if not torch.isfinite(query).all().item():
            raise ValueError('Nonfinite query')
        result = F.normalize(query.float(), dim=-1)
        if not torch.isfinite(result).all().item():
            raise ValueError('Query is outside finite FP32 range')
        return result

    @torch.no_grad()
    def topk(self, query, k=20, temperature=.07):
        q = self._query(query)
        if not 0 < k <= len(self.rows) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid retrieval k/temperature')
        best_scores = q.new_empty((len(q), 0))
        best_rows = self.rows.new_empty((len(q), 0))
        self.last_scan_rows = 0
        for start in range(0, len(self.rows), self.chunk_size):
            stop = min(start + self.chunk_size, len(self.rows))
            scores = (q @ self.vectors[start:stop].T) / temperature
            chunk_scores, chunk_rows = exact_topk(scores, self.rows[start:stop], min(k, stop - start))
            best_scores, best_rows = _sort_small(torch.cat([best_scores, chunk_scores], dim=1),
                                                torch.cat([best_rows, chunk_rows], dim=1), k)
            self.last_scan_rows += stop - start
        assert self.last_scan_rows == len(self.rows)
        return best_rows, best_scores

    @torch.no_grad()
    def score_beams(self, query, beam_rows, temperature=.07):
        q = self._query(query)
        rows = unique_beam_rows(beam_rows)
        if rows.shape[0] != len(q) or (rows >= len(self.content)).any() or temperature <= 0 or not np.isfinite(temperature):
            raise ValueError('Invalid beam rows/temperature')
        valid = rows >= 0
        flat = rows[valid]
        positions = np.searchsorted(self.rows_np, flat)
        safe = np.minimum(positions, len(self.rows_np) - 1)
        cold = (positions < len(self.rows_np)) & (self.rows_np[safe] == flat)
        vectors = q.new_empty((len(flat), self.vectors.shape[1]))
        if cold.any():
            vectors[torch.as_tensor(cold, device=self.device)] = self.vectors[torch.as_tensor(positions[cold], device=self.device)]
        if (~cold).any():
            vectors[torch.as_tensor(~cold, device=self.device)] = self._project(flat[~cold])
        batch_indices = np.nonzero(valid)[0]
        scores = q.new_full(rows.shape, -torch.inf)
        scores[torch.as_tensor(valid, device=self.device)] = (q[torch.as_tensor(batch_indices, device=self.device)] * vectors).sum(-1) / temperature
        return rows, scores


@torch.no_grad()
def full_cold_rank(cache, query, beam_rows, k=20):
    """Exact top-k of all cold items union valid generated beams; no GT input."""
    cold_rows, cold_scores = cache.topk(query, k)
    beams, beam_scores = cache.score_beams(query, beam_rows)
    cold_rows, cold_scores, beam_scores = cold_rows.cpu().numpy(), cold_scores.cpu().numpy(), beam_scores.cpu().numpy()
    # Cold beams already participate in the complete scan. Excluding them from
    # the merge prevents score rounding across kernels from changing a tie.
    positions = np.searchsorted(cache.rows_np, np.maximum(beams, 0))
    safe = np.minimum(positions, len(cache.rows_np) - 1)
    is_cold = (beams >= 0) & (positions < len(cache.rows_np)) & (cache.rows_np[safe] == beams)
    hybrid, only_beam = [], []
    for i in range(len(beams)):
        hybrid.append(rank_unique_rows(np.concatenate([cold_rows[i], np.where(is_cold[i], -1, beams[i])]),
                                       np.concatenate([cold_scores[i], beam_scores[i]]), k))
        only_beam.append(rank_unique_rows(beams[i], beam_scores[i], k))
    return dict(hybrid_rows=np.stack([r for r, _ in hybrid]), hybrid_scores=np.stack([s for _, s in hybrid]),
                beam_only_rows=np.stack([r for r, _ in only_beam]), beam_only_scores=np.stack([s for _, s in only_beam]),
                valid_beam_rows=beams, beam_scores=beam_scores)


def ranking_metrics(rows, targets, cutoffs=(5, 10, 20)):
    rows, targets = np.asarray(rows), np.asarray(targets)
    if (rows.ndim != 2 or targets.shape != (len(rows),) or not len(rows) or (targets < 0).any()
            or not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(targets.dtype, np.integer)
            or (rows < -1).any() or not cutoffs or min(cutoffs) < 1 or max(cutoffs) > rows.shape[1]):
        raise ValueError('Invalid ranking/targets/cutoffs')
    if any(len(r[r >= 0]) != len(np.unique(r[r >= 0])) for r in rows):
        raise ValueError('Ranked candidates must be unique')
    matches = rows == targets[:, None]
    metrics = {}
    for k in cutoffs:
        found = matches[:, :k]
        metrics[f'Recall@{k}'] = found.any(axis=1).mean().item()
        metrics[f'NDCG@{k}'] = (found / np.log2(np.arange(k) + 2)).sum(axis=1).mean().item()
    return metrics
