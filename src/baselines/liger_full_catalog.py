"""First-order full-catalog CE with bounded projection activations.

No negative sampling: forward visits every active row, backward replays each
chunk with its original dropout RNG. Masked rows contribute exp(-100) exactly
as in upstream training.py, but need no projection. Chunking changes dropout
draw layout versus a monolithic call; it does not reproduce its RNG trajectory.
"""
import math

import numpy as np
import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F


class RowCatalog:
    """Frozen CPU/mmap content addressed by explicit, ordered asset rows."""
    def __init__(self, content, rows):
        if content.ndim != 2 or rows.ndim != 1 or not len(rows):
            raise ValueError('Expected content matrix and nonempty row vector')
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError('Rows must be integer asset identities')
        if (rows < 0).any() or (rows >= len(content)).any():
            raise ValueError('Catalog row outside asset')
        if not (np.diff(rows) > 0).all():
            raise ValueError('Catalog rows must be sorted and unique')
        self.content, self.rows = content, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, selection):
        return self.content[self.rows[selection]]


def _logits(query, projection, catalog, start, stop, temperature):
    # Only this chunk crosses the device boundary; source features stay frozen.
    raw = torch.as_tensor(catalog[start:stop], device=query.device, dtype=query.dtype)
    items = F.normalize(projection(raw), dim=-1)
    return (F.normalize(query, dim=-1) @ items.T) / temperature


class _FullCatalogCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, labels, projection, catalog, chunk_size, temperature,
                masked_count, *parameters):
        batch, device = len(query), query.device
        log_z = query.new_full((batch,), math.log(masked_count) - 100 if masked_count else -math.inf)
        positive = query.new_zeros(batch)
        rng = []
        for start in range(0, len(catalog), chunk_size):
            stop = min(start + chunk_size, len(catalog))
            rng.append((torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
            logits = _logits(query, projection, catalog, start, stop, temperature)
            log_z = torch.logaddexp(log_z, torch.logsumexp(logits, dim=1))
            inside = (labels >= start) & (labels < stop)
            selected = logits.gather(1, (labels - start).clamp(0, stop - start - 1)[:, None]).squeeze(1)
            positive += torch.where(inside, selected, 0.)
        ctx.save_for_backward(query, labels, log_z, *parameters)
        ctx.projection, ctx.catalog = projection, catalog
        ctx.chunk_size, ctx.temperature, ctx.rng = chunk_size, temperature, rng
        ctx.training_modes = tuple(m.training for m in projection.modules())
        return (log_z - positive).mean()

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient):
        query, labels, log_z, *parameters = ctx.saved_tensors
        if tuple(m.training for m in ctx.projection.modules()) != ctx.training_modes:
            raise RuntimeError('Projection mode changed between forward and backward')
        q = query.detach().requires_grad_(True)
        grads = [torch.zeros_like(t) for t in (q, *parameters)]
        devices = [query.device.index] if query.device.type == 'cuda' else []
        # Recompute must not advance the caller's RNG, including when it fails.
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            for index, start in enumerate(range(0, len(ctx.catalog), ctx.chunk_size)):
                torch.set_rng_state(ctx.rng[index][0])
                if devices:
                    torch.cuda.set_rng_state(ctx.rng[index][1], query.device)
                stop = min(start + ctx.chunk_size, len(ctx.catalog))
                logits = _logits(q, ctx.projection, ctx.catalog, start, stop, ctx.temperature)
                with torch.no_grad():
                    weights = (logits - log_z[:, None]).exp()
                    inside = (labels >= start) & (labels < stop)
                    batch_rows = inside.nonzero(as_tuple=True)[0]
                    weights[batch_rows, labels[inside] - start] -= 1
                    weights *= output_gradient / len(query)
                local = torch.autograd.grad(logits, (q, *parameters), grad_outputs=weights)
                for total, grad in zip(grads, local):
                    total.add_(grad)
                del logits, local, weights
        return (grads[0], None, None, None, None, None, None, *grads[1:])


def full_catalog_cross_entropy(query, labels, projection, catalog, *, chunk_size=8192,
                               temperature=.07, masked_count=0):
    """Mean CE over ALL catalog rows plus masked_count constant -100 logits.

    Labels index catalog positions, not global asset rows. Supports first-order
    gradients for query and trainable projection parameters, including shared
    query-side parameters. Raw content must be frozen; parameters, content and
    row identities must not change until backward completes. Single device only;
    explicit FP32/FP64 path (outside autocast), no higher-order derivatives.
    """
    if (query.ndim != 2 or not len(query) or query.dtype not in (torch.float32, torch.float64)
            or query.device.type not in ('cpu', 'cuda')):
        raise ValueError('Expected nonempty FP32/FP64 query matrix on CPU/CUDA')
    if torch.is_autocast_enabled(query.device.type):
        raise ValueError('Full-catalog loss must run outside autocast')
    if (labels.shape != (len(query),) or labels.dtype != torch.long or labels.device != query.device
            or not len(catalog) or (labels < 0).any() or (labels >= len(catalog)).any()):
        raise ValueError('Labels must be active catalog positions')
    if (not isinstance(chunk_size, int) or chunk_size < 1 or not math.isfinite(temperature)
            or temperature <= 0 or not isinstance(masked_count, int) or masked_count < 0):
        raise ValueError('Invalid chunk size, temperature or masked count')
    if isinstance(catalog, torch.Tensor) and catalog.requires_grad:
        raise ValueError('Catalog content must be frozen')
    if any(True for _ in projection.buffers()):
        raise ValueError('Stateful projection buffers are unsupported')
    parameters = tuple(p for p in projection.parameters() if p.requires_grad)
    if not parameters or any(p.dtype != query.dtype or p.device != query.device for p in projection.parameters()):
        raise ValueError('Trainable projection and query must share dtype/device')
    return _FullCatalogCE.apply(query, labels, projection, catalog, chunk_size,
                               temperature, masked_count, *parameters)
