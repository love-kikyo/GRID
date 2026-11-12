"""Isolated 384-wide diagnostic runtime and one logical optimizer update."""
import hashlib
import math
import random
import time

import numpy as np
import torch

from src.baselines.liger_prefix_runtime import PrefixRuntime
from src.baselines.liger_state import BudgetSchedule, tree_digest
from src.baselines.liger_prefix_training import initial_progress, prefix_identity_digest


def build_context(m, config, *, microbatch=512, devices=1):
    from src.baselines.liger_capacity import build_capacity_model
    from src.baselines.liger_standardization import StandardizedContent
    from src.baselines.liger_retrieval import cold_rows_from_seen
    from src.baselines.liger_feature_cache import build_cached_seen, verify_cached_seen
    from scripts.probe_liger_sampled_scaling import numpy_memory_policy
    started = time.monotonic()
    ctx = PrefixRuntime(m, build_model=False)
    ctx.numpy_policy = numpy_memory_policy()
    assert torch.cuda.device_count() == devices
    ctx.gpus = []
    for i in range(devices):
        prop = torch.cuda.get_device_properties(i)
        assert '4090' in prop.name
        ctx.gpus.append(dict(name=prop.name, uuid=str(prop.uuid), total_bytes=prop.total_memory))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    ctx.tf = tf
    ctx.sids = np.load(m['assets']['frozen_sid']['path'], mmap_mode='r')
    ctx.raw_content = np.load(m['assets']['model_embedding_copy']['path'], mmap_mode='r')
    ctx.content = StandardizedContent(ctx.raw_content, ctx.standardization,
        expected_content_sha256=m['assets']['model_embedding_copy']['sha256'])
    ctx.cold = cold_rows_from_seen(len(ctx.content), ctx.seen)
    assert len(ctx.cold) == 23915712
    random.seed(42)
    np.random.seed(42)
    ctx.model, ctx.method, ctx.official, ctx.parameter_report = build_capacity_model(
        m['upstream_root'], config['model'], seed=42, content_dim=128, device='cuda:0')
    ctx.initial_sha = tree_digest(ctx.model.state_dict())
    ctx.binding = dict(purpose='capacity384_resource_diagnostic_only',
        manifest_sources=m['execution_sources'], config_sha256=m['capacity_config']['sha256'],
        microbatch=microbatch, logical_batch_size=2048, num_negatives=32768,
        rng_order='one catalog projection before all query microbatches',
        asset_sha256={k: v['sha256'] for k, v in m['assets'].items()},
        standardization_sha256=m['standardization']['sha256'])
    ctx.optimizer = torch.optim.AdamW(ctx.model.parameters(), lr=.0003, weight_decay=.035)
    ctx.scheduler = BudgetSchedule(ctx.optimizer, total=22460, warmup=1123, base_lr=.0003)
    ctx.scaler = torch.amp.GradScaler('cuda')
    ctx.feature_cache_report = None
    if devices == 1:
        expected = dict(standardization_stats_sha256=ctx.standardization['stats_sha256'],
            seen_rows_sha256=ctx.standardization['seen_rows_sha256'],
            source_sha256=m['assets']['model_embedding_copy']['sha256'])
        begin = time.monotonic()
        content, catalog, built = build_cached_seen(ctx.content, ctx.seen, expected_binding=expected)
        verified = verify_cached_seen(content, catalog, expected_binding=expected)
        assert built['cache_sha256'] == verified['cache_sha256'] == m['expected_feature_cache_sha256']
        ctx.content, ctx.catalog = content, catalog
        ctx.feature_cache_report = dict(build=built, verification=verified, seconds=time.monotonic()-begin)
    ctx.setup_seconds = time.monotonic() - started
    ctx.cpu_check()
    return ctx


def new_progress():
    return initial_progress(max_optimizer_steps=22460, max_attempts=22460)


def logical_step(ctx, records, progress, microbatch, *, expected_gradients=152):
    """One pool / projection draw / clip / AMP step / LR step per logical batch.

    This fixes the stochastic objective across query chunks. It does not claim
    the historical dropout RNG layout is identical to the monolithic runner.
    """
    from src.baselines.liger_supervision import pack_supervised_batch
    from src.baselines.liger_row_lookup import lookup_sorted_rows
    from src.baselines.liger_logical_batch import prepare_candidate_pool, microbatch_dense_loss
    assert microbatch in (512, 1024) and 0 < len(records) <= 2048
    assert ctx.model.training and progress['updates'] == ctx.scheduler.updates
    ctx.optimizer.zero_grad(set_to_none=True)
    inputs, identity = pack_supervised_batch(records, ctx.sids, ctx.content)
    positions = lookup_sorted_rows(ctx.seen, identity['target_rows'])
    assert np.array_equal(ctx.seen[positions], identity['target_rows'])
    device = next(ctx.model.parameters()).device
    labels = torch.as_tensor(positions, dtype=torch.long, device=device)
    pool = prepare_candidate_pool(ctx.model.emb_proj, ctx.catalog, labels,
        num_negatives=32768, seed=42, attempt=progress['attempts'],
        masked_count=len(ctx.content)-len(ctx.seen))
    total_sid, total_dense = 0., 0.
    for start in range(0, len(records), microbatch):
        stop = min(start + microbatch, len(records))
        batch = {k: torch.from_numpy(v[start:stop]) for k, v in inputs.items()}
        outputs, input_kwargs = ctx.official.model_forward(ctx.model, batch, device, 4, ctx.method)
        del input_kwargs
        sid = outputs.loss * pool.microbatch_weight(start, stop)
        dense = microbatch_dense_loss(ctx.model.predicted_embedding.float(), pool, start, stop)
        loss = sid + dense
        if not torch.isfinite(loss).item(): raise ValueError('Nonfinite logical loss')
        total_sid += sid.item()
        total_dense += dense.item()
        ctx.scaler.scale(loss).backward(retain_graph=stop < len(records))
        ctx.model.predicted_embedding = None
        del outputs, sid, dense, loss, batch
    sample_sha = hashlib.sha256(pool.sample.positions.tobytes()).hexdigest()
    sample_size = len(pool.sample.positions)
    del pool
    ctx.scaler.unscale_(ctx.optimizer)
    gradients = [p.grad for p in ctx.model.parameters() if p.grad is not None]
    assert len(gradients) == expected_gradients
    finite = all(torch.isfinite(g).all().item() for g in gradients)
    norm = torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), 1.) if finite else None
    if norm is not None and not torch.isfinite(norm).item(): raise ValueError('Nonfinite norm')
    lr, before = ctx.optimizer.param_groups[0]['lr'], ctx.scaler.get_scale()
    ctx.scaler.step(ctx.optimizer)
    ctx.scaler.update()
    skipped = ctx.scaler.get_scale() < before
    assert skipped != finite
    if not skipped:
        ctx.scheduler.step()
        progress['updates'] += 1
    else:
        progress['skipped_updates'] += 1
    identity_sha = prefix_identity_digest(identity)
    progress['attempts'] += 1
    progress['batches_consumed'] += 1
    progress['next_batch'] += 1
    progress['prefix_count'] += len(records)
    progress['samples_in_epoch'] += len(records)
    progress['final_count'] += int(identity['is_final'].sum())
    progress['data_chain'] = hashlib.sha256((progress['data_chain']+identity_sha).encode()).hexdigest()
    assert len(ctx.optimizer.state) == expected_gradients
    assert all(int(v['step'].item()) == progress['updates'] for v in ctx.optimizer.state.values())
    return dict(attempt=progress['attempts'], updates=progress['updates'], rows=len(records),
        microbatch=microbatch, microbatches=math.ceil(len(records)/microbatch),
        sid_loss=total_sid, dense_loss=total_dense, total_loss=total_sid+total_dense,
        skipped=skipped, scale_before=before, scale_after=ctx.scaler.get_scale(),
        learning_rate=lr, grad_norm=norm.item() if norm is not None else None,
        gradient_tensors=len(gradients), candidate_classes=sample_size,
        candidate_pool_sha256=sample_sha, prefix_identity_sha256=identity_sha,
        final_target_count=int(identity['is_final'].sum()), data_chain=progress['data_chain'])
