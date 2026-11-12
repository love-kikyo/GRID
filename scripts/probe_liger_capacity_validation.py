"""Two-GPU, full-cold validation bridge for diagnostic 384-wide weights only."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

from scripts.liger_capacity_manifest import binding
from scripts.probe_liger_cpu4 import sha, write_new


def sync_devices():
    import torch
    for i in range(2): torch.cuda.synchronize(i)


def run(path):
    import numpy as np
    import torch
    from scripts.liger_capacity_runtime import build_context
    from src.baselines.liger_data import pack_batch
    from src.baselines.liger_candidates import unique_beam_rows
    from src.baselines.liger_inference import generate_queries
    from src.baselines.liger_sharded_retrieval import ShardedColdCatalogCache
    from src.baselines.liger_retrieval import ColdCatalogCache, full_cold_rank
    from src.baselines.liger_validation import PredictionWriter
    from src.baselines.liger_eval_audit import audit_predictions
    from src.baselines.liger_state import tree_digest
    from src.baselines.liger_efficiency_timing import process_memory_snapshot
    m, c = binding(path)
    receipt_path = path.parent/'training_acceptance.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'passed_resource_training' and receipt['manifest_sha256'] == sha(path)
    assert receipt['source_git_commit'] == m['source_git_commit'] and receipt['exact_fresh_process_replay']
    assert not receipt['formal_training_ready'] and not receipt['test_records_read']
    for entry in receipt['reports']: assert sha(entry['path']) == entry['sha256']
    checkpoint = receipt['validation_checkpoint']
    assert sha(checkpoint['path']) == checkpoint['sha256']
    output = path.parent/'validation'
    assert not output.exists()
    report = dict(status='failed', phase=m['phase'], manifest_sha256=sha(path),
        job_id=os.environ['SLURM_JOB_ID'], source_git_commit=m['source_git_commit'],
        training_acceptance=dict(path=str(receipt_path), sha256=sha(receipt_path)),
        checkpoint=checkpoint, test_records_read=False, automatic_successor=False,
        formal_training_ready=False, formal_checkpoint_produced=False)
    started = time.monotonic()
    deadline = started+c['deadline_seconds']
    def check_time():
        if time.monotonic() > deadline: raise TimeoutError('Two-GPU validation deadline')
    cache = None
    try:
        ctx = build_context(m, c, microbatch=512, devices=2)
        state = torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
        assert tree_digest(state) == checkpoint['full_state_sha256']
        assert state['binding'] == ctx.binding and state['progress']['updates'] == 13
        ctx.model.load_state_dict(state['model'], strict=True)
        ctx.model.eval()
        del state
        before = tree_digest(ctx.model.state_dict())
        writer = PredictionWriter(output)
        batches = list(ctx.batches(training=False, validation_limit=65))
        assert [len(records) for _, records in batches] == [32, 32, 1]
        first_inputs, _ = pack_batch(batches[0][1], ctx.sids, ctx.content)
        q, generation = generate_queries(ctx.model, first_inputs, ctx.method, ctx.official, ctx.index, 'cuda:0')
        # Exact topology check on a small identical catalog that fits one GPU.
        subset = ctx.cold[:2048]
        with ShardedColdCatalogCache(ctx.model.emb_proj, ctx.content, subset,
                                     chunk_size=256) as tiny:
            mono = ColdCatalogCache(ctx.model.emb_proj, ctx.content, subset, device='cuda:0', chunk_size=256)
            a = tiny.rank(q, generation['valid_beam_rows'])
            b = full_cold_rank(mono, q, generation['valid_beam_rows'])
            for key in ('hybrid_rows', 'beam_only_rows', 'valid_beam_rows'):
                np.testing.assert_array_equal(a[key], b[key])
            for key in ('hybrid_scores', 'beam_only_scores'):
                np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-5)
            del mono
        del tiny, q, a, b
        ctx.model.predicted_embedding = None
        for i in range(2):
            with torch.cuda.device(i): torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(i)
        check_time()
        sync_devices()
        begin = time.monotonic()
        cache = ShardedColdCatalogCache(ctx.model.emb_proj, ctx.content, ctx.cold,
                                       chunk_size=c['validation']['chunk_size'])
        sync_devices()
        report.update(cache_seconds=time.monotonic()-begin,
            cache_bytes=list(cache.cache_bytes), shard_sizes=list(cache.shard_sizes),
            small_catalog_single_vs_two_gpu_passed=True,
            gpus=ctx.gpus, resources=ctx.resources, parameters=ctx.parameter_report)
        totals, unique, identity, timings = {}, set(), hashlib.sha256(), []
        for _, records in batches:
            check_time()
            begin = time.monotonic()
            inputs, meta = pack_batch(records, ctx.sids, ctx.content)
            packing_seconds = time.monotonic()-begin
            sync_devices()
            begin = time.monotonic()
            q, generation = generate_queries(ctx.model, inputs, ctx.method, ctx.official, ctx.index, 'cuda:0')
            ranked = cache.rank(q, generation['valid_beam_rows'])
            sync_devices()
            timings.append(dict(rows=len(records), seconds=time.monotonic()-begin,
                                packing_seconds=packing_seconds, scanned_rows=cache.last_scan_rows))
            assert cache.last_scan_rows == 23915712
            assert not unique.intersection(meta['user_ids'].tolist())
            unique.update(meta['user_ids'].tolist())
            arrays = dict(user_ids=meta['user_ids'], target_rows=meta['target_rows'],
                record_sha256=np.asarray(meta['serialized_sha256'], dtype='S64'), **generation,
                **{k: v for k, v in ranked.items() if k != 'valid_beam_rows'})
            identity.update(arrays['record_sha256'].tobytes())
            writer.append(arrays)
            for mode, values in audit_predictions(arrays, ctx.seen, m['asset_rows'], len(records)).items():
                for key, value in values.items():
                    totals.setdefault(mode, {}).setdefault(key, 0.)
                    totals[mode][key] += value*len(records)
            ctx.cpu_check()
            print('[liger-capacity-val] '+json.dumps(timings[-1]), flush=True)
        writer.flush()
        assert len(unique) == 65
        # The inference API must not depend on the true labels field.
        changed = dict(first_inputs, labels_sids=np.full_like(first_inputs['labels_sids'], 123))
        q1, g1 = generate_queries(ctx.model, first_inputs, ctx.method, ctx.official, ctx.index, 'cuda:0')
        q2, g2 = generate_queries(ctx.model, changed, ctx.method, ctx.official, ctx.index, 'cuda:0')
        torch.testing.assert_close(q1, q2, atol=0, rtol=0)
        np.testing.assert_array_equal(g1['beam_tokens'], g2['beam_tokens'])
        del q1, q2
        measured = []
        bench = c['validation']['benchmark']
        for i in range(bench['warmup_repeats']+bench['measured_repeats']):
            check_time()
            sync_devices()
            begin = time.monotonic()
            q, g = generate_queries(ctx.model, first_inputs, ctx.method, ctx.official, ctx.index, 'cuda:0')
            cache.rank(q, g['valid_beam_rows'])
            sync_devices()
            seconds = time.monotonic()-begin
            if i >= bench['warmup_repeats']: measured.append(seconds)
        ctx.model.predicted_embedding = None
        assert tree_digest(ctx.model.state_dict()) == before
        cache.close()
        ctx.cpu_check()
        # Independently reread serialized output, including SID reverse lookup.
        count = 0
        recomputed = {}
        for entry in writer.artifacts:
            assert sha(entry['path']) == entry['sha256']
            with np.load(entry['path'], allow_pickle=False) as arrays:
                np.testing.assert_array_equal(ctx.index.lookup_tokens(arrays['beam_tokens']), arrays['raw_beam_rows'])
                np.testing.assert_array_equal(unique_beam_rows(arrays['raw_beam_rows']), arrays['valid_beam_rows'])
                n = len(arrays['user_ids'])
                for mode, values in audit_predictions(arrays, ctx.seen, m['asset_rows'], n).items():
                    for key, value in values.items():
                        recomputed.setdefault(mode, {}).setdefault(key, 0.)
                        recomputed[mode][key] += value*n
                count += n
        assert count == 65
        for mode, values in recomputed.items():
            for key, value in values.items(): assert abs(value-totals[mode][key]) < 1e-12
        report.update(status='passed', records=65, batches=timings, artifacts=writer.artifacts,
            identity_sha256=identity.hexdigest(), model_state_sha256=before,
            model_weights_unchanged=True, gt_independence_verified=True,
            diagnostic_metrics_not_learning_evidence={mode:{k:v/65 for k,v in values.items()} for mode,values in totals.items()},
            benchmark=dict(records=32, warmup=bench['warmup_repeats'], seconds=measured,
                mean_seconds=statistics.mean(measured), repeated_same_inputs=True,
                forecast_scope='generation and ranking plus cache; excludes dataset reading, packing and final audit',
                estimated_full_validation_hours=(report['cache_seconds']+10857*statistics.mean(measured))/3600),
            peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
            peak_reserved_bytes=[torch.cuda.max_memory_reserved(i) for i in range(2)],
            cpu_checks=ctx.cpu_checks, memory=process_memory_snapshot())
        binding(path)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        if cache is not None: cache.close()
        report.update(elapsed_seconds=time.monotonic()-started, checked_at=datetime.now().astimezone().isoformat())
        write_new(path.parent/'validation_report.json', report)
    write_new(path.parent/'validation_acceptance.json', dict(status='passed_resource_validation',
        report=dict(path=str(path.parent/'validation_report.json'), sha256=sha(path.parent/'validation_report.json')),
        manifest_sha256=sha(path), job_id=os.environ['SLURM_JOB_ID'], records=65, cold_rows=23915712,
        formal_training_ready=False, automatic_successor=False, test_records_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    run(parser.parse_args().manifest.resolve())
