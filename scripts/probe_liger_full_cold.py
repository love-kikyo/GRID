"""128-row real-validation API/cost gate, with the complete cold universe.

Random initialization only: diagnostic metrics are not recommendation results.
No training updates, checkpoint, test reader, or automatic successor.
"""
import argparse
from datetime import datetime
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from scripts.probe_liger_cpu4 import ROOT, sha, git, write_new, verify_hashes
from src.baselines.liger_data import split_files

BATCH = ROOT / 'logs/training_launch/liger_batch_2026-10-19_v1'
INDEX = ROOT / 'logs/training_launch/liger_index_2026-10-19_v1'
INDEX_SHA = '7135394b01db961c91b99687eecc616398c61aa5c0c0ce80147cc8cbeac16109'
CONFIG = ROOT / 'configs/baselines/liger_history_last_v1.json'
SOURCES = ['scripts/probe_liger_full_cold.py', 'src/baselines/liger_retrieval.py',
    'src/baselines/liger_inference.py', 'src/baselines/liger_data.py', 'src/baselines/liger_candidates.py',
    'tests/test_liger_retrieval.py', 'tests/test_liger_inference.py', 'tests/test_liger_data.py',
    'tests/test_liger_candidates.py', 'tests/test_liger_full_cold_probe.py', 'run_liger_full_cold_cpu4.sh',
    'src/baselines/liger_eval_audit.py', 'tests/test_liger_eval_audit.py']


def prepare(path):
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    assert sha(INDEX / 'index_acceptance.json') == INDEX_SHA
    receipt = json.loads((INDEX / 'index_acceptance.json').read_text())
    assert receipt['status'] == 'passed' and sha(receipt['report']['path']) == receipt['report']['sha256']
    index = json.loads(Path(receipt['report']['path']).read_text())
    assert index['identity'] == dict(rows_checked=35458467, sid_aliases=0, row_zero_preserved=True)
    assert sha(BATCH / 'batch_acceptance.json') == 'cc176d687e3dacace6681dddd1af17f0c8208541a88d5be3e6e980b0191a956f'
    previous = json.loads((BATCH / 'manifest.json').read_text())
    m = {k: previous[k] for k in ('upstream_root', 'upstream_commit', 'upstream_files', 'assets',
         'input_metadata', 'execution_sources', 'catalog', 'asset_rows', 'seed', 'data_root')}
    for source in (CONFIG, BATCH / 'manifest.json', BATCH / 'batch_acceptance.json',
                   INDEX / 'manifest.json', INDEX / 'index_report.json', INDEX / 'index_acceptance.json'):
        m['input_metadata'][str(source)] = sha(source)
    for key in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[key])
    config = json.loads(CONFIG.read_text())
    assert config['candidate_universe_finalized'] and not config['formal_training_ready']
    assert config['inference']['cutoffs'] == [5, 10, 20] and config['inference']['generation_beams'] == 20
    root = Path(m['data_root'])
    train = json.loads((root / 'protocol_manifest_train.json').read_text())
    m.update(phase='full_cold_validation_scale_only', source_git_commit=git(ROOT, 'rev-parse', 'HEAD'),
        dirty_at_submission=False, config=str(CONFIG), index_artifacts=index['artifacts'],
        validation_files=split_files(train, root, 'evaluation'), validation_records=train['outputs']['evaluation']['samples'],
        batch_size=32, batches=4, warmup_batches=1, chunk_size=65536, history_items=7,
        candidate_universe_finalized=True, formal_training_ready=False, trained_checkpoint_used=False)
    m['execution_sources'].update({str(ROOT / name): sha(ROOT / name) for name in SOURCES})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])), flush=True)


def binding(path):
    assert sha(path) == os.environ['LIGER_COLD_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and not m['dirty_at_submission']
    for key in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[key])
    root = Path(m['data_root'])
    train = json.loads((root / 'protocol_manifest_train.json').read_text())
    assert list(split_files(train, root, 'evaluation')) == m['validation_files']
    assert sha(m['catalog']['path']) == m['catalog']['sha256']
    for entry in m['index_artifacts'].values():
        assert sha(entry['path']) == entry['sha256']
    for entry in m['assets'].values():
        stat = Path(entry['path']).stat()
        assert stat.st_size == entry['size_bytes'] and stat.st_mtime_ns == entry['mtime_ns']
    return m


def cuda_oracle(device):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from src.baselines.liger_retrieval import ColdCatalogCache, full_cold_rank
    from src.baselines.liger_candidates import rank_unique_rows
    projection = torch.nn.Linear(4, 4, bias=False).to(device).eval()
    with torch.no_grad(): projection.weight.copy_(torch.eye(4, device=device))
    raw = np.random.default_rng(2027).normal(size=(73, 4)).astype(np.float32)
    cold = np.arange(3, 73, dtype=np.int32)
    cache = ColdCatalogCache(projection, raw, cold, device=device, chunk_size=11)
    q = torch.tensor([[1., 2., -1., .5], [0., 0., 0., 0.]], device=device)
    beams = np.array([[0, 1, 17, -1, 1], [0, 1, 17, -1, 1]])
    actual = full_cold_rank(cache, q, beams)
    full = (F.normalize(q, dim=-1) @ F.normalize(torch.as_tensor(raw, device=device), dim=-1).T / .07).cpu().numpy()
    for i in range(2):
        universe = np.unique(np.concatenate([cold, beams[i][beams[i] >= 0]]))
        rows, scores = rank_unique_rows(universe, full[i, universe], 20)
        np.testing.assert_array_equal(actual['hybrid_rows'][i], rows)
        np.testing.assert_allclose(actual['hybrid_scores'][i], scores, atol=3e-6, rtol=1e-5)
    return dict(passed=True, full_union_rows=72, queries=2, includes_all_score_ties=True,
                includes_duplicate_invalid_and_seen_beams=True)


def run(path):
    m = binding(path)
    output = path.parent / 'cold_probe.json'
    assert not output.exists() and not (path.parent / 'predictions.npz').exists()
    report = dict(status='failed', phase=m['phase'], source_git_commit=m['source_git_commit'],
        manifest=dict(path=str(path), sha256=sha(path)), job_id=os.environ['SLURM_JOB_ID'],
        candidate_universe_finalized=True, formal_training_ready=False, trained_checkpoint_used=False,
        metrics_are_diagnostic_only=True, test_records_read=False, validation_records_read=True, batches=[], cpu_checks=[])
    try:
        import torch
        from src.runtime_cpu import THREAD_VARIABLES, configure_torch
        from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot
        assert all(os.environ.get(key) == '1' for key in THREAD_VARIABLES)
        configure_torch(torch)
        budget = check_export_allocation()
        report['resources'] = budget
        report['cpu_checks'].append(export_cpu_snapshot(budget['allowed_cpus']))
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        import numpy as np
        from scripts.probe_liger_batch_scaling import new_model
        from src.baselines.liger_data import record_stream, pack_batch
        from src.baselines.liger_candidates import SidRowIndex
        from src.baselines.liger_retrieval import ColdCatalogCache, cold_rows_from_seen, full_cold_rank, ranking_metrics
        from src.baselines.liger_inference import generate_queries
        assert torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        report['gpu'] = dict(name=props.name, uuid=str(props.uuid), total_bytes=props.total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['versions'] = dict(torch=torch.__version__, cuda=torch.version.cuda, tensorflow=tf.__version__, numpy=np.__version__)
        print('[liger-cold] verify fixed assets and load global SID index', flush=True)
        for asset in m['assets'].values(): assert sha(asset['path']) == asset['sha256']
        content = np.load(m['assets']['model_embedding_copy']['path'], mmap_mode='r', allow_pickle=False)
        sids = np.load(m['assets']['frozen_sid']['path'], mmap_mode='r', allow_pickle=False)
        index = SidRowIndex(*(np.load(m['index_artifacts'][name]['path'], mmap_mode='r', allow_pickle=False)
                              for name in ('sid_keys', 'asset_rows')))
        seen = np.load(m['catalog']['path'], mmap_mode='r', allow_pickle=False)
        cold = cold_rows_from_seen(len(content), seen)
        config = json.loads(Path(m['config']).read_text())
        assert len(cold) == config['cold_rows'] == 23915712 and len(seen) == config['seen_rows'] == 11542755
        assert len(content) == len(sids) == config['asset_rows'] == m['asset_rows']
        report['cuda_oracle'] = cuda_oracle('cuda:0')
        model, method, evaluation, initial_hash = new_model(m, content.shape[1])
        model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        print(f'[liger-cold] project all {len(cold)} cold rows once', flush=True)
        cache = ColdCatalogCache(model.emb_proj, content, cold, device='cuda:0', chunk_size=m['chunk_size'])
        torch.cuda.synchronize()
        report.update(initial_state_sha256=initial_hash, cold_rows=len(cold), seen_rows=len(seen),
            cold_cache_seconds=time.monotonic() - started, cache_vector_bytes=cache.vectors.numel() * 4,
            cache_build_peak_allocated_bytes=torch.cuda.max_memory_allocated(), chunk_size=m['chunk_size'], tf32=False)
        stream = record_stream(tf, m['validation_files'], len(sids))
        arrays = {}
        for i in range(m['batches']):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            records = list(itertools.islice(stream, m['batch_size']))
            assert len(records) == m['batch_size']
            inputs, metadata = pack_batch(records, sids, content)
            data_done = time.monotonic()
            query, generation = generate_queries(model, inputs, method, evaluation, index, 'cuda:0')
            torch.cuda.synchronize()
            generated = time.monotonic()
            ranked = full_cold_rank(cache, query, generation['valid_beam_rows'], 20)
            torch.cuda.synchronize()
            finished = time.monotonic()
            measurement = dict(batch=i + 1, warmup=i < m['warmup_batches'], rows=len(records),
                data_seconds=data_done - started, generation_seconds=generated - data_done,
                full_cold_rank_seconds=finished - generated, total_seconds=finished - started,
                cold_rows_scanned=cache.last_scan_rows, invalid_beam_slots=int((generation['raw_beam_rows'] < 0).sum()),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                serialized_sha256=metadata['serialized_sha256'])
            report['batches'].append(measurement)
            if i == 0:
                changed = dict(inputs, labels_sids=np.full_like(inputs['labels_sids'], 123))
                other_query, other = generate_queries(model, changed, method, evaluation, index, 'cuda:0')
                torch.testing.assert_close(query, other_query, atol=0, rtol=0)
                np.testing.assert_array_equal(generation['beam_tokens'], other['beam_tokens'])
                report['gt_independence_verified'] = True
                del other_query
            values = dict(user_ids=metadata['user_ids'], target_rows=metadata['target_rows'], **generation,
                          **{key: value for key, value in ranked.items() if key != 'valid_beam_rows'})
            for key, value in values.items(): arrays.setdefault(key, []).append(value)
            report['cpu_checks'].append(export_cpu_snapshot(budget['allowed_cpus']))
            print('[liger-cold] ' + json.dumps({k: v for k, v in measurement.items() if k != 'serialized_sha256'}), flush=True)
        stream.close()
        arrays = {key: np.concatenate(value) for key, value in arrays.items()}
        assert len(np.unique(arrays['user_ids'])) == m['batch_size'] * m['batches']
        report['diagnostic_metrics'] = {mode: ranking_metrics(arrays[f'{mode}_rows'], arrays['target_rows'])
                                        for mode in ('hybrid', 'beam_only')}
        artifact = path.parent / 'predictions.npz'
        with artifact.open('xb') as handle: np.savez(handle, **arrays)
        report['predictions'] = dict(path=str(artifact), sha256=sha(artifact))
        measured = report['batches'][m['warmup_batches']:]
        report['summary'] = dict(measured_batches=len(measured), measured_rows=sum(b['rows'] for b in measured),
            samples_per_second=sum(b['rows'] for b in measured) / sum(b['total_seconds'] for b in measured),
            median_batch_seconds=statistics.median(b['total_seconds'] for b in measured),
            median_rank_seconds=statistics.median(b['full_cold_rank_seconds'] for b in measured))
        binding(path)
        report['status'] = 'passed'
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(output, report)
        print('[liger-cold] ' + json.dumps(dict(status=report['status'], report=str(output))), flush=True)


def accept(path):
    import numpy as np
    from src.baselines.liger_candidates import SidRowIndex, unique_beam_rows
    from src.baselines.liger_eval_audit import audit_predictions
    m = binding(path)
    report_path = path.parent / 'cold_probe.json'
    r = json.loads(report_path.read_text())
    assert r['status'] == 'passed' and r['phase'] == 'full_cold_validation_scale_only'
    assert r['source_git_commit'] == m['source_git_commit'] and r['job_id'] == os.environ['SLURM_JOB_ID']
    assert r['manifest'] == dict(path=str(path), sha256=sha(path))
    assert all(r[k] is False for k in ('formal_training_ready', 'trained_checkpoint_used', 'test_records_read', 'tf32'))
    assert all(r[k] is True for k in ('candidate_universe_finalized', 'validation_records_read', 'metrics_are_diagnostic_only', 'gt_independence_verified'))
    assert r['cuda_oracle']['passed'] and r['cold_rows'] == 23915712 and r['seen_rows'] == 11542755
    assert r['cache_vector_bytes'] == r['cold_rows'] * 128 * 4 and r['cold_cache_seconds'] > 0
    assert len(r['batches']) == m['batches'] == 4
    for i, b in enumerate(r['batches']):
        assert b['batch'] == i + 1 and b['rows'] == 32 and b['warmup'] == (i == 0)
        assert b['cold_rows_scanned'] == r['cold_rows'] and len(b['serialized_sha256']) == b['rows']
        assert all(math.isfinite(b[key]) and b[key] > 0 for key in ('data_seconds', 'generation_seconds', 'full_cold_rank_seconds', 'total_seconds'))
        assert abs(b['data_seconds'] + b['generation_seconds'] + b['full_cold_rank_seconds'] - b['total_seconds']) < 1e-6
        assert 0 < b['peak_allocated_bytes'] <= b['peak_reserved_bytes'] <= r['gpu']['total_bytes']
    measured = r['batches'][1:]
    expected = dict(measured_batches=3, measured_rows=96, samples_per_second=96 / sum(b['total_seconds'] for b in measured),
        median_batch_seconds=statistics.median(b['total_seconds'] for b in measured),
        median_rank_seconds=statistics.median(b['full_cold_rank_seconds'] for b in measured))
    assert r['summary'] == expected
    assert r['resources']['allocated_cpus'] == r['resources']['user_cpu_ceiling'] == 4
    allowed = r['resources']['allowed_cpus']
    assert 0 < len(allowed) <= 4 and len(r['cpu_checks']) == 5
    assert all(c['allowed_cpus'] == allowed and c['processes'] and all(p['threads_checked'] > 0 for p in c['processes']) for c in r['cpu_checks'])
    assert sha(r['predictions']['path']) == r['predictions']['sha256']
    index = SidRowIndex(*(np.load(m['index_artifacts'][name]['path'], mmap_mode='r', allow_pickle=False)
                          for name in ('sid_keys', 'asset_rows')))
    seen = np.load(m['catalog']['path'], mmap_mode='r', allow_pickle=False)
    assert len(seen) == r['seen_rows'] and len(index.keys) == m['asset_rows']
    with np.load(r['predictions']['path'], allow_pickle=False) as a:
        assert len(a['user_ids']) == len(np.unique(a['user_ids'])) == 128
        assert a['target_rows'].shape == (128,) and ((a['target_rows'] >= 0) & (a['target_rows'] < m['asset_rows'])).all()
        assert a['beam_tokens'].shape == (128, 20, 4)
        np.testing.assert_array_equal(index.lookup_tokens(a['beam_tokens']), a['raw_beam_rows'])
        np.testing.assert_array_equal(unique_beam_rows(a['raw_beam_rows']), a['valid_beam_rows'])
        metrics = audit_predictions(a, seen, m['asset_rows'], 128)
        for mode in metrics:
            for key, value in metrics[mode].items():
                assert abs(r['diagnostic_metrics'][mode][key] - value) < 1e-12
    write_new(path.parent / 'cold_acceptance.json', dict(status='passed', job_id=r['job_id'],
        source_git_commit=m['source_git_commit'], checked_at=datetime.now().astimezone().isoformat(),
        report=dict(path=str(report_path), sha256=sha(report_path)), records=128,
        candidate_universe_finalized=True, formal_training_ready=False, metrics_are_diagnostic_only=True,
        audit_scope='Independent CPU artifact, identity, rank-metric and timing audit; GPU not rerun'))
    print('[liger-cold] independent acceptance passed; diagnostic only, no formal checkpoint', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    globals()[args.mode](args.manifest.resolve())
