"""Real-train capacity/throughput probe; not formal training or batch selection."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from scripts.probe_liger_cpu4 import ROOT, sha, git, write_new, verify_hashes, import_official
from scripts.build_liger_train_catalog import training_files
from src.baselines.liger_batch_probe import take_batch, summarize, verify_cases

PRIOR = ROOT / 'logs/training_launch/liger_scale_linear_2026-10-19_v1'
PRIOR_ACCEPTANCE_SHA = 'eb51cc9e43d1242ba6b7991c0fe72b30b58667d1929bab5575013abafd2453f1'
SOURCES = ['scripts/probe_liger_batch_scaling.py', 'src/baselines/liger_batch_probe.py',
           'tests/test_liger_batch_probe.py', 'tests/test_liger_batch_launcher.py',
           'run_liger_batch_scaling_cpu4.sh', 'scripts/build_liger_train_catalog.py',
           'scripts/verify_taobao_training_data.py']


def prepare(path):
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    receipt = PRIOR / 'scale_acceptance.json'
    assert sha(receipt) == PRIOR_ACCEPTANCE_SHA
    accepted = json.loads(receipt.read_text())
    assert accepted['status'] == 'passed' and sha(accepted['report']['path']) == accepted['report']['sha256']
    report = json.loads(Path(accepted['report']['path']).read_text())
    assert sha(report['manifest']['path']) == report['manifest']['sha256']
    previous = json.loads(Path(report['manifest']['path']).read_text())
    assert previous['projection_backend'] == 'linear' and previous['chunk_size'] == 65536
    for key in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(previous[key])
    data_root = ROOT / 'data/taobao_history_last_user_v1'
    train = json.loads((data_root / 'protocol_manifest_train.json').read_text())
    entries = training_files(train, data_root)
    assert train['outputs']['train']['samples'] == 6573674
    m = {k: previous[k] for k in ('upstream_root', 'upstream_commit', 'upstream_files',
         'assets', 'input_metadata', 'catalog', 'asset_rows', 'masked_count', 'execution_sources')}
    for source in (receipt, Path(accepted['report']['path']), Path(report['manifest']['path'])):
        m['input_metadata'][str(source)] = sha(source)
    m['execution_sources'].update({str(ROOT / p): sha(ROOT / p) for p in SOURCES})
    m.update(phase='real_train_batch_capacity_only', source_git_commit=git(ROOT, 'rev-parse', 'HEAD'),
        dirty_at_submission=False, data_root=str(data_root), train_files=entries, train_records=6573674,
        batch_sizes=[128, 512, 1024], warmup_steps=1, measured_steps=3, chunk_size=65536,
        projection_backend='linear', seed=42, history_items=7, max_consumed_records=6656,
        distinct_records_across_cases=True, stop_after_oom=True, dense_dtype='float32',
        formal_training_ready=False, candidate_universe_finalized=False, formal_batch_selected=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])), flush=True)


def binding(path):
    assert sha(path) == os.environ['LIGER_BATCH_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and not m['dirty_at_submission']
    for key in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[key])
    train = json.loads((Path(m['data_root']) / 'protocol_manifest_train.json').read_text())
    assert training_files(train, Path(m['data_root'])) == m['train_files']
    assert sha(m['catalog']['path']) == m['catalog']['sha256']
    for asset in m['assets'].values():
        stat = Path(asset['path']).stat()
        assert stat.st_size == asset['size_bytes'] and stat.st_mtime_ns == asset['mtime_ns']
    return m


def real_records(tf, files):
    """Sequential, single-threaded reads of only the manifest's training files."""
    for entry in files:
        iterator = tf.compat.v1.io.tf_record_iterator(entry['path'], options='GZIP')
        try:
            for raw in iterator:
                fields = tf.train.Example.FromString(raw).features.feature
                assert set(fields) == {'user_id', 'sequence_data'}
                assert all(v.WhichOneof('kind') == 'int64_list' for v in fields.values())
                users = fields['user_id'].int64_list.value
                assert len(users) == 1
                yield int(users[0]), list(fields['sequence_data'].int64_list.value), raw
        finally:
            del iterator


def new_model(m, content_dim):
    import torch
    from transformers import T5Config
    from omegaconf import OmegaConf
    from src.baselines.liger_projection import use_linear_residuals
    torch.manual_seed(m['seed'])
    upstream = Path(m['upstream_root'])
    official = import_official('_grid_liger_batch_model', upstream / 'src/tiger.py')
    evaluation = import_official('_grid_liger_batch_eval', upstream / 'src/evaluation.py')
    method = OmegaConf.to_container(OmegaConf.merge(
        OmegaConf.load(upstream / 'configs/method/base.yaml'),
        OmegaConf.load(upstream / 'configs/method/setting.yaml')), resolve=True)
    method['embedding_head_dict']['text_embedding_dim'] = content_dim
    config = T5Config(num_layers=6, num_decoder_layers=6, d_model=128, d_ff=1024,
        num_heads=6, d_kv=64, dropout_rate=.2, vocab_size=4 * 4096 + 2,
        pad_token_id=0, eos_token_id=4 * 4096 + 1, decoder_start_token_id=0,
        feed_forward_proj='relu', layer_norm_epsilon=1e-8, initializer_factor=.02)
    model = official.TIGER(config, n_semantic_codebook=3, max_items_per_seq=7,
        flag_use_output_embedding=True, flag_use_learnable_text_embed=True,
        embedding_head_dict=method['embedding_head_dict'])
    assert use_linear_residuals(model.emb_proj) == 3
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return model.cuda().train(), method, evaluation, digest.hexdigest()


def measure_case(m, case, records, seen_users, sids, content, rows, catalog):
    import numpy as np
    import torch
    from src.baselines.liger_inputs import pack_sequences
    from src.baselines.liger_full_catalog import full_catalog_cross_entropy
    setup = time.monotonic()
    model, method, evaluation, case['initial_state_sha256'] = new_model(m, content.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.035)
    scaler = torch.amp.GradScaler('cuda')
    torch.cuda.synchronize()
    case['setup_seconds_excluded'] = time.monotonic() - setup
    for index in range(m['warmup_steps'] + m['measured_steps']):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        sequences, identity = take_batch(records, case['batch_size'], seen_users)
        case['attempts'].append(identity)
        packed = pack_sequences(sequences, sids, content, history_items=m['history_items'])
        targets = packed.pop('target_rows')
        packed.pop('history_rows')
        positions = np.searchsorted(rows, targets)
        assert (positions < len(rows)).all()
        np.testing.assert_array_equal(rows[positions], targets)
        batch = {k: torch.from_numpy(v) for k, v in packed.items()}
        data_done = time.monotonic()
        labels = torch.from_numpy(positions).long().cuda()
        outputs, _ = evaluation.model_forward(model, batch, 'cuda', 4, method)
        dense_loss = full_catalog_cross_entropy(model.predicted_embedding.float(), labels,
            model.emb_proj, catalog, chunk_size=m['chunk_size'], temperature=.07,
            masked_count=m['masked_count'])
        loss = outputs.loss + dense_loss
        assert torch.isfinite(loss).item()
        torch.cuda.synchronize()
        forward_done = time.monotonic()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert len(grads) == 152 and all(torch.isfinite(g).all().item() for g in grads.values())
        for prefix in ('encoder.', 'decoder.', 'emb_proj.'):
            assert any(n.startswith(prefix) and g.abs().sum().item() > 0 for n, g in grads.items())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm).item()
        scaler.step(optimizer)
        scaler.update()
        assert len(optimizer.state) == 152
        assert all(int(s['step'].item()) == index + 1 for s in optimizer.state.values())
        torch.cuda.synchronize()
        finished = time.monotonic()
        step = dict(step=index + 1, warmup=index < m['warmup_steps'],
            data_seconds=data_done - started, forward_seconds=forward_done - data_done,
            backward_update_seconds=finished - forward_done, total_seconds=finished - started,
            sid_loss=outputs.loss.item(), dense_loss=dense_loss.item(), total_loss=loss.item(),
            grad_norm=norm.item(), gradient_tensors=len(grads), optimizer_step_verified=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        case['steps'].append(step)
        print('[liger-batch] ' + json.dumps(dict(batch_size=case['batch_size'], **step)), flush=True)
        del outputs, dense_loss, loss, grads
    case['summary'] = summarize(case['steps'], case['batch_size'], m['train_records'])
    case['status'] = 'completed'


def run(path):
    m = binding(path)
    output = path.parent / 'batch_probe.json'
    assert not output.exists() and not list(path.parent.glob('batch_case_*.json'))
    report = dict(status='failed', phase=m['phase'], source_git_commit=m['source_git_commit'],
        manifest=dict(path=str(path), sha256=sha(path)), job_id=os.environ['SLURM_JOB_ID'],
        formal_training_ready=False, candidate_universe_finalized=False, held_out_records_read=False,
        formal_batch_selected=False, includes_train_read_decode_pack=True, cases=[], cpu_checks=[])
    try:
        import torch
        from src.runtime_cpu import THREAD_VARIABLES, configure_torch
        from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot
        assert all(os.environ.get(k) == '1' for k in THREAD_VARIABLES)
        configure_torch(torch)
        budget = check_export_allocation()
        report['resources'] = budget
        # Accepted runtime order: Torch first; TensorFlow only reads records on CPU.
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        import numpy as np
        from src.baselines.liger_full_catalog import RowCatalog
        report['cpu_checks'].append(export_cpu_snapshot(budget['allowed_cpus']))
        assert torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        report['gpu'] = dict(name=props.name, uuid=str(props.uuid), total_bytes=props.total_memory)
        report['versions'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                  numpy=np.__version__, tensorflow=tf.__version__)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print('[liger-batch] verify frozen content and SID asset bytes', flush=True)
        for asset in m['assets'].values():
            assert sha(asset['path']) == asset['sha256']
        content = np.load(m['assets']['model_embedding_copy']['path'], mmap_mode='r', allow_pickle=False)
        sids = np.load(m['assets']['frozen_sid']['path'], mmap_mode='r', allow_pickle=False)
        rows = np.load(m['catalog']['path'], mmap_mode='r', allow_pickle=False)
        assert len(rows) == m['catalog']['rows'] and len(content) == len(sids) == m['asset_rows']
        catalog = RowCatalog(content, rows)
        records, seen_users, stopped = real_records(tf, m['train_files']), set(), False
        report.update(catalog_rows=len(catalog), masked_count=m['masked_count'],
                      chunk_size=m['chunk_size'], projection_backend='linear', dense_dtype='float32', tf32=False)
        for size in m['batch_sizes']:
            case = dict(batch_size=size, status='skipped_after_oom' if stopped else 'running', steps=[], attempts=[])
            report['cases'].append(case)
            if not stopped:
                print(f'[liger-batch] start batch={size}', flush=True)
                try:
                    measure_case(m, case, records, seen_users, sids, content, rows, catalog)
                except torch.cuda.OutOfMemoryError as exc:
                    case.update(status='oom', error_type='OutOfMemoryError', error=str(exc))
                    stopped = True
                    print(f'[liger-batch] OOM batch={size}; stop larger sizes', flush=True)
                torch.cuda.empty_cache()
            write_new(path.parent / f'batch_case_{size}.json', case)
            report['cpu_checks'].append(export_cpu_snapshot(budget['allowed_cpus']))
        records.close()
        report['sweep_summary'] = verify_cases(report['cases'], m['batch_sizes'], m['warmup_steps'],
            m['measured_steps'], m['train_records'], props.total_memory)
        assert report['sweep_summary']['unique_consumed_users'] == len(seen_users) <= m['max_consumed_records']
        binding(path)
        report['status'] = 'passed'
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(output, report)
        print('[liger-batch] ' + json.dumps(dict(status=report['status'], report=str(output))), flush=True)


def accept(path):
    m = binding(path)
    report_path = path.parent / 'batch_probe.json'
    r = json.loads(report_path.read_text())
    assert r['status'] == 'passed' and r['phase'] == 'real_train_batch_capacity_only'
    assert r['source_git_commit'] == m['source_git_commit'] and r['job_id'] == os.environ['SLURM_JOB_ID']
    assert r['manifest'] == dict(path=str(path), sha256=sha(path))
    assert all(r[k] is False for k in ('formal_training_ready', 'candidate_universe_finalized',
                                      'held_out_records_read', 'formal_batch_selected', 'tf32'))
    assert r['includes_train_read_decode_pack'] is True
    assert r['catalog_rows'] == m['catalog']['rows'] and r['masked_count'] == m['masked_count']
    assert r['catalog_rows'] + r['masked_count'] == m['asset_rows']
    assert r['chunk_size'] == m['chunk_size'] and r['dense_dtype'] == 'float32' and r['projection_backend'] == 'linear'
    summary = verify_cases(r['cases'], m['batch_sizes'], m['warmup_steps'], m['measured_steps'],
                           m['train_records'], r['gpu']['total_bytes'])
    assert summary == r['sweep_summary'] and summary['unique_consumed_users'] <= m['max_consumed_records']
    for case in r['cases']:
        assert json.loads((path.parent / f"batch_case_{case['batch_size']}.json").read_text()) == case
    allowed = r['resources']['allowed_cpus']
    assert r['resources']['allocated_cpus'] == r['resources']['user_cpu_ceiling'] == 4 and 0 < len(allowed) <= 4
    assert len(r['cpu_checks']) == 1 + len(m['batch_sizes'])
    for check in r['cpu_checks']:
        assert check['allowed_cpus'] == allowed and check['processes']
        assert all(p['threads_checked'] > 0 for p in check['processes'])
    write_new(path.parent / 'batch_acceptance.json', dict(status='passed',
        checked_at=datetime.now().astimezone().isoformat(), job_id=r['job_id'],
        source_git_commit=m['source_git_commit'], report=dict(path=str(report_path), sha256=sha(report_path)),
        sweep_summary=summary, formal_training_ready=False, candidate_universe_finalized=False,
        formal_batch_selected=False,
        audit_scope='Separate source/identity/timing arithmetic audit; TFRecords and GPU not rerun'))
    print('[liger-batch] acceptance passed ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    globals()[args.mode](args.manifest.resolve())
