"""Two-step real-catalog scale probe; not a formal LIGER training run."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

from scripts.probe_liger_cpu4 import ROOT, sha, git, write_new, verify_hashes, import_official

INVENTORY = ROOT / 'logs/training_launch/liger_catalog_2026-10-19_v1'
PRIOR = ROOT / 'logs/training_launch/liger_probe_2026-10-12_v2'
BASELINE_SCALE = ROOT / 'logs/training_launch/liger_scale_2026-10-19_v1'
SOURCES = ['scripts/probe_liger_full_catalog.py', 'src/baselines/liger_full_catalog.py',
           'tests/test_liger_full_catalog.py', 'tests/test_liger_scale_probe.py',
           'run_liger_full_catalog_probe_cpu4.sh', 'scripts/probe_liger_cpu4.py',
           'src/baselines/liger_inputs.py', 'src/runtime_cpu.py', 'src/runtime_export_cpu.py',
           'src/baselines/liger_projection.py', 'tests/test_liger_projection.py',
           'scripts/liger_projection_diagnostic.py']


def prepare(path, projection_backend='conv1d', chunk_size=8192):
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    assert projection_backend in {'conv1d', 'linear'} and chunk_size in {8192, 65536}
    acceptance = json.loads((INVENTORY / 'catalog_acceptance.json').read_text())
    assert acceptance['status'] == 'passed' and acceptance['records'] == 6573674
    assert sha(acceptance['report']['path']) == acceptance['report']['sha256']
    inventory = json.loads(Path(acceptance['report']['path']).read_text())
    assert sha(inventory['manifest']['path']) == inventory['manifest']['sha256']
    im = json.loads(Path(inventory['manifest']['path']).read_text())
    assert inventory['held_out_records_read'] is False
    assert sha(PRIOR / 'acceptance.json') == 'c00bd6e50016e8c5d87fbc6191d28c2c43c8bfbbca3c42084a45a3180e28fcd6'
    prior = json.loads((PRIOR / 'acceptance.json').read_text())
    assert prior['status'] == 'passed' and sha(PRIOR / 'probe.json') == prior['report_sha256']
    inputs = dict(im['input_metadata'])
    for source in [INVENTORY / 'catalog_acceptance.json', INVENTORY / 'catalog_report.json',
                   INVENTORY / 'manifest.json', PRIOR / 'probe.json', PRIOR / 'acceptance.json']:
        inputs[str(source)] = sha(source)
    catalog = inventory['artifacts']['input_target_rows']
    assert sha(catalog['path']) == catalog['sha256']
    if projection_backend == 'linear':
        baseline_acceptance = BASELINE_SCALE / 'scale_acceptance.json'
        assert sha(baseline_acceptance) == 'e931f812319fd68d00a8e5d46031dfa8b82e626bcdce49ad97408977a7818030'
        accepted = json.loads(baseline_acceptance.read_text())
        assert accepted['status'] == 'passed' and sha(accepted['report']['path']) == accepted['report']['sha256']
        for source in [baseline_acceptance, BASELINE_SCALE / 'scale_probe.json', BASELINE_SCALE / 'manifest.json']:
            inputs[str(source)] = sha(source)
    manifest = dict(phase='full_catalog_scale_only', source_git_commit=git(ROOT, 'rev-parse', 'HEAD'),
        dirty_at_submission=False, upstream_commit=im['upstream_commit'],
        upstream_root=str(ROOT.parent / 'liger-upstream'), upstream_files=im['upstream_files'],
        input_metadata=inputs, assets=im['assets'], catalog=catalog, asset_rows=inventory['asset_rows'],
        fixture_report=str(PRIOR / 'probe.json'), seed=42, batch_size=32, history_items=7,
        chunk_size=chunk_size, projection_backend=projection_backend, optimizer_steps=2, dense_dtype='float32',
        masked_count=inventory['asset_rows'] - catalog['rows'],
        catalog_definition='All consumed last-seven history and GT rows from training, for scale diagnostic only',
        candidate_universe_finalized=False, formal_training_ready=False,
        execution_sources={str(ROOT / p): sha(ROOT / p) for p in SOURCES})
    for key in ('input_metadata', 'upstream_files'):
        verify_hashes(manifest[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, manifest)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=manifest['source_git_commit'])), flush=True)


def binding(path):
    assert sha(path) == os.environ['LIGER_SCALE_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and not m['dirty_at_submission']
    for key in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[key])
    assert sha(m['catalog']['path']) == m['catalog']['sha256']
    for asset in m['assets'].values():
        stat = Path(asset['path']).stat()
        assert stat.st_size == asset['size_bytes'] and stat.st_mtime_ns == asset['mtime_ns']
    return m


def official_projection_check(projection):
    """CUDA float64 oracle: monolithic no-dropout and replayed chunk dropout."""
    import copy
    import torch
    import torch.nn.functional as F
    from src.baselines.liger_full_catalog import full_catalog_cross_entropy
    reports = []
    for training in (False, True):
        p = copy.deepcopy(projection).double().train(training)
        q = torch.randn(3, 128, dtype=torch.float64, device='cuda', requires_grad=True)
        raw = torch.randn(13, 128, dtype=torch.float64, device='cuda')
        labels = torch.tensor([0, 8, 12], device='cuda')
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        items = torch.cat([p(raw[i:i + 5]) for i in range(0, 13, 5)]) if training else p(raw)
        logits = (F.normalize(q, dim=-1)[:, None] * F.normalize(items, dim=-1)[None]).sum(-1) / .07
        logits = torch.cat([logits, logits.new_full((3, 7), -100)], dim=1)
        expected = F.cross_entropy(logits, labels)
        a = torch.autograd.grad(expected, (q, *p.parameters()))
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
        actual = full_catalog_cross_entropy(q, labels, p, raw, chunk_size=5, masked_count=7)
        after_cpu, after_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
        b = torch.autograd.grad(actual, (q, *p.parameters()))
        assert torch.equal(after_cpu, torch.get_rng_state()) and torch.equal(after_gpu, torch.cuda.get_rng_state())
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=1e-8, rtol=1e-7)
        reports.append(dict(dropout_training=training, loss_abs_error=abs(actual.item() - expected.item()),
            gradient_max_abs_error=max((x - y).abs().max().item() for x, y in zip(a, b)),
            gradient_tensors=len(a), backward_restores_rng=True))
    return reports


def run(path):
    m = binding(path)
    output = path.parent / 'scale_probe.json'
    assert not output.exists()
    report = dict(status='failed', phase=m['phase'], source_git_commit=m['source_git_commit'],
        manifest=dict(path=str(path), sha256=sha(path)), job_id=os.environ['SLURM_JOB_ID'],
        formal_training_ready=False, candidate_universe_finalized=False, held_out_records_read=False)
    try:
        import numpy as np
        import torch
        from transformers import T5Config
        from omegaconf import OmegaConf
        from src.runtime_cpu import THREAD_VARIABLES, configure_torch
        from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot
        from src.baselines.liger_full_catalog import RowCatalog, full_catalog_cross_entropy
        from src.baselines.liger_inputs import pack_sequences
        assert all(os.environ.get(k) == '1' for k in THREAD_VARIABLES)
        configure_torch(torch)
        budget = check_export_allocation()
        report['resources'] = budget
        report['cpu_start'] = export_cpu_snapshot(budget['allowed_cpus'])
        assert torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        report['gpu'] = dict(name=props.name, uuid=str(props.uuid), total_bytes=props.total_memory)
        report['versions'] = dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(m['seed'])
        print('[liger-scale] verify frozen content and SID asset bytes', flush=True)
        for asset in m['assets'].values():
            assert sha(asset['path']) == asset['sha256']
        content = np.load(m['assets']['model_embedding_copy']['path'], mmap_mode='r', allow_pickle=False)
        sids = np.load(m['assets']['frozen_sid']['path'], mmap_mode='r', allow_pickle=False)
        rows = np.load(m['catalog']['path'], mmap_mode='r', allow_pickle=False)
        assert len(rows) == m['catalog']['rows'] and len(content) == m['asset_rows']
        catalog = RowCatalog(content, rows)
        fixture = json.loads(Path(m['fixture_report']).read_text())['fixture']
        sequences = [h + [t] for h, t in zip(fixture['history_rows'], fixture['target_rows'])]
        assert len(sequences) == m['batch_size']
        packed = pack_sequences(sequences, sids, content, history_items=m['history_items'])
        targets = packed.pop('target_rows')
        packed.pop('history_rows')
        positions = np.searchsorted(rows, targets)
        np.testing.assert_array_equal(rows[positions], targets)
        batch = {k: torch.from_numpy(v) for k, v in packed.items()}
        labels = torch.from_numpy(positions).long().cuda()
        upstream = Path(m['upstream_root'])
        official = import_official('_grid_liger_scale_model', upstream / 'src/tiger.py')
        evaluation = import_official('_grid_liger_scale_eval', upstream / 'src/evaluation.py')
        method = OmegaConf.to_container(OmegaConf.merge(
            OmegaConf.load(upstream / 'configs/method/base.yaml'),
            OmegaConf.load(upstream / 'configs/method/setting.yaml')), resolve=True)
        method['embedding_head_dict']['text_embedding_dim'] = content.shape[1]
        config = T5Config(num_layers=6, num_decoder_layers=6, d_model=128, d_ff=1024,
            num_heads=6, d_kv=64, dropout_rate=.2, vocab_size=4 * 4096 + 2,
            pad_token_id=0, eos_token_id=4 * 4096 + 1, decoder_start_token_id=0,
            feed_forward_proj='relu', layer_norm_epsilon=1e-8, initializer_factor=.02)
        model = official.TIGER(config, n_semantic_codebook=3, max_items_per_seq=7,
            flag_use_output_embedding=True, flag_use_learnable_text_embed=True,
            embedding_head_dict=method['embedding_head_dict']).cuda()
        report['projection_backend'] = m.get('projection_backend', 'conv1d')
        if report['projection_backend'] == 'linear':
            from scripts.liger_projection_diagnostic import diagnose, verify_diagnostic
            from src.baselines.liger_projection import use_linear_residuals
            print('[liger-scale] residual kernel correctness and timing diagnostic', flush=True)
            report['projection_diagnostic'] = diagnose(model.emb_proj)
            verify_diagnostic(report['projection_diagnostic'])
            assert use_linear_residuals(model.emb_proj) == 3
        print('[liger-scale] check official projection loss/gradient equivalence', flush=True)
        report['official_projection_checks'] = official_projection_check(model.emb_proj)
        torch.cuda.empty_cache()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.035)
        scaler = torch.amp.GradScaler('cuda')
        report.update(catalog_rows=len(catalog), masked_count=m['masked_count'],
            chunk_size=m['chunk_size'], fixture_users=fixture['user_ids'], batch_size=len(sequences),
            dense_dtype='float32', dense_target_projection_autocast=False,
            tf32=False, diagnostic_only=True, steps=[])
        for step in range(m['optimizer_steps']):
            print(f'[liger-scale] full-catalog step {step + 1}; active rows={len(catalog)}', flush=True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            outputs, _ = evaluation.model_forward(model, batch, 'cuda', 4, method)
            dense_loss = full_catalog_cross_entropy(model.predicted_embedding.float(), labels,
                model.emb_proj, catalog, chunk_size=m['chunk_size'], temperature=.07,
                masked_count=m['masked_count'])
            loss = outputs.loss + dense_loss
            assert torch.isfinite(loss).item()
            torch.cuda.synchronize()
            forward_seconds = time.monotonic() - started
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
            assert all(torch.isfinite(g).all().item() for g in grads.values())
            for prefix in ('encoder.', 'decoder.', 'emb_proj.'):
                assert any(n.startswith(prefix) and g.abs().sum().item() > 0 for n, g in grads.items())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm).item()
            scaler.step(optimizer)
            scaler.update()
            assert all(int(s['step'].item()) == step + 1 for s in optimizer.state.values())
            torch.cuda.synchronize()
            measurement = dict(step=step + 1, sid_loss=outputs.loss.item(), dense_loss=dense_loss.item(),
                total_loss=loss.item(), gradient_tensors=len(grads), grad_norm=norm.item(),
                forward_seconds=forward_seconds, total_step_seconds=time.monotonic() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), optimizer_step_verified=True)
            report['steps'].append(measurement)
            print('[liger-scale] ' + json.dumps(measurement), flush=True)
            del outputs, dense_loss, loss, grads
        report['cpu_end'] = export_cpu_snapshot(budget['allowed_cpus'])
        binding(path)
        report['status'] = 'passed'
        report['remaining_gates'] = ['seen/cold candidate policy', 'item identity and valid candidate mapping',
                                    'formal data adapter, validation selection and training budget']
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['checked_at'] = datetime.now().astimezone().isoformat()
        write_new(output, report)
        print('[liger-scale] ' + json.dumps(dict(status=report['status'], report=str(output))), flush=True)


def accept(path):
    import math
    m = binding(path)
    report_path = path.parent / 'scale_probe.json'
    r = json.loads(report_path.read_text())
    assert r['status'] == 'passed' and r['phase'] == 'full_catalog_scale_only'
    assert r['source_git_commit'] == m['source_git_commit'] and r['job_id'] == os.environ['SLURM_JOB_ID']
    assert r['manifest'] == dict(path=str(path), sha256=sha(path))
    assert not r['formal_training_ready'] and not r['candidate_universe_finalized'] and not r['held_out_records_read']
    assert r['catalog_rows'] == m['catalog']['rows'] and r['masked_count'] == m['masked_count']
    assert r['catalog_rows'] + r['masked_count'] == m['asset_rows']
    assert r['batch_size'] == len(set(r['fixture_users'])) == 32
    assert r['chunk_size'] == m['chunk_size'] and r['dense_dtype'] == 'float32'
    assert r.get('projection_backend', 'conv1d') == m.get('projection_backend', 'conv1d')
    if m.get('projection_backend') == 'linear':
        from scripts.liger_projection_diagnostic import verify_diagnostic
        verify_diagnostic(r['projection_diagnostic'])
    assert len(r['steps']) == m['optimizer_steps'] == 2
    assert [c['dropout_training'] for c in r['official_projection_checks']] == [False, True]
    assert all(c['backward_restores_rng'] and c['gradient_tensors'] > 0 and
               math.isfinite(c['gradient_max_abs_error']) and c['gradient_max_abs_error'] < 1e-6 and
               math.isfinite(c['loss_abs_error']) and c['loss_abs_error'] < 1e-8
               for c in r['official_projection_checks'])
    for index, step in enumerate(r['steps'], 1):
        assert step['step'] == index and step['optimizer_step_verified'] and step['gradient_tensors'] > 0
        assert all(math.isfinite(step[k]) and step[k] > 0 for k in
                   ('sid_loss', 'dense_loss', 'total_loss', 'grad_norm', 'forward_seconds', 'total_step_seconds'))
        assert abs(step['sid_loss'] + step['dense_loss'] - step['total_loss']) < 1e-4
        assert step['peak_allocated_bytes'] <= step['peak_reserved_bytes'] <= r['gpu']['total_bytes']
    assert r['resources']['allocated_cpus'] == 4
    cpus = r['resources']['allowed_cpus']
    assert 0 < len(cpus) <= 4
    for key in ('cpu_start', 'cpu_end'):
        assert r[key]['allowed_cpus'] == cpus and r[key]['processes']
        assert all(p['threads_checked'] > 0 for p in r[key]['processes'])
    write_new(path.parent / 'scale_acceptance.json', dict(status='passed', job_id=r['job_id'],
        checked_at=datetime.now().astimezone().isoformat(), source_git_commit=m['source_git_commit'],
        report=dict(path=str(report_path), sha256=sha(report_path)),
        audit_scope='Separate report/source/resource arithmetic audit; GPU computation not repeated',
        formal_training_ready=False, candidate_universe_finalized=False))
    print('[liger-scale-acceptance] passed; no successor training', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'run', 'accept'])
    p.add_argument('manifest', type=Path)
    p.add_argument('--projection-backend', choices=['conv1d', 'linear'], default='conv1d', help='Preparation only')
    p.add_argument('--chunk-size', type=int, choices=[8192, 65536], default=8192, help='Preparation only')
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args.manifest.resolve(), args.projection_backend, args.chunk_size)
    else:
        assert args.projection_backend == 'conv1d' and args.chunk_size == 8192, 'Run/accept settings come only from manifest'
        {'run': run, 'accept': accept}[args.mode](args.manifest.resolve())
