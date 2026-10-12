"""Bounded official-LIGER compatibility probe, NOT a benchmark or full training.

Prepare provenance on the login node; execute --run only through CPU4 Slurm.
Uses 32 train examples, 7 history items and a <=2048-row diagnostic catalog.
No test examples, downloads, package installs, checkpoints or performance metrics.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'configs/baselines/liger_upstream_lock.json'
SOURCES = ['scripts/probe_liger_cpu4.py', 'src/baselines/liger_inputs.py',
           'tests/test_liger_inputs.py', 'tests/test_liger_probe.py',
           'tests/test_export_cpu_budget.py', 'tests/test_repository_launchers.py',
           'run_probe_liger_cpu4.sh',
           'src/runtime_cpu.py', 'src/runtime_export_cpu.py',
           'scripts/verify_taobao_training_data.py', 'configs/baselines/liger_upstream_lock.json']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def verify_hashes(entries):
    for path, expected in entries.items():
        require(sha(path) == expected, f'Source/input fingerprint changed: {path}')


def prepare(path, upstream):
    require(not git(ROOT, 'status', '--porcelain'), 'Commit source changes before preparation')
    lock = json.loads(LOCK.read_text())
    require(git(upstream, 'rev-parse', 'HEAD') == lock['commit'], 'Upstream commit mismatch')
    require(not git(upstream, 'status', '--porcelain'), 'Upstream working tree is dirty')
    upstream_files = {str(upstream / k): v for k, v in lock['files'].items()}
    verify_hashes(upstream_files)
    data_root = ROOT / 'data/taobao_history_last_user_v1'
    audit_report = ROOT / 'logs/training_launch/source_snapshot_2026-10-12/report.json'
    require(sha(audit_report) == 'b577b7385a68e9885a08fcc9a2c7402f49f028b476e6c7694349aae052ea5ec6',
            'Frozen source audit changed')
    audit = json.loads(audit_report.read_text())
    assets = {r['role']: r for r in audit['files'] if r['role'] in {'frozen_sid', 'model_embedding_copy'}}
    require(len(assets) == 2, 'Missing audited assets')
    inputs = [data_root / name for name in
              ('protocol_manifest_train.json', 'protocol_manifest_test.json', 'protocol_audit.json')]
    inputs.append(audit_report)
    manifest = {'source_git_commit': git(ROOT, 'rev-parse', 'HEAD'), 'dirty_at_submission': False,
                'upstream_commit': lock['commit'], 'upstream_root': str(upstream),
                'data_root': str(data_root), 'assets': assets,
                'execution_sources': {str(ROOT / p): sha(ROOT / p) for p in SOURCES},
                'upstream_files': upstream_files, 'input_metadata': {str(p): sha(p) for p in inputs},
                'seed': 42, 'batch_size': 32, 'history_items': 7, 'catalog_cap': 2048,
                'steps': 2, 'phase': 'compatibility_only', 'full_training_ready': False}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, manifest)
    print(json.dumps({'manifest': str(path), 'sha256': sha(path),
                      'source_git_commit': manifest['source_git_commit']}), flush=True)


def import_official(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_train_fixture(m):
    # TF reads TFRecords on CPU only, without creating any tf.data worker pool.
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from scripts.verify_taobao_training_data import (
        _assert_protocol_manifest, _assert_manifest_outputs, _assert_audit,
    )
    root = Path(m['data_root'])
    train = json.loads((root / 'protocol_manifest_train.json').read_text())
    test = json.loads((root / 'protocol_manifest_test.json').read_text())
    audit = json.loads((root / 'protocol_audit.json').read_text())
    _assert_protocol_manifest(train, 'train', 'history-last-user-v1')
    _assert_protocol_manifest(test, 'test', 'history-last-user-v1')
    _assert_manifest_outputs(train, root)
    _assert_manifest_outputs(test, root)
    _assert_audit(audit, train, test, 'history-last-user-v1')
    shard = train['outputs']['train']['files'][0]
    iterator = tf.compat.v1.io.tf_record_iterator(shard['path'], options='GZIP')
    sequences, users, digest = [], [], hashlib.sha256()
    for raw in itertools.islice(iterator, m['batch_size']):
        digest.update(raw)
        example = tf.train.Example.FromString(raw).features.feature
        require(set(example) == {'user_id', 'sequence_data'}, 'Unexpected record fields')
        sequences.append(list(example['sequence_data'].int64_list.value))
        users.append(int(example['user_id'].int64_list.value[0]))
    del iterator
    require(len(sequences) == m['batch_size'] == len(set(users)), 'Missing/duplicate probe users')
    return sequences, {'shard': shard, 'split': 'train', 'records': len(users),
                       'serialized_record_sha256': digest.hexdigest(), 'user_ids': users}


def run_probe(m, report):
    from src.runtime_cpu import THREAD_VARIABLES, configure_torch
    from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot
    require(all(os.environ.get(k) == '1' for k in THREAD_VARIABLES), 'Library thread budget missing')
    budget = check_export_allocation()
    report['cpu_start'] = export_cpu_snapshot(budget['allowed_cpus'])
    report['resources'] = budget
    report['hostname'] = os.uname().nodename
    for group in ('execution_sources', 'upstream_files', 'input_metadata'):
        verify_hashes(m[group])
    for asset in m['assets'].values():
        stat = Path(asset['path']).stat()
        require(stat.st_size == asset['size_bytes'] and stat.st_mtime_ns == asset['mtime_ns'],
                'Audited asset stat changed')
    print('[liger-probe] initialize PyTorch before TensorFlow, matching GRID runtime', flush=True)
    import numpy as np
    import torch
    configure_torch(torch)
    print('[liger-probe] reading 32 frozen train records', flush=True)
    sequences, report['fixture'] = read_train_fixture(m)
    print('[liger-probe] fixture decoded; loading Transformers and mmap assets', flush=True)
    from transformers import T5Config, __version__ as transformers_version
    from omegaconf import OmegaConf
    from src.baselines.liger_inputs import pack_sequences, catalog_memory_lower_bounds
    torch.manual_seed(m['seed'])
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one allocated GPU')
    props = torch.cuda.get_device_properties(0)
    require('4090' in props.name, 'Expected authorized RTX 4090 node')
    report['gpu'] = {'name': props.name, 'uuid': str(props.uuid), 'total_bytes': props.total_memory}
    report['versions'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'transformers': transformers_version, 'numpy': np.__version__}
    sids = np.load(m['assets']['frozen_sid']['path'], mmap_mode='r', allow_pickle=False)
    content = np.load(m['assets']['model_embedding_copy']['path'], mmap_mode='r', allow_pickle=False)
    packed = pack_sequences(sequences, sids, content, history_items=m['history_items'])
    targets = packed.pop('target_rows')
    histories = packed.pop('history_rows')
    report['fixture'].update(target_rows=targets.tolist(), history_rows=histories,
                             input_shapes={k: list(v.shape) for k, v in packed.items()})
    # Small artificial catalog used ONLY to exercise the official loss path.
    required = np.unique(np.concatenate([targets] + [np.asarray(x) for x in histories]))
    require(len(required) <= m['catalog_cap'], 'Fixture exceeds diagnostic catalog cap')
    fill = np.setdiff1d(np.arange(min(m['catalog_cap'], len(content))), required)
    rows = np.sort(np.concatenate([required, fill[:m['catalog_cap'] - len(required)]]))
    report['diagnostic_catalog_rows'] = len(rows)
    report['asset_rows_not_verified_active_catalog'] = len(content)
    report['catalog_buffer_lower_bounds'] = catalog_memory_lower_bounds(len(content))
    batch = {k: torch.from_numpy(v) for k, v in packed.items()}
    print('[liger-probe] create GPU diagnostic catalog and official model', flush=True)
    labels = torch.from_numpy(np.searchsorted(rows, targets)).long().cuda()
    catalog = torch.from_numpy(content[rows].astype(np.float32)).cuda()
    upstream = Path(m['upstream_root'])
    official = import_official('_grid_probe_liger_model', upstream / 'src/tiger.py')
    evaluation = import_official('_grid_probe_liger_eval', upstream / 'src/evaluation.py')
    base = OmegaConf.load(upstream / 'configs/method/base.yaml')
    setting = OmegaConf.load(upstream / 'configs/method/setting.yaml')
    method = OmegaConf.to_container(OmegaConf.merge(base, setting), resolve=True)
    method['embedding_head_dict']['text_embedding_dim'] = content.shape[1]
    config = T5Config(num_layers=6, num_decoder_layers=6, d_model=128, d_ff=1024,
                      num_heads=6, d_kv=64, dropout_rate=.2, vocab_size=4 * 4096 + 2,
                      pad_token_id=0, eos_token_id=4 * 4096 + 1, decoder_start_token_id=0,
                      feed_forward_proj='relu', layer_norm_epsilon=1e-8, initializer_factor=.02)
    model = official.TIGER(config, n_semantic_codebook=3, max_items_per_seq=m['history_items'],
                           flag_use_output_embedding=True, flag_use_learnable_text_embed=True,
                           embedding_head_dict=method['embedding_head_dict']).cuda()
    report['probe_adaptations'] = {'fixed_existing_sid': True, 'raw_content_float32': True,
                                   'content_dimension': content.shape[1], 'history_items': 7,
                                   'catalog_is_diagnostic_subset': True, 'cold_start_not_exercised': True}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.035)
    scaler = torch.amp.GradScaler('cuda')
    report['steps'] = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(m['steps']):
        print(f'[liger-probe] optimizer step {step + 1}', flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        outputs, _ = evaluation.model_forward(model, batch, 'cuda', 4, method)
        _, logits = evaluation.get_target_embed(model.predicted_embedding, model, method, catalog)
        dense_loss = torch.nn.functional.cross_entropy(logits, labels)
        loss = outputs.loss + dense_loss
        require(torch.isfinite(loss).item(), 'Nonfinite probe loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        require(all(torch.isfinite(g).all().item() for g in gradients.values()), 'Nonfinite gradients')
        for prefix in ('encoder.', 'decoder.', 'emb_proj.'):
            require(any(n.startswith(prefix) and g.abs().sum().item() > 0
                        for n, g in gradients.items()), f'Missing active gradients: {prefix}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        report['steps'].append({'step': step + 1, 'sid_loss': outputs.loss.item(),
                                'dense_loss': dense_loss.item(), 'total_loss': loss.item(),
                                'elapsed_seconds_not_benchmark': time.monotonic() - started,
                                'gradient_tensors': len(gradients)})
    model.eval()
    print('[liger-probe] beam generation API check', flush=True)
    with torch.no_grad():
        # API smoke only: no trie/cold-start catalog and no effectiveness claims.
        one = {k: v[:1] for k, v in batch.items()}
        _, inputs = evaluation.model_forward(model, one, 'cuda', 4, method, skip_forward=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            generated = model.generate(**inputs, max_new_tokens=4, min_new_tokens=4,
                                       num_beams=20, num_return_sequences=20, do_sample=False)
        require(generated.shape == (20, 5), 'Unexpected generation API shape')
    report['generation_api_shape_not_quality'] = list(generated.shape)
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    report['cpu_end'] = export_cpu_snapshot(budget['allowed_cpus'])
    for group in ('execution_sources', 'upstream_files', 'input_metadata'):
        verify_hashes(m[group])
    report['status'] = 'probe_passed'
    report['full_training_ready'] = False
    report['remaining_gates'] = ['full_catalog_projection_and_loss_scaling',
                                'seen_catalog_and_cold_start_protocol',
                                'formal_adapter_and_item_identity_evaluation']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--run', type=Path)
    parser.add_argument('--upstream', type=Path, default=ROOT.parent / 'liger-upstream')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve(), args.upstream.resolve())
        return
    path = args.run.resolve()
    require(sha(path) == os.environ.get('BASELINE_MANIFEST_SHA256'), 'Manifest binding mismatch')
    m = json.loads(path.read_text())
    require(m['source_git_commit'] == os.environ.get('SOURCE_GIT_COMMIT'), 'Commit binding mismatch')
    output = path.parent / 'probe.json'
    require(not output.exists(), 'Refusing to overwrite an existing probe')
    report = {'status': 'failed', 'manifest_sha256': sha(path), 'manifest': str(path),
              'source_git_commit': m['source_git_commit'], 'upstream_commit': m['upstream_commit'],
              'job_id': os.environ.get('SLURM_JOB_ID'), 'full_training_ready': False}
    try:
        run_probe(m, report)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        write_new(output, report)
        print(json.dumps({'status': report['status'], 'report': str(output),
                          'full_training_ready': False}), flush=True)


if __name__ == '__main__':
    main()
