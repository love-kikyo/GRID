"""Four-cell, train-only learning diagnostic. Never resumes formal weights.

The 4096-item loss is explicitly diagnostic, NOT a replacement for the approved
full-catalog training objective. Each case starts from the same random model.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
import traceback

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new

CONFIG = ROOT / 'configs/baselines/liger_learning_probe_v2.json'
PRIOR = ROOT / 'logs/training_launch/liger_train_seed42_epoch02_2026-10-21_v1/manifest.json'
PRIOR_SHA = 'e4339dc70f0ec440003618cf25b23c9d370756a74e0c4da8b7d19b371da1f420'
SOURCES = ['scripts/probe_liger_learning.py', 'src/baselines/liger_standardization.py',
    'src/baselines/liger_supervision.py', 'src/baselines/liger_learning_probe.py',
    'tests/test_liger_standardization.py', 'tests/test_liger_supervision.py',
    'tests/test_liger_learning_probe.py', 'tests/test_liger_learning_launcher.py',
    'run_liger_learning_probe_cpu4.sh']


def check_config(c):
    assert c['phase'] == 'train_only_learning_diagnostic'
    assert (c['fit_parents'], c['reserved_parents'], c['batch_size']) == (128, 128, 32)
    assert c['catalog_rows'] == 4096 and c['attempts_per_case'] == 1024
    assert c['evaluation_attempts'] == [0, 256, 1024]
    assert c['full_catalog_smoke_attempts'] == 2 and c['deadline_seconds'] == 2400
    assert c['history_items'] == 7 and c['warmup_steps'] == 51
    assert c['learning_rate'] == .0003 and c['weight_decay'] == .035
    assert c['seed'] == 42 and c['split_seed'] == 20261022
    assert c['loss_weighting'] == 'equal_prefix_examples'
    assert c['standardization_fit_scope'] == 'all_frozen_train_seen_item_rows'
    assert not any(c[k] for k in ('formal_training_ready', 'validation_test_access', 'automatic_successor'))
    assert c['cases'] == [dict(name=name, standardized=standardized, supervision=supervision)
        for name, standardized, supervision in [('raw_final', False, 'final_only'),
          ('standardized_final', True, 'final_only'), ('raw_prefix', False, 'all_prefix'),
          ('standardized_prefix', True, 'all_prefix')]]


def prepare(path):
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    assert sha(PRIOR) == PRIOR_SHA
    old = json.loads(PRIOR.read_text())
    for key in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(old[key])
    config = json.loads(CONFIG.read_text())
    check_config(config)
    m = {k: old[k] for k in ('upstream_root', 'upstream_commit', 'upstream_files',
        'execution_sources', 'input_metadata', 'assets', 'catalog', 'index_artifacts',
        'asset_rows', 'data_root', 'train_files', 'seed')}
    m['execution_sources'].update({str(ROOT / p): sha(ROOT / p) for p in SOURCES})
    # The previously locked files did not cover content preprocessing source.
    source = Path(m['upstream_root']) / 'ID_generation/utils.py'
    m['upstream_files'][str(source)] = sha(source)
    m['input_metadata'][str(PRIOR)] = PRIOR_SHA
    m['input_metadata'][str(CONFIG)] = sha(CONFIG)
    m.update(config=str(CONFIG), config_sha256=sha(CONFIG),
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        phase=config['phase'], formal_training_ready=False, automatic_successor=False,
        validation_test_records_read=False, initialization='fresh seed42 per case; no checkpoint load')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])))


def binding(path):
    assert sha(path) == os.environ['LIGER_LEARNING_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and not m['dirty_at_submission']
    assert not any(m[k] for k in ('formal_training_ready', 'automatic_successor', 'validation_test_records_read'))
    assert 'validation_files' not in m and 'resume_checkpoint' not in m
    for key in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[key])
    c = json.loads(Path(m['config']).read_text())
    check_config(c)
    return m, c


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Diagnostic budget expired; partial cases are not an accepted four-cell comparison')


def fresh(m, c):
    import torch
    from scripts.probe_liger_batch_scaling import new_model
    from src.baselines.liger_state import BudgetSchedule
    model, method, official, initial = new_model(m, 128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    schedule = BudgetSchedule(optimizer, total=c['attempts_per_case'], warmup=c['warmup_steps'], base_lr=c['learning_rate'])
    return model, method, official, optimizer, schedule, torch.amp.GradScaler('cuda'), initial


def step(parts, examples, sids, content, rows, catalog, *, masked_count=0, chunk_size=4096):
    import numpy as np
    import torch
    from src.baselines.liger_supervision import pack_supervised_batch
    from src.baselines.liger_full_catalog import full_catalog_cross_entropy
    from src.baselines.liger_state import tree_digest
    model, method, official, optimizer, schedule, scaler, _ = parts
    model.train()
    inputs, identity = pack_supervised_batch(examples, sids, content)
    labels = np.searchsorted(rows, identity['target_rows'])
    assert (labels < len(rows)).all() and np.array_equal(rows[labels], identity['target_rows'])
    assert (identity['loss_weights'] == 1).all()
    optimizer.zero_grad(set_to_none=True)
    outputs, _ = official.model_forward(model, {k: torch.from_numpy(v) for k, v in inputs.items()}, 'cuda', 4, method)
    dense = full_catalog_cross_entropy(model.predicted_embedding.float(), torch.as_tensor(labels, device='cuda'),
        model.emb_proj, catalog, temperature=.07, chunk_size=chunk_size, masked_count=masked_count)
    loss = outputs.loss + dense
    assert torch.isfinite(loss).item()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) == 152
    finite = all(torch.isfinite(g).all().item() for g in grads)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.) if finite else None
    assert norm is None or torch.isfinite(norm).item()
    old_scale, lr = scaler.get_scale(), optimizer.param_groups[0]['lr']
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < old_scale
    assert skipped != finite
    if not skipped: schedule.step()
    return dict(sid_loss=outputs.loss.item(), dense_loss=dense.item(), grad_norm=None if norm is None else norm.item(),
        skipped=skipped, successful_updates=schedule.updates, learning_rate=lr,
        parent_users=identity['parent_user_ids'].tolist(), prefix_indices=identity['prefix_target_indices'].tolist(),
        final_target_count=int(identity['is_final'].sum()), batch_identity=tree_digest(identity))


def evaluate(parts, examples, sids, content, index, path, deadline):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from src.baselines.liger_supervision import pack_supervised_batch
    from src.baselines.liger_inference import generate_queries
    from src.baselines.liger_learning_probe import generation_metrics
    model, method, official = parts[:3]
    model.eval()
    arrays = {k: [] for k in ('labels', 'tokens', 'raw_rows', 'ce', 'top1', 'top20')}
    with torch.no_grad():
        for start in range(0, len(examples), 32):
            check_deadline(deadline)
            inputs, _ = pack_supervised_batch(examples[start:start + 32], sids, content)
            labels = torch.as_tensor(inputs['labels_sids'], device='cuda')
            out, _ = official.model_forward(model, {k: torch.from_numpy(v) for k, v in inputs.items()}, 'cuda', 4, method)
            ce = F.cross_entropy(out.logits.float().flatten(0, 1), labels.flatten(), reduction='none').reshape(-1, 4)
            arrays['ce'].append(ce.cpu().numpy())
            arrays['top1'].append((out.logits.argmax(-1) == labels).cpu().numpy())
            arrays['top20'].append((out.logits.topk(20, dim=-1).indices == labels[:, :, None]).any(-1).cpu().numpy())
            # generate_queries replaces labels with dummy zeros before generation.
            _, predictions = generate_queries(model, inputs, method, official, index, 'cuda')
            arrays['labels'].append(inputs['labels_sids'])
            arrays['tokens'].append(predictions['beam_tokens'])
            arrays['raw_rows'].append(predictions['raw_beam_rows'])
    values = {k: np.concatenate(v) for k, v in arrays.items()}
    with path.open('xb') as stream: np.savez_compressed(stream, **values)
    return dict(path=str(path), sha256=sha(path),
        natural_beam=generation_metrics(values['tokens'], values['labels']),
        teacher_ce=values['ce'].mean(0).tolist(), teacher_top1=values['top1'].mean(0).tolist(),
        teacher_top20=values['top20'].mean(0).tolist(), valid_beam_fraction=float((values['raw_rows'] >= 0).mean()))


def run(path):
    m, c = binding(path)
    output = path.parent / 'learning_probe.json'
    assert not output.exists()
    started = time.monotonic()
    deadline = started + c['deadline_seconds']
    report = dict(status='running', phase=m['phase'], manifest_sha256=sha(path),
        source_git_commit=m['source_git_commit'], job_id=os.environ.get('SLURM_JOB_ID'),
        formal_training_ready=False, automatic_successor=False, validation_test_records_read=False, cases=[])
    try:
        import numpy as np
        import torch
        from src.runtime_cpu import THREAD_VARIABLES, configure_torch
        from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot
        assert all(os.environ.get(k) == '1' for k in THREAD_VARIABLES)
        configure_torch(torch)
        resources = check_export_allocation()
        report['resources'] = resources
        report['cpu_checks'] = [export_cpu_snapshot(resources['allowed_cpus'])]
        assert torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['gpu'] = dict(name=torch.cuda.get_device_name(0), uuid=str(torch.cuda.get_device_properties(0).uuid))
        # TF must only be imported after Torch on the bound compute node.
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from src.baselines.liger_data import record_stream, split_files
        from src.baselines.liger_learning_probe import split_parents, diagnostic_catalog, deterministic_batches
        from src.baselines.liger_supervision import expand_training_records
        from src.baselines.liger_standardization import fit_standardization, StandardizedContent
        from src.baselines.liger_full_catalog import RowCatalog
        from src.baselines.liger_candidates import SidRowIndex
        train = json.loads((Path(m['data_root']) / 'protocol_manifest_train.json').read_text())
        assert list(split_files(train, Path(m['data_root']), 'train')) == m['train_files']
        for asset in m['assets'].values(): assert sha(asset['path']) == asset['sha256']
        assert sha(m['catalog']['path']) == m['catalog']['sha256']
        for entry in m['index_artifacts'].values(): assert sha(entry['path']) == entry['sha256']
        sids = np.load(m['assets']['frozen_sid']['path'], mmap_mode='r')
        raw = np.load(m['assets']['model_embedding_copy']['path'], mmap_mode='r')
        seen = np.load(m['catalog']['path'], mmap_mode='r')
        assert len(seen) == 11542755 and len(raw) == 35458467
        stream = record_stream(tf, m['train_files'], len(raw))
        try: records = list(itertools.islice(stream, c['fit_parents'] + c['reserved_parents']))
        finally: stream.close()
        fit, reserved = split_parents(records, seed=c['split_seed'])
        parent_path = path.parent / 'parents.json'
        write_new(parent_path, {group: [dict(user_id=r.user_id, sequence=list(r.sequence),
            serialized_sha256=r.serialized_sha256) for r in values] for group, values in [('fit', fit), ('reserved', reserved)]})
        report['parents'] = dict(path=str(parent_path), sha256=sha(parent_path))
        rows = diagnostic_catalog(fit, seen, size=c['catalog_rows'], seed=c['split_seed'])
        rows_path = path.parent / 'diagnostic_catalog.npy'
        with rows_path.open('xb') as f: np.save(f, rows, allow_pickle=False)
        report['catalog'] = dict(path=str(rows_path), sha256=sha(rows_path), count=len(rows), formal_catalog=False)
        stats = fit_standardization(raw, seen, content_sha256=m['assets']['model_embedding_copy']['sha256'])
        stats_path = path.parent / 'train_seen_standardization.json'
        write_new(stats_path, stats)
        report['standardization'] = dict(path=str(stats_path), sha256=sha(stats_path))
        standardized = StandardizedContent(raw, stats, expected_content_sha256=m['assets']['model_embedding_copy']['sha256'])
        index = SidRowIndex(*(np.load(m['index_artifacts'][key]['path'], mmap_mode='r') for key in ('sid_keys', 'asset_rows')))
        final_tasks = {group: expand_training_records(values, mode='final_only') for group, values in [('fit', fit), ('reserved', reserved)]}
        prefixes = expand_training_records(fit)
        prefix_sequences = {e.sequence for e in prefixes}
        report['reserved_exact_training_prefix_overlap'] = sum(e.sequence in prefix_sequences for e in final_tasks['reserved'])
        report['reserved_is_formal_validation'] = False
        assert report['reserved_exact_training_prefix_overlap'] == 0, 'Reserved final tasks overlap optimized prefixes; stop diagnostic'
        # Entire seen catalog smoke, separate weights; never used to initialize any case.
        parts = fresh(m, c)
        report['initial_model_sha256'] = parts[-1]
        full_catalog = RowCatalog(standardized, seen)
        report['full_catalog_smoke'] = []
        batches = deterministic_batches(prefixes, steps=c['full_catalog_smoke_attempts'])
        for batch in batches:
            check_deadline(deadline)
            result = step(parts, batch, sids, standardized, seen, full_catalog,
                masked_count=len(raw) - len(seen), chunk_size=65536)
            assert not result['skipped'], 'Full-catalog integration smoke skipped an update'
            report['full_catalog_smoke'].append(result)
        del parts, full_catalog
        torch.cuda.empty_cache()
        for setting in c['cases']:
            check_deadline(deadline)
            parts = fresh(m, c)
            assert parts[-1] == report['initial_model_sha256']
            content = standardized if setting['standardized'] else raw
            catalog = RowCatalog(content, rows)
            examples = expand_training_records(fit, mode=setting['supervision'])
            case = dict(**setting, initial_model_sha256=parts[-1], pool_size=len(examples), evaluations=[],
                final_target_exposures=0, attempts=0, skipped_updates=0)
            report['cases'].append(case)
            trace_path = path.parent / f"{setting['name']}-steps.jsonl"
            with trace_path.open('x') as trace:
                for attempt, batch in enumerate(itertools.chain([None], deterministic_batches(examples,
                        batch_size=c['batch_size'], steps=c['attempts_per_case'], seed=c['seed']))):
                    check_deadline(deadline)
                    if batch is not None:
                        measurement = step(parts, batch, sids, content, rows, catalog)
                        trace.write(json.dumps(dict(attempt=attempt, **measurement), allow_nan=False) + '\n')
                        case['attempts'] += 1
                        case['final_target_exposures'] += measurement['final_target_count']
                        case['skipped_updates'] += int(measurement['skipped'])
                    if attempt in c['evaluation_attempts']:
                        # Evaluation must not alter the training RNG trajectory.
                        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                            for group, tasks in final_tasks.items():
                                location = path.parent / f"{setting['name']}-step{attempt:04d}-{group}.npz"
                                case['evaluations'].append(dict(attempt=attempt, group=group,
                                    **evaluate(parts, tasks, sids, content, index, location, deadline)))
                        report['cpu_checks'].append(export_cpu_snapshot(resources['allowed_cpus']))
                    if attempt % 128 == 0:
                        trace.flush()
                        print('[liger-learning]', setting['name'], attempt, flush=True)
            case['successful_updates'] = parts[4].updates
            case['trace'] = dict(path=str(trace_path), sha256=sha(trace_path))
            # Diagnostic weights only: deliberately no formal checkpoint schema/optimizer.
            weight_path = path.parent / f"{setting['name']}-diagnostic-only.pt"
            with weight_path.open('xb') as f:
                torch.save(dict(purpose='diagnostic_only_never_formal_resume', model=parts[0].state_dict(),
                    manifest_sha256=sha(path), case=setting, standardization_sha256=stats['stats_sha256']
                    if setting['standardized'] else None), f)
            case['diagnostic_weights'] = dict(path=str(weight_path), sha256=sha(weight_path))
            del parts, catalog
            torch.cuda.empty_cache()
        report['status'] = 'completed_diagnostic'
        report['strict_equal_successful_updates'] = all(x['skipped_updates'] == 0 for x in report['cases'])
    except BaseException:
        report['status'] = 'failed_or_incomplete'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        write_new(output, report)


def accept(path):
    import numpy as np
    from src.baselines.liger_data import Record
    from src.baselines.liger_learning_probe import generation_metrics, diagnostic_catalog, deterministic_batches
    from src.baselines.liger_supervision import expand_training_records, pack_supervised_batch
    from src.baselines.liger_candidates import SidRowIndex
    from src.baselines.liger_standardization import validate_standardization
    from src.baselines.liger_state import tree_digest
    m, c = binding(path)
    source = path.parent / 'learning_probe.json'
    r = json.loads(source.read_text())
    assert r['status'] == 'completed_diagnostic' and r['manifest_sha256'] == sha(path)
    assert r['source_git_commit'] == m['source_git_commit'] and r['job_id']
    assert not any(r[k] for k in ('formal_training_ready', 'automatic_successor', 'validation_test_records_read'))
    assert not r['reserved_is_formal_validation']
    assert '4090' in r['gpu']['name'] and r['resources']['allocated_cpus'] == 4
    assert 1 <= len(r['resources']['allowed_cpus']) <= 4 and r['resources']['library_threads'] == 1
    assert len(r['cpu_checks']) == 13
    for check in r['cpu_checks']:
        assert check['allowed_cpus'] == r['resources']['allowed_cpus'] and check['processes']
        assert all(p['threads_checked'] >= 1 for p in check['processes'])
    assert len(r['full_catalog_smoke']) == 2 and not any(x['skipped'] for x in r['full_catalog_smoke'])
    assert len(r['cases']) == 4
    for field in ('parents', 'catalog', 'standardization'):
        assert sha(r[field]['path']) == r[field]['sha256']
    parents = json.loads(Path(r['parents']['path']).read_text())
    assert set(parents) == {'fit', 'reserved'} and all(len(v) == 128 for v in parents.values())
    records = {group: [Record(**entry) for entry in entries] for group, entries in parents.items()}
    # Validate identities across both groups, including original sequence lengths.
    expand_training_records(records['fit'] + records['reserved'], mode='final_only')
    sids = np.load(m['assets']['frozen_sid']['path'], mmap_mode='r')
    raw = np.load(m['assets']['model_embedding_copy']['path'], mmap_mode='r')
    seen = np.load(m['catalog']['path'], mmap_mode='r')
    np.testing.assert_array_equal(np.load(r['catalog']['path']),
        diagnostic_catalog(records['fit'], seen, size=4096, seed=c['split_seed']))
    assert not r['catalog']['formal_catalog'] and r['catalog']['count'] == 4096
    index = SidRowIndex(*(np.load(m['index_artifacts'][key]['path'], mmap_mode='r') for key in ('sid_keys', 'asset_rows')))
    tasks = {group: expand_training_records(values, mode='final_only') for group, values in records.items()}
    prefixes = expand_training_records(records['fit'])
    prefix_sequences = {e.sequence for e in prefixes}
    assert r['reserved_exact_training_prefix_overlap'] == sum(e.sequence in prefix_sequences for e in tasks['reserved'])
    assert r['reserved_exact_training_prefix_overlap'] == 0
    group_labels = {group: np.asarray(sids[[e.sequence[-1] for e in examples]], dtype=np.int64)
        + np.asarray([1, 4097, 8193, 12289]) for group, examples in tasks.items()}
    stats = json.loads(Path(r['standardization']['path']).read_text())
    validate_standardization(stats, content=raw, expected_content_sha256=m['assets']['model_embedding_copy']['sha256'])
    assert stats['n_samples_seen'] == 11542755
    row_hash = hashlib.sha256()
    for start in range(0, len(seen), 65536):
        row_hash.update(np.asarray(seen[start:start + 65536], dtype='<i8').tobytes())
    assert row_hash.hexdigest() == stats['seen_rows_sha256']
    for case, expected in zip(r['cases'], c['cases']):
        assert all(case[k] == v for k, v in expected.items())
        assert case['initial_model_sha256'] == r['initial_model_sha256']
        assert case['attempts'] == 1024 and case['successful_updates'] + case['skipped_updates'] == 1024
        for field in ('trace', 'diagnostic_weights'):
            assert sha(case[field]['path']) == case[field]['sha256']
        trace = [json.loads(line) for line in Path(case['trace']['path']).read_text().splitlines()]
        assert len(trace) == 1024 and [x['attempt'] for x in trace] == list(range(1, 1025))
        assert sum(x['final_target_count'] for x in trace) == case['final_target_exposures']
        assert sum(x['skipped'] for x in trace) == case['skipped_updates']
        pool = expand_training_records(records['fit'], mode=case['supervision'])
        assert case['pool_size'] == len(pool)
        updates = 0
        for t, batch in zip(trace, deterministic_batches(pool, seed=c['seed'])):
            _, identity = pack_supervised_batch(batch, sids, raw)
            assert t['batch_identity'] == tree_digest(identity)
            assert t['parent_users'] == identity['parent_user_ids'].tolist()
            assert t['prefix_indices'] == identity['prefix_target_indices'].tolist()
            assert t['final_target_count'] == int(identity['is_final'].sum())
            assert all(math.isfinite(t[k]) and t[k] >= 0 for k in ('sid_loss', 'dense_loss'))
            assert (t['grad_norm'] is None) == t['skipped']
            assert t['grad_norm'] is None or math.isfinite(t['grad_norm']) and t['grad_norm'] >= 0
            next_step = updates + 1
            scale = next_step / 51 if next_step <= 51 else .5 * (1 + math.cos(math.pi * (next_step - 51) / 973))
            assert math.isclose(t['learning_rate'], .0003 * scale, rel_tol=1e-10, abs_tol=1e-15)
            updates += int(not t['skipped'])
            assert t['successful_updates'] == updates
        assert updates == case['successful_updates']
        assert len(case['evaluations']) == 6
        assert {(e['attempt'], e['group']) for e in case['evaluations']} == {
            (step, group) for step in c['evaluation_attempts'] for group in ('fit', 'reserved')}
        for e in case['evaluations']:
            assert sha(e['path']) == e['sha256']
            with np.load(e['path']) as a:
                assert a['labels'].shape == a['ce'].shape == (128, 4)
                np.testing.assert_array_equal(a['labels'], group_labels[e['group']])
                np.testing.assert_array_equal(a['raw_rows'], index.lookup_tokens(a['tokens']))
                assert a['top1'].shape == a['top20'].shape == (128, 4)
                assert a['top1'].dtype == a['top20'].dtype == np.bool_
                assert np.all(~a['top1'] | a['top20'])
                assert np.isfinite(a['ce']).all()
                assert generation_metrics(a['tokens'], a['labels']) == e['natural_beam']
                for key, arr in [('teacher_ce', 'ce'), ('teacher_top1', 'top1'), ('teacher_top20', 'top20')]:
                    np.testing.assert_allclose(a[arr].mean(0), e[key], rtol=1e-6, atol=1e-7)
                assert float((a['raw_rows'] >= 0).mean()) == e['valid_beam_fraction']
    assert r['strict_equal_successful_updates'] == all(x['skipped_updates'] == 0 for x in r['cases'])
    write_new(path.parent / 'learning_acceptance.json', dict(status='passed_diagnostic_integrity',
        report=dict(path=str(source), sha256=sha(source)), job_id=r['job_id'],
        formal_training_ready=False, quality_decision='author_review_required', automatic_successor=False))
    print('[liger-learning] independent artifact/metric audit passed; NO formal successor', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    globals()[args.command](args.manifest.resolve())
