"""Single raw-final learning-budget diagnostic; no formal training or resume.

Reuse the accepted v2 train fixtures and immutable model/loss helpers. Only the
attempt budget and corresponding cosine horizon change; warmup stays at 51.
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import time
import traceback

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new
from scripts.probe_liger_learning import fresh, step, evaluate

CONFIG = ROOT / 'configs/baselines/liger_learning_budget_v3.json'
PRIOR = ROOT / 'logs/training_launch/liger_learning_2026-10-22_v2'
PRIOR_ACCEPTANCE_SHA = 'f321360ecff238bcd3d5ab5bec82520896f355fb7d1415864fe27eac07fe2e47'
SOURCES = ['scripts/probe_liger_learning_budget.py', 'tests/test_liger_learning_budget.py',
           'run_liger_learning_budget_cpu4.sh']


def check_config(c):
    assert c['phase'] == 'train_only_learning_budget_diagnostic'
    assert (c['fit_parents'], c['reserved_parents'], c['batch_size']) == (128, 128, 32)
    assert c['catalog_rows'] == 4096 and c['attempts_per_case'] == 8192
    assert c['evaluation_attempts'] == [0, 256] + list(range(512, 8193, 512))
    assert c['deadline_seconds'] == 2400 and c['warmup_steps'] == 51
    assert c['history_items'] == 7 and c['seed'] == 42 and c['split_seed'] == 20261022
    assert c['learning_rate'] == .0003 and c['weight_decay'] == .035
    assert c['sid_loss_weight'] == c['dense_loss_weight'] == 1.
    assert c['loss_weighting'] == 'equal_prefix_examples' and c['fresh_initialization'] is True
    assert c['case'] == dict(name='raw_final', standardized=False, supervision='final_only')
    assert not any(c[k] for k in ('formal_training_ready', 'validation_test_access', 'automatic_successor'))
    assert not {'cases', 'resume_checkpoint', 'validation_files'} & c.keys()


def prepare(path):
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    receipt_path = PRIOR / 'learning_acceptance.json'
    assert sha(receipt_path) == PRIOR_ACCEPTANCE_SHA
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'passed_diagnostic_integrity' and receipt['job_id'] == '147957'
    assert sha(receipt['report']['path']) == receipt['report']['sha256']
    r = json.loads(Path(receipt['report']['path']).read_text())
    assert r['status'] == 'completed_diagnostic' and not r['formal_training_ready']
    prior_manifest = PRIOR / 'manifest.json'
    assert sha(prior_manifest) == r['manifest_sha256']
    old = json.loads(prior_manifest.read_text())
    for key in ('execution_sources', 'input_metadata', 'upstream_files'): verify_hashes(old[key])
    baseline = next(x for x in r['cases'] if x['name'] == 'raw_final')
    assert baseline['attempts'] == baseline['successful_updates'] == 1024 and baseline['skipped_updates'] == 0
    c = json.loads(CONFIG.read_text())
    check_config(c)
    m = {k: old[k] for k in ('upstream_root', 'upstream_commit', 'upstream_files', 'execution_sources',
        'input_metadata', 'assets', 'catalog', 'index_artifacts', 'asset_rows', 'data_root', 'train_files', 'seed')}
    m.update(parents=r['parents'], diagnostic_catalog=r['catalog'],
        previous_trace=baseline['trace'], expected_initial_model_sha256=baseline['initial_model_sha256'],
        previous_initial_evaluations=[e for e in baseline['evaluations'] if e['attempt'] == 0])
    inputs = [dict(path=str(receipt_path), sha256=PRIOR_ACCEPTANCE_SHA), receipt['report'],
        dict(path=str(prior_manifest), sha256=sha(prior_manifest)), r['parents'], r['catalog'], baseline['trace'],
        *m['previous_initial_evaluations']]
    for entry in inputs:
        assert sha(entry['path']) == entry['sha256']
        m['input_metadata'][entry['path']] = entry['sha256']
    m['input_metadata'][str(CONFIG)] = sha(CONFIG)
    m['execution_sources'].update({str(ROOT / p): sha(ROOT / p) for p in SOURCES})
    m.update(config=str(CONFIG), config_sha256=sha(CONFIG), source_git_commit=git(ROOT, 'rev-parse', 'HEAD'),
        dirty_at_submission=False, phase=c['phase'], fresh_initialization=True, formal_training_ready=False,
        validation_test_records_read=False, automatic_successor=False, previous_job_id='147957')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])))


def binding(path):
    assert sha(path) == os.environ['LIGER_BUDGET_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and not m['dirty_at_submission']
    assert m['fresh_initialization'] is True
    assert not any(m[k] for k in ('formal_training_ready', 'automatic_successor', 'validation_test_records_read'))
    assert not {'validation_files', 'resume_checkpoint'} & m.keys()
    for key in ('execution_sources', 'input_metadata', 'upstream_files'): verify_hashes(m[key])
    c = json.loads(Path(m['config']).read_text())
    check_config(c)
    return m, c


def beam_format(tokens):
    import numpy as np
    tokens = np.asarray(tokens)
    assert tokens.ndim == 3 and tokens.shape[1:] == (20, 4) and len(tokens)
    assert np.issubdtype(tokens.dtype, np.integer)
    offsets = np.asarray([1, 4097, 8193, 12289])
    valid = (tokens >= offsets) & (tokens < offsets + 4096)
    return dict(codebook_valid_fraction=valid.mean((0, 1)).tolist(),
                all_codebooks_valid_fraction=float(valid.all(-1).mean()))


def load_inputs(m):
    import numpy as np
    from src.baselines.liger_data import Record
    from src.baselines.liger_supervision import expand_training_records
    from src.baselines.liger_learning_probe import diagnostic_catalog
    from src.baselines.liger_candidates import SidRowIndex
    for entry in [*m['assets'].values(), m['catalog'], *m['index_artifacts'].values(),
                  m['parents'], m['diagnostic_catalog']]:
        assert sha(entry['path']) == entry['sha256']
    parents = json.loads(Path(m['parents']['path']).read_text())
    assert set(parents) == {'fit', 'reserved'} and all(len(v) == 128 for v in parents.values())
    records = {group: [Record(**entry) for entry in entries] for group, entries in parents.items()}
    expand_training_records(records['fit'] + records['reserved'], mode='final_only')
    tasks = {group: expand_training_records(values, mode='final_only') for group, values in records.items()}
    assert not {e.sequence for e in tasks['fit']} & {e.sequence for e in tasks['reserved']}
    sids = np.load(m['assets']['frozen_sid']['path'], mmap_mode='r')
    content = np.load(m['assets']['model_embedding_copy']['path'], mmap_mode='r')
    assert content.dtype == np.int8 and content.shape == (35458467, 128)
    seen = np.load(m['catalog']['path'], mmap_mode='r')
    rows = np.load(m['diagnostic_catalog']['path'])
    np.testing.assert_array_equal(rows, diagnostic_catalog(records['fit'], seen, size=4096, seed=20261022))
    index = SidRowIndex(*(np.load(m['index_artifacts'][key]['path'], mmap_mode='r') for key in ('sid_keys', 'asset_rows')))
    return tasks, sids, content, rows, index


def run(path):
    m, c = binding(path)
    output = path.parent / 'budget_probe.json'
    assert not output.exists()
    started = time.monotonic()
    deadline = started + c['deadline_seconds']
    report = dict(status='running', phase=m['phase'], manifest_sha256=sha(path),
        source_git_commit=m['source_git_commit'], job_id=os.environ.get('SLURM_JOB_ID'),
        formal_training_ready=False, automatic_successor=False, validation_test_records_read=False,
        fresh_initialization=True, evaluations=[], attempts=0, skipped_updates=0, final_target_exposures=0)
    try:
        import numpy as np
        import torch
        from src.runtime_cpu import THREAD_VARIABLES, configure_torch
        from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot
        from src.baselines.liger_learning_probe import deterministic_batches
        from src.baselines.liger_full_catalog import RowCatalog
        assert all(os.environ.get(k) == '1' for k in THREAD_VARIABLES)
        configure_torch(torch)
        resources = check_export_allocation()
        report['resources'] = resources
        report['cpu_checks'] = [export_cpu_snapshot(resources['allowed_cpus'])]
        assert torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['gpu'] = dict(name=torch.cuda.get_device_name(0), uuid=str(torch.cuda.get_device_properties(0).uuid))
        # Already audited train fixtures: no TensorFlow import or split scan.
        tasks, sids, content, rows, index = load_inputs(m)
        parts = fresh(m, c)
        assert parts[-1] == m['expected_initial_model_sha256']
        report['initial_model_sha256'] = parts[-1]
        catalog = RowCatalog(content, rows)
        previous = [json.loads(line) for line in Path(m['previous_trace']['path']).read_text().splitlines()]
        trace_path = path.parent / 'raw_final-steps.jsonl'
        report['warmup_loss_max_abs_delta_from_v2'] = dict(sid_loss=0., dense_loss=0.)
        with trace_path.open('x') as trace:
            for attempt, batch in enumerate(itertools.chain([None], deterministic_batches(tasks['fit'],
                    batch_size=c['batch_size'], steps=c['attempts_per_case'], seed=c['seed']))):
                if time.monotonic() >= deadline: raise TimeoutError('8192-attempt diagnostic budget expired')
                if batch is not None:
                    measurement = step(parts, batch, sids, content, rows, catalog)
                    if attempt <= len(previous):
                        assert measurement['batch_identity'] == previous[attempt - 1]['batch_identity']
                    if attempt <= c['warmup_steps']:
                        for key in ('sid_loss', 'dense_loss'):
                            report['warmup_loss_max_abs_delta_from_v2'][key] = max(
                                report['warmup_loss_max_abs_delta_from_v2'][key],
                                abs(measurement[key] - previous[attempt - 1][key]))
                    trace.write(json.dumps(dict(attempt=attempt, **measurement), allow_nan=False) + '\n')
                    report['attempts'] += 1
                    report['final_target_exposures'] += measurement['final_target_count']
                    report['skipped_updates'] += int(measurement['skipped'])
                if attempt in c['evaluation_attempts']:
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        for group, examples in tasks.items():
                            location = path.parent / f'raw_final-step{attempt:05d}-{group}.npz'
                            e = dict(attempt=attempt, group=group,
                                **evaluate(parts, examples, sids, content, index, location, deadline))
                            with np.load(location) as values: e.update(beam_format(values['tokens']))
                            report['evaluations'].append(e)
                            print('[liger-budget-eval]', json.dumps(e), flush=True)
                    report['cpu_checks'].append(export_cpu_snapshot(resources['allowed_cpus']))
                if attempt % 128 == 0:
                    trace.flush()
                    print('[liger-budget]', attempt, 'updates', parts[4].updates, flush=True)
        report['successful_updates'] = parts[4].updates
        report['trace'] = dict(path=str(trace_path), sha256=sha(trace_path))
        weight_path = path.parent / 'raw_final-diagnostic-only.pt'
        with weight_path.open('xb') as stream:
            torch.save(dict(purpose='diagnostic_only_never_formal_resume', model=parts[0].state_dict(),
                manifest_sha256=sha(path), case=c['case'], standardization_sha256=None), stream)
        report['diagnostic_weights'] = dict(path=str(weight_path), sha256=sha(weight_path))
        report['status'] = 'completed_diagnostic'
    except BaseException:
        report['status'] = 'failed_or_incomplete'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        write_new(output, report)


def accept(path):
    import numpy as np
    from src.baselines.liger_learning_probe import deterministic_batches, generation_metrics
    from src.baselines.liger_supervision import pack_supervised_batch
    from src.baselines.liger_state import tree_digest
    m, c = binding(path)
    source = path.parent / 'budget_probe.json'
    r = json.loads(source.read_text())
    assert r['status'] == 'completed_diagnostic' and r['manifest_sha256'] == sha(path)
    assert r['source_git_commit'] == m['source_git_commit'] and r['job_id'] and r['fresh_initialization']
    assert not any(r[k] for k in ('formal_training_ready', 'automatic_successor', 'validation_test_records_read'))
    assert r['initial_model_sha256'] == m['expected_initial_model_sha256']
    assert r['attempts'] == r['successful_updates'] + r['skipped_updates'] == 8192
    assert r['final_target_exposures'] == 8192 * 32
    assert '4090' in r['gpu']['name'] and r['resources']['allocated_cpus'] == 4
    assert 1 <= len(r['resources']['allowed_cpus']) <= 4 and r['resources']['library_threads'] == 1
    assert len(r['cpu_checks']) == 1 + len(c['evaluation_attempts'])
    for snapshot in r['cpu_checks']:
        assert snapshot['allowed_cpus'] == r['resources']['allowed_cpus'] and snapshot['processes']
        assert all(p['threads_checked'] >= 1 for p in snapshot['processes'])
    tasks, sids, content, rows, index = load_inputs(m)
    for key in ('trace', 'diagnostic_weights'): assert sha(r[key]['path']) == r[key]['sha256']
    trace = [json.loads(line) for line in Path(r['trace']['path']).read_text().splitlines()]
    previous = [json.loads(line) for line in Path(m['previous_trace']['path']).read_text().splitlines()]
    assert len(trace) == 8192 and [t['attempt'] for t in trace] == list(range(1, 8193))
    updates = 0
    warmup_delta = dict(sid_loss=0., dense_loss=0.)
    for t, batch in zip(trace, deterministic_batches(tasks['fit'], steps=8192, seed=c['seed'])):
        _, identity = pack_supervised_batch(batch, sids, content)
        assert t['batch_identity'] == tree_digest(identity)
        assert t['parent_users'] == identity['parent_user_ids'].tolist()
        assert t['prefix_indices'] == identity['prefix_target_indices'].tolist()
        assert t['final_target_count'] == 32 and identity['is_final'].all()
        assert all(math.isfinite(t[k]) and t[k] >= 0 for k in ('sid_loss', 'dense_loss'))
        assert (t['grad_norm'] is None) == t['skipped']
        assert t['grad_norm'] is None or math.isfinite(t['grad_norm']) and t['grad_norm'] >= 0
        n = updates + 1
        scale = n / 51 if n <= 51 else .5 * (1 + math.cos(math.pi * (n - 51) / (8192 - 51)))
        assert math.isclose(t['learning_rate'], .0003 * scale, rel_tol=1e-10, abs_tol=1e-15)
        updates += int(not t['skipped'])
        assert t['successful_updates'] == updates
        if t['attempt'] <= len(previous): assert t['batch_identity'] == previous[t['attempt'] - 1]['batch_identity']
        if t['attempt'] <= 51:
            for key in warmup_delta:
                warmup_delta[key] = max(warmup_delta[key], abs(t[key] - previous[t['attempt'] - 1][key]))
    assert updates == r['successful_updates'] and sum(t['skipped'] for t in trace) == r['skipped_updates']
    assert warmup_delta == r['warmup_loss_max_abs_delta_from_v2']
    assert len(r['evaluations']) == 2 * len(c['evaluation_attempts'])
    assert {(e['attempt'], e['group']) for e in r['evaluations']} == {
        (s, g) for s in c['evaluation_attempts'] for g in ('fit', 'reserved')}
    initial_comparison = {}
    for e in r['evaluations']:
        assert sha(e['path']) == e['sha256']
        inputs, _ = pack_supervised_batch(tasks[e['group']], sids, content)
        with np.load(e['path']) as a:
            np.testing.assert_array_equal(a['labels'], inputs['labels_sids'])
            np.testing.assert_array_equal(a['raw_rows'], index.lookup_tokens(a['tokens']))
            assert a['ce'].shape == a['top1'].shape == a['top20'].shape == (128, 4)
            assert np.isfinite(a['ce']).all() and (a['ce'] >= 0).all()
            assert a['top1'].dtype == a['top20'].dtype == np.bool_ and np.all(~a['top1'] | a['top20'])
            assert generation_metrics(a['tokens'], a['labels']) == e['natural_beam']
            assert all(e[k] == value for k, value in beam_format(a['tokens']).items())
            for key, arr in [('teacher_ce', 'ce'), ('teacher_top1', 'top1'), ('teacher_top20', 'top20')]:
                np.testing.assert_allclose(a[arr].mean(0), e[key], rtol=1e-6, atol=1e-7)
            assert float((a['raw_rows'] >= 0).mean()) == e['valid_beam_fraction']
            if e['attempt'] == 0:
                prior = next(p for p in m['previous_initial_evaluations'] if p['group'] == e['group'])
                with np.load(prior['path']) as before:
                    np.testing.assert_array_equal(a['labels'], before['labels'])
                    initial_comparison[e['group']] = dict(tokens_equal=bool(np.array_equal(a['tokens'], before['tokens'])),
                        max_ce_abs_delta=float(np.abs(a['ce'] - before['ce']).max()))
    write_new(path.parent / 'budget_acceptance.json', dict(status='passed_diagnostic_integrity',
        report=dict(path=str(source), sha256=sha(source)), job_id=r['job_id'], initial_comparison=initial_comparison,
        formal_training_ready=False, quality_decision='author_review_required', automatic_successor=False))
    print('[liger-budget] independent audit passed; NO formal successor', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'accept'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    globals()[args.command](args.manifest.resolve())
