"""Create-only provenance and acceptance gates for LIGER training."""
import json
import os
import copy
from pathlib import Path

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new
from src.baselines.liger_data import split_files

PRIOR = ROOT / 'logs/training_launch/liger_cold_2026-10-19_v1'
PRIOR_RECEIPT_SHA = '04cab5f63f9de86b67cba09af0dcd5be7848e801e2244ab285e7473642160a1d'
CONFIG = ROOT / 'configs/baselines/liger_train_seed42_v1.json'
SOURCES = ['scripts/liger_training_manifest.py', 'scripts/train_liger_cpu4.py',
    'scripts/probe_liger_resume.py', 'src/baselines/liger_runtime.py',
    'src/baselines/liger_state.py', 'src/baselines/liger_training.py', 'src/baselines/liger_validation.py',
    'tests/test_liger_state.py', 'tests/test_liger_training.py', 'tests/test_liger_validation.py',
    'tests/test_liger_training_entrypoints.py', 'tests/test_liger_continuation.py',
    'run_liger_resume_smoke_cpu4.sh', 'run_liger_train_cpu4.sh']

# Explicit migration boundary: no model, loss, data, inference, optimizer or
# scheduler implementation may change between the accepted epochs.
CONTINUATION_REVISIONS = {
    'scripts/liger_training_manifest.py', 'scripts/train_liger_cpu4.py',
    'tests/test_liger_training_entrypoints.py', 'tests/test_liger_continuation.py',
    'run_liger_train_cpu4.sh',
}


def stage_bounds(m, config):
    stage = m.get('stage', dict(start_epoch=0, stop_after_epoch=1))
    start, stop = stage['start_epoch'], stage['stop_after_epoch']
    assert 0 <= start < stop <= config['training']['max_epochs'] == 3
    assert stop == start + 1, 'Only one additional epoch is authorized per launch'
    assert (start == 0) == (m['phase'] == 'first_epoch_formal_training')
    return start, stop


def read_bound(entry):
    assert sha(entry['path']) == entry['sha256']
    return json.loads(Path(entry['path']).read_text())


def verify_continuation(m):
    """Verify the accepted predecessor and narrowly scoped source migration."""
    link = m['continuation']
    old = read_bound(link['manifest'])
    receipt = read_bound(link['acceptance'])
    r = read_bound(receipt['report'])
    assert old['formal_training_ready'] and r['status'] == receipt['status'] == 'passed'
    assert r['manifest_sha256'] == link['manifest']['sha256']
    assert r['source_git_commit'] == old['source_git_commit']
    assert r['job_id'] == receipt['job_id']
    assert not r['test_records_read'] and receipt['no_test_access']
    assert r['stopped_for_author_acceptance'] and not r['automatic_successor']
    for key in ('config', 'config_sha256', 'input_metadata', 'upstream_root', 'upstream_commit',
                'upstream_files', 'assets', 'catalog', 'index_artifacts', 'asset_rows',
                'seed', 'data_root', 'train_files', 'validation_files', 'smoke_acceptance'):
        assert m[key] == old[key], f'Continuation changed frozen input: {key}'
    allowed = {str(ROOT / name) for name in CONTINUATION_REVISIONS}
    old_sources, new_sources = old['execution_sources'], m['execution_sources']
    assert set(old_sources) <= set(new_sources), 'Cannot drop source bindings'
    changed = {p for p in new_sources if new_sources[p] != old_sources.get(p)}
    assert changed <= allowed, f'Training/evaluation core changed: {changed - allowed}'
    assert sorted(changed) == link['changed_orchestration_sources']
    config = json.loads(Path(m['config']).read_text())
    start, stop = stage_bounds(m, config)
    p = r['final_progress']
    from src.baselines.liger_training import check_progress
    check_progress(p, config['training'])
    assert start == p['epoch'] == receipt['completed_epochs']
    assert p['next_batch'] == p['samples_in_epoch'] == 0
    assert r['best'] == p['best'] == receipt['best']
    assert receipt['updates'] == p['updates'] and receipt['skipped_updates'] == p['skipped_updates']
    assert receipt['original_max_epochs'] == 3
    assert receipt['full_training_records'] == config['training']['records_per_epoch']
    assert receipt['full_validation_records'] == config['validation']['records']
    checkpoint = r['checkpoints'][-1]
    assert checkpoint == m['resume_checkpoint']
    assert checkpoint['epoch'] == start and checkpoint['next_batch'] == 0
    assert checkpoint['updates'] == p['updates']
    assert sha(checkpoint['path']) == checkpoint['sha256']
    assert Path(checkpoint['path']).resolve().is_relative_to((ROOT / 'logs/train/runs').resolve())
    for v in p['validations']:
        for key in ('report', 'acceptance', 'checkpoint'):
            assert sha(v[key]['path']) == v[key]['sha256']
    return old, r


def prepare_continuation(path, previous_manifest, previous_acceptance):
    path = Path(path).resolve()
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    previous_manifest, previous_acceptance = Path(previous_manifest).resolve(), Path(previous_acceptance).resolve()
    old = json.loads(previous_manifest.read_text())
    receipt = json.loads(previous_acceptance.read_text())
    r = read_bound(receipt['report'])
    m = copy.deepcopy(old)
    start = r['final_progress']['epoch']
    m.update(source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        phase='epoch_continuation', formal_training_ready=True, test_records_read=False,
        automatic_successor=False, initialization='accepted formal checkpoint; original three-epoch budget',
        stage=dict(start_epoch=start, stop_after_epoch=start + 1),
        run_directory=str(ROOT / 'logs/train/runs' / path.parent.name),
        resume_checkpoint=r['checkpoints'][-1])
    m['execution_sources'].update({str(ROOT / name): sha(ROOT / name) for name in SOURCES})
    m['continuation'] = dict(manifest=dict(path=str(previous_manifest), sha256=sha(previous_manifest)),
        acceptance=dict(path=str(previous_acceptance), sha256=sha(previous_acceptance)),
        changed_orchestration_sources=sorted(p for p, h in m['execution_sources'].items()
                                            if h != old['execution_sources'].get(p)))
    verify_config(json.loads(CONFIG.read_text()), json.loads((ROOT / 'configs/baselines/liger_history_last_v1.json').read_text()))
    verify_continuation(m)
    for key in ('execution_sources', 'input_metadata', 'upstream_files'): verify_hashes(m[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'],
                         stage=m['stage'])), flush=True)


def verify_config(config, candidate):
    t, b = config['training'], candidate['training_budget']
    for key in ('batch_size', 'drop_last', 'records_per_epoch', 'batches_per_epoch',
                'max_epochs', 'max_optimizer_steps', 'warmup_steps'):
        assert t[key] == b[key]
    assert t['stop_after_epoch'] == b['stop_for_acceptance_after_epochs'] == 1
    assert config['optimizer'] == dict(name='AdamW', learning_rate=.0003, weight_decay=.035, clip_norm=1.)
    assert config['validation']['selection'] == ['hybrid Recall@5 descending', 'hybrid NDCG@5 descending', 'epoch ascending']
    assert config['validation']['records'] == 347416 and config['validation']['batch_size'] == 32
    assert not config['validation']['test_access'] and candidate['candidate_universe_finalized']
    assert t['chunk_size'] == 65536 and t['shuffle_buffer'] == 8192
    assert config['smoke'] == dict(train_steps=3, save_after_step=1, validate_after_step=2, validation_records=65)


def prepare(path, *, smoke_receipt=None, resume_checkpoint=None):
    path = Path(path).resolve()
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    assert sha(PRIOR / 'cold_acceptance.json') == PRIOR_RECEIPT_SHA
    receipt = json.loads((PRIOR / 'cold_acceptance.json').read_text())
    assert receipt['status'] == 'passed' and sha(receipt['report']['path']) == receipt['report']['sha256']
    previous = json.loads((PRIOR / 'manifest.json').read_text())
    for key in ('execution_sources', 'input_metadata', 'upstream_files'): verify_hashes(previous[key])
    m = {key: previous[key] for key in ('upstream_root', 'upstream_commit', 'upstream_files',
        'assets', 'input_metadata', 'execution_sources', 'catalog', 'asset_rows', 'seed', 'data_root', 'index_artifacts')}
    for source in (PRIOR / 'manifest.json', PRIOR / 'cold_probe.json', PRIOR / 'cold_acceptance.json', CONFIG):
        m['input_metadata'][str(source)] = sha(source)
    m['execution_sources'].update({str(ROOT / name): sha(ROOT / name) for name in SOURCES})
    config = json.loads(CONFIG.read_text())
    verify_config(config, json.loads((ROOT / config['candidate_config']).read_text()))
    root = Path(m['data_root'])
    train = json.loads((root / 'protocol_manifest_train.json').read_text())
    m.update(source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        config=str(CONFIG), config_sha256=sha(CONFIG), train_files=split_files(train, root, 'train'),
        validation_files=split_files(train, root, 'evaluation'),
        phase='first_epoch_formal_training' if smoke_receipt else 'training_resume_smoke',
        candidate_universe_finalized=True, formal_training_ready=bool(smoke_receipt),
        test_records_read=False, automatic_successor=False, initialization='seed42 random; never smoke checkpoint')
    m['run_directory'] = str(ROOT / 'logs/train/runs' / path.parent.name)
    if smoke_receipt:
        smoke_receipt = Path(smoke_receipt).resolve()
        r = json.loads(smoke_receipt.read_text())
        assert r['status'] == 'passed' and r['fresh_process_exact_resume']
        assert r['execution_sources'] == m['execution_sources'] and r['config_sha256'] == m['config_sha256']
        for entry in r['reports']: assert sha(entry['path']) == entry['sha256']
        m['smoke_acceptance'] = dict(path=str(smoke_receipt), sha256=sha(smoke_receipt))
    if resume_checkpoint:
        assert smoke_receipt
        resume = Path(resume_checkpoint).resolve()
        assert resume.is_relative_to((ROOT / 'logs/train/runs').resolve())
        m['resume_checkpoint'] = dict(path=str(resume), sha256=sha(resume))
        m['initialization'] = 'hash-bound formal checkpoint; same original budget'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])), flush=True)


def binding(path):
    path = Path(path).resolve()
    assert sha(path) == os.environ['LIGER_TRAIN_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and not m['dirty_at_submission']
    for key in ('execution_sources', 'input_metadata', 'upstream_files'): verify_hashes(m[key])
    root = Path(m['data_root'])
    train = json.loads((root / 'protocol_manifest_train.json').read_text())
    for key, split in (('train_files', 'train'), ('validation_files', 'evaluation')):
        assert list(split_files(train, root, split)) == m[key]
    assert sha(m['config']) == m['config_sha256'] and sha(m['catalog']['path']) == m['catalog']['sha256']
    for entry in m['index_artifacts'].values(): assert sha(entry['path']) == entry['sha256']
    for asset in m['assets'].values():
        stat = Path(asset['path']).stat()
        assert stat.st_size == asset['size_bytes'] and stat.st_mtime_ns == asset['mtime_ns']
    if m['formal_training_ready']:
        assert m['phase'] in ('first_epoch_formal_training', 'epoch_continuation')
        if m['phase'] == 'epoch_continuation':
            verify_continuation(m)
            return m
        r = m['smoke_acceptance']
        assert sha(r['path']) == r['sha256']
        receipt = json.loads(Path(r['path']).read_text())
        assert receipt['status'] == 'passed' and receipt['fresh_process_exact_resume']
        assert receipt['execution_sources'] == m['execution_sources'] and receipt['config_sha256'] == m['config_sha256']
        for entry in receipt['reports']: assert sha(entry['path']) == entry['sha256']
    else:
        assert m['phase'] == 'training_resume_smoke'
    if 'resume_checkpoint' in m:
        assert m['formal_training_ready']
        assert sha(m['resume_checkpoint']['path']) == m['resume_checkpoint']['sha256']
    return m


def checkpoint_binding(m):
    # Stable across new launch IDs; any execution/config/data change requires an explicit migration.
    return dict(purpose='formal' if m['formal_training_ready'] else 'smoke',
                config_sha256=m['config_sha256'], execution_sources=m['execution_sources'],
                input_metadata=m['input_metadata'], upstream_commit=m['upstream_commit'],
                assets={k: v['sha256'] for k, v in m['assets'].items()},
                catalog_sha256=m['catalog']['sha256'], index_artifacts=m['index_artifacts'])
