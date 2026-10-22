"""Create-only binding for the new prefix/standardization integration gate.

Old formal epochs and diagnostic weights are never resumed. Exact prefix
counts are derived from the already audited last-eight catalog inventory,
without scanning training TFRecords again or claiming a new length histogram.
"""
import json
import math
import os
from pathlib import Path

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new

CONFIG = ROOT / 'configs/baselines/liger_prefix_gate_v3.json'
PRIOR = ROOT / 'logs/training_launch/liger_train_seed42_epoch02_2026-10-21_v1/manifest.json'
PRIOR_SHA = 'e4339dc70f0ec440003618cf25b23c9d370756a74e0c4da8b7d19b371da1f420'
LEARNING = ROOT / 'logs/training_launch/liger_learning_2026-10-22_v2/manifest.json'
LEARNING_RECEIPT = LEARNING.parent / 'learning_acceptance.json'
LEARNING_RECEIPT_SHA = 'f321360ecff238bcd3d5ab5bec82520896f355fb7d1415864fe27eac07fe2e47'
INVENTORY = ROOT / 'logs/training_launch/liger_catalog_2026-10-19_v1/catalog_report.json'
INVENTORY_RECEIPT = INVENTORY.parent / 'catalog_acceptance.json'
SOURCES = ['scripts/liger_prefix_manifest.py', 'scripts/probe_liger_prefix.py',
    'src/baselines/liger_prefix_stream.py', 'src/baselines/liger_prefix_runtime.py',
    'src/baselines/liger_prefix_training.py', 'tests/test_liger_prefix_stream.py',
    'tests/test_liger_prefix_runtime.py', 'tests/test_liger_prefix_training.py',
    'tests/test_liger_prefix_gate.py', 'run_liger_prefix_gate_cpu4.sh']


def check_config(c):
    assert c['protocol'] == 'liger-fixed-sid-scl-prefix-standardized-v3'
    assert c['phase'] == 'prefix_standardized_integration_gate' and c['seed'] == 42
    assert not c['formal_training_ready'] and not c['automatic_successor']
    assert c['candidate_config'] == 'configs/baselines/liger_history_last_v1.json'
    assert c['optimizer'] == dict(name='AdamW', learning_rate=.0003, weight_decay=.035, clip_norm=1.)
    t = c['training']
    assert (t['batch_size'], t['shuffle_buffer'], t['prefix_shuffle_buffer'], t['history_items']) == (1024, 8192, 8192, 7)
    assert (t['max_attempts'], t['max_optimizer_steps'], t['warmup_steps'], t['chunk_size']) == (3, 19260, 963, 65536)
    assert t['supervision'] == 'all_nonempty_history_prefixes_in_last8'
    assert t['loss_weighting'] == 'equal_prefix_examples_both_sid_and_dense' and not t['drop_last']
    assert c['standardization'] == dict(fit_scope='frozen_train_seen', reuse_verified_statistics=True)
    assert c['validation'] == dict(split='evaluation', batch_size=32, candidate_mode='full_cold', test_access=False)
    assert c['smoke'] == dict(train_steps=3, save_after_step=1, validate_after_step=2, validation_records=65)
    assert c['resources'] == dict(gpus=1, gpu_model='RTX 4090', cpu_ceiling=4, library_threads=1)


def inventory_counts(report, batch_size=1024):
    """For each retained window, prefixes = window length - one first item."""
    assert report['status'] == 'passed' and report['history_items'] == 7
    assert not report['held_out_records_read']
    parents = report['records']
    assert parents == report['unique_users'] == 6573674
    window_items = report['consumed_input_plus_target_occurrences']
    assert 2 * parents <= window_items <= 8 * parents
    prefixes = window_items - parents
    return dict(parents_per_pass=parents, retained_window_items=window_items,
        prefix_examples_per_pass=prefixes, final_targets_per_pass=parents,
        prefix_batches_per_pass=math.ceil(prefixes / batch_size), batch_size=batch_size,
        final_batch_size=(prefixes - 1) % batch_size + 1,
        derivation='sum(window_length - 1) = audited last8 occurrences - parent records',
        source_inventory=str(INVENTORY), source_inventory_sha256=sha(INVENTORY),
        new_tfrecord_scan=False, formal_budget_approved=False)


def prepare(path):
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    assert sha(PRIOR) == PRIOR_SHA and sha(LEARNING_RECEIPT) == LEARNING_RECEIPT_SHA
    old = json.loads(PRIOR.read_text())
    learned = json.loads(LEARNING.read_text())
    for m in (old, learned):
        for group in ('execution_sources', 'input_metadata', 'upstream_files'):
            verify_hashes(m[group])
    receipt = json.loads(LEARNING_RECEIPT.read_text())
    assert receipt['status'] == 'passed_diagnostic_integrity'
    assert sha(receipt['report']['path']) == receipt['report']['sha256']
    report = json.loads(Path(receipt['report']['path']).read_text())
    assert report['manifest_sha256'] == sha(LEARNING)
    assert report['source_git_commit'] == learned['source_git_commit']
    stats = report['standardization']
    assert sha(stats['path']) == stats['sha256']
    inventory = json.loads(INVENTORY.read_text())
    inventory_receipt = json.loads(INVENTORY_RECEIPT.read_text())
    assert inventory_receipt['status'] == 'passed'
    assert inventory_receipt['report']['sha256'] == sha(INVENTORY)
    c = json.loads(CONFIG.read_text())
    check_config(c)
    m = {k: old[k] for k in ('upstream_root', 'upstream_commit', 'upstream_files',
        'execution_sources', 'input_metadata', 'assets', 'catalog', 'index_artifacts',
        'asset_rows', 'data_root', 'train_files', 'validation_files', 'seed')}
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        m[group].update(learned[group])
    m['execution_sources'].update({str(ROOT / p): sha(ROOT / p) for p in SOURCES})
    m['input_metadata'].update({str(p): sha(p) for p in (PRIOR, LEARNING, LEARNING_RECEIPT,
        Path(receipt['report']['path']), Path(stats['path']), INVENTORY, INVENTORY_RECEIPT, CONFIG)})
    m.update(config=str(CONFIG), config_sha256=sha(CONFIG), standardization=stats,
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'), dirty_at_submission=False,
        phase=c['phase'], formal_training_ready=False, automatic_successor=False,
        initialization='fresh seed42; no old formal or diagnostic checkpoint',
        counts=inventory_counts(inventory), test_records_read=False)
    assert inventory['artifacts']['input_target_rows'] == m['catalog']
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])))


def binding(path):
    assert sha(path) == os.environ['LIGER_PREFIX_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT'] and not m['dirty_at_submission']
    assert m['phase'] == 'prefix_standardized_integration_gate'
    assert not any(m[k] for k in ('formal_training_ready', 'automatic_successor', 'test_records_read'))
    assert 'resume_checkpoint' not in m and 'test_files' not in m
    for group in ('execution_sources', 'input_metadata', 'upstream_files'):
        verify_hashes(m[group])
    assert sha(m['config']) == m['config_sha256']
    assert sha(m['standardization']['path']) == m['standardization']['sha256']
    c = json.loads(Path(m['config']).read_text())
    check_config(c)
    assert m['counts'] == inventory_counts(json.loads(INVENTORY.read_text()))
    from src.baselines.liger_data import split_files
    train = json.loads((Path(m['data_root']) / 'protocol_manifest_train.json').read_text())
    for split, key in (('train', 'train_files'), ('evaluation', 'validation_files')):
        assert list(split_files(train, Path(m['data_root']), split)) == m[key]
    assert sha(m['catalog']['path']) == m['catalog']['sha256']
    for entry in m['index_artifacts'].values():
        assert sha(entry['path']) == entry['sha256']
    for entry in m['assets'].values():
        stat = Path(entry['path']).stat()
        assert stat.st_size == entry['size_bytes'] and stat.st_mtime_ns == entry['mtime_ns']
    return m
