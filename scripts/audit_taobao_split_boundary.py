#!/usr/bin/env python3
"""Measure cross-split overlap without changing frozen data or training preflight."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.verify_taobao_training_data import (
    _assert_audit, _assert_manifest_outputs, _assert_protocol_manifest,
    _identity_with_sha256, _load_json, _sha256,
)


def compare_sequences(train, test, semantic_ids, tail_items):
    """Exact comparisons, including the generator's cropped input/target view."""
    if tail_items < 2 or min(len(train), len(test)) < 2:
        raise ValueError('Sequences and generator view must contain at least two items')
    train_tail, test_tail = train[-tail_items:], test[-tail_items:]
    train_sid, test_sid = semantic_ids[train_tail], semantic_ids[test_tail]
    return {
        'full_sequence_equal': np.array_equal(train, test),
        'full_input_history_equal': np.array_equal(train[:-1], test[:-1]),
        'target_item_equal': bool(train[-1] == test[-1]),
        'generator_sample_equal_item': np.array_equal(train_tail, test_tail),
        'generator_input_equal_sid': np.array_equal(train_sid[:-1], test_sid[:-1]),
        'generator_sample_equal_sid': np.array_equal(train_sid, test_sid),
        'target_sid_equal': np.array_equal(train_sid[-1], test_sid[-1]),
        'test_target_in_training_input': bool(np.any(train[:-1] == test[-1])),
        'training_target_in_test_input': bool(np.any(test[:-1] == train[-1])),
        'training_sequence_is_test_prefix': np.array_equal(train, test[:len(train)]),
        'test_sequence_is_training_prefix': np.array_equal(test, train[:len(test)]),
    }


def checked_sequence(sequence, semantic_ids):
    sequence = np.asarray(sequence)
    if sequence.ndim != 1 or len(sequence) < 2 or sequence.dtype.kind not in 'iu':
        raise ValueError('Invalid retained sequence')
    if int(sequence.min()) < 0 or int(sequence.max()) >= len(semantic_ids):
        raise ValueError('Item row outside frozen Semantic-ID mapping')
    return sequence


def measure_boundaries(test_records, train_records, validation_records, semantic_ids, tail_items=8):
    # Exact uint32 storage: ~7.2 GB for the frozen test split. No digest-only
    # equality claim and no second full read of the test split are needed.
    if len(semantic_ids) > 2**32:
        raise ValueError('Mapping exceeds compact item-row storage')
    test_index = {}
    for user, sequence in test_records:
        user = int(user)
        if user in test_index:
            raise ValueError('Duplicate user within test')
        sequence = checked_sequence(sequence, semantic_ids)
        test_index[user] = sequence.astype('<u4').tobytes()
    pairs, counts, train_users = {}, {'test': len(test_index)}, set()
    metric_names = compare_sequences(np.array([0, 0]), np.array([0, 0]), semantic_ids, tail_items)
    for split, records in [('train', train_records), ('evaluation', validation_records)]:
        seen = train_users if split == 'train' else set()
        metrics = Counter({key: 0 for key in metric_names})
        overlapping, examples = 0, []
        for user, sequence in records:
            user = int(user)
            if user in seen:
                raise ValueError(f'Duplicate user within {split}')
            if split == 'evaluation' and user in train_users:
                raise ValueError('Train/validation user overlap')
            seen.add(user)
            sequence = checked_sequence(sequence, semantic_ids)
            if user not in test_index:
                continue
            other = np.frombuffer(test_index[user], dtype='<u4')
            flags = compare_sequences(sequence, other, semantic_ids, tail_items)
            metrics.update({key: int(value) for key, value in flags.items()})
            overlapping += 1
            if flags['generator_sample_equal_sid'] and len(examples) < 10:
                examples.append({'user_id': user, 'source_length': len(sequence),
                                 'test_length': len(other), **flags})
        counts[split] = len(seen)
        pairs[f'{split}_test'] = {
            'overlap_users': overlapping, 'comparisons_within_overlapping_users': dict(metrics),
            'fraction_of_test_users': overlapping / len(test_index) if test_index else 0.0,
            'duplicate_examples_local_only': examples,
        }
    return {'counts': counts, 'train_validation_overlap_users': 0, 'pairs': pairs}


def iter_records(files, split, batch_size, max_records=None):
    import tensorflow as tf

    description = {'user_id': tf.io.FixedLenFeature([], tf.int64),
                   'sequence_data': tf.io.VarLenFeature(tf.int64)}
    dataset = tf.data.TFRecordDataset([str(p) for p in files], compression_type='GZIP',
                                    num_parallel_reads=4)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    if max_records is not None:
        dataset = dataset.take(max_records)
    count = 0
    for serialized in dataset.batch(batch_size).prefetch(1):
        parsed = tf.io.parse_example(serialized, description)
        users = parsed['user_id'].numpy()
        ragged = tf.RaggedTensor.from_sparse(parsed['sequence_data'])
        offsets, values = ragged.row_splits.numpy(), ragged.values.numpy()
        for i, user in enumerate(users):
            yield int(user), values[offsets[i]:offsets[i + 1]]
        count += len(users)
        if count // 100000 != (count - len(users)) // 100000:
            print(f'[boundary] split={split} processed={count}', flush=True)
    print(f'[boundary] split={split} finished={count}', flush=True)


def run_audit(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite boundary report: {output}')
    data_root, run_dir = args.data_root.resolve(), args.run_dir.resolve()
    cfg_path = run_dir / '.hydra/config.yaml'
    cfg = OmegaConf.load(cfg_path)
    preflight_path = args.training_preflight.resolve()
    preflight = _load_json(preflight_path)
    protocol = 'history-last-user-v1'
    if (preflight['status'] != 'passed' or preflight['protocol'] != protocol
            or Path(preflight['data_root']).resolve() != data_root
            or Path(cfg.data_dir).resolve() != data_root):
        raise ValueError('Run / preflight / data root mismatch')
    paths = {name: data_root / filename for name, filename in {
        'train': 'protocol_manifest_train.json', 'test': 'protocol_manifest_test.json',
        'audit': 'protocol_audit.json'}.items()}
    manifests = {key: _load_json(path) for key, path in paths.items()}
    for split in ('train', 'test'):
        _assert_protocol_manifest(manifests[split], split, protocol)
        _assert_manifest_outputs(manifests[split], data_root)
    _assert_audit(manifests['audit'], manifests['train'], manifests['test'], protocol)
    for name, path in paths.items():
        if _sha256(path) != preflight['source_manifests'][name]['sha256']:
            raise ValueError(f'Changed {name} manifest since teacher preflight')
    sid_path = Path(preflight['semantic_id']['path'])
    if Path(cfg.semantic_id_path).resolve() != sid_path.resolve():
        raise ValueError('Run Semantic-ID path does not match preflight')
    if _sha256(sid_path) != preflight['semantic_id']['sha256']:
        raise ValueError('Changed Semantic-ID mapping since teacher preflight')
    sid_map = np.load(sid_path, mmap_mode='r', allow_pickle=False)
    hierarchy, tokens = int(cfg.num_hierarchies), int(cfg.sequence_length)
    if tokens % hierarchy or tokens // hierarchy < 2 or sid_map.shape[1] != hierarchy:
        raise ValueError('Unsupported generator sequence layout')
    for split in ('train', 'val', 'test'):
        loader = cfg.data_loading[f'{split}_dataloader_config'].dataloader
        if int(loader.sequence_length) != tokens or not loader.collate_fn._target_.endswith('.collate_fn_train'):
            raise ValueError('Generator view needs review for this collator / sequence length')
    fingerprints = {name: _identity_with_sha256(path) for name, path in {
        **paths, 'training_preflight': preflight_path, 'run_config': cfg_path,
        'auditor': Path(__file__), 'protocol_verifier': ROOT / 'scripts/verify_taobao_training_data.py',
        'collator': ROOT / 'src/data/loading/components/collate_functions.py',
        'preprocessing': ROOT / 'src/data/loading/components/pre_processing.py',
    }.items()}
    streams = {}
    for split in ('test', 'train', 'evaluation'):
        manifest = manifests['test' if split == 'test' else 'train']
        files = [Path(entry['path']) for entry in manifest['outputs'][split]['files']]
        streams[split] = iter_records(files, split, args.batch_size, args.max_records_per_split)
    measured = measure_boundaries(streams['test'], streams['train'], streams['evaluation'], sid_map, tokens // hierarchy)
    expected = {key: int(value['records']) for key, value in manifests['audit']['destinations'].items()}
    for key, count in measured['counts'].items():
        required = expected[key] if args.max_records_per_split is None else min(expected[key], args.max_records_per_split)
        if count != required or preflight['counts'][key] != expected[key]:
            raise ValueError(f'Unexpected scan count for {key}: {count}, expected {required}')
    # Recheck immutable inputs after the complete read; never rewrite them.
    for record in fingerprints.values():
        if _sha256(Path(record['path'])) != record['sha256']:
            raise ValueError(f'Input changed during audit: {record["path"]}')
    for split in ('train', 'test'):
        _assert_manifest_outputs(manifests[split], data_root)
    report = {
        'status': 'completed', 'protocol': protocol, 'data_root': str(data_root),
        'partial_run': args.max_records_per_split is not None,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_git_commit': os.environ.get('SOURCE_GIT_COMMIT'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'fingerprints': fingerprints,
        'semantic_id': preflight['semantic_id'], 'generator_items_including_target': tokens // hierarchy,
        'comparison_scope': 'same user across splits; exact item-row arrays and mapped SID tuples, not hashes',
        'limitations': ['No event timestamps: overlap is not by itself proof of future-information leakage.',
                        'Cross-user duplicate sequences and task redesign are outside this audit.',
                        'Generator view is not the full candidate-aware reranker input.'],
        **measured,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'report': str(output), 'partial_run': report['partial_run'],
                      'counts': measured['counts'], 'pairs': measured['pairs']}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'run-dir', 'training-preflight', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--max-records-per-split', type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_records_per_split is not None and args.max_records_per_split < 1):
        parser.error('Batch size and record limit must be positive')
    run_audit(args)


if __name__ == '__main__':
    main()
