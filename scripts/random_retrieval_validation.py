"""Full, fresh validation and frozen fusion selection for random-init job 158562.

This wrapper does not edit training code, resume training, or select a checkpoint.
It fixes the already selected 90k checkpoint and gates the subsequent slot study.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'logs/train/runs/2026-11-10/click-listwise-random-seed42-158562'
CHECKPOINT = RUN / 'checkpoints/checkpoint_000_090000.ckpt'
PREFLIGHT = ROOT / 'logs/training_preflight/listwise_random_seed42_158562.json'
AUDITS = ROOT / 'logs/random_init_audit/seed42_158562/formal'
DATA = ROOT / 'data/taobao_history_last_user_v1'
REFERENCE = ROOT / 'logs/analysis/paired_evaluation/listwise_seed42_132049_val_full_v1.h5'
REFERENCE_SHA = 'f6f09d4936c52be2fe613ec9b0d97c16a2e199d82f6ef53632074d5bc6048f7c'
N = 347416
ORDER = ['ndcg@10', 'ndcg@5', 'recall@10', 'recall@5']
PROTOCOL = dict(metric_order=ORDER, tie_breaker='smaller_rerank_weight', beam_weight=1.0,
                hash_sample=dict(word=0, modulus=1, remainder=0),
                rerank_weight_grid=dict(start='0.00', stop='0.50', step='0.01'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_commit(expected):
    require(bool(re.fullmatch('[0-9a-f]{40}', expected)), 'An exact execution commit is required')
    require(os.environ.get('SOURCE_GIT_COMMIT') == expected, 'Execution environment commit mismatch')
    subprocess.run(['git', 'diff', '--exit-code', expected, '--', 'src', 'scripts', 'tests', 'configs'],
                   cwd=ROOT, check=True, capture_output=True)
    for path in ('src', 'scripts', 'tests', 'configs'):
        result = subprocess.check_output(['git', 'status', '--porcelain', '--', path], cwd=ROOT, text=True)
        require(not result.strip(), f'Execution source is dirty: {path}')


def verify_random_initialization(audit_root):
    evidence = []
    for rank in range(3):
        for phase in ('initialization', 'train_start', 'updates', 'finished'):
            path = Path(audit_root) / f'rank_{rank}_{phase}.json'
            report = read(path)
            require(report.get('status') == 'passed' and report.get('phase') == phase
                    and report.get('global_rank') == rank and report.get('world_size') == 3
                    and report.get('seed') == 42 and report.get('smoke') is False
                    and report.get('method') == 'joint-listwise-random-init', 'Invalid formal audit identity')
            step = {'initialization': 0, 'train_start': 0, 'updates': 2, 'finished': 115000}[phase]
            require(report.get('global_step') == step, 'Unexpected formal audit step')
            if phase == 'initialization':
                require(report.get('constructed_parameters_unchanged') is True
                        and report.get('optimizer_initially_empty') is True
                        and 'initialization_checkpoint' in report and report['initialization_checkpoint'] is None
                        and 'resume_checkpoint' in report and report['resume_checkpoint'] is None,
                        'Training was not audited as fresh random initialization')
            if phase == 'finished':
                require(report.get('first_two_updates_verified') is True, 'Missing update evidence')
            evidence.append(record(path))
    return evidence


def validate_config(cfg):
    require(cfg.seed == 42 and cfg.ckpt_path is None and cfg.sequence_length == 32
            and cfg.hist_itemkey_length == 100, 'Unexpected random-init run configuration')
    model = cfg.model
    require(model.manual_ckpt_path is None and model.load_only_weights is False
            and model.rerank_objective == 'listwise' and model.top_k_for_generation == 20
            and model.top_k_for_score == 10 and model.active_top_k_for_score is None
            and model.beam_log_prob_weight == 1.0 and model.rerank_logit_weight == 0.1,
            'Model initialization or frozen inference settings changed')


def validate_checkpoint_header(payload):
    require(isinstance(payload, dict) and payload.get('global_step') == 90000,
            'Checkpoint must be the selected 90000-update checkpoint, not the 115000-step terminal run')
    require(isinstance(payload.get('state_dict'), dict) and bool(payload['state_dict']),
            'Missing checkpoint state_dict')


def loader_fields():
    return dict(batch_size_per_device=32, num_workers=1, persistent_workers=False,
                drop_last=False, should_shuffle_rows=False, assign_all_files_per_worker=False)


@contextmanager
def argv(values):
    original = sys.argv
    sys.argv = values
    try:
        yield
    finally:
        sys.argv = original


def export_validation(output, binding):
    """Scoped wrappers enforce strict loading and explicitly nonpersistent fresh reading."""
    import torch
    from scripts import export_paired_evaluation as export
    from scripts.profile_efficiency_cpu4 import gpu_info, instantiate
    original = {name: getattr(export, name) for name in
                ('instantiate_model', 'configure_dataloader', 'instantiate_datamodule', 'get_loader', 'write_metadata')}
    evidence = dict(configurations=0, datamodules=0, loaders=0)

    def configure(cfg, split, batch_size, num_workers):
        require(split == 'val' and batch_size == 32 and num_workers == 1, 'Validation loader scope changed')
        validate_config(cfg)
        original['configure_dataloader'](cfg, split, batch_size, num_workers)
        dl = cfg.data_loading.val_dataloader_config.dataloader
        dl.persistent_workers = False
        require(all(getattr(dl, key) == value for key, value in loader_fields().items()), 'Invalid fresh-loader configuration')
        evidence['configurations'] += 1

    def datamodule(cfg):
        dm = original['instantiate_datamodule'](cfg)
        require(dm.trainer.world_size == 1 and dm.trainer.global_rank == 0, 'Validation must use a single rank')
        evidence['datamodules'] += 1
        return dm

    def get_loader(dm, split):
        require(split == 'val', 'Unexpected split')
        loader = original['get_loader'](dm, split)
        require(loader.num_workers == 1 and loader.persistent_workers is False
                and loader.dataset.is_for_training is False, 'Loader must be a fresh finite validation iterator')
        evidence['loaders'] += 1
        return loader

    def strict_model(cfg, checkpoint, device):
        require(Path(checkpoint).resolve() == CHECKPOINT.resolve(), 'Wrong checkpoint path')
        hardware = gpu_info(torch)
        require(torch.cuda.device_count() == 1 and '4090' in hardware['name'], 'Exactly one 4090 is required')
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        validate_checkpoint_header(payload)
        del payload
        gc.collect()
        model, loading = instantiate(cfg, {'model': 'listwise', 'checkpoint': binding['checkpoint']}, device, torch)
        require(loading == {'missing': 0, 'unexpected': 0}, 'Checkpoint parameter mismatch')
        require(all(bool(torch.isfinite(tensor).all()) for tensor in model.parameters()), 'Nonfinite loaded parameters')
        model.requires_grad_(False)
        evidence.update(checkpoint_loading=loading, checkpoint_global_step=90000,
                        loaded_parameters_finite=True, hardware=hardware)
        return model

    def metadata(path, values, num_samples):
        require(all(evidence.get(key) == 1 for key in ('configurations', 'datamodules', 'loaders')),
                'Validation loader was reconstructed or reused unexpectedly')
        require(num_samples == N and values['limits'] == {'max_batches': None, 'max_samples': None}, 'Incomplete validation')
        values = dict(values, execution_wrapper=dict(source=binding['wrapper'],
                      source_git_commit=binding['source_git_commit'], fresh_single_pass=True,
                      loader_settings=loader_fields(), **evidence))
        original['write_metadata'](path, values, num_samples)

    replacements = dict(instantiate_model=strict_model, configure_dataloader=configure,
                        instantiate_datamodule=datamodule, get_loader=get_loader, write_metadata=metadata)
    for name, replacement in replacements.items():
        setattr(export, name, replacement)
    try:
        with argv(['export_paired_evaluation', '--run-dir', str(RUN), '--checkpoint', str(CHECKPOINT),
                   '--training-preflight', str(PREFLIGHT), '--expected-data-root', str(DATA),
                   '--expected-protocol', 'history-last-user-v1', '--split', 'val', '--output', str(output),
                   '--batch-size', '32', '--num-workers', '1', '--flush-samples', '8192',
                   '--device', 'cuda', '--amp', 'bf16']):
            export.main()
    finally:
        for name, function in original.items():
            setattr(export, name, function)
    return evidence


def validate_metadata(meta, binding, expected=N):
    require(meta.get('schema_version') == 3 and meta.get('split') == 'val'
            and meta.get('sample_unit') == 'unique_user_history' and meta.get('num_samples') == expected,
            'Wrong validation metadata schema/split/count')
    require(meta.get('limits') == {'max_batches': None, 'max_samples': None}, 'Partial validation is forbidden')
    require(meta.get('checkpoint') == binding['checkpoint']['path']
            and meta.get('checkpoint_sha256') == binding['checkpoint']['sha256']
            and meta.get('run_dir') == binding['run_dir']
            and meta.get('hydra_config_sha256') == binding['config']['sha256']
            and meta.get('git_commit') == binding['source_git_commit'], 'Validation source/checkpoint mismatch')
    require(meta.get('batch_size') == 32 and meta.get('num_workers') == 1 and meta.get('amp') == 'bf16'
            and meta.get('top_k_for_generation') == 20 and meta.get('num_hierarchies') == 4
            and meta.get('num_embeddings_per_hierarchy') == 4096
            and meta.get('beam_log_prob_weight') == 1.0 and meta.get('rerank_logit_weight') == 0.1,
            'Validation inference settings changed')
    prov = meta.get('dataset_provenance', {})
    require(prov.get('protocol') == 'history-last-user-v1' and prov.get('expected_samples') == expected
            and prov.get('data_root') == binding['data_root'] and prov.get('initialization_checkpoint') is None
            and prov.get('source_compatibility') is None, 'Validation dataset provenance changed')
    require({key: prov.get('training_preflight', {}).get(key) for key in ('path', 'sha256')} == binding['preflight'],
            'Preflight binding mismatch')
    require(prov.get('manifests') == binding['manifests'], 'Dataset manifest binding mismatch')
    wrap = meta.get('execution_wrapper', {})
    require(wrap.get('source') == binding['wrapper'] and wrap.get('source_git_commit') == binding['source_git_commit']
            and wrap.get('fresh_single_pass') is True and wrap.get('loader_settings') == loader_fields()
            and all(wrap.get(key) == 1 for key in ('configurations', 'datamodules', 'loaders'))
            and wrap.get('checkpoint_loading') == dict(missing=0, unexpected=0)
            and wrap.get('checkpoint_global_step') == 90000 and wrap.get('loaded_parameters_finite') is True,
            'Missing strict loading / fresh iterator evidence')


def independent_metrics(target, beam, scores):
    """Count GT outrankers, independently of selector sorting; preserve beam-order ties."""
    import numpy as np
    hit_matrix = np.all(beam == target[:, None, :], axis=-1)
    require(bool((hit_matrix.sum(-1) <= 1).all()), 'Duplicate target candidate')
    hits = hit_matrix.any(-1)
    pos = hit_matrix.argmax(-1)
    gt_score = scores[np.arange(len(pos)), pos][:, None]
    ahead = (scores > gt_score) | ((scores == gt_score) & (np.arange(scores.shape[1])[None, :] < pos[:, None]))
    rank = ahead.sum(-1) + 1
    rank[~hits] = scores.shape[1] + 1
    discount = 1.0 / np.log2(rank + 1.0)
    return np.asarray([(discount * (rank <= 10)).sum(), (discount * (rank <= 5)).sum(),
                       (rank <= 10).sum(), (rank <= 5).sum()], dtype=np.float64)


def audit_export(output, reference, expected=N):
    import h5py
    import numpy as np
    shapes = dict(user_id=(expected,), example_hash=(expected, 2), target_sid=(expected, 4),
                  beam_sid=(expected, 20, 4), rerank_sid=(expected, 20, 4), beam_score=(expected, 20),
                  raw_rerank_logit=(expected, 20), rerank_score=(expected, 20))
    totals = np.zeros((51, 4), dtype=np.float64)
    runtime_totals = np.zeros(4, dtype=np.float64)
    # Model-facing SID tokens are shifted by +1: 1..4096, with 0 reserved
    # for padding. Base4097 is injective even across the padded token space.
    powers = np.array([4097 ** 3, 4097 ** 2, 4097, 1], dtype=np.int64)
    with h5py.File(output, 'r') as f, h5py.File(reference, 'r') as ref:
        require(set(f.keys()) == set(shapes) and int(f.attrs.get('num_samples', -1)) == expected, 'Wrong H5 schema/count')
        require(all(f[key].shape == value for key, value in shapes.items()), 'Wrong H5 shapes')
        require(ref['user_id'].shape == (expected,), 'Wrong reference size')
        require(f['user_id'].dtype == np.int64 and f['example_hash'].dtype == np.uint64, 'Wrong identity dtype')
        users, hashes = f['user_id'][:], f['example_hash'][:]
        require(len(np.unique(users)) == expected and len(np.unique(hashes, axis=0)) == expected, 'Duplicate user/hash')
        ref_users = ref['user_id'][:]
        require(len(np.unique(ref_users)) == expected, 'Duplicate reference users')
        order, ref_order = np.argsort(users), np.argsort(ref_users)
        for key in ('user_id', 'example_hash', 'target_sid'):
            require(np.array_equal(f[key][:][order], ref[key][:][ref_order]), f'Validation identity differs: {key}')
        for start in range(0, expected, 8192):
            a = {key: f[key][start:start + 8192] for key in shapes if key not in ('user_id', 'example_hash')}
            for key in ('target_sid', 'beam_sid', 'rerank_sid'):
                require(a[key].dtype == np.int16 and bool(((a[key] >= 1) & (a[key] <= 4096)).all()), 'Invalid SID dtype/range')
            for key in ('beam_score', 'raw_rerank_logit', 'rerank_score'):
                require(a[key].dtype == np.float32 and bool(np.isfinite(a[key]).all()), 'Nonfinite or non-FP32 scores')
            for key in ('beam_score', 'rerank_score'):
                require(bool((np.diff(a[key], axis=1) <= 0).all()), 'Scores are not sorted')
            beam_keys = a['beam_sid'].astype(np.int64) @ powers
            rerank_keys = a['rerank_sid'].astype(np.int64) @ powers
            require(bool((np.diff(np.sort(beam_keys, axis=1), axis=1) > 0).all()), 'Duplicate candidate')
            require(np.array_equal(np.sort(beam_keys, axis=1), np.sort(rerank_keys, axis=1)), 'Rerank candidate set changed')
            # Runtime BF16 fusion is preserved, not reconstructed as FP32 fusion.
            # Report-layer FP32 beta scanning is independently recomputed below.
            runtime_totals += independent_metrics(a['target_sid'], a['rerank_sid'], a['rerank_score'])
            for index in range(51):
                scores = 1.0 * a['beam_score'] + (index / 100) * a['raw_rerank_logit']
                require(bool(np.isfinite(scores).all()), 'Nonfinite fusion')
                totals[index] += independent_metrics(a['target_sid'], a['beam_sid'], scores)
    return dict(num_examples=expected, unique_users=expected, unique_input_hashes=expected,
                input_identity_matches_reference=True, candidate_integrity_passed=True,
                independent_grid_metrics=(totals / expected).tolist(),
                runtime_metrics=dict(zip(ORDER, (runtime_totals / expected).tolist())))


def validate_selection(selection, output, sidecar, checkpoint, metrics, expected=N, commit=None):
    import numpy as np
    require(selection.get('schema_version') == 2 and selection.get('selection_scope') == 'validation_only'
            and selection.get('input_split') == 'val' and selection.get('num_examples') == expected,
            'Selection schema/scope/count mismatch')
    require(selection.get('input') == output['path'] and selection.get('input_sha256') == output['sha256']
            and selection.get('input_sidecar_sha256') == sidecar['sha256']
            and selection.get('input_checkpoint_sha256') == checkpoint['sha256'], 'Selection input binding mismatch')
    if commit is not None:
        require(selection.get('source_git_commit') == commit, 'Selector commit mismatch')
    require(selection.get('selection_protocol') == PROTOCOL, 'Selection grid/metric rule changed')
    rows = selection.get('candidates', [])
    require(len(rows) == 51, 'Wrong grid length')
    for index, row in enumerate(rows):
        require(row.get('beam_weight') == 1.0 and row.get('rerank_weight') == index / 100
                and row.get('variant') == f'fusion@1@{index / 100:.2f}'
                and set(row.get('metrics', {})) == set(ORDER), 'Grid candidate changed')
    actual = np.array([[row['metrics'][key] for key in ORDER] for row in rows], dtype=np.float64)
    require(bool(np.isfinite(actual).all()) and np.allclose(actual, metrics, rtol=0, atol=1e-12),
            'Independent selection metric recomputation failed')
    best = max(range(51), key=lambda i: tuple(actual[i]) + (-rows[i]['rerank_weight'],))
    require(selection.get('selected') == rows[best], 'Wrong selected beta/tie break')
    return dict(candidates_checked=51, metric_values_checked=204,
                max_absolute_metric_difference=float(np.max(np.abs(actual - metrics))), selected=rows[best])


def run(output_dir, expected_commit):
    verify_commit(expected_commit)
    output_dir = Path(output_dir)
    require(output_dir.is_absolute() and output_dir.resolve() == output_dir
            and output_dir.is_relative_to(ROOT / 'logs/analysis') and not output_dir.exists(), 'Unsafe/existing output directory')
    from src.runtime_cpu import THREAD_VARIABLES
    from src.runtime_export_cpu import check_export_allocation
    require(os.environ.get('GRID_EXPORT_CPU_BUDGET') == '4' and os.environ.get('GRID_FUSION_CPU_BUDGET') == '4'
            and all(os.environ.get(name) == '1' for name in THREAD_VARIABLES), 'CPU4 and single-thread libraries must be set before imports')
    allocation = check_export_allocation()
    audits = verify_random_initialization(AUDITS)
    preflight = read(PREFLIGHT)
    require(preflight.get('status') == 'passed' and preflight.get('initialization_checkpoint') is None
            and preflight.get('counts') == dict(evaluation=N, test=2061101, train=6573674), 'Wrong formal preflight')
    require(sha(REFERENCE) == REFERENCE_SHA, 'Accepted validation reference changed')
    manifests = {key: record(DATA / name) for key, name in
                 [('train', 'protocol_manifest_train.json'), ('test', 'protocol_manifest_test.json'), ('audit', 'protocol_audit.json')]}
    binding = dict(checkpoint=record(CHECKPOINT), config=record(RUN / '.hydra/config.yaml'),
                   preflight=record(PREFLIGHT), wrapper=record(__file__), manifests=manifests,
                   run_dir=str(RUN), data_root=str(DATA), source_git_commit=expected_commit)
    sources = [record(ROOT / name) for name in ('scripts/export_paired_evaluation.py', 'scripts/select_score_fusion.py',
               'scripts/profile_efficiency_cpu4.py', 'scripts/analyze_paired_evaluation.py', 'src/runtime_cpu.py', 'src/runtime_export_cpu.py')]
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new(output_dir / 'preparation.json', dict(status='prepared', binding=binding, audits=audits,
              reference=dict(path=str(REFERENCE), sha256=REFERENCE_SHA), execution_sources=sources, cpu_budget=allocation))
    output = output_dir / 'validation.h5'
    loading = export_validation(output, binding)
    metadata = read(output.with_suffix('.json'))
    validate_metadata(metadata, binding)
    require(not list(output_dir.glob('*.incomplete.*')), 'Incomplete export remains')
    audited = audit_export(output, REFERENCE)
    with argv(['select_score_fusion', '--input', str(output), '--beam-weight', '1.0',
               '--rerank-weight-start', '0.00', '--rerank-weight-stop', '0.50', '--rerank-weight-step', '0.01',
               '--metric-order', *ORDER, '--sample-modulus', '1', '--sample-remainder', '0',
               '--output-json', str(output_dir / 'fusion.json'), '--output-md', str(output_dir / 'fusion.md')]):
        from scripts import select_score_fusion
        select_score_fusion.main()
    output_record, sidecar = record(output), record(output.with_suffix('.json'))
    fusion = read(output_dir / 'fusion.json')
    selected = validate_selection(fusion, output_record, sidecar, binding['checkpoint'],
                                  audited['independent_grid_metrics'], commit=expected_commit)
    verify_commit(expected_commit)
    for entry in sources + audits + [binding['wrapper'], binding['config'], binding['preflight'], binding['checkpoint'], *manifests.values()]:
        require(sha(entry['path']) == entry['sha256'], f'Input/source changed during validation: {entry["path"]}')
    require(sha(REFERENCE) == REFERENCE_SHA, 'Validation identity reference changed during export')
    for entry in metadata['source_files'] + fusion['source_files']:
        require(sha(entry['path']) == entry['sha256'], 'Recorded execution source changed')
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  checkpoint=binding['checkpoint'], validation=output_record, sidecar=sidecar,
                  selection=record(output_dir / 'fusion.json'), selection_markdown=record(output_dir / 'fusion.md'),
                  num_examples=N, checkpoint_loading=loading['checkpoint_loading'], checkpoint_global_step=90000,
                  source_git_commit=expected_commit, fresh_single_pass=True, execution_wrapper=binding['wrapper'],
                  training_audits=audits, training_final_step=115000, cpu_budget=allocation,
                  hardware=loading['hardware'], reference=dict(path=str(REFERENCE), sha256=REFERENCE_SHA),
                  export_integrity=audited, fusion_acceptance=selected,
                  scope='Fixed 90k checkpoint; full validation-only sim10 beta selection before five-policy test; no training or test tuning',
                  limitation='Does not retrospectively validate the original training-internal checkpoint selection; only 90k weights are evaluated here')
    write_new(output_dir / 'acceptance.json', result)
    print(json.dumps(dict(status='passed', acceptance=str(output_dir / 'acceptance.json'),
                         num_examples=N, beta=fusion['selected']['rerank_weight'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    run(args.output_dir, args.expected_commit)
