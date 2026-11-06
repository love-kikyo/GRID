"""Freeze the authorized seed42 inference-only sweep. Standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'retrieval': [[20, 0], [20, 5], [20, 10], [20, 20]],
          'beam': [[10, 10], [40, 10]]}
SOURCES = [
    'scripts/prepare_inference_sensitivity.py', 'scripts/run_inference_sensitivity.py',
    'scripts/sensitivity_core.py', 'scripts/audit_inference_sensitivity.py',
    'scripts/export_paired_evaluation.py', 'scripts/profile_rerank_efficiency.py',
    'scripts/profile_efficiency_cpu4.py', 'run_inference_sensitivity_cpu4.sh',
    'src/runtime_cpu.py', 'src/runtime_export_cpu.py',
    'src/models/modules/semantic_id/tiger_generation_model.py',
    'src/models/modules/huggingface/transformer_base_module.py',
    'src/models/modules/base_module.py',
    'src/data/loading/components/dataloading.py',
    'src/data/loading/components/iterators.py',
    'src/data/loading/components/interfaces.py',
    'src/data/loading/components/pre_processing.py',
    'src/data/loading/components/collate_functions.py',
    'tests/test_sensitivity_core.py', 'tests/test_sensitivity_audit.py',
    'tests/test_inference_sensitivity_runner.py',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f'Changed accepted source: {path}')
    return {'path': str(path), 'sha256': digest}


def write_new(path, payload):
    with Path(path).open('x') as f:
        f.write(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def prepare(path):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit the execution sources before creating the manifest')
    analysis = ROOT / 'logs/analysis/paired_evaluation'
    old = json.loads((analysis / 'test_acceptance_seed42_full_v1.json').read_text())
    if old['status'] != 'passed':
        raise ValueError('Original full-test acceptance missing')
    bound = old['models']['listwise']
    sidecar = analysis / 'listwise_seed42_132049_test_full_v1.json'
    sidecar_record = record(sidecar, bound['sidecar_sha256'])
    meta = json.loads(sidecar.read_text())
    selection = analysis / 'listwise_seed42_132049_fusion_val.json'
    selection_record = record(selection, bound['selection_sha256'])
    selected = json.loads(selection.read_text())['selected']
    if selected['beam_weight'] != 1 or selected['rerank_weight'] != .5:
        raise ValueError('Unexpected frozen validation weights')
    checkpoint = Path(meta['checkpoint'])
    stat = checkpoint.stat()
    if (stat.st_size != meta['checkpoint_stat']['size_bytes'] or
            stat.st_mtime_ns != meta['checkpoint_stat']['mtime_ns']):
        raise ValueError('Checkpoint stat differs from accepted export')
    run_dir = Path(meta['run_dir'])
    config = record(run_dir / '.hydra/config.yaml', meta['hydra_config_sha256'])
    metadata = [sidecar_record, config, selection_record,
                record(analysis / 'test_acceptance_seed42_full_v1.json'),
                record(ROOT / 'logs/training_preflight/listwise_seed42_132049.json')]
    for entry in meta['dataset_provenance']['manifests'].values():
        metadata.append(record(entry['path'], entry['sha256']))
    payload = {
        'schema_version': 1, 'seed': 42, 'protocol': 'history-last-user-v1',
        'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'dirty_at_submission': False, 'groups': GROUPS,
        'beta': .5, 'beam_weight': 1., 'runtime_beta': .1,
        'batch_size': 32, 'amp': 'bf16', 'num_workers': 1,
        'sequence_length': 32, 'hist_itemkey_length': 100,
        'expected_test_samples': 2061101, 'smoke_samples': 65,
        'run_dir': str(run_dir),
        'data_root': str(ROOT / 'data/taobao_history_last_user_v1'),
        'training_preflight': str(ROOT / 'logs/training_preflight/listwise_seed42_132049.json'),
        'checkpoint': {'path': str(checkpoint), 'sha256': bound['checkpoint_sha256']},
        'reference_test': {'path': str(sidecar.with_suffix('.h5')), 'sha256': bound['h5_sha256']},
        'frozen_beta': selection_record,
        'execution_sources': [record(ROOT / name) for name in SOURCES],
        'input_metadata': metadata,
        'scope': 'Fixed checkpoint inference sensitivity only; no retraining, no beta search',
        'candidate_policy': 'Fresh beam20 per batch shared across all M; beam10 and beam40 searched independently',
        'retrieval_policy': 'Actual top_k_for_score changes; active mask disabled; prompt length/position change disclosed',
        'old_main_results': 'Unchanged; same inputs/GT required, not cross-hardware bitwise predictions',
        'hardware_requirement': 'Single RTX 4090; entire job at most four CPU cores',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, payload)
    print(json.dumps({'manifest': str(path), 'sha256': sha(path), 'source_git_commit': payload['source_git_commit']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    prepare(parser.parse_args().manifest.resolve())
