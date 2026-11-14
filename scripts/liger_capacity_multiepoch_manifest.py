"""Frozen fresh twelve-epoch authority; keep the accepted one-pass lineage intact.

The old manifest is evidence, never a source of initial model weights. New
runtime and formal configuration refer to the same approved global budget.
Gate and formal purposes cannot exchange checkpoints.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new
from scripts import liger_capacity_training_manifest as prior_contract

PRIOR = ROOT / 'logs/training_launch/liger_capacity384_train_seed42_2026-11-12_v1/manifest.json'
PRIOR_SHA = 'e148c1ef542bf9e2ca797766a712c108eb0b269b0ec68fcc39d60ca6f3b560b6'
PRIOR_RUN = ROOT / 'logs/train/runs/liger_capacity384_train_seed42_2026-11-12_v1'
CONFIG = ROOT / 'configs/baselines/liger_capacity384_multiepoch_seed42_v1.json'
CONFIG_SHA = 'c0088c1907b78fad0a809910927bc38921ebcfa86a89829d9197b05ba53d9293'
GATE_PHASE = 'capacity384_multiepoch_gate'
FORMAL_PHASE = 'capacity384_multiepoch_formal'
RECEIPTS = {
    'training': ('training_acceptance.json', 'd428824d4e32e3fd0fe55a6f045d38d1f34e9222cf02e911ef55ed1c10c71b8d'),
    'validation': ('acceptance.json', 'd687e48c8f8f0a27e1dea5a356edabccc62b3d2a1f119570c70b9990b9a94731'),
}
SOURCES = [
    'scripts/liger_capacity_multiepoch_manifest.py',
    'src/baselines/liger_early_stopping.py',
    'scripts/train_liger_capacity_multiepoch.py',
    'scripts/validate_liger_capacity_multiepoch.py',
    'scripts/probe_liger_capacity_multiepoch.py',
    'tests/test_liger_capacity_multiepoch_manifest.py',
    'tests/test_liger_early_stopping.py',
    'tests/test_liger_capacity_multiepoch_training.py',
    'tests/test_liger_capacity_multiepoch_validation.py',
    'tests/test_liger_capacity_multiepoch_gate.py',
    'run_liger_capacity_multiepoch_gate_cpu4.sh',
    'run_liger_capacity_multiepoch_train_cpu4.sh',
    'run_liger_capacity_multiepoch_val_cpu4.sh',
]
CHANGED = frozenset({'phase', 'formal_training_ready', 'initialization',
    'source_git_commit', 'execution_sources', 'input_metadata', 'config',
    'config_sha256', 'formal_config', 'run_directory', 'gate_acceptance'})
ADDED = frozenset({'multiepoch_predecessor', 'multiepoch_predecessor_acceptances'})


def read_bound(entry):
    assert isinstance(entry, dict) and set(entry) == {'path', 'sha256'}
    assert sha(entry['path']) == entry['sha256'], 'Referenced artifact bytes changed'
    return json.loads(Path(entry['path']).read_text())


def check_config(c):
    assert sha(CONFIG) == CONFIG_SHA, 'Approved multiepoch configuration changed'
    prior_contract._exact(c, json.loads(CONFIG.read_text()))
    t, e = c['training'], c['early_stopping']
    assert t['batch_size'] == t['logical_batch_size'] == t['physical_batch_size'] * t['accumulation_steps'] == 2048
    assert t['batches_per_epoch'] == (t['prefixes_per_epoch'] + 2047) // 2048 == 22460
    assert t['final_batch_size'] == t['prefixes_per_epoch'] - (t['batches_per_epoch'] - 1) * 2048 == 81
    assert t['min_epochs'] == e['min_epochs'] == 6
    assert t['max_epochs'] == e['max_epochs'] == 12 and e['patience'] == 5
    assert t['max_attempts'] == t['max_optimizer_steps'] == 12 * t['batches_per_epoch'] == 269520
    assert t['warmup_steps'] == 1123 and e['min_delta'] == 0
    return c


def predecessor():
    assert sha(PRIOR) == PRIOR_SHA, 'Accepted one-pass manifest changed'
    old = json.loads(PRIOR.read_text())
    prior_contract.verify(old)
    assert old['phase'] == prior_contract.FORMAL_PHASE
    receipts, values = {}, {}
    evidence = {str(PRIOR): PRIOR_SHA}
    for key, (filename, digest) in RECEIPTS.items():
        path = PRIOR_RUN / filename
        receipts[key] = dict(path=str(path), sha256=digest)
        values[key] = read_bound(receipts[key])
        evidence[str(path)] = digest
    train, val = values['training'], values['validation']
    assert train['status'] == 'passed_training' and train['job_id'] == '160438'
    assert val['status'] == 'passed' and val['job_id'] == '160439'
    for r in (train, val):
        assert r['manifest_sha256'] == PRIOR_SHA and r['source_git_commit'] == old['source_git_commit']
        assert r['test_records_read'] is False and r['automatic_successor'] is False
        report = read_bound(r['report'])
        assert report['status'] == 'passed' and report['manifest_sha256'] == PRIOR_SHA
        evidence[r['report']['path']] = r['report']['sha256']
    assert val['completed_passes'] == 1 and val['full_validation_records'] == 347416
    assert val['validation_model_weights_unchanged'] is True
    assert train['counts']['prefix_count'] == val['full_training_prefixes'] == 45996113
    assert train['counts']['final_count'] == val['full_training_parents'] == 6573674
    assert val['training_acceptance'] == receipts['training']
    assert train['initial_state_sha256'] == old['expected_initial_state_sha256']
    return old, receipts, evidence


def checkpoint_binding(m):
    assert m['phase'] in (GATE_PHASE, FORMAL_PHASE)
    # Reuse the unchanged metadata layout, changing only this explicitly new
    # purpose. The old helper does not read or restore an old checkpoint.
    surrogate = dict(m, phase=prior_contract.GATE_PHASE)
    result = prior_contract.checkpoint_binding(surrogate)
    c = check_config(read_bound(m['formal_config']))
    result.update(purpose=m['phase'], training_budget={k:c['training'][k] for k in (
        'max_epochs', 'min_epochs', 'max_attempts', 'max_optimizer_steps', 'warmup_steps')},
        early_stopping=copy.deepcopy(c['early_stopping']))
    return result


def verify_gate(entry, m):
    r = read_bound(entry)
    assert r['status'] == 'passed_capacity384_multiepoch_gate' and r['phase'] == GATE_PHASE
    for key in ('exact_fresh_process_replay', 'cross_epoch_exact_replay',
                'validation_model_weights_unchanged', 'early_stopping_policy_verified'):
        assert r[key] is True
    assert r['two_gpu_validation_records'] == 65
    for key in ('formal_training_ready', 'formal_checkpoint_produced', 'test_records_read', 'automatic_successor'):
        assert r[key] is False
    gate = read_bound(r['gate_manifest'])
    c = verify(gate)
    assert gate['phase'] == GATE_PHASE and gate['gate_acceptance'] is None
    assert r['manifest_sha256'] == r['gate_manifest']['sha256']
    assert r['source_git_commit'] == gate['source_git_commit']
    assert r['formal_config_sha256'] == m['formal_config']['sha256']
    assert r['initial_state_sha256'] == m['expected_initial_state_sha256']
    assert r['execution_sources'] == m['execution_sources']
    assert isinstance(r['job_id'], str) and r['job_id']
    for key in set(m) - {'phase', 'formal_training_ready', 'source_git_commit', 'run_directory', 'gate_acceptance'}:
        prior_contract._exact(gate[key], m[key], f'gate/formal.{key}')
    assert gate['run_directory'] != m['run_directory']
    directory = Path(gate['run_directory'])
    assert Path(entry['path']).resolve() == directory/'gate_acceptance.json'
    from scripts.probe_liger_capacity_multiepoch import verify_gate_reports
    reports = verify_gate_reports(gate, c, directory)
    canonical = lambda values: sorted(json.dumps(x, sort_keys=True) for x in values)
    assert canonical(r['reports']) == canonical(reports), 'Gate report set changed'
    for ref in reports: assert sha(ref['path']) == ref['sha256']
    first = json.loads((directory/'continuous.json').read_text())
    assert first['manifest_sha256'] == r['manifest_sha256'] and first['job_id'] == r['job_id']
    return r


def verify(m):
    old, receipts, evidence = predecessor()
    assert set(m) == set(old) | ADDED, 'Unknown or missing manifest authority'
    assert m['phase'] in (GATE_PHASE, FORMAL_PHASE)
    assert m['formal_training_ready'] is (m['phase'] == FORMAL_PHASE)
    assert all(m[k] is False for k in ('automatic_successor', 'test_records_read', 'dirty_at_submission'))
    assert isinstance(m['source_git_commit'], str) and re.fullmatch('[0-9a-f]{40}', m['source_git_commit'])
    for key in set(old) - CHANGED: prior_contract._exact(m[key], old[key], key)
    assert m['multiepoch_predecessor'] == dict(path=str(PRIOR), sha256=PRIOR_SHA)
    assert m['multiepoch_predecessor_acceptances'] == receipts
    c = check_config(json.loads(CONFIG.read_text()))
    assert m['formal_config'] == dict(path=str(CONFIG), sha256=CONFIG_SHA)
    assert m['config'] == str(CONFIG) and m['config_sha256'] == CONFIG_SHA
    assert m['initialization'] == c['initialization']
    sources = dict(old['execution_sources'])
    for name in SOURCES:
        path = str(ROOT/name)
        assert path not in sources, 'New execution sources may not replace inherited bytes'
        sources[path] = sha(path)
    assert m['execution_sources'] == sources
    assert m['input_metadata'] == dict(old['input_metadata'], **evidence, **{str(CONFIG): CONFIG_SHA})
    for group in ('execution_sources', 'input_metadata', 'upstream_files'): verify_hashes(m[group])
    run = Path(m['run_directory'])
    assert run.parent == ROOT/'logs/train/runs' and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', run.name)
    if m['phase'] == GATE_PHASE:
        assert m['gate_acceptance'] is None
    else:
        assert isinstance(m['gate_acceptance'], dict)
        verify_gate(m['gate_acceptance'], m)
    return c


def prepare(path, phase='gate', gate_receipt=None):
    path = Path(path).resolve()
    assert phase in ('gate', 'formal') and (phase == 'formal') == (gate_receipt is not None)
    assert not path.exists() and not git(ROOT, 'status', '--porcelain'), 'Commit source changes first'
    assert os.environ.get('NUMPY_MADVISE_HUGEPAGE') == '0'
    old, receipts, evidence = predecessor()
    c = check_config(json.loads(CONFIG.read_text()))
    m = copy.deepcopy(old)
    m.update(phase=FORMAL_PHASE if phase == 'formal' else GATE_PHASE,
        formal_training_ready=phase == 'formal', initialization=c['initialization'],
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'),
        config=str(CONFIG), config_sha256=CONFIG_SHA,
        formal_config=dict(path=str(CONFIG), sha256=CONFIG_SHA),
        multiepoch_predecessor=dict(path=str(PRIOR), sha256=PRIOR_SHA),
        multiepoch_predecessor_acceptances=receipts,
        execution_sources=dict(old['execution_sources'], **{str(ROOT/name):sha(ROOT/name) for name in SOURCES}),
        input_metadata=dict(old['input_metadata'], **evidence, **{str(CONFIG): CONFIG_SHA}),
        run_directory=str(ROOT/'logs/train/runs'/path.parent.name), gate_acceptance=None)
    if gate_receipt:
        gate_receipt = Path(gate_receipt).resolve()
        m['gate_acceptance'] = dict(path=str(gate_receipt), sha256=sha(gate_receipt))
    verify(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, m)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'], phase=m['phase'])))
    return m


def binding(path):
    path = Path(path).resolve()
    assert sha(path) == os.environ['LIGER_MULTIEPOCH_MANIFEST_SHA256']
    m = json.loads(path.read_text())
    assert m['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT']
    assert os.environ.get('NUMPY_MADVISE_HUGEPAGE') == '0'
    return m, verify(m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare-gate', 'prepare-formal'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--gate-receipt', type=Path)
    args = parser.parse_args()
    if (args.mode == 'prepare-formal') != (args.gate_receipt is not None):
        parser.error('Exactly prepare-formal requires --gate-receipt')
    prepare(args.manifest, phase=args.mode.removeprefix('prepare-'), gate_receipt=args.gate_receipt)
