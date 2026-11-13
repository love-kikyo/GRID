"""Fresh 384-wide first-pass authority; diagnostics never initialize formal weights.

The inherited runtime config supplies the already accepted logical batch/stream
schedule. The separately frozen formal config authorizes the architecture,
physical microbatch, one-pass scope and one independent two-GPU validation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new
from scripts import liger_capacity_manifest as resource_contract

PRIOR = ROOT / 'logs/training_launch/liger_capacity384_2026-11-12_v1/manifest.json'
PRIOR_SHA = 'cb3096f0678e2015fe7349114903c84fee889959bdb7c7003b78963faf226d33'
CONFIG = ROOT / 'configs/baselines/liger_capacity384_train_seed42_v1.json'
CONFIG_SHA = '6b08807bcbd3d5a869a778e89d510b65adcb05bd9e67528361eb81b0cc8818e6'
GATE_PHASE = 'capacity384_training_gate'
FORMAL_PHASE = 'capacity384_first_pass_formal'
RECEIPTS = {
    'training': ('training_acceptance.json', '9f79a364a52df901b80ebe479dca99daea86bb3e4f7a6c98a60843b4c91ab4dc'),
    'validation': ('validation_acceptance.json', 'fa01d4b51b323e0f2b0daf01a6929455df481cdc551085c382bfcb2b6f1832be'),
    'routes': ('cuda_route_acceptance.json', 'c0b922365f639deee0c7cde29e7b1f3638ae885f1f4d3a986476385e4fce2427'),
}
SOURCES = [
    'scripts/liger_capacity_training_manifest.py', 'tests/test_liger_capacity_training_manifest.py',
    'scripts/train_liger_capacity_cpu4.py', 'tests/test_liger_capacity_training_entrypoints.py',
    'scripts/validate_liger_capacity_cpu4.py', 'tests/test_liger_capacity_full_validation.py',
    'scripts/probe_liger_capacity_training.py', 'tests/test_liger_capacity_training_gate.py',
    'run_liger_capacity_training_gate_cpu4.sh', 'run_liger_capacity_formal_train_cpu4.sh',
    'run_liger_capacity_formal_val_cpu4.sh',
]
CHANGED = frozenset({'phase', 'formal_training_ready', 'initialization',
    'source_git_commit', 'execution_sources', 'input_metadata'})
ADDED = frozenset({'formal_config', 'capacity_predecessor', 'resource_acceptances',
    'expected_initial_state_sha256', 'run_directory', 'gate_acceptance'})


def _exact(actual, wanted, path='config'):
    assert type(actual) is type(wanted), f'{path}: type drift'
    if isinstance(wanted, dict):
        assert set(actual) == set(wanted), f'{path}: fields drift'
        for key in wanted: _exact(actual[key], wanted[key], f'{path}.{key}')
    elif isinstance(wanted, list):
        assert len(actual) == len(wanted), f'{path}: length drift'
        for i, (a, b) in enumerate(zip(actual, wanted)): _exact(a, b, f'{path}[{i}]')
    else:
        assert actual == wanted, f'{path}: value drift'


def check_config(c):
    # Hash fixes the complete human-readable authorization, including unknown
    # fields/type distinctions; neither a changed file nor a partial dict passes.
    assert sha(CONFIG) == CONFIG_SHA, 'Approved formal configuration changed'
    _exact(c, json.loads(CONFIG.read_text()))
    t = c['training']
    assert t['physical_batch_size'] * t['accumulation_steps'] == t['logical_batch_size'] == t['batch_size'] == 2048
    assert t['batches_per_epoch'] == (t['prefixes_per_epoch'] + 2047) // 2048
    assert t['final_batch_size'] == t['prefixes_per_epoch'] - (t['batches_per_epoch'] - 1) * 2048 == 81
    return c


def read_bound(entry):
    assert set(entry) == {'path', 'sha256'} and sha(entry['path']) == entry['sha256']
    return json.loads(Path(entry['path']).read_text())


def predecessor():
    assert sha(PRIOR) == PRIOR_SHA
    old = json.loads(PRIOR.read_text())
    resource_contract.verify(old)
    evidence = {str(PRIOR): PRIOR_SHA}
    receipts, payloads = {}, {}
    for name, (filename, digest) in RECEIPTS.items():
        path = PRIOR.parent / filename
        assert sha(path) == digest
        receipts[name] = dict(path=str(path), sha256=digest)
        payloads[name] = read_bound(receipts[name])
        evidence[str(path)] = digest
    train, val, routes = (payloads[key] for key in ('training', 'validation', 'routes'))
    assert train['status'] == 'passed_resource_training' and train['job_id'] == '160306'
    assert train['manifest_sha256'] == PRIOR_SHA and train['source_git_commit'] == old['source_git_commit']
    assert train['exact_fresh_process_replay'] is True
    assert set(train['cases']) == {'512', '1024'} and all(x['status'] == 'passed' for x in train['cases'].values())
    assert all(train[k] is False for k in ('test_records_read', 'formal_training_ready', 'automatic_successor'))
    reports = {}
    for entry in train['reports']:
        reports[entry['path']] = read_bound(entry)
        evidence[entry['path']] = entry['sha256']
    initial_report = reports[str(PRIOR.parent / 'micro1024/continuous.json')]
    assert initial_report['status'] == 'passed' and initial_report['resumed'] is False
    assert initial_report['microbatch'] == 1024 and initial_report['logical_batch_size'] == 2048
    initial = initial_report['initial_state_sha256']
    assert re.fullmatch('[0-9a-f]{64}', initial)
    assert all(r['initial_state_sha256'] == initial for r in reports.values())
    assert val['status'] == 'passed_resource_validation' and val['job_id'] == '160308'
    assert val['manifest_sha256'] == PRIOR_SHA and (val['records'], val['cold_rows']) == (65, 23915712)
    assert all(val[k] is False for k in ('test_records_read', 'formal_training_ready', 'automatic_successor'))
    vr = read_bound(val['report'])
    evidence[val['report']['path']] = val['report']['sha256']
    assert vr['status'] == 'passed' and vr['records'] == 65
    assert vr['model_weights_unchanged'] is vr['gt_independence_verified'] is True
    assert vr['small_catalog_single_vs_two_gpu_passed'] is True
    assert vr['shard_sizes'] == [11957856, 11957856]
    assert routes['status'] == 'passed_cuda_route_coverage' and routes['job_id'] == '160392'
    assert routes['skipped'] == [] and len(routes['tests']) == 10
    assert routes['formal_training'] is routes['dataset_records_read'] is False
    # Route supplement has its own exact source binding and did not modify the
    # accepted capacity sources. Include those dependency sources as well.
    from scripts.probe_liger_capacity_routes import verify as verify_routes
    route_path = PRIOR.parent / 'routes_manifest.json'
    route = verify_routes(route_path)
    assert sha(route_path) == routes['manifest_sha256']
    assert route['source_git_commit'] == routes['source_git_commit']
    evidence[str(route_path)] = sha(route_path)
    sources = dict(old['execution_sources'])
    for path, digest in route['execution_sources'].items():
        assert path not in sources or sources[path] == digest
        sources[path] = digest
    return old, receipts, evidence, initial, sources


def checkpoint_binding(m):
    """Same purpose for train/validation; different purpose for gate/formal.

    Validation loads MODEL weights only from this binding, never restores a
    single-GPU optimizer/RNG state into a two-GPU process.
    """
    assert m['phase'] in (GATE_PHASE, FORMAL_PHASE)
    stats = read_bound({'path': m['standardization']['path'], 'sha256': m['standardization']['sha256']})
    return dict(purpose=m['phase'], formal_config_sha256=m['formal_config']['sha256'],
        runtime_config_sha256=m['config_sha256'], capacity_config_sha256=m['capacity_config']['sha256'],
        execution_sources=m['execution_sources'], input_metadata=m['input_metadata'],
        upstream_commit=m['upstream_commit'], assets={k:v['sha256'] for k,v in m['assets'].items()},
        catalog_sha256=m['catalog']['sha256'], index_artifacts=m['index_artifacts'],
        standardization_sha256=m['standardization']['sha256'],
        standardization_stats_sha256=stats['stats_sha256'],
        standardization_seen_rows_sha256=stats['seen_rows_sha256'],
        supervision_tag='bounded_last8_all_prefix_equal_weight_v1',
        feature_cache_sha256=m['expected_feature_cache_sha256'],
        expected_initial_state_sha256=m['expected_initial_state_sha256'],
        model=dict(d_model=384, d_ff=2560, num_layers=6, num_decoder_layers=6, num_heads=6, d_kv=64),
        logical_batch_size=2048, physical_batch_size=1024, accumulation_steps=2,
        dense_objective=m['dense_objective'],
        rng_order='one catalog projection before all query microbatches')


def verify_gate(entry, m):
    """Check the actual new handoff/replay reports, not just summary booleans."""
    r = read_bound(entry)
    assert r['status'] == 'passed_capacity384_training_gate' and r['phase'] == GATE_PHASE
    assert r['exact_fresh_process_replay'] is True and r['two_gpu_validation_records'] == 65
    assert r['validation_model_weights_unchanged'] is True
    assert all(r[k] is False for k in ('formal_training_ready', 'formal_checkpoint_produced',
                                     'test_records_read', 'automatic_successor'))
    gate = read_bound(r['gate_manifest'])
    verify(gate)
    assert gate['phase'] == GATE_PHASE and gate['gate_acceptance'] is None
    assert r['manifest_sha256'] == r['gate_manifest']['sha256']
    assert r['source_git_commit'] == gate['source_git_commit']
    assert r['initial_state_sha256'] == m['expected_initial_state_sha256']
    assert r['formal_config_sha256'] == m['formal_config']['sha256']
    assert r['execution_sources'] == m['execution_sources']
    assert isinstance(r['job_id'], str) and r['job_id']
    for key in set(m) - {'phase', 'formal_training_ready', 'source_git_commit', 'run_directory', 'gate_acceptance'}:
        assert gate[key] == m[key], f'Gate/formal binding differs: {key}'
    assert gate['run_directory'] != m['run_directory']
    expected_paths = {
        'train_report': 'continuous.json', 'resumed_report': 'resumed.json',
        'validation_report': 'validation/report.json',
        'validation_acceptance': 'validation/acceptance.json',
        'validation_gate_report': 'validation_gate.json',
    }
    expected_entries = []
    values = {}
    for key, filename in expected_paths.items():
        ref = r[key]
        assert Path(ref['path']).resolve() == Path(gate['run_directory']) / filename
        values[key] = read_bound(ref)
        expected_entries.append(ref)
    a, b = values['train_report'], values['resumed_report']
    from scripts.probe_liger_capacity import check_replay
    check_replay(a, b)
    assert a['initial_state_sha256'] == m['expected_initial_state_sha256']
    assert a['checkpoint_binding'] == checkpoint_binding(gate)
    assert len(a['steps']) == 3 and [s['attempt'] for s in a['steps']] == [1, 2, 3]
    assert a['final_progress']['updates'] == a['final_progress']['attempts'] == 3
    assert a['final_progress']['prefix_count'] == 6144 and a['final_progress']['skipped_updates'] == 0
    cp = a['validation_checkpoint']
    assert cp['attempts'] == cp['updates'] == cp['next_batch'] == 3
    assert cp['full_state_sha256'] == a['state_after_three']
    vr, va, wrapper = (values[k] for k in ('validation_report', 'validation_acceptance', 'validation_gate_report'))
    assert vr['status'] == va['status'] == wrapper['status'] == 'passed'
    assert vr['records'] == va['records'] == 65 and vr['cold_rows'] == 23915712
    assert vr['batches'] == 3 and vr['model_weights_unchanged'] is True
    assert vr['checkpoint'] == va['checkpoint'] == cp
    assert va['report'] == r['validation_report'] and va['metrics'] == vr['metrics']
    assert vr['gt_independence_verified'] is True and vr['test_records_read'] is False
    assert wrapper['checkpoint'] == cp
    assert wrapper['proof']['model_weights_unchanged'] is True
    assert wrapper['proof']['before_model_sha256'] == wrapper['proof']['after_model_sha256'] == vr['model_sha256']
    assert wrapper['proof']['load_scope'] == 'model_weights_only'
    assert wrapper['loaded']['checkpoint'] == cp
    assert wrapper['loaded']['model_state_sha256'] == vr['model_sha256']
    assert wrapper['loaded']['load_scope'] == 'model_weights_only'
    for key in ('validation_report', 'validation_acceptance'):
        assert wrapper[key] == r[key]
    for report in (a, b, wrapper):
        assert report['job_id'] == r['job_id'] and report['manifest_sha256'] == r['manifest_sha256']
        assert report['phase'] == GATE_PHASE and report['source_git_commit'] == gate['source_git_commit']
        assert all(report[k] is False for k in ('formal_training_ready', 'formal_checkpoint_produced',
                                              'test_records_read', 'automatic_successor'))
    expected_entries += vr['artifacts'] + [a['step_one_checkpoint'], cp]
    canonical = lambda values: sorted(json.dumps(x, sort_keys=True) for x in values)
    assert canonical(r['reports']) == canonical(expected_entries)
    for artifact in r['reports']:
        assert sha(artifact['path']) == artifact['sha256']
    return r


def verify(m):
    old, receipts, evidence, initial, inherited_sources = predecessor()
    assert set(m) == set(old) | ADDED
    assert m['phase'] in (GATE_PHASE, FORMAL_PHASE)
    assert m['formal_training_ready'] is (m['phase'] == FORMAL_PHASE)
    assert all(m[k] is False for k in ('automatic_successor', 'test_records_read', 'dirty_at_submission'))
    assert isinstance(m['source_git_commit'], str) and re.fullmatch('[0-9a-f]{40}', m['source_git_commit'])
    for key in set(old) - CHANGED: _exact(m[key], old[key], key)
    assert m['capacity_predecessor'] == dict(path=str(PRIOR), sha256=PRIOR_SHA)
    assert m['resource_acceptances'] == receipts
    assert m['expected_initial_state_sha256'] == initial
    c = check_config(json.loads(CONFIG.read_text()))
    assert m['formal_config'] == dict(path=str(CONFIG), sha256=CONFIG_SHA)
    assert m['initialization'] == c['initialization']
    expected_sources = dict(inherited_sources, **{str(ROOT / name):sha(ROOT / name) for name in SOURCES})
    assert m['execution_sources'] == expected_sources
    expected_metadata = dict(old['input_metadata'], **evidence, **{str(CONFIG): CONFIG_SHA})
    assert m['input_metadata'] == expected_metadata
    for group in ('execution_sources', 'input_metadata', 'upstream_files'): verify_hashes(m[group])
    # The inherited Runtime is intentionally unchanged; prove its consumed
    # stream/optimizer settings agree with the new explicit authorization.
    legacy = json.loads(Path(m['config']).read_text())
    assert legacy['optimizer'] == c['optimizer'] and legacy['dense_sampling'] == c['dense_sampling']
    for key, value in legacy['training'].items(): assert c['training'][key] == value
    assert legacy['validation']['batch_size'] == c['validation']['batch_size']
    run = Path(m['run_directory'])
    assert run.parent == ROOT / 'logs/train/runs'
    assert re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', run.name)
    if m['phase'] == GATE_PHASE:
        assert m['gate_acceptance'] is None
    else:
        assert isinstance(m['gate_acceptance'], dict)
        verify_gate(m['gate_acceptance'], m)
    return c


def prepare(path, phase='gate', gate_receipt=None):
    path = Path(path).resolve()
    assert phase in ('gate', 'formal') and (phase == 'formal') == (gate_receipt is not None)
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    assert os.environ.get('NUMPY_MADVISE_HUGEPAGE') == '0'
    old, receipts, evidence, initial, sources = predecessor()
    c = check_config(json.loads(CONFIG.read_text()))
    m = copy.deepcopy(old)
    m.update(phase=FORMAL_PHASE if phase == 'formal' else GATE_PHASE,
        formal_training_ready=phase == 'formal', initialization=c['initialization'],
        source_git_commit=git(ROOT, 'rev-parse', 'HEAD'),
        formal_config=dict(path=str(CONFIG), sha256=CONFIG_SHA),
        capacity_predecessor=dict(path=str(PRIOR), sha256=PRIOR_SHA),
        resource_acceptances=receipts, expected_initial_state_sha256=initial,
        execution_sources=dict(sources, **{str(ROOT / name):sha(ROOT / name) for name in SOURCES}),
        input_metadata=dict(old['input_metadata'], **evidence, **{str(CONFIG): CONFIG_SHA}),
        run_directory=str(ROOT / 'logs/train/runs' / path.parent.name), gate_acceptance=None)
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
    assert sha(path) == os.environ['LIGER_CAPACITY_TRAIN_MANIFEST_SHA256']
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
