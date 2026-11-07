"""A bound, full-only verification extension; frozen inference sources stay intact."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from scripts.prepare_inference_sensitivity import ASSET_ROOT, ROOT, record, sha, write_new
from scripts.prepare_retrieval_content import scope_fields
from scripts.compare_retrieval_paired_reports import compare_paired_reports

PARENT_PATH = ASSET_ROOT / 'logs/training_launch/retrieval_content_seed42_full_2026-11-07_v1/manifest.json'
PARENT_SHA = 'b3bdced637c5ef0744b7452896dade6b2f89a69003c76e75befab2a56c503322'
INFERENCE_COMMIT = '1c8fbe94b2d32d5f3d47e9aaf6c4f68140ae9579'
SMOKE_DIR = ASSET_ROOT / 'logs/analysis/retrieval_content_seed42_full_2026-11-07_v1/smoke'
SMOKE_ACCEPTANCE_SHA = 'b298bf9852ae7b5cffe6b69aca4c3ab26b3860ff31f3a2ebe6fbfec3b89e7638'
SMOKE_PAIRED_SHA = 'e4a693b079e87a890512caa2d6ef5096f4e60956cbd2ef0fabe76d1850ad69ab'
SOURCES = ['scripts/compare_retrieval_paired_reports.py', 'scripts/retrieval_content_verified.py',
           'tests/test_retrieval_paired_report_comparison.py',
           'tests/test_retrieval_content_verified.py', 'run_retrieval_content_verified_cpu4.sh']
RULE = {'absolute_tolerance': 1e-12, 'relative_tolerance': 0.,
        'scope': 'Allowlisted finite statistical floats only; structure, types, identity, counts, hashes and settings exact'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _parent():
    record(PARENT_PATH, PARENT_SHA)
    m = json.loads(PARENT_PATH.read_text())
    require(m['source_git_commit'] == INFERENCE_COMMIT, 'Wrong frozen inference commit')
    require(all(m.get(k) == v for k, v in scope_fields().items()), 'Parent scientific scope changed')
    for entry in m['execution_sources'] + m['input_metadata']:
        record(entry['path'], entry['sha256'])
    m['_manifest_sha256'] = PARENT_SHA
    return m


def prepare(path, output_dir):
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Commit extension sources before freezing')
    require(not path.exists() and not output_dir.exists(), 'Refusing existing manifest/output')
    _parent()
    manifest = {
        'schema_version': 1, 'experiment': 'retrieval_content_gate_extension_v1',
        'execution_root': str(ROOT), 'inference_source_commit': INFERENCE_COMMIT,
        'extension_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'parent_manifest': record(PARENT_PATH, PARENT_SHA), 'smoke_dir': str(SMOKE_DIR),
        'smoke_acceptance': record(SMOKE_DIR / 'acceptance.json', SMOKE_ACCEPTANCE_SHA),
        'smoke_paired': record(SMOKE_DIR / 'paired.json', SMOKE_PAIRED_SHA),
        'full_output_dir': str(output_dir),
        'prelaunch_acceptance_path': str(path.parent / 'smoke_reacceptance.json'),
        'runtime_acceptance_path': str(path.parent / 'runtime_gate.json'),
        'comparison_rule': RULE, 'extension_sources': [record(ROOT / p) for p in SOURCES],
        'mode': 'Scoped in-process replacement of require_gate only; parent files and all prediction/statistics algorithms unchanged',
        'resources': {'gpus': 1, 'gpu_type': '4090', 'cpus': 4, 'memory_gib': 24},
        'scope': 'Reuse accepted 65-row GPU smoke; seed42 five-policy full test only; no retraining or parameter selection',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, manifest)
    print(json.dumps({'path': str(path), 'sha256': sha(path),
                      'extension_source_commit': manifest['extension_source_commit'],
                      'inference_source_commit': INFERENCE_COMMIT}))


def load_extension(path):
    path = Path(path).resolve()
    digest = sha(path)
    require(digest == os.environ.get('CONTENT_EXTENSION_MANIFEST_SHA256'), 'Extension manifest hash mismatch')
    e = json.loads(path.read_text())
    require(e.get('schema_version') == 1 and e.get('experiment') == 'retrieval_content_gate_extension_v1',
            'Wrong extension schema')
    require(e.get('execution_root') == str(ROOT) and e.get('inference_source_commit') == INFERENCE_COMMIT,
            'Wrong execution root/inference source')
    require(e['extension_source_commit'] == os.environ.get('CONTENT_EXTENSION_GIT_COMMIT'),
            'Extension source commit mismatch')
    require(os.environ.get('SOURCE_GIT_COMMIT') == INFERENCE_COMMIT
            and os.environ.get('CONTENT_MANIFEST_SHA256') == PARENT_SHA, 'Parent environment binding mismatch')
    require(e.get('parent_manifest') == {'path': str(PARENT_PATH), 'sha256': PARENT_SHA}, 'Parent binding mismatch')
    require(e.get('smoke_dir') == str(SMOKE_DIR) and e.get('comparison_rule') == RULE, 'Smoke/rule scope changed')
    require(e.get('resources') == {'gpus': 1, 'gpu_type': '4090', 'cpus': 4, 'memory_gib': 24},
            'Resource scope changed')
    require(e.get('prelaunch_acceptance_path') == str(path.parent / 'smoke_reacceptance.json')
            and e.get('runtime_acceptance_path') == str(path.parent / 'runtime_gate.json'), 'Receipt paths changed')
    output = Path(e['full_output_dir'])
    require(output.is_absolute() and output.resolve() == output
            and output.is_relative_to(ASSET_ROOT / 'logs/analysis') and output.name == 'full',
            'Invalid full output path')
    expected_sources = {str((ROOT / p).resolve()) for p in SOURCES}
    require(len(e['extension_sources']) == len(expected_sources)
            and {r['path'] for r in e['extension_sources']} == expected_sources, 'Extension source list changed')
    for entry in e['extension_sources']:
        record(entry['path'], entry['sha256'])
    for key, expected_sha, filename in (('smoke_acceptance', SMOKE_ACCEPTANCE_SHA, 'acceptance.json'),
                                       ('smoke_paired', SMOKE_PAIRED_SHA, 'paired.json')):
        require(e.get(key) == {'path': str(SMOKE_DIR / filename), 'sha256': expected_sha}, 'Smoke receipt binding changed')
        record(e[key]['path'], expected_sha)
    _parent()
    e['_path'], e['_sha256'] = str(path), digest
    return e


def verify_smoke(e):
    from scripts.audit_retrieval_content import audit
    from scripts.analyze_retrieval_content import analyze
    m = _parent()
    stored = json.loads((SMOKE_DIR / 'acceptance.json').read_text())
    record(SMOKE_DIR / 'acceptance.json', SMOKE_ACCEPTANCE_SHA)
    fresh = audit(SMOKE_DIR, m, 'smoke')
    require({k: v for k, v in stored.items() if k != 'checked_at'}
            == {k: v for k, v in fresh.items() if k != 'checked_at'}, 'Smoke artifact acceptance changed')
    paired = json.loads((SMOKE_DIR / 'paired.json').read_text())
    record(SMOKE_DIR / 'paired.json', SMOKE_PAIRED_SHA)
    fresh_paired = analyze(SMOKE_DIR, m, 'smoke')
    comparison = compare_paired_reports(paired, fresh_paired)

    def scientific_sha(report):
        content = {k: v for k, v in report.items() if k not in ('checked_at', 'analysis_cpu_affinity')}
        return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode('utf-8')).hexdigest()

    return {
        'status': 'passed', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'extension_manifest': {'path': e['_path'], 'sha256': e['_sha256']},
        'extension_source_commit': e['extension_source_commit'], 'inference_source_commit': INFERENCE_COMMIT,
        'parent_manifest': e['parent_manifest'], 'smoke_acceptance': e['smoke_acceptance'],
        'smoke_paired': e['smoke_paired'], 'smoke_run': record(SMOKE_DIR / 'run.json'),
        'comparison_rule': RULE, 'paired_comparison': comparison,
        'paired_recorded_scientific_sha256': scientific_sha(paired),
        'paired_recomputed_scientific_sha256': scientific_sha(fresh_paired),
        'num_examples': 65, 'policies': m['policies'], 'artifact_acceptance_exact': True,
        'parent_execution_sources_verified': True, 'model_inference_rerun': False,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
    }


def _check_proof(proof, e):
    require(proof.get('status') == 'passed' and proof.get('num_examples') == 65
            and proof.get('artifact_acceptance_exact') is True
            and proof.get('parent_execution_sources_verified') is True
            and proof.get('model_inference_rerun') is False, 'Invalid verification proof')
    for key in ('extension_source_commit', 'inference_source_commit', 'parent_manifest',
                'smoke_acceptance', 'smoke_paired', 'comparison_rule'):
        require(proof.get(key) == e[key], f'Verification binding mismatch: {key}')
    require(proof.get('extension_manifest') == {'path': e['_path'], 'sha256': e['_sha256']},
            'Verification extension identity mismatch')
    require(proof.get('smoke_run') == record(SMOKE_DIR / 'run.json'), 'Verification smoke run changed')
    require(proof.get('policies') == scope_fields()['policies'], 'Verification policy mismatch')
    comparison = proof.get('paired_comparison', {})
    require(comparison.get('status') == 'passed'
            and comparison.get('tolerance') == {'abs_tol': 1e-12, 'rel_tol': 0.}
            and comparison.get('ignored_runtime_fields') == ['checked_at', 'analysis_cpu_affinity'],
            'Missing or changed paired numeric verification')
    differences = comparison.get('tolerated_differences')
    require(type(differences) is list
            and comparison.get('exact_equal_excluding_runtime_fields') is (not bool(differences)),
            'Inconsistent numeric difference summary')
    for difference in differences:
        require(type(difference) is dict
                and set(difference) == {'path', 'recorded', 'fresh', 'absolute_difference'}
                and type(difference['path']) is str, 'Malformed numeric difference')
        require(all(type(difference[k]) is float and math.isfinite(difference[k])
                    for k in ('recorded', 'fresh', 'absolute_difference'))
                and difference['absolute_difference'] == abs(difference['recorded'] - difference['fresh'])
                and 0 < difference['absolute_difference'] <= 1e-12, 'Invalid tolerated difference')
    maximum = comparison.get('max_abs_difference')
    require(type(maximum) is float and math.isfinite(maximum)
            and maximum == max((d['absolute_difference'] for d in differences), default=0.0),
            'Invalid maximum numerical difference')


def verify_and_write(e):
    destination = Path(e['prelaunch_acceptance_path'])
    require(not destination.exists(), 'Prelaunch verification already exists')
    proof = verify_smoke(e)
    _check_proof(proof, e)
    write_new(destination, proof)
    print(json.dumps({'status': proof['status'], 'receipt': record(destination),
                      'paired_comparison': proof['paired_comparison']}))


@contextmanager
def patched_gate(runner, e, prelaunch_record):
    previous = runner.require_gate

    def checked_gate(m, directory):
        require(m.get('_manifest_sha256') == PARENT_SHA and m.get('source_git_commit') == INFERENCE_COMMIT
                and Path(directory).resolve() == SMOKE_DIR, 'Runtime gate inputs changed')
        destination = Path(e['runtime_acceptance_path'])
        require(not destination.exists(), 'Runtime verification already exists')
        require(record(prelaunch_record['path']) == prelaunch_record, 'Prelaunch proof changed before runtime gate')
        job_id = os.environ.get('SLURM_JOB_ID')
        require(type(job_id) is str and job_id.isascii() and job_id.isdigit(),
                'Runtime gate requires a real Slurm job ID')
        proof = verify_smoke(e)  # Repeat on the actual allocated CPU; do not require identical rounding differences.
        _check_proof(proof, e)
        proof['prelaunch_acceptance'] = prelaunch_record
        proof['job_id'] = job_id
        write_new(destination, proof)
        smoke_run = json.loads((SMOKE_DIR / 'run.json').read_text())
        return {'path': str(SMOKE_DIR / 'acceptance.json'), 'sha256': SMOKE_ACCEPTANCE_SHA,
                'paired_analysis': {'path': str(SMOKE_DIR / 'paired.json'), 'sha256': SMOKE_PAIRED_SHA},
                'verification': record(destination)}, smoke_run['smoke_gate']

    runner.require_gate = checked_gate
    try:
        yield
    finally:
        runner.require_gate = previous


def run_full(e):
    from scripts import run_retrieval_content as runner
    prelaunch = Path(e['prelaunch_acceptance_path'])
    _check_proof(json.loads(prelaunch.read_text()), e)
    require(not Path(e['full_output_dir']).exists(), 'Full output already exists')
    args = SimpleNamespace(manifest=PARENT_PATH, output_dir=Path(e['full_output_dir']),
                           phase='full', smoke_dir=SMOKE_DIR)
    with patched_gate(runner, e, record(prelaunch)):
        runner.run(args)


def accept_full(e):
    """Add explicit extension lineage to the unchanged export/statistical acceptance."""
    output = Path(e['full_output_dir'])
    destination = output / 'lineage_acceptance.json'
    require(not destination.exists(), 'Full lineage acceptance already exists')
    run_path, artifact_path, paired_path = (output / p for p in ('run.json', 'acceptance.json', 'paired.json'))
    run, artifact, paired = (json.loads(p.read_text()) for p in (run_path, artifact_path, paired_path))
    m = _parent()
    for report in (run, artifact, paired):
        require(report.get('seed') == 42 and report.get('phase') == 'full'
                and report.get('source_git_commit') == INFERENCE_COMMIT
                and report.get('manifest_sha256') == PARENT_SHA and report.get('policies') == m['policies'],
                'Full scientific/source identity mismatch')
    require(run.get('status') == 'complete' and run.get('processed') == 2061101
            and run.get('training_performed') is False, 'Incomplete full export')
    for report in (artifact, paired):
        require(report.get('status') == 'passed' and report.get('num_examples') == 2061101
                and report.get('run') == record(run_path), 'Full acceptance/run mismatch')
    require(paired.get('acceptance') == record(artifact_path)
            and paired.get('artifacts') == artifact.get('artifacts')
            and paired.get('statistics') == m['statistics']
            and paired.get('all_point_estimates_match_acceptance') is True, 'Full statistics binding mismatch')
    prelaunch = record(e['prelaunch_acceptance_path'])
    runtime = record(e['runtime_acceptance_path'])
    proof = json.loads(Path(runtime['path']).read_text())
    _check_proof(proof, e)
    job_id = run.get('job_id')
    require(type(job_id) is str and job_id.isascii() and job_id.isdigit(),
            'Full export requires a real Slurm job ID')
    require(proof.get('prelaunch_acceptance') == prelaunch and proof.get('job_id') == job_id,
            'Runtime verification provenance mismatch')
    prior = run.get('prior_smoke_acceptance', {})
    require(prior == {'path': str(SMOKE_DIR / 'acceptance.json'), 'sha256': SMOKE_ACCEPTANCE_SHA,
                     'paired_analysis': {'path': str(SMOKE_DIR / 'paired.json'), 'sha256': SMOKE_PAIRED_SHA},
                     'verification': runtime}, 'Full export omitted or changed runtime verification')
    require(artifact.get('prior_smoke_acceptance') == prior, 'Artifact acceptance omitted runtime verification')
    write_new(destination, {'status': 'passed', 'num_examples': 2061101,
                           'extension_manifest': {'path': e['_path'], 'sha256': e['_sha256']},
                           'extension_source_commit': e['extension_source_commit'],
                           'inference_source_commit': INFERENCE_COMMIT,
                           'prelaunch_acceptance': prelaunch, 'runtime_acceptance': runtime,
                           'artifact_acceptance': record(artifact_path), 'paired_acceptance': record(paired_path),
                           'run': record(run_path), 'comparison_rule': RULE,
                           'checked_at': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'status': 'passed', 'phase': 'full', 'lineage': record(destination)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--extension', type=Path, required=True)
    prep.add_argument('--output-dir', type=Path, required=True)
    for action in ('verify', 'run', 'accept-full'):
        sub.add_parser(action).add_argument('--extension', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args.extension.resolve(), args.output_dir.resolve())
    else:
        e = load_extension(args.extension)
        {'verify': verify_and_write, 'run': run_full, 'accept-full': accept_full}[args.command](e)


if __name__ == '__main__':
    main()
