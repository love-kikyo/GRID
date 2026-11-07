"""Exercise the full-only verification extension using small, bound local files."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_retrieval_content as parent
from scripts import retrieval_content_verified as verified
from scripts.compare_retrieval_paired_reports import COMPARISONS, METRICS, POLICIES, VARIANTS


def put(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
    return verified.record(path)


def paired_fixture():
    metric = dict(left_mean=.3, right_mean=.2, delta=.1, relative_gain=.5,
                  paired_bootstrap_ci=[.09, .11], confidence=.95,
                  nonzero_user_deltas=17, role='secondary_exploratory')
    return dict(
        status='passed', checked_at='old', analysis_cpu_affinity=[1],
        point_estimates={p: {v: {m: .25 for m in METRICS} for v in VARIANTS} for p in POLICIES},
        comparisons={c: dict(left=['sim10', 'fused'], right=['recent10', 'fused'],
                             direction='left_minus_right',
                             metrics={m: copy.deepcopy(metric) for m in METRICS}) for c in COMPARISONS})


@pytest.fixture
def bound(tmp_path, monkeypatch):
    root, assets = tmp_path / 'sources', tmp_path / 'assets'
    smoke = assets / 'logs/analysis/old/smoke'
    parent_path = assets / 'logs/training_launch/old/manifest.json'
    for module in (verified, parent):
        monkeypatch.setattr(module, 'ROOT', root)
        monkeypatch.setattr(module, 'ASSET_ROOT', assets)
    monkeypatch.setattr(verified, 'SMOKE_DIR', smoke)
    monkeypatch.setattr(verified, 'PARENT_PATH', parent_path)
    for relative in verified.SOURCES:
        put(root / relative, {'source': relative})
    frozen_source = put(root / 'frozen.py', {'unchanged': 'inference'})
    metadata = put(assets / 'small-metadata.json', {'protocol': 'fixture'})
    m = dict(parent.scope_fields(), source_git_commit=verified.INFERENCE_COMMIT,
             execution_sources=[frozen_source], input_metadata=[metadata])
    monkeypatch.setattr(verified, 'PARENT_SHA', put(parent_path, m)['sha256'])
    acceptance = dict(status='passed', phase='smoke', num_examples=65, checked_at='old')
    monkeypatch.setattr(verified, 'SMOKE_ACCEPTANCE_SHA',
                        put(smoke / 'acceptance.json', acceptance)['sha256'])
    paired = paired_fixture()
    monkeypatch.setattr(verified, 'SMOKE_PAIRED_SHA', put(smoke / 'paired.json', paired)['sha256'])
    put(smoke / 'run.json', {'smoke_gate': {'status': 'passed', 'checks': ['original']}})

    def git_output(command, **kwargs):
        assert kwargs == {'cwd': root, 'text': True}
        return {('git', 'status', '--porcelain'): '',
                ('git', 'rev-parse', 'HEAD'): 'extension-commit\n'}[tuple(command)]

    monkeypatch.setattr(verified, 'subprocess', SimpleNamespace(check_output=git_output))
    manifest = assets / 'logs/training_launch/extension/manifest.json'
    output = assets / 'logs/analysis/extension/full'
    verified.prepare(manifest, output)
    monkeypatch.setenv('CONTENT_EXTENSION_MANIFEST_SHA256', verified.sha(manifest))
    monkeypatch.setenv('CONTENT_EXTENSION_GIT_COMMIT', 'extension-commit')
    monkeypatch.setenv('SOURCE_GIT_COMMIT', verified.INFERENCE_COMMIT)
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', verified.PARENT_SHA)
    monkeypatch.setenv('SLURM_JOB_ID', '12345')
    e = verified.load_extension(manifest)
    return SimpleNamespace(e=e, manifest=manifest, output=output, m=m,
                           acceptance=acceptance, paired=paired, monkeypatch=monkeypatch)


def proof(bound):
    e = bound.e
    result = {k: copy.deepcopy(e[k]) for k in (
        'extension_source_commit', 'inference_source_commit', 'parent_manifest',
        'smoke_acceptance', 'smoke_paired', 'comparison_rule')}
    result.update(status='passed', num_examples=65, artifact_acceptance_exact=True,
                  parent_execution_sources_verified=True, model_inference_rerun=False,
                  extension_manifest={'path': e['_path'], 'sha256': e['_sha256']},
                  smoke_run=verified.record(verified.SMOKE_DIR / 'run.json'),
                  policies=list(bound.m['policies']),
                  paired_comparison=verified.compare_paired_reports(bound.paired, bound.paired))
    return result


def prelaunch(bound):
    return put(bound.e['prelaunch_acceptance_path'], proof(bound))


def refresh_manifest(bound, mutation):
    document = json.loads(bound.manifest.read_text())
    mutation(document)
    put(bound.manifest, document)
    bound.monkeypatch.setenv('CONTENT_EXTENSION_MANIFEST_SHA256', verified.sha(bound.manifest))


def test_prepare_and_load_bind_both_source_generations_without_creating_output(bound):
    e = bound.e
    assert e['parent_manifest'] == verified.record(verified.PARENT_PATH)
    assert e['inference_source_commit'] == verified.INFERENCE_COMMIT
    assert e['extension_source_commit'] == 'extension-commit'
    assert e['_sha256'] == verified.sha(bound.manifest)
    assert {Path(s['path']).relative_to(verified.ROOT).as_posix()
            for s in e['extension_sources']} == set(verified.SOURCES)
    assert not bound.output.exists()


@pytest.mark.parametrize('field,value', [
    ('parent_manifest', {'path': '/wrong/manifest.json', 'sha256': 'wrong'}),
    ('execution_root', '/wrong/root'), ('inference_source_commit', 'wrong'),
    ('extension_source_commit', 'wrong'), ('smoke_dir', '/wrong/smoke'),
    ('comparison_rule', {'absolute_tolerance': 1.}),
    ('resources', {'gpus': 3, 'gpu_type': '4090', 'cpus': 4, 'memory_gib': 24}),
    ('prelaunch_acceptance_path', '/wrong/proof.json'),
    ('runtime_acceptance_path', '/wrong/runtime.json'),
    ('full_output_dir', 'relative/full'), ('full_output_dir', '/tmp/outside/full'),
    ('smoke_acceptance', {'path': '/wrong/acceptance.json', 'sha256': 'wrong'}),
    ('smoke_paired', {'path': '/wrong/paired.json', 'sha256': 'wrong'}),
])
def test_manifest_scope_mutations_rejected_even_when_manifest_hash_is_rebound(bound, field, value):
    refresh_manifest(bound, lambda m: m.update({field: value}))
    with pytest.raises(ValueError):
        verified.load_extension(bound.manifest)


@pytest.mark.parametrize('mutation', [
    lambda m: m['extension_sources'].pop(),
    lambda m: m['extension_sources'].append(m['extension_sources'][0]),
    lambda m: m['extension_sources'].__setitem__(0, m['extension_sources'][1]),
])
def test_source_list_must_be_exact_and_unique(bound, mutation):
    refresh_manifest(bound, mutation)
    with pytest.raises(ValueError, match='source list'):
        verified.load_extension(bound.manifest)


@pytest.mark.parametrize('which', ['extension', 'frozen', 'metadata', 'parent', 'acceptance', 'paired'])
def test_actual_bound_file_hash_changes_fail_closed(bound, which):
    path = {'extension': Path(bound.e['extension_sources'][0]['path']),
            'frozen': Path(bound.m['execution_sources'][0]['path']),
            'metadata': Path(bound.m['input_metadata'][0]['path']),
            'parent': verified.PARENT_PATH,
            'acceptance': verified.SMOKE_DIR / 'acceptance.json',
            'paired': verified.SMOKE_DIR / 'paired.json'}[which]
    path.write_text(path.read_text() + '\nchanged')
    with pytest.raises(ValueError, match='Changed accepted source'):
        verified.load_extension(bound.manifest)


@pytest.mark.parametrize('variable', ['CONTENT_EXTENSION_MANIFEST_SHA256', 'CONTENT_EXTENSION_GIT_COMMIT',
                                     'SOURCE_GIT_COMMIT', 'CONTENT_MANIFEST_SHA256'])
def test_environment_bindings_required(bound, variable):
    bound.monkeypatch.delenv(variable)
    with pytest.raises(ValueError):
        verified.load_extension(bound.manifest)


def test_prepare_refuses_dirty_sources_and_existing_manifest(bound):
    original = bound.manifest.read_bytes()
    with pytest.raises(ValueError, match='existing manifest/output'):
        verified.prepare(bound.manifest, bound.output)
    bound.monkeypatch.setattr(verified.subprocess, 'check_output', lambda *a, **k: ' M source.py')
    with pytest.raises(ValueError, match='Commit extension sources'):
        verified.prepare(bound.manifest.parent / 'new.json', bound.output)
    assert bound.manifest.read_bytes() == original


@pytest.mark.parametrize('field,value', [('source_git_commit', 'new-inference'), ('seed', 43)])
def test_parent_rejects_scientific_changes_even_with_updated_parent_digest(bound, field, value):
    document = copy.deepcopy(bound.m)
    document[field] = value
    bound.monkeypatch.setattr(verified, 'PARENT_SHA', put(verified.PARENT_PATH, document)['sha256'])
    with pytest.raises(ValueError):
        verified._parent()


def test_verify_smoke_recomputes_audit_and_analysis_without_model_rerun(bound):
    from scripts import analyze_retrieval_content, audit_retrieval_content
    calls = []

    def audit(directory, m, phase):
        calls.append(('audit', directory, phase, m['_manifest_sha256']))
        return dict(bound.acceptance, checked_at='new')

    def analyze(directory, m, phase):
        calls.append(('analyze', directory, phase, m['_manifest_sha256']))
        return dict(bound.paired, checked_at='new', analysis_cpu_affinity=[2])

    bound.monkeypatch.setattr(audit_retrieval_content, 'audit', audit)
    bound.monkeypatch.setattr(analyze_retrieval_content, 'analyze', analyze)
    result = verified.verify_smoke(bound.e)
    verified._check_proof(result, bound.e)
    assert calls == [(name, verified.SMOKE_DIR, 'smoke', verified.PARENT_SHA)
                     for name in ('audit', 'analyze')]
    assert result['paired_comparison']['status'] == 'passed'
    assert result['model_inference_rerun'] is False
    assert result['paired_recorded_scientific_sha256'] == result['paired_recomputed_scientific_sha256']
    assert len(result['paired_recorded_scientific_sha256']) == 64
    bound.monkeypatch.setattr(audit_retrieval_content, 'audit',
                             lambda *a: dict(bound.acceptance, num_examples=66))
    with pytest.raises(ValueError, match='artifact acceptance changed'):
        verified.verify_smoke(bound.e)


@pytest.mark.parametrize('field,value', [
    ('status', 'failed'), ('num_examples', 64), ('artifact_acceptance_exact', 1),
    ('parent_execution_sources_verified', False), ('model_inference_rerun', True),
    ('extension_source_commit', 'other'), ('inference_source_commit', 'other'),
    ('extension_manifest', {}), ('parent_manifest', {}), ('smoke_acceptance', {}),
    ('smoke_paired', {}), ('smoke_run', {}), ('comparison_rule', {}),
    ('policies', ['sim10']), ('paired_comparison', {}),
])
def test_proof_cannot_change_identity_or_claim_skipped_checks(bound, field, value):
    candidate = proof(bound)
    candidate[field] = value
    with pytest.raises(ValueError):
        verified._check_proof(candidate, bound.e)


@pytest.mark.parametrize('field,value', [
    ('tolerance', {'abs_tol': 1., 'rel_tol': 1.}),
    ('ignored_runtime_fields', ['statistics']),
    ('exact_equal_excluding_runtime_fields', False),
    ('max_abs_difference', 1e-13), ('max_abs_difference', float('nan')),
    ('tolerated_differences', [{'path': '/untrusted', 'recorded': 0., 'fresh': 1.,
                              'absolute_difference': 1e-13}]),
])
def test_numeric_receipt_cannot_relax_rule_or_misstate_its_comparison(bound, field, value):
    candidate = proof(bound)
    candidate['paired_comparison'][field] = value
    with pytest.raises(ValueError):
        verified._check_proof(candidate, bound.e)


def test_prelaunch_receipt_exclusive_write(bound):
    bound.monkeypatch.setattr(verified, 'verify_smoke', lambda e: proof(bound))
    verified.verify_and_write(bound.e)
    destination = Path(bound.e['prelaunch_acceptance_path'])
    original = destination.read_bytes()
    verified._check_proof(json.loads(original), bound.e)
    with pytest.raises(ValueError, match='already exists'):
        verified.verify_and_write(bound.e)
    assert destination.read_bytes() == original


def test_patched_gate_preserves_old_smoke_identity_adds_runtime_receipt_and_restores(bound):
    receipt = prelaunch(bound)
    original_gate = lambda *a: None
    runner = SimpleNamespace(require_gate=original_gate)
    bound.monkeypatch.setattr(verified, 'verify_smoke', lambda e: proof(bound))
    with verified.patched_gate(runner, bound.e, receipt):
        prior, smoke_gate = runner.require_gate(verified._parent(), verified.SMOKE_DIR)
        assert prior['path'] == str(verified.SMOKE_DIR / 'acceptance.json')
        assert prior['sha256'] == verified.SMOKE_ACCEPTANCE_SHA
        assert prior['paired_analysis'] == bound.e['smoke_paired']
        assert prior['verification'] == verified.record(bound.e['runtime_acceptance_path'])
        assert smoke_gate == {'status': 'passed', 'checks': ['original']}
        runtime = json.loads(Path(bound.e['runtime_acceptance_path']).read_text())
        assert runtime['prelaunch_acceptance'] == receipt and runtime['job_id'] == '12345'
        with pytest.raises(ValueError, match='Runtime verification already exists'):
            runner.require_gate(verified._parent(), verified.SMOKE_DIR)
    assert runner.require_gate is original_gate


@pytest.mark.parametrize('kind', ['parent_hash', 'source_commit', 'smoke_dir', 'prelaunch_hash'])
def test_runtime_gate_rejects_changed_inputs_and_restores_after_exception(bound, kind):
    receipt = prelaunch(bound)
    original_gate = lambda *a: None
    runner = SimpleNamespace(require_gate=original_gate)
    m, directory = verified._parent(), verified.SMOKE_DIR
    if kind == 'parent_hash':
        m['_manifest_sha256'] = 'other'
    elif kind == 'source_commit':
        m['source_git_commit'] = 'other'
    elif kind == 'smoke_dir':
        directory = verified.SMOKE_DIR.parent / 'other-smoke'
    else:
        Path(receipt['path']).write_text('{}')
    with pytest.raises(ValueError):
        with verified.patched_gate(runner, bound.e, receipt):
            runner.require_gate(m, directory)
    assert runner.require_gate is original_gate
    assert not Path(bound.e['runtime_acceptance_path']).exists()


def test_runtime_gate_requires_slurm_job_id_and_restores_on_missing_identity(bound):
    receipt = prelaunch(bound)
    original_gate = lambda *a: None
    runner = SimpleNamespace(require_gate=original_gate)
    bound.monkeypatch.delenv('SLURM_JOB_ID')
    bound.monkeypatch.setattr(verified, 'verify_smoke', lambda e: proof(bound))
    with pytest.raises(ValueError):
        with verified.patched_gate(runner, bound.e, receipt):
            runner.require_gate(verified._parent(), verified.SMOKE_DIR)
    assert runner.require_gate is original_gate
    assert not Path(bound.e['runtime_acceptance_path']).exists()


@pytest.mark.parametrize('raise_during_run', [False, True])
def test_run_full_is_full_only_and_restores_gate_even_if_runner_fails(bound, raise_during_run):
    from scripts import run_retrieval_content as runner
    prelaunch(bound)
    original_gate = runner.require_gate
    calls = []
    bound.monkeypatch.setattr(verified, 'verify_smoke', lambda e: proof(bound))

    def run(args):
        calls.append(args)
        assert runner.require_gate is not original_gate
        runner.require_gate(verified._parent(), args.smoke_dir)
        if raise_during_run:
            raise RuntimeError('runner failed')

    bound.monkeypatch.setattr(runner, 'run', run)
    if raise_during_run:
        with pytest.raises(RuntimeError, match='runner failed'):
            verified.run_full(bound.e)
    else:
        verified.run_full(bound.e)
    assert len(calls) == 1
    assert vars(calls[0]) == dict(manifest=verified.PARENT_PATH, output_dir=bound.output,
                                phase='full', smoke_dir=verified.SMOKE_DIR)
    assert runner.require_gate is original_gate


def test_run_full_refuses_existing_output_before_calling_runner(bound):
    from scripts import run_retrieval_content as runner
    prelaunch(bound)
    bound.output.mkdir(parents=True)
    bound.monkeypatch.setattr(runner, 'run', lambda args: pytest.fail('Existing output must not be touched'))
    with pytest.raises(ValueError, match='Full output already exists'):
        verified.run_full(bound.e)


def full_reports(bound):
    prelaunch_record = prelaunch(bound)
    runtime = proof(bound)
    runtime.update(prelaunch_acceptance=prelaunch_record, job_id='12345')
    runtime_record = put(bound.e['runtime_acceptance_path'], runtime)
    identity = dict(seed=42, phase='full', source_git_commit=verified.INFERENCE_COMMIT,
                    manifest_sha256=verified.PARENT_SHA, policies=list(bound.m['policies']))
    prior = dict(bound.e['smoke_acceptance'], paired_analysis=bound.e['smoke_paired'],
                 verification=runtime_record)
    run = dict(identity, status='complete', processed=2061101, training_performed=False,
               job_id='12345', prior_smoke_acceptance=prior)
    artifact = dict(identity, status='passed', num_examples=2061101,
                    prior_smoke_acceptance=copy.deepcopy(prior), artifacts=[{'fixture': 'small'}])
    paired = dict(identity, status='passed', num_examples=2061101,
                  statistics=bound.m['statistics'], artifacts=copy.deepcopy(artifact['artifacts']),
                  all_point_estimates_match_acceptance=True)
    reports = dict(run=run, acceptance=artifact, paired=paired, runtime=runtime)
    write_full_reports(bound, reports)
    return reports


def write_full_reports(bound, reports):
    run_record = put(bound.output / 'run.json', reports['run'])
    reports['acceptance']['run'] = run_record
    acceptance_record = put(bound.output / 'acceptance.json', reports['acceptance'])
    reports['paired'].update(run=run_record, acceptance=acceptance_record)
    put(bound.output / 'paired.json', reports['paired'])


def test_full_acceptance_records_all_lineage_and_never_overwrites(bound):
    full_reports(bound)
    verified.accept_full(bound.e)
    destination = bound.output / 'lineage_acceptance.json'
    original = destination.read_bytes()
    lineage = json.loads(original)
    assert lineage['status'] == 'passed' and lineage['num_examples'] == 2061101
    assert lineage['extension_manifest'] == verified.record(bound.manifest)
    assert lineage['extension_source_commit'] == 'extension-commit'
    assert lineage['inference_source_commit'] == verified.INFERENCE_COMMIT
    for field, path in [('prelaunch_acceptance', bound.e['prelaunch_acceptance_path']),
                        ('runtime_acceptance', bound.e['runtime_acceptance_path']),
                        ('artifact_acceptance', bound.output / 'acceptance.json'),
                        ('paired_acceptance', bound.output / 'paired.json'),
                        ('run', bound.output / 'run.json')]:
        assert lineage[field] == verified.record(path)
    with pytest.raises(ValueError, match='already exists'):
        verified.accept_full(bound.e)
    assert destination.read_bytes() == original


@pytest.mark.parametrize('mutation', [
    lambda r: r['run'].pop('prior_smoke_acceptance'),
    lambda r: r['run']['prior_smoke_acceptance'].pop('verification'),
    lambda r: r['run']['prior_smoke_acceptance']['verification'].update(sha256='wrong'),
    lambda r: r['run'].update(job_id='different-job'),
    lambda r: r['run'].update(processed=65),
    lambda r: r['run'].update(training_performed=True),
    lambda r: r['acceptance'].pop('prior_smoke_acceptance'),
    lambda r: r['acceptance'].update(source_git_commit='extension-not-inference'),
    lambda r: r['paired'].update(manifest_sha256='extension-not-parent'),
    lambda r: r['paired'].update(statistics={}),
    lambda r: r['paired'].update(artifacts=[]),
    lambda r: r['paired'].update(all_point_estimates_match_acceptance=False),
])
def test_full_acceptance_rejects_wrong_lineage_even_with_fresh_report_hashes(bound, mutation):
    reports = full_reports(bound)
    mutation(reports)
    write_full_reports(bound, reports)
    with pytest.raises(ValueError):
        verified.accept_full(bound.e)
    assert not (bound.output / 'lineage_acceptance.json').exists()


@pytest.mark.parametrize('field,value', [('extension_manifest', {}), ('extension_source_commit', 'other'),
                                        ('prelaunch_acceptance', {}), ('job_id', 'different-job')])
def test_full_acceptance_rejects_bad_runtime_proof_even_when_receipt_rebound(bound, field, value):
    reports = full_reports(bound)
    reports['runtime'][field] = value
    runtime_record = put(bound.e['runtime_acceptance_path'], reports['runtime'])
    for key in ('run', 'acceptance'):
        reports[key]['prior_smoke_acceptance']['verification'] = runtime_record
    write_full_reports(bound, reports)
    with pytest.raises(ValueError):
        verified.accept_full(bound.e)
    assert not (bound.output / 'lineage_acceptance.json').exists()


@pytest.mark.parametrize('missing', [True, False])
def test_full_acceptance_requires_a_real_job_identity_not_matching_missing_values(bound, missing):
    reports = full_reports(bound)
    for key in ('runtime', 'run'):
        if missing:
            reports[key].pop('job_id')
        else:
            reports[key]['job_id'] = None
    runtime_record = put(bound.e['runtime_acceptance_path'], reports['runtime'])
    for key in ('run', 'acceptance'):
        reports[key]['prior_smoke_acceptance']['verification'] = runtime_record
    write_full_reports(bound, reports)
    with pytest.raises(ValueError):
        verified.accept_full(bound.e)
    assert not (bound.output / 'lineage_acceptance.json').exists()


@pytest.mark.parametrize('path_key', ['prelaunch_acceptance_path', 'runtime_acceptance_path'])
def test_full_acceptance_requires_receipt_file(bound, path_key):
    full_reports(bound)
    Path(bound.e[path_key]).unlink()
    with pytest.raises(FileNotFoundError):
        verified.accept_full(bound.e)
    assert not (bound.output / 'lineage_acceptance.json').exists()


@pytest.mark.parametrize('filename', ['run.json', 'acceptance.json'])
def test_full_acceptance_rejects_stale_report_hashes(bound, filename):
    full_reports(bound)
    path = bound.output / filename
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError):
        verified.accept_full(bound.e)
