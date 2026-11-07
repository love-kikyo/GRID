import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.prepare_retrieval_content import load_manifest, scope_fields, sha, write_new
from scripts import prepare_retrieval_content as content


def fixture(tmp_path, monkeypatch):
    m = copy.deepcopy(scope_fields())
    m.update(source_git_commit='test', execution_sources=[], input_metadata=[])
    path = tmp_path / 'manifest.json'
    write_new(path, m)
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'test')
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(path))
    return path, m


def test_scope_and_manifest_hash(tmp_path, monkeypatch):
    p, _ = fixture(tmp_path, monkeypatch)
    assert load_manifest(p)['full_samples'] == 2061101
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', 'wrong')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_manifest(p)


@pytest.mark.parametrize('key,value', [('seed', 43), ('full_samples', 16384),
    ('smoke_samples', 64), ('policies', ['sim10']), ('physical_slots', 5),
    ('C', 40), ('beta', .1), ('runtime_beta', .5), ('num_workers', 4),
    ('asset_root', '/wrong'), ('execution_root', '/wrong'), ('dirty_at_submission', True)])
def test_scope_drift(tmp_path, monkeypatch, key, value):
    p, m = fixture(tmp_path, monkeypatch)
    m[key] = value
    p.write_text(json.dumps(m))
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(p))
    with pytest.raises(ValueError, match='authorized scope'):
        load_manifest(p)


@pytest.mark.parametrize('key,value', [('bootstrap_samples', 100), ('bootstrap_seed', 42),
    ('confidence', .90), ('primary_comparison', 'sim10_minus_mask0:fused'),
    ('primary_metric', 'recall@10'), ('ks', [10])])
def test_statistics_frozen(tmp_path, monkeypatch, key, value):
    p, m = fixture(tmp_path, monkeypatch)
    m['statistics'][key] = value
    p.write_text(json.dumps(m))
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(p))
    with pytest.raises(ValueError, match='authorized scope'):
        load_manifest(p)


def test_source_and_commit_hash(tmp_path, monkeypatch):
    p, m = fixture(tmp_path, monkeypatch)
    source = tmp_path / 'source.py'
    source.write_text('original')
    m['execution_sources'] = [{'path': str(source), 'sha256': sha(source)}]
    p.write_text(json.dumps(m))
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(p))
    source.write_text('changed')
    with pytest.raises(ValueError, match='Changed accepted source'):
        load_manifest(p)
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'wrong')
    with pytest.raises(ValueError, match='commit binding'):
        load_manifest(p)


def test_counts_require_integers(tmp_path, monkeypatch):
    p, m = fixture(tmp_path, monkeypatch)
    m['full_samples'] = float(m['full_samples'])
    p.write_text(json.dumps(m))
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(p))
    with pytest.raises(ValueError, match='integer scope'):
        load_manifest(p)


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_training_seed_is_separate_from_fixed_history_sampling_seed(seed):
    scope = scope_fields(seed)
    assert scope['seed'] == seed
    assert scope['experiment'] == f'retrieval_content_seed{seed}_full_v1'
    assert scope['random_seed'] == 42
    assert scope['statistics'] == scope_fields()['statistics']


@pytest.mark.parametrize('seed', [True, 43., '43', 0, 41, 45, None])
def test_unsupported_or_mistyped_seed_rejected(seed):
    with pytest.raises(ValueError, match='authorized scope'):
        scope_fields(seed)


def put_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return content.record(path)


@pytest.fixture(params=[43, 44])
def multiseed_bound(tmp_path, monkeypatch, request):
    seed = request.param
    assets, root, old_root = tmp_path / 'assets', tmp_path / 'new-root', tmp_path / 'old-root'
    monkeypatch.setattr(content, 'ASSET_ROOT', assets)
    monkeypatch.setattr(content, 'ROOT', root)
    monkeypatch.setattr(content, 'MULTISEED_SOURCES', ['runtime.py', 'prepare.py'])
    sources = [put_json(root / name, {'name': name}) for name in content.MULTISEED_SOURCES]
    frozen = put_json(old_root / 'runtime.py', {'name': 'runtime.py'})
    run_id = content.PRIOR_SCANS[seed]['run_id']
    run = assets / 'logs/train/runs/date' / f'click-listwise-seed{seed}-{run_id}'
    checkpoint = {'path': str(run / f"checkpoints/checkpoint_000_{content.PRIOR_SCANS[seed]['best_step']}.ckpt"),
                  'sha256': f'checkpoint-{seed}'}
    analysis = assets / 'logs/analysis/paired_evaluation'
    selection = dict(input_split='val', selection_scope='validation_only',
                     input_checkpoint_sha256=checkpoint['sha256'],
                     selected={'beam_weight': 1., 'rerank_weight': .5})
    selection_record = put_json(analysis / f'listwise_seed{seed}_{run_id}_fusion_val.json', selection)
    prior = dict(seed=seed, protocol='history-last-user-v1', expected_test_samples=2061101,
                 execution_root=str(old_root), execution_sources=[frozen],
                 run_dir=str(run), data_root=str(assets / 'data/taobao_history_last_user_v1'),
                 training_preflight=str(assets / f'logs/training_preflight/listwise_seed{seed}_{run_id}.json'),
                 checkpoint=checkpoint,
                 reference_test={'path': str(analysis / f'listwise_seed{seed}_{run_id}_test_full_v1.h5'),
                                 'sha256': f'reference-{seed}'},
                 frozen_beta=selection_record, input_metadata=[selection_record])
    prior_path = assets / f'logs/training_launch/sensitivity_seed{seed}_2026-11-07_v1/manifest.json'
    prior_record = put_json(prior_path, prior)
    accepted = dict(status='passed', phase='full', group='retrieval', seed=seed,
                    num_examples=2061101, unique_users=2061101, manifest_sha256=prior_record['sha256'],
                    **{key: prior[key] for key in ('checkpoint', 'reference_test', 'frozen_beta')})
    accepted_path = assets / f'logs/analysis/inference_sensitivity_seed{seed}_2026-11-07_v1/full_retrieval/acceptance.json'
    accepted_record = put_json(accepted_path, accepted)
    bindings = copy.deepcopy(content.PRIOR_SCANS)
    bindings[seed].update(manifest_sha256=prior_record['sha256'], acceptance_sha256=accepted_record['sha256'])
    monkeypatch.setattr(content, 'PRIOR_SCANS', bindings)

    def git(command, **kwargs):
        assert kwargs == {'cwd': root, 'text': True}
        return {('git', 'status', '--porcelain'): '', ('git', 'rev-parse', 'HEAD'): 'fixture-commit\n'}[tuple(command)]

    monkeypatch.setattr(content, 'subprocess', SimpleNamespace(check_output=git))
    manifest = assets / f'logs/training_launch/new-seed{seed}/manifest.json'
    content.prepare(manifest, seed)
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'fixture-commit')
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(manifest))
    return SimpleNamespace(seed=seed, manifest=manifest, prior=prior, accepted=accepted,
                           prior_path=prior_path, accepted_path=accepted_path, root=root,
                           sources=sources, monkeypatch=monkeypatch)


def rewrite_manifest(bound, mutation):
    value = json.loads(bound.manifest.read_text())
    mutation(value)
    put_json(bound.manifest, value)
    bound.monkeypatch.setenv('CONTENT_MANIFEST_SHA256', sha(bound.manifest))


def test_multiseed_prepare_load_binds_own_assets_and_local_source_files(multiseed_bound):
    b = multiseed_bound
    m = content.load_manifest(b.manifest)
    assert m['seed'] == b.seed and m['random_seed'] == 42
    assert m['execution_sources'] == b.sources
    assert m['checkpoint'] == b.prior['checkpoint']
    assert m['reference_test'] == b.prior['reference_test']
    assert m['frozen_beta'] == b.prior['frozen_beta']
    assert m['selection'] == content.SELECTION
    # No checkpoint/H5 fixtures exist: preparation never opened either large asset.
    assert not Path(m['checkpoint']['path']).exists()
    assert not Path(m['reference_test']['path']).exists()


@pytest.mark.parametrize('field', content.INHERITED_FIELDS)
def test_rebound_manifest_cannot_mix_training_seed_assets(multiseed_bound, field):
    b = multiseed_bound
    rewrite_manifest(b, lambda m: m.update({field: 'wrong-seed-asset'}))
    with pytest.raises(ValueError, match='accepted training seed'):
        content.load_manifest(b.manifest)


@pytest.mark.parametrize('mutation', [
    lambda m: m['execution_sources'].pop(),
    lambda m: m['execution_sources'].append(m['execution_sources'][0]),
    lambda m: m['input_metadata'].pop(),
    lambda m: m.update(selection='change retrieval ordering'),
    lambda m: m.update(seed=42, experiment='retrieval_content_seed43_full_v1'),
    lambda m: m.update(random_seed=43),
])
def test_multiseed_scope_and_provenance_cannot_be_relaxed(multiseed_bound, mutation):
    b = multiseed_bound
    rewrite_manifest(b, mutation)
    with pytest.raises(ValueError):
        content.load_manifest(b.manifest)


@pytest.mark.parametrize('which', ['prior', 'acceptance', 'source', 'frozen_beta'])
def test_multiseed_accepted_file_changes_fail_closed(multiseed_bound, which):
    b = multiseed_bound
    path = {'prior': b.prior_path, 'acceptance': b.accepted_path,
            'source': Path(b.sources[0]['path']),
            'frozen_beta': Path(b.prior['frozen_beta']['path'])}[which]
    path.write_text(path.read_text() + '\nchanged')
    with pytest.raises(ValueError, match='Changed accepted source'):
        content.load_manifest(b.manifest)


def test_multiseed_manifest_cannot_be_overwritten(multiseed_bound):
    b = multiseed_bound
    original = b.manifest.read_bytes()
    with pytest.raises(FileExistsError, match='existing manifest'):
        content.prepare(b.manifest, b.seed)
    assert b.manifest.read_bytes() == original


def test_multiseed_prepare_requires_clean_sources(multiseed_bound):
    b = multiseed_bound
    b.monkeypatch.setattr(content.subprocess, 'check_output', lambda *args, **kwargs: ' M runtime.py')
    with pytest.raises(RuntimeError, match='Commit sources'):
        content.prepare(b.manifest.parent / 'new.json', b.seed)
