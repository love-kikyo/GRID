import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts import random_retrieval_validation as v


def bindings():
    return dict(checkpoint=dict(path='/run/checkpoints/checkpoint_000_090000.ckpt', sha256='c' * 64),
                config=dict(path='/run/.hydra/config.yaml', sha256='d' * 64),
                preflight=dict(path='/preflight.json', sha256='e' * 64),
                wrapper=dict(path='/wrapper.py', sha256='f' * 64),
                manifests={key: dict(path='/' + key, sha256=str(i) * 64)
                           for i, key in enumerate(('train', 'test', 'audit'))},
                run_dir='/run', data_root='/data', source_git_commit='a' * 40)


def metadata(binding):
    return dict(schema_version=3, split='val', sample_unit='unique_user_history', num_samples=3,
                limits=dict(max_batches=None, max_samples=None), checkpoint=binding['checkpoint']['path'],
                checkpoint_sha256=binding['checkpoint']['sha256'], run_dir=binding['run_dir'],
                hydra_config_sha256=binding['config']['sha256'], git_commit=binding['source_git_commit'],
                batch_size=32, num_workers=1, amp='bf16', top_k_for_generation=20, num_hierarchies=4,
                num_embeddings_per_hierarchy=4096, beam_log_prob_weight=1.0, rerank_logit_weight=0.1,
                dataset_provenance=dict(protocol='history-last-user-v1', expected_samples=3,
                     data_root=binding['data_root'], initialization_checkpoint=None, source_compatibility=None,
                     training_preflight=dict(binding['preflight'], status='passed'), manifests=binding['manifests']),
                execution_wrapper=dict(source=binding['wrapper'], source_git_commit=binding['source_git_commit'],
                     fresh_single_pass=True, loader_settings=v.loader_fields(), configurations=1, datamodules=1,
                     loaders=1, checkpoint_loading=dict(missing=0, unexpected=0), checkpoint_global_step=90000,
                     loaded_parameters_finite=True))


def test_metadata_accepts_exact_full_binding():
    b = bindings()
    v.validate_metadata(metadata(b), b, expected=3)


@pytest.mark.parametrize('key,value', [
    ('schema_version', 2), ('split', 'test'), ('num_samples', 2),
    ('limits', dict(max_batches=2, max_samples=None)),
    ('checkpoint_sha256', '0' * 64), ('checkpoint', '/wrong.ckpt'),
    ('hydra_config_sha256', '0' * 64), ('git_commit', '0' * 40),
    ('rerank_logit_weight', 0.5), ('num_workers', 2),
])
def test_metadata_rejects_scope_hash_partial_or_checkpoint_mismatch(key, value):
    b = bindings()
    meta = metadata(b)
    meta[key] = value
    with pytest.raises(ValueError):
        v.validate_metadata(meta, b, expected=3)


@pytest.mark.parametrize('key,value', [
    ('fresh_single_pass', False), ('loaders', 2), ('datamodules', 2),
    ('checkpoint_loading', dict(missing=1, unexpected=0)), ('checkpoint_global_step', 115000),
    ('loaded_parameters_finite', False), ('source', dict(path='/wrong', sha256='f' * 64)),
])
def test_metadata_requires_strict_loading_and_fresh_single_pass(key, value):
    b = bindings()
    meta = metadata(b)
    meta['execution_wrapper'][key] = value
    with pytest.raises(ValueError):
        v.validate_metadata(meta, b, expected=3)


def test_fresh_loader_explicitly_disables_persistence():
    fields = v.loader_fields()
    assert fields == dict(batch_size_per_device=32, num_workers=1, persistent_workers=False,
                          drop_last=False, should_shuffle_rows=False, assign_all_files_per_worker=False)
    b = bindings()
    meta = metadata(b)
    meta['execution_wrapper']['loader_settings']['persistent_workers'] = True
    with pytest.raises(ValueError, match='fresh iterator'):
        v.validate_metadata(meta, b, expected=3)


@pytest.mark.parametrize('payload', [{}, dict(global_step=115000, state_dict={'x': 1}),
                                    dict(global_step=90000, state_dict={}),
                                    dict(global_step=90000, state_dict=[])])
def test_reject_wrong_checkpoint_header(payload):
    with pytest.raises(ValueError):
        v.validate_checkpoint_header(payload)


def test_checkpoint_header_uses_saved_step_not_terminal_training_step():
    v.validate_checkpoint_header(dict(global_step=90000, state_dict={'x': 1}))


def make_exports(tmp_path):
    n = 3
    beam = np.ones((n, 20, 4), dtype=np.int16)
    beam[:, :, 0] = np.arange(1, 21, dtype=np.int16)
    target = beam[:, 2].copy()
    target[2, 0] = 30
    scores = np.tile(-np.arange(20, dtype=np.float32), (n, 1))
    raw = np.tile(np.linspace(-3, 2, 20, dtype=np.float32), (n, 1))
    fused = scores + 0.1 * raw
    order = np.argsort(-fused, axis=1, kind='stable')
    arrays = dict(user_id=np.array([19, 7, 12], dtype=np.int64),
                  example_hash=np.array([[19, 9], [7, 9], [12, 9]], dtype=np.uint64),
                  target_sid=target, beam_sid=beam, beam_score=scores, raw_rerank_logit=raw,
                  rerank_sid=np.take_along_axis(beam, order[..., None], axis=1),
                  rerank_score=np.take_along_axis(fused, order, axis=1))
    output, ref = tmp_path / 'validation.h5', tmp_path / 'reference.h5'
    with h5py.File(output, 'w') as f:
        f.attrs['num_samples'] = n
        for key, value in arrays.items():
            f[key] = value
    # Different record order is permitted; only identity is compared.
    with h5py.File(ref, 'w') as f:
        for key in ('user_id', 'example_hash', 'target_sid'):
            f[key] = arrays[key][[2, 0, 1]]
    return output, ref


def test_full_export_integrity_and_independent_metrics(tmp_path):
    output, ref = make_exports(tmp_path)
    audited = v.audit_export(output, ref, expected=3)
    assert audited['unique_users'] == audited['unique_input_hashes'] == 3
    assert audited['input_identity_matches_reference'] is True
    assert np.asarray(audited['independent_grid_metrics']).shape == (51, 4)
    assert audited['independent_grid_metrics'][0][2] == pytest.approx(2 / 3)


@pytest.mark.parametrize('mutation', ['count', 'duplicate_user', 'duplicate_hash', 'candidate', 'target',
                                     'nan', 'float64', 'permutation', 'sorting', 'extra_field', 'zero_sid'])
def test_bad_exports_rejected(tmp_path, mutation):
    output, ref = make_exports(tmp_path)
    with h5py.File(output, 'r+') as f:
        if mutation == 'count': f.attrs['num_samples'] = 2
        elif mutation == 'duplicate_user': f['user_id'][1] = f['user_id'][0]
        elif mutation == 'duplicate_hash': f['example_hash'][1] = f['example_hash'][0]
        elif mutation == 'candidate': f['beam_sid'][0, 1] = f['beam_sid'][0, 0]
        elif mutation == 'target': f['target_sid'][0, 0] = 9
        elif mutation == 'nan': f['raw_rerank_logit'][0, 0] = np.nan
        elif mutation == 'float64':
            old = f['raw_rerank_logit'][:].astype(np.float64)
            del f['raw_rerank_logit']
            f['raw_rerank_logit'] = old
        elif mutation == 'permutation': f['rerank_sid'][0, 0, 0] = 99
        elif mutation == 'sorting': f['beam_score'][0, 2] = 10
        elif mutation == 'extra_field': f['extra'] = np.zeros(3)
        elif mutation == 'zero_sid': f['beam_sid'][0, 0, 3] = 0
    with pytest.raises(ValueError):
        v.audit_export(output, ref, expected=3)


def selection_fixture():
    output = dict(path='/validation.h5', sha256='a' * 64)
    sidecar = dict(path='/validation.json', sha256='b' * 64)
    checkpoint = dict(path='/model.ckpt', sha256='c' * 64)
    metrics = np.tile([0.3, 0.2, 0.4, 0.35], (51, 1))
    rows = [dict(variant=f'fusion@1@{i / 100:.2f}', beam_weight=1.0, rerank_weight=i / 100,
                 metrics=dict(zip(v.ORDER, metrics[i]))) for i in range(51)]
    selection = dict(schema_version=2, selection_scope='validation_only', input_split='val', num_examples=3,
                     input=output['path'], input_sha256=output['sha256'], input_sidecar_sha256=sidecar['sha256'],
                     input_checkpoint_sha256=checkpoint['sha256'], selection_protocol=copy.deepcopy(v.PROTOCOL),
                     candidates=rows, selected=rows[0], source_git_commit='d' * 40)
    return selection, output, sidecar, checkpoint, metrics


def test_selection_matches_independent_metrics_and_smallest_beta_tie():
    selected, output, sidecar, checkpoint, metrics = selection_fixture()
    result = v.validate_selection(selected, output, sidecar, checkpoint, metrics, expected=3, commit='d' * 40)
    assert result['selected']['rerank_weight'] == 0
    assert result['metric_values_checked'] == 204


@pytest.mark.parametrize('mutation', ['partial', 'checkpoint', 'sidecar', 'grid', 'rule', 'metric', 'tie', 'nan', 'commit'])
def test_invalid_fusion_selection_rejected(mutation):
    selected, output, sidecar, checkpoint, metrics = selection_fixture()
    if mutation == 'partial': selected['num_examples'] = 2
    elif mutation == 'checkpoint': selected['input_checkpoint_sha256'] = 'x' * 64
    elif mutation == 'sidecar': selected['input_sidecar_sha256'] = 'x' * 64
    elif mutation == 'grid': selected['candidates'][2]['rerank_weight'] = .75
    elif mutation == 'rule': selected['selection_protocol']['metric_order'] = list(reversed(v.ORDER))
    elif mutation == 'metric': selected['candidates'][2]['metrics']['ndcg@10'] += .1
    elif mutation == 'tie': selected['selected'] = selected['candidates'][50]
    elif mutation == 'nan': selected['candidates'][2]['metrics']['ndcg@10'] = float('nan')
    elif mutation == 'commit': selected['source_git_commit'] = 'x' * 40
    with pytest.raises(ValueError):
        v.validate_selection(selected, output, sidecar, checkpoint, metrics, expected=3, commit='d' * 40)


def test_stable_score_ties_are_resolved_by_original_beam_position():
    beam = np.ones((1, 20, 4), dtype=np.int16)
    beam[0, :, 0] = np.arange(1, 21)
    target = beam[:, 6].copy()
    values = v.independent_metrics(target, beam, np.zeros((1, 20), dtype=np.float32))
    np.testing.assert_allclose(values, [1 / np.log2(8), 0, 1, 0], rtol=0, atol=1e-15)


def test_shifted_sid_4096_is_legal(tmp_path):
    output, ref = make_exports(tmp_path)
    with h5py.File(output, 'r+') as f:
        f['beam_sid'][0, 19, 3] = 4096
        f['rerank_sid'][0, 19, 3] = 4096
    assert v.audit_export(output, ref, expected=3)['candidate_integrity_passed'] is True


def test_base4096_alias_with_padding_is_rejected_before_candidate_comparison(tmp_path):
    # A raw zero (padding, illegal as a generated digit) can alias digit4096
    # under base4096. Base4097 separates both; the domain guard must reject 0.
    pairs = np.array([[1, 1, 1, 4096], [1, 1, 2, 0]], dtype=np.int64)
    old = pairs @ np.array([4096**3, 4096**2, 4096, 1], dtype=np.int64)
    new = pairs @ np.array([4097**3, 4097**2, 4097, 1], dtype=np.int64)
    assert old[0] == old[1] and new[0] != new[1]
    output, ref = make_exports(tmp_path)
    with h5py.File(output, 'r+') as f:
        f['beam_sid'][0, :2] = pairs.astype(np.int16)
        f['rerank_sid'][0, :2] = pairs.astype(np.int16)
    with pytest.raises(ValueError, match='SID dtype/range'):
        v.audit_export(output, ref, expected=3)


def test_fp32_beta_arithmetic_matches_existing_selector(tmp_path):
    from decimal import Decimal
    from scripts.select_score_fusion import build_candidates, evaluate_variant
    output, _ = make_exports(tmp_path)
    with h5py.File(output, 'r') as f:
        payload = {key: f[key][:] for key in ('target_sid', 'beam_sid', 'beam_score', 'raw_rerank_logit')}
    candidates = build_candidates(SimpleNamespace(
        beam_weight=1.0, rerank_weight_start=Decimal('0.00'), rerank_weight_stop=Decimal('0.50'),
        rerank_weight_step=Decimal('0.01'), metric_order=v.ORDER))
    for index, candidate in enumerate(candidates):
        assert candidate['rerank_weight'] == index / 100
        scores = 1.0 * payload['beam_score'] + (index / 100) * payload['raw_rerank_logit']
        assert scores.dtype == np.float32
        independent = v.independent_metrics(payload['target_sid'], payload['beam_sid'], scores) / 3
        selected = evaluate_variant(payload, candidate['variant'], v.ORDER)
        np.testing.assert_allclose(independent, [selected[k] for k in v.ORDER], rtol=0, atol=1e-15)


def test_config_rejects_teacher_and_changed_inference_weights():
    cfg = SimpleNamespace(seed=42, ckpt_path=None, sequence_length=32, hist_itemkey_length=100,
              model=SimpleNamespace(manual_ckpt_path=None, load_only_weights=False, rerank_objective='listwise',
                       top_k_for_generation=20, top_k_for_score=10, active_top_k_for_score=None,
                       beam_log_prob_weight=1.0, rerank_logit_weight=.1))
    v.validate_config(cfg)
    cfg.model.manual_ckpt_path = '/teacher.ckpt'
    with pytest.raises(ValueError): v.validate_config(cfg)


def test_formal_audit_requires_all_three_ranks_and_correct_steps(tmp_path):
    for rank in range(3):
        for phase, step in [('initialization', 0), ('train_start', 0), ('updates', 2), ('finished', 115000)]:
            report = dict(status='passed', phase=phase, global_rank=rank, world_size=3, seed=42,
                          smoke=False, method='joint-listwise-random-init', global_step=step,
                          constructed_parameters_unchanged=True, optimizer_initially_empty=True,
                          initialization_checkpoint=None, resume_checkpoint=None, first_two_updates_verified=True)
            (tmp_path / f'rank_{rank}_{phase}.json').write_text(json.dumps(report))
    assert len(v.verify_random_initialization(tmp_path)) == 12
    p = tmp_path / 'rank_2_finished.json'
    report = json.loads(p.read_text())
    report['global_step'] = 90000
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='step'):
        v.verify_random_initialization(tmp_path)


def test_cli_rejects_nonexistent_commit_before_any_output(tmp_path, monkeypatch):
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'not-a-commit')
    with pytest.raises(ValueError, match='exact execution commit'):
        v.run(tmp_path / 'new', 'not-a-commit')
    assert not (tmp_path / 'new').exists()
