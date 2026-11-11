import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import prepare_random_retrieval_content as prepare


def selection(beta=.17):
    candidates = [dict(beam_weight=1., rerank_weight=i / 100,
                       variant=f'fusion@1@{i / 100}',
                       metrics={key: .3 if i == round(beta * 100) else .2
                                for key in ('ndcg@10', 'ndcg@5', 'recall@10', 'recall@5')})
                  for i in range(51)]
    return dict(input_split='val', selection_scope='validation_only', num_examples=347416,
                input_checkpoint_sha256='a' * 64,
                selection_protocol=dict(metric_order=['ndcg@10', 'ndcg@5', 'recall@10', 'recall@5'],
                                        tie_breaker='smaller_rerank_weight', beam_weight=1.,
                                        hash_sample={'word': 0, 'modulus': 1, 'remainder': 0},
                                        rerank_weight_grid={'start': '0.00', 'stop': '0.50', 'step': '0.01'}),
                candidates=candidates, selected=candidates[round(beta * 100)])


@pytest.mark.parametrize('beta', [0., .01, .17, .33, .49, .5])
def test_own_validation_may_choose_any_original_grid_weight(beta):
    assert prepare.validate_selection(selection(beta), 'a' * 64) == beta


@pytest.mark.parametrize('field,value', [('input_split', 'test'), ('num_examples', 65),
    ('selection_scope', 'test'), ('input_checkpoint_sha256', 'b' * 64)])
def test_wrong_source_rejected(field, value):
    report = selection(); report[field] = value
    with pytest.raises(ValueError): prepare.validate_selection(report, 'a' * 64)


def test_wrong_selected_and_grid_rejected():
    report = selection(); report['selected'] = report['candidates'][50]
    with pytest.raises(ValueError): prepare.validate_selection(report, 'a' * 64)
    report = selection(); report['candidates'].pop()
    with pytest.raises(ValueError): prepare.validate_selection(report, 'a' * 64)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.1, 1.1, True])
def test_bad_metric_rejected(bad):
    report = selection(); report['candidates'][2]['metrics']['ndcg@10'] = bad
    with pytest.raises(ValueError): prepare.validate_selection(report, 'a' * 64)


def test_tie_goes_to_smaller_beta():
    report = selection(.17)
    report['candidates'][0]['metrics'] = dict(report['selected']['metrics'])
    with pytest.raises(ValueError): prepare.validate_selection(report, 'a' * 64)
    report['selected'] = report['candidates'][0]
    assert prepare.validate_selection(report, 'a' * 64) == 0.


def test_fixed_scope_does_not_inherit_teacher_or_old_weights():
    fields = prepare.scope()
    assert fields['seed'] == 42 and fields['physical_slots'] == 10 and fields['C'] == 20
    assert fields['full_samples'] == 2061101 and fields['smoke_samples'] == 65
    assert fields['policies'] == ['sim10', 'recent10', 'random10', 'mask5', 'mask0']
    assert 'beta' not in fields and 'frozen_beta' not in fields
    assert 'random-seed42-158562' in fields['run_dir']
    assert prepare.CHECKPOINT.name == 'checkpoint_000_090000.ckpt'
    assert 'Only user_id' in fields['reference_role']


def test_manifest_checksum_rejected_before_read(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'; path.write_text('{}')
    monkeypatch.setenv('CONTENT_MANIFEST_SHA256', 'a' * 64)
    with pytest.raises(ValueError, match='Manifest hash'): prepare.load_manifest(path)


def test_launcher_limits_and_order():
    path = prepare.ROOT / 'run_random_retrieval_content_cpu4.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    source = path.read_text()
    for value in ['--cpus-per-task=4', '--ntasks=1', '--mem=24G', '--gres=gpu:4090:1',
                  '--partition=rtx4090', 'set -euo pipefail']:
        assert value in source
    names = ['-m scripts.random_retrieval_validation', '-m scripts.prepare_random_retrieval_content',
             '-m scripts.run_random_retrieval_content', '-m scripts.audit_retrieval_content',
             '-m scripts.analyze_retrieval_content']
    offsets = [source.index(name) for name in names]
    assert offsets == sorted(offsets)
    assert '--phase full --smoke-dir' in source
    assert 'sbatch' not in source and 'scancel' not in source


@pytest.mark.parametrize('args', [[], ['../bad'], ['random_init_retrieval_seed43_158562_2026-11-11_v1']])
def test_launcher_rejects_bad_name_without_side_effects(args):
    result = subprocess.run(['bash', str(prepare.ROOT / 'run_random_retrieval_content_cpu4.sh'), *args],
                            capture_output=True, text=True)
    assert result.returncode == 2
