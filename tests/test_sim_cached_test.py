import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from scripts.export_sim_cached_test import audit_predictions, check_selection, N
from scripts.smoke_sim_cached_rerank import score_cached_batch


def test_locked_selection_rejects_test_split_or_changed_selection():
    selected = dict(beam_weight=1., rerank_weight=.09, variant='fusion@1@0.09')
    fusion = dict(status='passed', num_samples=347416, candidates_checked=51, metric_values_checked=204,
                  grid_expanded=False, checkpoint_sha256='ckpt', input_sha256='val', selected=selected)
    acceptance = dict(status='passed', num_samples=347416, checkpoint_sha256='ckpt',
                      h5_sha256='val', sidecar_sha256='sidecar')
    selection = dict(num_examples=347416, input_split='val', input_sha256='val',
                     input_sidecar_sha256='sidecar', selected=selected)
    check_selection(fusion, selection, acceptance)
    for key, value in [('input_split', 'test'), ('input_sha256', 'other'),
                       ('selected', dict(beam_weight=1., rerank_weight=.1, variant='fusion@1@0.10'))]:
        bad = copy.deepcopy(selection)
        bad[key] = value
        with pytest.raises(AssertionError):
            check_selection(fusion, bad, acceptance)
    fusion['selected'] = selection['selected'] = dict(beam_weight=1., rerank_weight=.1, variant='fusion@1@0.10')
    with pytest.raises(AssertionError):
        check_selection(fusion, selection, acceptance)


@pytest.fixture
def full_test_tail(tmp_path):
    n = N % 32  # Full test ends with 13 rows, not validation's 24.
    assert n == 13
    beam = np.broadcast_to(np.arange(80).reshape(1, 20, 4), (n, 20, 4)).copy()
    scores = -np.tile(np.arange(20, dtype=np.float32), (n, 1))
    logits = torch.linspace(-3, 2, 20).to(torch.bfloat16).repeat(n, 1)
    model = SimpleNamespace(top_k_for_generation=20, num_hierarchies=4,
        powers=torch.tensor([4096**3, 4096**2, 4096, 1]),
        reranker=SimpleNamespace(score=lambda h, c, auxiliary: {'esu_logits': logits}),
        _combine_rerank_scores=lambda b, r: b + .1 * r)
    batch = SimpleNamespace(transformed_sequences={'hist_itemkey': torch.ones((n, 5), dtype=torch.long)})
    result = score_cached_batch(model, batch, beam, scores, torch.device('cpu'), 'none')
    arrays = dict(user_id=np.arange(n, dtype=np.int64), example_hash=np.arange(n * 2, dtype=np.uint64).reshape(n, 2),
                  target_sid=beam[:, 0], beam_sid=beam, beam_score=scores,
                  raw_rerank_logit=logits.float().numpy(), rerank_sid=result['rerank']['ids'].numpy(),
                  rerank_score=result['rerank']['scores'].numpy())
    with h5py.File(tmp_path / 'out.h5', 'w') as out, h5py.File(tmp_path / 'cache.h5', 'w') as ref:
        for key, value in arrays.items():
            out[key] = value
            if key in ('user_id', 'example_hash', 'target_sid', 'beam_sid', 'beam_score'):
                ref[key] = value[::-1]  # Audit must align, not assume storage order.
        yield out, ref, n


def test_full_test_partial_batch_and_reordered_reference(full_test_tail):
    out, ref, n = full_test_tail
    assert audit_predictions(out, ref, n) == n


@pytest.mark.parametrize('key', ['example_hash', 'target_sid', 'beam_sid', 'beam_score',
                                 'raw_rerank_logit', 'rerank_score', 'user_id'])
def test_independent_test_audit_rejects_corruption(full_test_tail, key):
    out, ref, n = full_test_tail
    if key == 'user_id':
        out[key][0] = out[key][1]
    else:
        value = out[key][0]
        value.flat[0] += 1
        out[key][0] = value
    with pytest.raises(AssertionError):
        audit_predictions(out, ref, n)


def test_launcher_never_searches_or_submits_successors():
    root = Path(__file__).resolve().parents[1]
    launcher = root / 'run_sim_cached_test_cpu4.sh'
    subprocess.run(['bash', '-n', str(launcher)], check=True)
    text = launcher.read_text()
    assert '#SBATCH --cpus-per-task=4' in text and '#SBATCH --gres=gpu:1' in text
    assert '--cpu-bind=verbose,cores' in text and '--cpu-bind=verbose,threads' in text
    assert 'sbatch ' not in text and 'run_select_score_fusion' not in text
    assert text.index('-m pytest') < text.index('-m scripts.export_sim_cached_test') < text.index('--acceptance "$ACCEPTANCE"')
    source = (root / 'scripts/export_sim_cached_test.py').read_text()
    assert source.index('frozen, previous = validate_frozen_validation()') < source.index("cfg, 'test',")
    assert 'cached.score_cached_batch(' in source and 'cached.verify_inference(' not in source
