from pathlib import Path
import json
import subprocess

import numpy as np
import pytest

from scripts.analyze_sim_seed42_paired import (arguments, independent_values, resolved_comparisons,
                                             COMPARISONS, VARIANTS, METRICS, verify_markdown,
                                             verify_source_record, ORIGINAL_AUDIT_SHA)
from scripts.analyze_paired_evaluation import ranked_candidates, metric_contributions, markdown_report


@pytest.mark.parametrize('variant', ['beam', 'click', 'fusion@1@0.09', 'fusion@1@0.50'])
@pytest.mark.parametrize('ties', [True, False])
def test_independent_gt_rank_metrics_match_production(variant, ties):
    rng = np.random.default_rng(42)
    n = 85
    beam = np.arange(n * 20 * 4, dtype=np.int64).reshape(n, 20, 4)
    target = beam[np.arange(n), np.arange(n) % 20].copy()
    target[::7] = -1
    raw = rng.integers(-3, 3, (n, 20)).astype(np.float32) if ties else rng.normal(size=(n, 20)).astype(np.float32)
    beam_scores = np.sort(rng.integers(-5, 0, (n, 20)).astype(np.float32), axis=1)[:, ::-1].copy()
    payload = dict(target_sid=target, beam_sid=beam, raw_rerank_logit=raw, beam_score=beam_scores)
    actual = independent_values(payload, variant)
    ranked = ranked_candidates(payload, variant)
    for k in (5, 10):
        expected = metric_contributions(target, ranked, k)
        for name in ('recall', 'ndcg'):
            np.testing.assert_array_equal(actual[f'{name}@{k}'], expected[name])


def test_fixed_pairs_use_only_validation_selected_weights():
    args = arguments()
    assert args.compare == COMPARISONS and len(COMPARISONS) == 5
    assert args.bootstrap_samples == 5000 and args.bootstrap_batch_size == 500
    assert args.ks == [5, 10] and args.confidence == .95 and args.seed == 2027
    assert args.expected_split == 'test' and args.chunk_size == 65536
    assert VARIANTS == dict(sim='fusion@1@0.09', separate='fusion@1@0.50')
    assert resolved_comparisons() == ['sid:beam,sim:click', 'sid:beam,sim:fusion@1@0.09',
        'sim:click,sim:fusion@1@0.09', 'sim:fusion@1@0.09,separate:fusion@1@0.50',
        'sim:click,separate:click']


def test_launcher_is_cpu_only_sequential_and_has_no_successor():
    path = Path(__file__).resolve().parents[1] / 'run_sim_paired_cpu4.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    assert '#SBATCH --cpus-per-task=4' in text and '#SBATCH --gres=gpu:0' in text
    assert '--cpu-bind=verbose,cores' in text and '--cpu-bind=verbose,threads' in text
    assert 'SIM_TEST_ACCEPTANCE_SHA256' in text
    assert text.index('-m pytest') < text.index('analyze_sim_seed42_paired run') < text.index('analyze_sim_seed42_paired audit')
    assert 'sbatch ' not in text and 'select_score_fusion' not in text


def test_markdown_json_roundtrip_preserves_original_metric_display_order():
    metric = dict(baseline_mean=.1, challenger_mean=.2, absolute_gain=.1,
                  relative_gain_percent=100., paired_bootstrap_ci=[.09, .11])
    original = dict(sample_unit='unique_user_history', bootstrap_samples=5000, seed=2027,
                    comparisons=[dict(baseline='a:beam', challenger='b:click',
                                      metrics={key: dict(metric) for key in METRICS})])
    text = markdown_report(original)
    loaded = json.loads(json.dumps(original, sort_keys=True))
    assert markdown_report(loaded) != text  # Reproduce the old display-order failure.
    verify_markdown(loaded, text)
    with pytest.raises(AssertionError):
        verify_markdown(loaded, text.replace('0.200000', '0.900000'))
    loaded['comparisons'][0]['metrics']['recall@5']['challenger_mean'] = .9
    with pytest.raises(AssertionError):
        verify_markdown(loaded, text)


def test_recovery_source_exception_is_exact_and_scoped(tmp_path):
    root = Path(__file__).resolve().parents[1]
    old = dict(path=str(root / 'scripts/analyze_sim_seed42_paired.py'), sha256=ORIGINAL_AUDIT_SHA)
    verify_source_record(old, recovery=True)
    with pytest.raises(AssertionError):
        verify_source_record(old, recovery=False)
    with pytest.raises(AssertionError):
        verify_source_record({**old, 'sha256': 'wrong'}, recovery=True)
    path = tmp_path / 'unrelated.py'
    path.write_text('untouched')
    with pytest.raises(AssertionError):
        verify_source_record({**old, 'path': str(path)}, recovery=True)


def test_recovery_launcher_only_reaudits_existing_report():
    path = Path(__file__).resolve().parents[1] / 'run_sim_paired_audit_recovery_cpu1.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    assert '#SBATCH --cpus-per-task=1' in text and '#SBATCH --gres=gpu:0' in text
    assert 'analyze_sim_seed42_paired audit-recovery' in text
    assert 'analyze_sim_seed42_paired run' not in text and 'sbatch ' not in text
