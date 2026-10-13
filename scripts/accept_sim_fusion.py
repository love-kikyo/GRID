"""Independently count GT outrankers for the fixed SIM validation beta grid."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

ORDER = ['ndcg@10', 'ndcg@5', 'recall@10', 'recall@5']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def contributions(target, beam, scores):
    """No sorting: count strict outrankers and earlier original positions at ties."""
    match = (beam == target[:, None, :]).all(-1)
    assert (match.sum(-1) <= 1).all()
    pos = match.argmax(-1)
    gt_score = scores[np.arange(len(pos)), pos][:, None]
    ahead = (scores > gt_score) | ((scores == gt_score) & (np.arange(scores.shape[1])[None, :] < pos[:, None]))
    rank = ahead.sum(-1) + 1
    rank[~match.any(-1)] = max(11, scores.shape[1] + 1)
    discount = 1. / np.log2(rank + 1.)
    return np.stack([discount * (rank <= 10), discount * (rank <= 5), rank <= 10, rank <= 5], axis=1)


def audit(selection, validation, output):
    assert len(os.sched_getaffinity(0)) == 1 and not output.exists()
    a, s = json.loads(validation.read_text()), json.loads(selection.read_text())
    n = 347416
    assert a['status'] == 'passed' and a['num_samples'] == n and a['cached_beam_exact']
    assert a['teacher_regenerated'] is False
    assert s['slurm_job_id'] == os.environ['SLURM_JOB_ID']
    assert s['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT']
    assert s['input_split'] == 'val' and s['selection_scope'] == 'validation_only' and s['num_examples'] == n
    assert s['input'] == a['output']
    assert s['input_sha256'] == a['h5_sha256'] == sha(s['input'])
    assert s['input_sidecar_sha256'] == a['sidecar_sha256'] == sha(Path(s['input']).with_suffix('.json'))
    assert s['input_checkpoint_sha256'] == a['checkpoint_sha256']
    assert s['selection_protocol'] == dict(metric_order=ORDER, tie_breaker='smaller_rerank_weight',
        beam_weight=1., hash_sample=dict(word=0, modulus=1, remainder=0),
        rerank_weight_grid=dict(start='0.00', stop='0.50', step='0.01'))
    assert s['cpu_budget']['allocated_cpus'] == s['cpu_budget']['user_cpu_ceiling'] == 4
    assert s['cpu_budget']['library_threads'] == 1 and len(s['cpu_process_tree_samples']) == 2
    assert 0 < len(s['cpu_budget']['allowed_cpus']) <= 4
    for sample in s['cpu_process_tree_samples']:
        assert sample['allowed_cpus'] == s['cpu_budget']['allowed_cpus'] and sample['processes']
        assert all(p['threads_checked'] > 0 for p in sample['processes'])
    for entry in s['source_files']:
        assert sha(entry['path']) == entry['sha256']
    rows = s['candidates']
    assert len(rows) == 51 and [r['rerank_weight'] for r in rows] == [i / 100 for i in range(51)]
    assert all(r['beam_weight'] == 1 for r in rows)
    assert [r['variant'] for r in rows] == [f'fusion@1@{i / 100:.2f}' for i in range(51)]
    totals, pure, pure_reference = np.zeros((51, 4)), np.zeros(4), np.zeros(4)
    from scripts.analyze_paired_evaluation import ranked_candidates, metric_contributions
    with h5py.File(s['input'], 'r') as f:
        assert len(f['target_sid']) == n
        for start in range(0, n, 8192):
            target, beam, bs, rs = [f[k][start:start + 8192] for k in
                                   ('target_sid', 'beam_sid', 'beam_score', 'raw_rerank_logit')]
            for i, row in enumerate(rows):
                totals[i] += contributions(target, beam, bs + row['rerank_weight'] * rs).sum(0)
            pure += contributions(target, beam, rs).sum(0)
            ranked = ranked_candidates(dict(beam_sid=beam, raw_rerank_logit=rs), 'click')
            for j, key in enumerate(ORDER):
                metric, k = key.split('@')
                pure_reference[j] += metric_contributions(target, ranked, int(k))[metric].sum()
    actual = np.array([[r['metrics'][key] for key in ORDER] for r in rows])
    expected = totals / n
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pure / n, pure_reference / n, rtol=0, atol=1e-12)
    best = max(range(51), key=lambda i: tuple(actual[i]) + (-rows[i]['rerank_weight'],))
    assert s['selected'] == rows[best]
    result = dict(status='passed', checked_at=datetime.now().astimezone().isoformat(),
        scope='full validation-only fixed-grid SIM fusion; not test', num_samples=n,
        job_id=s['slurm_job_id'], source_git_commit=s['source_git_commit'],
        candidates_checked=51, metric_values_checked=204,
        independent_method='GT outranker counting with earlier original beam position at ties',
        max_absolute_metric_difference=float(np.abs(actual - expected).max()),
        selected=s['selected'], beta_0_1_fp32=rows[10], beam_only=rows[0],
        pure_sim_esu=dict(zip(ORDER, (pure / n).tolist())), pure_sim_double_checked=True,
        selection_at_grid_boundary=best in (0, 50), grid_expanded=False,
        checkpoint_sha256=a['checkpoint_sha256'], input_sha256=s['input_sha256'],
        validation_acceptance=str(validation), validation_acceptance_sha256=sha(validation),
        selection=str(selection), selection_sha256=sha(selection), markdown_sha256=sha(selection.with_suffix('.md')),
        cpu_budget=s['cpu_budget'], audit_script_sha256=sha(__file__))
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('[sim-fusion-acceptance] ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--validation-acceptance', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.selection, args.validation_acceptance, args.output)
