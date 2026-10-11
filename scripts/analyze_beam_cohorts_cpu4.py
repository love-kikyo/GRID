"""Descriptive natural-beam cohorts from accepted exports; no inference or tuning."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re

from src.runtime_cpu import THREAD_VARIABLES
from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot

MODELS = ('sid', 'listwise', 'pointwise', 'separate')
BASELINES = ('sid', 'pointwise', 'separate')
GROUPS = ('neither', 'baseline_only', 'challenger_only', 'both')
WIDTH = 21  # Rank 0 means GT absent from the complete natural beam@20.


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def guard():
    if any(os.environ.get(name) != '1' for name in THREAD_VARIABLES):
        raise RuntimeError('Library thread limits must be set before Python starts')
    return check_export_allocation()


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'Metric reconstruction mismatch: {a} vs {b}')


def check_training_seed(seed, config_text):
    # Read only the unambiguous top-level integer in the frozen Hydra YAML.
    values = re.findall(r'^seed:[ \t]*([0-9]+)[ \t]*$', config_text, re.MULTILINE)
    if type(seed) is not int or seed not in (42, 43, 44) or values != [str(seed)]:
        raise ValueError('Training seed does not match the frozen Hydra config')


def accepted_metrics(accepted, seed):
    if 'independently_recomputed_metrics' in accepted:
        return accepted['independently_recomputed_metrics']
    if seed == 42 and 'verified_metrics' in accepted:
        return accepted['verified_metrics']  # Original seed42 receipt schema; do not rewrite it.
    raise ValueError('Missing accepted per-variant metrics')


def summarize_matrix(matrix, population):
    if population <= 0 or len(matrix) != WIDTH or any(len(r) != WIDTH for r in matrix):
        raise ValueError('Invalid rank matrix geometry or population')
    if any(type(n) is not int or n < 0 for row in matrix for n in row):
        raise ValueError('Rank counts must be nonnegative integers')
    count = sum(map(sum, matrix))
    if count > population:
        raise ValueError('Cohort exceeds population')
    result = {'num_examples': count, 'population_fraction': count / population, 'metrics': {}}
    for k in (5, 10):
        for metric in ('recall', 'ndcg'):
            value = [0.] + [(float(r <= k) if metric == 'recall' else
                            (1 / math.log2(r + 1) if r <= k else 0.)) for r in range(1, WIDTH)]
            a = math.fsum(n * value[i] for i, row in enumerate(matrix) for n in row)
            b = math.fsum(n * value[j] for row in matrix for j, n in enumerate(row))
            result['metrics'][f'{metric}@{k}'] = {
                'baseline_mean': a / count if count else None,
                'challenger_mean': b / count if count else None,
                'conditional_delta': (b - a) / count if count else None,
                'population_delta_contribution': (b - a) / population,
            }
    return result


def ranks(target, candidates):
    import numpy as np
    if candidates.ndim != 3 or candidates.shape[1] != 20 or target.shape != (len(candidates), candidates.shape[2]):
        raise ValueError('Expected exact SID targets and a beam of 20 candidates')
    hit = np.all(candidates == target[:, None, :], axis=2)
    if np.any(hit.sum(1) > 1):
        raise ValueError('Duplicate GT in candidates')
    return np.where(hit.any(1), hit.argmax(1) + 1, 0)


def accumulate(hist, baseline, challenger):
    import numpy as np
    group = (baseline > 0).astype(np.int64) + 2 * (challenger > 0)
    index = group * WIDTH * WIDTH + baseline * WIDTH + challenger
    hist += np.bincount(index, minlength=4 * WIDTH * WIDTH).reshape(4, WIDTH, WIDTH)


def run(manifest_path, output):
    budget = guard()  # Must precede numeric imports.
    snapshots = [export_cpu_snapshot(budget['allowed_cpus'])]
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == os.environ['COHORT_MANIFEST_SHA256']
    assert manifest['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT']
    seed = manifest['training_seed']
    assert type(seed) is int and seed in (42, 43, 44) and manifest['expected_samples'] == 2061101
    for record in manifest['sources'] + [manifest['paired_report'], manifest['paired_acceptance']]:
        assert sha(record['path']) == record['sha256'], record['path']
    report = json.loads(Path(manifest['paired_report']['path']).read_text())
    accepted = json.loads(Path(manifest['paired_acceptance']['path']).read_text())
    assert accepted['status'] == 'passed' and accepted['report_sha256'] == manifest['paired_report']['sha256']
    n = report['num_examples']
    assert n == accepted['num_samples_per_model'] == manifest['expected_samples']
    assert accepted['cross_model_user_hash_target_equal'] and accepted['duplicate_candidates_per_row'] == 0
    assert set(report['inputs']) == set(MODELS)
    verified_metrics = accepted_metrics(accepted, seed)
    variants = {'sid': 'beam'}
    for model in MODELS:
        path = Path(report['inputs'][model]); identity = accepted['models'][model]
        assert sha(path) == identity['h5_sha256'] and sha(path.with_suffix('.json')) == identity['sidecar_sha256']
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['split'] == 'test' and meta['num_samples'] == n
        assert meta['checkpoint_sha256'] == identity['checkpoint_sha256']
        assert meta['limits'] == {'max_batches': None, 'max_samples': None}
        config_path = Path(meta['run_dir']) / '.hydra/config.yaml'
        assert sha(config_path) == meta['hydra_config_sha256']
        check_training_seed(seed, config_path.read_text())
        if model != 'sid':
            selection = report['fusion_selections'][model]
            assert sha(selection['path']) == selection['sha256'] == identity['selection_sha256']
            selected = json.loads(Path(selection['path']).read_text())['selected']
            assert selected['variant'] == selection['variant'] == 'fusion@1@0.50'
            variants[model] = selection['variant']
        print('[bound]', model, flush=True)
    import h5py
    import numpy as np
    from scripts.analyze_paired_evaluation import ranked_candidates, metric_contributions

    hist = {m: np.zeros((4, WIDTH, WIDTH), dtype=np.int64) for m in BASELINES}
    sums = {m: {f'{metric}@{k}': 0. for k in (5, 10) for metric in ('recall', 'ndcg')} for m in MODELS}
    coverage = dict.fromkeys(MODELS, 0)
    with ExitStack() as stack:
        files = {m: stack.enter_context(h5py.File(report['inputs'][m], 'r')) for m in MODELS}
        assert all(len(f['user_id']) == n for f in files.values())
        for start in range(0, n, 32768):
            stop = min(start + 32768, n)
            payload = {m: {k: f[k][start:stop] for k in
                       ('user_id', 'example_hash', 'target_sid', 'beam_sid', 'beam_score', 'raw_rerank_logit')}
                       for m, f in files.items()}
            rr = {}
            for m, p in payload.items():
                assert all(np.array_equal(p[k], payload['sid'][k]) for k in ('user_id', 'example_hash', 'target_sid'))
                assert np.isfinite(p['beam_score']).all() and np.isfinite(p['raw_rerank_logit']).all()
                candidates = ranked_candidates(p, variants[m])
                rr[m] = ranks(p['target_sid'], candidates)
                natural = ranks(p['target_sid'], p['beam_sid'])
                assert np.array_equal(rr[m] > 0, natural > 0)
                coverage[m] += int((natural > 0).sum())
                # Independent rank computation by score comparisons and original-index tie breaking.
                if m != 'sid':
                    score = p['beam_score'] + .5 * p['raw_rerank_logit']
                    assert np.isfinite(score).all()
                    index = np.maximum(natural - 1, 0)
                    target_score = score[np.arange(len(index)), index][:, None]
                    before = (score > target_score) | ((score == target_score) & (np.arange(20) < index[:, None]))
                    assert np.array_equal(rr[m], np.where(natural > 0, before.sum(1) + 1, 0))
                for k in (5, 10):
                    for metric, values in metric_contributions(p['target_sid'], candidates, k).items():
                        sums[m][f'{metric}@{k}'] += float(values.sum())
            assert np.array_equal(payload['sid']['beam_sid'], payload['separate']['beam_sid'])
            assert np.array_equal(payload['sid']['beam_score'], payload['separate']['beam_score'])
            for baseline in BASELINES:
                accumulate(hist[baseline], rr[baseline], rr['listwise'])
            if start == 0 or stop == n:
                snapshots.append(export_cpu_snapshot(budget['allowed_cpus']))
                print('[scanned]', stop, '/', n, flush=True)
    result = {}
    for m in MODELS:
        assert coverage[m] == accepted['models'][m]['coverage_count_at_20']
        for metric, total in sums[m].items():
            close(total / n, verified_metrics[m + ':' + variants[m]][metric])
    for baseline, counts in hist.items():
        assert int(counts.sum()) == n
        groups = {name: summarize_matrix(counts[i].tolist(), n) for i, name in enumerate(GROUPS)}
        overall = summarize_matrix(counts.sum(0).tolist(), n)
        existing = next(c for c in report['comparisons'] if c['baseline'] == baseline + ':' + variants[baseline]
                        and c['challenger'] == 'listwise:' + variants['listwise'])
        for metric, values in overall['metrics'].items():
            close(values['baseline_mean'], sums[baseline][metric] / n)
            close(values['challenger_mean'], sums['listwise'][metric] / n)
            close(values['conditional_delta'], existing['metrics'][metric]['absolute_gain'])
            close(math.fsum(g['metrics'][metric]['population_delta_contribution'] for g in groups.values()),
                  values['conditional_delta'])
        result[baseline + '_vs_listwise'] = {
            'baseline_variant': variants[baseline], 'challenger_variant': variants['listwise'],
            'groups': groups, 'overall': overall, 'rank_histograms': counts.tolist(),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(status='passed', checked_at=datetime.now().astimezone().isoformat(),
                   training_seed=seed, num_examples=n, comparisons=result, coverage=coverage,
                   source_git_commit=manifest['source_git_commit'], manifest_sha256=sha(manifest_path),
                   manifest_path=str(manifest_path.resolve()), cpu_budget=budget, cpu_process_tree_samples=snapshots,
                   scope='Descriptive post-hoc natural-beam cohorts; no new inference, selection, CI or causal claim',
                   cohort_order=list(GROUPS), rank_zero='GT absent from natural beam@20',
                   validation='Input hashes/alignment; independent score-count rank; per-user vs rank-histogram metrics; accepted full-test metric reconstruction',
                   limitations=['Both-hit is not a shared identical candidate pool',
                                'Model-dependent groups are descriptive, not a causal generator/reranker decomposition',
                                f'Single training seed{seed}; no history-length or item-semantic explanation in this stage'])
    with output.open('x') as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('[passed]', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)
