"""Fixed seed42 SIM/SID/Separate comparisons with independent metric audit.

Use the unchanged paired statistical core and validation choices. No inference,
weight search, cohort/history exploration, or new training is performed.
"""
from argparse import Namespace
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.stats import binomtest

from scripts.accept_sim_fusion import contributions, ORDER, sha
from scripts.analyze_paired_cpu4 import run as paired_run
from scripts.analyze_paired_evaluation import paired_bootstrap_ci_from_counts, markdown_report

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'logs/analysis/paired_evaluation'
SIM_ACCEPTANCE = ROOT / 'logs/training_launch/sim_seed42_test_2026-10-14/cached_full_v1_recovery.json'
OLD_ACCEPTANCE = BASE / 'test_acceptance_seed42_full_v1.json'
OLD_ACCEPTANCE_SHA = 'ace4fcaa5112f44f4b01853e00ed218c800249c20af4e6e2de53e221f9f92564'
INPUTS = dict(sid=BASE / 'sid_seed42_131908_test_full_v1.h5',
              sim=BASE / 'sim_soft_seed42_139478_test_cached_full_v1.h5',
              separate=BASE / 'separate_seed42_132051_test_full_v1.h5')
SELECTIONS = dict(sim=BASE / 'sim_soft_seed42_139478_fusion_val.json',
                  separate=BASE / 'separate_seed42_132051_fusion_val.json')
VARIANTS = dict(sim='fusion@1@0.09', separate='fusion@1@0.50')
COMPARISONS = ['sid:beam,sim:click', 'sid:beam,sim:selected', 'sim:click,sim:selected',
               'sim:selected,separate:selected', 'sim:click,separate:click']
METRICS = ['recall@5', 'ndcg@5', 'recall@10', 'ndcg@10']
N, CHUNK = 2061101, 65536
REPORT = BASE / 'paired_report_sim_seed42_full_v1.json'
PREFLIGHT = BASE / 'paired_sim_seed42_preflight_v1.json'
ACCEPTANCE = BASE / 'paired_sim_seed42_acceptance_v1.json'


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def arguments():
    return Namespace(input=[f'{k}={v}' for k, v in INPUTS.items()],
                     selection=[f'{k}={v}' for k, v in SELECTIONS.items()], compare=COMPARISONS,
                     ks=[5, 10], bootstrap_samples=5000, bootstrap_batch_size=500,
                     confidence=.95, seed=2027, chunk_size=CHUNK, expected_split='test',
                     output_json=REPORT, output_md=REPORT.with_suffix('.md'))


def verify_bindings():
    assert sha(OLD_ACCEPTANCE) == OLD_ACCEPTANCE_SHA
    assert sha(SIM_ACCEPTANCE) == os.environ['SIM_TEST_ACCEPTANCE_SHA256']
    old, sim = json.loads(OLD_ACCEPTANCE.read_text()), json.loads(SIM_ACCEPTANCE.read_text())
    assert old['status'] == sim['status'] == 'passed'
    assert old['num_samples_per_model'] == sim['num_samples'] == N
    assert sim['input_identity_alignment'] and sim['cached_beam_exact']
    assert sim['teacher_regenerated'] is False and sim['runtime_fusion_recomputed_exact']
    assert old['separate_and_sid_beam_sid_and_scores_exactly_equal']
    assert sha(ROOT / 'scripts/accept_sim_cached_test.py') == sim['audit_script_sha256']
    records = {}
    for name, path in INPUTS.items():
        accepted = sim if name == 'sim' else old['models'][name]
        assert sha(path) == accepted['h5_sha256']
        assert sha(path.with_suffix('.json')) == accepted['sidecar_sha256']
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['num_samples'] == N and meta['split'] == 'test'
        assert meta['checkpoint_sha256'] == accepted['checkpoint_sha256']
        records[name] = dict(h5=record(path), sidecar=record(path.with_suffix('.json')),
                             checkpoint_sha256=meta['checkpoint_sha256'])
        if name in SELECTIONS:
            selection_path = SELECTIONS[name]
            expected = sim['frozen_selection']['selection']['sha256'] if name == 'sim' else accepted['selection_sha256']
            assert sha(selection_path) == expected
            s = json.loads(selection_path.read_text())
            assert s['input_split'] == 'val' and s['selection_scope'] == 'validation_only'
            assert s['num_examples'] == 347416 and s['selected']['variant'] == VARIANTS[name]
            assert s['input_checkpoint_sha256'] == meta['checkpoint_sha256']
            assert sha(s['input']) == s['input_sha256']
            assert sha(Path(s['input']).with_suffix('.json')) == s['input_sidecar_sha256']
            records[name]['selection'] = record(selection_path)
    return dict(models=records, sim_acceptance=record(SIM_ACCEPTANCE), old_acceptance=record(OLD_ACCEPTANCE))


def run():
    assert not any(p.exists() for p in (PREFLIGHT, REPORT, REPORT.with_suffix('.md'), ACCEPTANCE))
    bindings = verify_bindings()
    preflight = dict(status='passed', job_id=os.environ['SLURM_JOB_ID'],
                     source_git_commit=os.environ['SOURCE_GIT_COMMIT'], bindings=bindings,
                     comparisons=COMPARISONS, source_files=[record(p) for p in (
                         Path(__file__), ROOT / 'run_sim_paired_cpu4.sh', ROOT / 'scripts/accept_sim_fusion.py')])
    with PREFLIGHT.open('x') as f:
        json.dump(preflight, f, indent=2)
        f.write('\n')
    print('[sim-paired] accepted predictions and validation selections bound', flush=True)
    paired_run(arguments())


def independent_values(payload, variant):
    beam = payload['beam_sid']
    if variant == 'beam':
        # Beam-only follows the stored original order, including score ties.
        scores = np.broadcast_to(-np.arange(beam.shape[1], dtype=np.float32), beam.shape[:2])
    elif variant == 'click':
        scores = payload['raw_rerank_logit']
    else:
        kind, bw, rw = variant.split('@')
        assert kind == 'fusion'
        scores = np.float32(bw) * payload['beam_score'] + np.float32(rw) * payload['raw_rerank_logit']
    values = contributions(payload['target_sid'], beam, scores)
    return {key: values[:, ORDER.index(key)] for key in METRICS}


def resolved_comparisons():
    result = []
    for comparison in COMPARISONS:
        for name, variant in VARIANTS.items():
            comparison = comparison.replace(f'{name}:selected', f'{name}:{variant}')
        result.append(comparison)
    return result


def audit():
    assert len(os.sched_getaffinity(0)) == 1 and not ACCEPTANCE.exists()
    preflight, report = json.loads(PREFLIGHT.read_text()), json.loads(REPORT.read_text())
    assert preflight['status'] == 'passed' and preflight['bindings'] == verify_bindings()
    assert preflight['job_id'] == report['slurm_job_id'] == os.environ['SLURM_JOB_ID']
    assert preflight['source_git_commit'] == report['source_git_commit'] == os.environ['SOURCE_GIT_COMMIT']
    assert preflight['comparisons'] == COMPARISONS
    for entry in preflight['source_files'] + report['source_files']:
        assert record(entry['path']) == entry
    assert report['analyzer_sha256'] == sha(ROOT / 'scripts/analyze_paired_evaluation.py')
    assert report['inputs'] == {k: str(p) for k, p in INPUTS.items()}
    assert report['num_examples'] == N and report['sample_unit'] == 'unique_user_history'
    assert report['bootstrap_samples'] == 5000 and report['bootstrap_batch_size'] == 500
    assert report['bootstrap_method'] == 'paired_nonparametric_multinomial_counts'
    assert report['confidence'] == .95 and report['seed'] == 2027 and report['stream_chunk_size'] == CHUNK
    assert report['cpu_budget']['allocated_cpus'] == 4 and report['cpu_budget']['library_threads'] == 1
    assert len(report['cpu_process_tree_samples']) == 2
    for sample in report['cpu_process_tree_samples']:
        assert sample['allowed_cpus'] == report['cpu_budget']['allowed_cpus']
        assert 0 < len(sample['allowed_cpus']) <= 4 and sample['processes']
    for name in SELECTIONS:
        assert report['fusion_selections'][name] == dict(
            **record(SELECTIONS[name]), variant=VARIANTS[name])
    comparisons = report['comparisons']
    assert [f"{c['baseline']},{c['challenger']}" for c in comparisons] == resolved_comparisons()
    labels = sorted({c[side] for c in comparisons for side in ('baseline', 'challenger')})
    sums = {label: {k: 0. for k in METRICS} for label in labels}
    counts = [{k: Counter() for k in METRICS} for _ in comparisons]
    discord = [{k: [0, 0] for k in METRICS if k.startswith('recall')} for _ in comparisons]
    coverage = 0
    with ExitStack() as stack:
        handles = {name: stack.enter_context(h5py.File(path, 'r')) for name, path in INPUTS.items()}
        for handle in handles.values():
            assert len(handle['user_id']) == len(np.unique(handle['user_id'][:])) == N
        for start in range(0, N, CHUNK):
            stop = min(start + CHUNK, N)
            payloads = {name: {k: handle[k][start:stop] for k in (
                'user_id', 'example_hash', 'target_sid', 'beam_sid', 'beam_score', 'raw_rerank_logit')}
                for name, handle in handles.items()}
            ref = payloads['sid']
            for name, payload in payloads.items():
                for key in ('user_id', 'example_hash', 'target_sid', 'beam_sid', 'beam_score'):
                    np.testing.assert_array_equal(payload[key], ref[key], err_msg=name + ':' + key)
                assert np.isfinite(payload['raw_rerank_logit']).all() and np.isfinite(payload['beam_score']).all()
            coverage += int((ref['beam_sid'] == ref['target_sid'][:, None, :]).all(-1).any(-1).sum())
            values = {}
            for label in labels:
                name, variant = label.split(':', 1)
                values[label] = independent_values(payloads[name], variant)
                for key in METRICS:
                    sums[label][key] += float(values[label][key].sum())
            for index, c in enumerate(comparisons):
                a, b = values[c['baseline']], values[c['challenger']]
                for key in METRICS:
                    delta, num = np.unique(b[key] - a[key], return_counts=True)
                    counts[index][key].update({float(v): int(n) for v, n in zip(delta, num)})
                    if key.startswith('recall'):
                        discord[index][key][0] += int(((a[key] == 0) & (b[key] == 1)).sum())
                        discord[index][key][1] += int(((a[key] == 1) & (b[key] == 0)).sum())
            if stop % (CHUNK * 8) == 0 or stop == N:
                print(f'[sim-paired-audit] samples={stop}/{N}', flush=True)
    assert coverage == 568629
    means = {label: {k: v / N for k, v in entries.items()} for label, entries in sums.items()}
    rng = np.random.default_rng(2027)
    crosses = []
    for index, c in enumerate(comparisons):
        assert c['num_examples'] == N
        for key in METRICS:
            result = c['metrics'][key]
            a, b = means[c['baseline']][key], means[c['challenger']][key]
            np.testing.assert_allclose([result['baseline_mean'], result['challenger_mean'], result['absolute_gain']],
                                       [a, b, b - a], rtol=0, atol=1e-12)
            np.testing.assert_allclose(result['relative_gain_percent'], (b - a) / a * 100, rtol=0, atol=1e-10)
            ci = paired_bootstrap_ci_from_counts(counts[index][key], 5000, 500, .95, rng)
            np.testing.assert_allclose(ci, result['paired_bootstrap_ci'], rtol=0, atol=1e-12)
            assert result['confidence'] == .95
            if ci[0] <= 0 <= ci[1]:
                crosses.append(dict(baseline=c['baseline'], challenger=c['challenger'], metric=key, ci=list(ci)))
            if key.startswith('recall'):
                win, loss = discord[index][key]
                binary = result['paired_binary_test']
                assert binary['baseline_miss_challenger_hit'] == win and binary['baseline_hit_challenger_miss'] == loss
                assert binary['discordant_pairs'] == win + loss
                assert binary['exact_two_sided_pvalue'] == (float(binomtest(win, win + loss, .5).pvalue) if win + loss else 1.)
    assert REPORT.with_suffix('.md').read_text() == markdown_report(report)
    result = dict(status='passed', checked_at=datetime.now().astimezone().isoformat(),
                  job_id=os.environ['SLURM_JOB_ID'], source_git_commit=os.environ['SOURCE_GIT_COMMIT'],
                  num_examples=N, comparisons_verified=5, metric_deltas_and_intervals_verified=20,
                  recall_paired_tests_verified=10, same_inputs_targets_candidates_and_beam_scores=True,
                  natural_beam_hits=coverage, verified_metrics=means, intervals_crossing_zero=crosses,
                  report=record(REPORT), markdown=record(REPORT.with_suffix('.md')), preflight=record(PREFLIGHT),
                  audit_script=record(Path(__file__)),
                  bootstrap_verification='Independent GT outranker metrics/delta counts; original bootstrap helper and RNG order reused',
                  limitations=['Single training seed; user-paired intervals do not measure training-seed variation',
                      'Individual 95% intervals, not simultaneous multiple-comparison coverage',
                      'SIM/Separate differ in objective and architecture; not a single-feature ablation',
                      'Existing split overlap retained; no new history analysis',
                      'Zero stored p-values may reflect floating-point underflow'])
    with ACCEPTANCE.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print('[sim-paired-acceptance] ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    assert sys.argv[1:] in (['run'], ['audit'])
    run() if sys.argv[1] == 'run' else audit()
