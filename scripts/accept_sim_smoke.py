"""Read-only smoke evidence checks, then write a new acceptance receipt.

Uses only the standard library. Does not repeat inference or large-file hashes.
Slurm terminal status must additionally be checked before formal submission.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accept(job_id):
    if not str(job_id).isdigit():
        raise ValueError('Expected numeric job ID')
    launch_path = ROOT / f'logs/training_launch/sim_soft_seed42_{job_id}.json'
    launch = json.loads(launch_path.read_text())
    assert launch['mode'] == 'two_step_ddp_smoke' and launch['seed'] == 42
    assert launch['job_id'] == str(job_id) and launch['source_dirty_at_submit'] is False
    preflight_path = ROOT / launch['preflight']
    preflight = json.loads(preflight_path.read_text())
    assert preflight['status'] == 'passed'
    for item in preflight['source_files']:
        assert digest(Path(item['path'])) == item['sha256'], item['path']
    evidence = {str(launch_path): digest(launch_path), str(preflight_path): digest(preflight_path)}
    cpus = set()
    report_dir = ROOT / launch['audit_dir']
    for rank in range(3):
        base_path, sim_path = report_dir / f'rank_{rank}.json', report_dir / f'sim_rank_{rank}.json'
        base, sim = json.loads(base_path.read_text()), json.loads(sim_path.read_text())
        assert base['status'] == sim['status'] == 'passed'
        assert base['global_rank'] == sim['global_rank'] == rank
        assert base['world_size'] == 3 and base['optimizer_steps'] == 2
        assert base['frozen_teacher_unchanged'] is True
        assert base['teacher_checkpoint'] == launch['teacher_checkpoint']
        assert base['exact_initial_teacher_parameters'] > 0 and base['validation_batches'] >= 1
        assert all(math.isfinite(x) for x in base['validation_metrics'].values())
        assert sim['natural_candidates_equal_teacher'] and sim['target_invariance']
        assert len(base['gradient_checks']) == len(sim['sim_gradient_checks']) == len(sim['cpu_samples']) == 2
        for check in base['gradient_checks'] + sim['sim_gradient_checks']:
            assert all(x > 0 for x in check.values())
        rank_cpus = {tuple(sample['allowed_cpus']) for sample in sim['cpu_samples']}
        assert len(rank_cpus) == 1
        cpu_tuple = rank_cpus.pop()
        assert len(cpu_tuple) == 1 and not cpus.intersection(cpu_tuple)
        cpus.update(cpu_tuple)
        for path in (base_path, sim_path):
            evidence[str(path)] = digest(path)
    stdout = ROOT / f'logs/slurm/GRID-SIM-HLastV1-{job_id}.out'
    stderr = stdout.with_suffix('.err')
    output = stdout.read_text()
    counts = re.findall(r'(\d+) passed', output)
    assert counts and max(map(int, counts)) >= 24
    assert launch['source_git_commit'] in output
    assert 'Traceback (most recent call last)' not in stderr.read_text()
    for path in (stdout, stderr):
        evidence[str(path)] = digest(path)
    return dict(status='passed', job_id=str(job_id), source_git_commit=launch['source_git_commit'],
                world_size=3, optimizer_steps=2, total_bound_cpus=len(cpus),
                cpu_affinity=sorted(cpus), evidence_sha256=evidence,
                scope='source + per-rank smoke receipts; not an effectiveness result',
                scheduler_terminal_status='must be checked separately')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = accept(args.job_id)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'evidence_sha256'}))
