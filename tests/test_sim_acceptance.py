import json

import pytest

from scripts import accept_sim_smoke as audit


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    def save(rel, obj):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))
    save('logs/training_launch/sim_soft_seed42_123.json', dict(
        job_id='123', mode='two_step_ddp_smoke', seed=42, source_dirty_at_submit=False,
        source_git_commit='fixed_source', teacher_checkpoint='/frozen/teacher',
        preflight='preflight.json', audit_dir='audit'))
    source = tmp_path / 'source.py'
    source.write_text('immutable')
    save('preflight.json', dict(status='passed', source_files=[
        dict(path=str(source), sha256=audit.digest(source))]))
    for rank in range(3):
        save(f'audit/rank_{rank}.json', dict(status='passed', global_rank=rank,
            world_size=3, optimizer_steps=2, frozen_teacher_unchanged=True,
            teacher_checkpoint='/frozen/teacher', exact_initial_teacher_parameters=10,
            validation_batches=2, validation_metrics={'val/loss': 1.},
            gradient_checks=[{'reranker': 7}] * 2))
        save(f'audit/sim_rank_{rank}.json', dict(status='passed', global_rank=rank,
            natural_candidates_equal_teacher=True, target_invariance=True,
            sim_gradient_checks=[{'gsu': 1, 'esu': 1}] * 2,
            cpu_samples=[{'allowed_cpus': [rank]}] * 2))
    output = tmp_path / 'logs/slurm/GRID-SIM-HLastV1-123.out'
    output.parent.mkdir(parents=True)
    output.write_text('fixed_source\n24 passed in 5s\n')
    output.with_suffix('.err').write_text('')
    return tmp_path


def test_accepts_complete_aligned_receipts(evidence):
    report = audit.accept(123)
    assert report['status'] == 'passed' and report['total_bound_cpus'] == 3
    assert len(report['evidence_sha256']) == 10


@pytest.mark.parametrize('file,field,value', [
    ('rank_0.json', 'frozen_teacher_unchanged', False),
    ('rank_1.json', 'world_size', 1),
    ('sim_rank_2.json', 'target_invariance', False),
    ('sim_rank_1.json', 'cpu_samples', [{'allowed_cpus': [0, 1]}] * 2),
    ('sim_rank_1.json', 'cpu_samples', [{'allowed_cpus': [0]}] * 2),
])
def test_rejects_failed_or_unbounded_evidence(evidence, file, field, value):
    path = evidence / 'audit' / file
    obj = json.loads(path.read_text())
    obj[field] = value
    path.write_text(json.dumps(obj))
    with pytest.raises(AssertionError):
        audit.accept(123)


def test_rejects_changed_source(evidence):
    (evidence / 'source.py').write_text('changed')
    with pytest.raises(AssertionError):
        audit.accept(123)
