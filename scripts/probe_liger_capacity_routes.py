"""GPU-only synthetic branch coverage supplement; no data reads or training."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from scripts.probe_liger_cpu4 import ROOT, git, sha, verify_hashes, write_new
from scripts.liger_capacity_manifest import verify as verify_parent

PARENT = ROOT / 'logs/training_launch/liger_capacity384_2026-11-12_v1/manifest.json'
PARENT_SHA = 'cb3096f0678e2015fe7349114903c84fee889959bdb7c7003b78963faf226d33'
SOURCES = ('scripts/probe_liger_capacity_routes.py',
           'tests/test_liger_sharded_cuda_routes.py', 'run_liger_capacity_routes_cpu4.sh')


def verify(path):
    m = json.loads(path.read_text())
    assert set(m) == {'phase', 'source_git_commit', 'execution_sources', 'parent',
                      'validation_acceptance', 'formal_training', 'dataset_records_read'}
    assert m['phase'] == 'synthetic_cuda_beam_route_coverage_only'
    assert m['formal_training'] is False and m['dataset_records_read'] is False
    assert len(m['source_git_commit']) == 40
    assert m['parent'] == dict(path=str(PARENT), sha256=PARENT_SHA)
    assert sha(PARENT) == PARENT_SHA
    parent = json.loads(PARENT.read_text())
    verify_parent(parent)
    assert set(m['execution_sources']) == {str(ROOT / name) for name in SOURCES}
    verify_hashes(m['execution_sources'])
    receipt = PARENT.parent / 'validation_acceptance.json'
    assert m['validation_acceptance'] == dict(path=str(receipt), sha256=sha(receipt))
    accepted = json.loads(receipt.read_text())
    assert accepted['status'] == 'passed_resource_validation' and accepted['job_id'] == '160308'
    assert accepted['manifest_sha256'] == PARENT_SHA
    assert sha(accepted['report']['path']) == accepted['report']['sha256']
    return m


def prepare(path):
    assert not path.exists() and not git(ROOT, 'status', '--porcelain')
    receipt = PARENT.parent / 'validation_acceptance.json'
    m = dict(phase='synthetic_cuda_beam_route_coverage_only', source_git_commit=git(ROOT, 'rev-parse', 'HEAD'),
        execution_sources={str(ROOT / name): sha(ROOT / name) for name in SOURCES},
        parent=dict(path=str(PARENT), sha256=PARENT_SHA),
        validation_acceptance=dict(path=str(receipt), sha256=sha(receipt)),
        formal_training=False, dataset_records_read=False)
    write_new(path, m)
    verify(path)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), source_git_commit=m['source_git_commit'])))


def run(path):
    import pytest
    import torch
    from src.runtime_cpu import configure_environment, configure_torch
    from src.runtime_export_cpu import check_export_allocation, export_cpu_snapshot
    assert sha(path) == os.environ['LIGER_ROUTE_MANIFEST_SHA256']
    m = verify(path)
    assert m['source_git_commit'] == os.environ['LIGER_ROUTE_SOURCE_COMMIT']
    output = path.parent / 'cuda_route_acceptance.json'
    assert not output.exists()
    configure_environment()
    configure_torch(torch)
    resources = check_export_allocation()
    assert torch.cuda.device_count() == 2
    assert all('4090' in torch.cuda.get_device_name(i) for i in range(2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpus = [dict(name=torch.cuda.get_device_name(i),
                 uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(2)]

    class Reports:
        def __init__(self):
            self.collected, self.passed, self.skipped = [], [], []

        def pytest_collection_finish(self, session):
            self.collected = [item.nodeid for item in session.items]

        def pytest_runtest_logreport(self, report):
            if report.skipped: self.skipped.append(report.nodeid)
            if report.when == 'call' and report.passed: self.passed.append(report.nodeid)

    reports = Reports()
    before = export_cpu_snapshot(resources['allowed_cpus'])
    code = pytest.main(['-q', '-s', str(ROOT / SOURCES[1])], plugins=[reports])
    assert code == 0 and reports.collected and reports.passed == reports.collected and not reports.skipped
    after = export_cpu_snapshot(resources['allowed_cpus'])
    verify(path)
    write_new(output, dict(status='passed_cuda_route_coverage', job_id=os.environ['SLURM_JOB_ID'],
        manifest_sha256=sha(path), source_git_commit=m['source_git_commit'],
        tests=reports.passed, skipped=reports.skipped, resources=resources, gpus=gpus, cpu_checks=[before, after],
        dataset_records_read=False, formal_training=False,
        scope='synthetic CUDA routing only; not new recommendation metrics or full-catalog timing',
        checked_at=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else run)(args.manifest.resolve())
