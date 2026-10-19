from pathlib import Path
import subprocess

import pytest

from scripts.accept_sim_cached_test import (e, cached, verify_sources, verify_export_identity,
                                           EXPORT_JOB, EXPORT_COMMIT, EXPORTER_SHA)


def test_json_string_source_paths_are_hashed_without_weakening_verification(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('immutable source\n')
    digest = e.file_sha256(source)
    # Reproduce the original failure explicitly; no inference is involved.
    with pytest.raises(AttributeError, match='open'):
        e.file_sha256(str(source))
    metadata = dict(source_files=[cached.record(source)], dataset_provenance=dict(
        manifests={'test': cached.record(source)}, verified_training_sources={str(source): digest}))
    verify_sources(metadata)
    metadata['dataset_provenance']['verified_training_sources'][str(source)] = 'wrong'
    with pytest.raises(AssertionError):
        verify_sources(metadata)
    metadata['dataset_provenance']['verified_training_sources'][str(source)] = digest
    source.write_text('changed source\n')
    with pytest.raises(AssertionError):
        verify_sources(metadata)


def test_export_identity_is_not_replaced_by_recovery_job(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'different-audit-job')
    monkeypatch.setenv('SOURCE_GIT_COMMIT', 'different-audit-commit')
    metadata = dict(git_commit=EXPORT_COMMIT, slurm_job_id=EXPORT_JOB, exporter_sha256=EXPORTER_SHA)
    verify_export_identity(metadata)
    for key in metadata:
        with pytest.raises(AssertionError):
            verify_export_identity({**metadata, key: 'changed'})


def test_recovery_launcher_has_no_gpu_inference_or_successor():
    path = Path(__file__).resolve().parents[1] / 'run_accept_sim_cached_test_cpu1.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    assert '#SBATCH --cpus-per-task=1' in text and '#SBATCH --gres=gpu:0' in text
    assert '--cpu-bind=verbose,threads' in text
    assert '-m scripts.accept_sim_cached_test' in text
    assert '-m scripts.export_sim_cached_test' not in text and 'sbatch ' not in text
