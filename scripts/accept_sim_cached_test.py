"""Recover only the CPU audit of SIM test job 139967, without re-exporting.

The immutable exporter passed a JSON string to a Path-only hash helper. This
audit converts those paths explicitly and separates export and audit identity.
No existing predictions, sidecars, source hashes or receipts are overwritten.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from scripts import export_sim_cached_test as original  # Imports torch first.
import h5py

e, cached = original.e, original.cached
EXPORT_JOB = '139967'
EXPORT_COMMIT = 'f481f1c3f2e1a9c8ee0e0ed358023e0b19ce96e8'
EXPORTER_SHA = '20ca8d852a4910a84d41898af29719e85b3996e0d3c335537d2f6befba0ebde9'
SIDECAR_SHA = 'ba7ba8eb0150a5e54fe24b444bd1e0dfcdc3cac209607e47e98c6e327d779a87'


def verify_sources(metadata):
    provenance = metadata['dataset_provenance']
    for entry in metadata['source_files'] + list(provenance['manifests'].values()):
        assert cached.record(entry['path']) == entry, entry['path']
    for path, digest in provenance['verified_training_sources'].items():
        assert e.file_sha256(Path(path)) == digest, path


def verify_export_identity(metadata):
    assert metadata['git_commit'] == EXPORT_COMMIT
    assert metadata['slurm_job_id'] == EXPORT_JOB
    assert metadata['exporter_sha256'] == EXPORTER_SHA


def accept(output, receipt):
    assert len(os.sched_getaffinity(0)) == 1 and not receipt.exists()
    assert os.environ['SLURM_JOB_ID'] != EXPORT_JOB
    assert e.file_sha256(Path(original.__file__)) == EXPORTER_SHA
    assert e.file_sha256(output.with_suffix('.json')) == SIDECAR_SHA
    frozen, previous = original.validate_frozen_validation()
    checkpoint = json.loads(cached.CHECKPOINT_ACCEPTANCE.read_text())
    source = original.validate_test_cache(checkpoint)
    m = json.loads(output.with_suffix('.json').read_text())
    verify_export_identity(m)
    assert m['num_samples'] == original.N and m['split'] == 'test' and m['schema_version'] == 3
    assert m['limits'] == dict(max_batches=None, max_samples=None)
    assert m['amp'] == 'bf16' and m['batch_size'] == 32 and m['num_workers'] == 1
    assert m['checkpoint_sha256'] == checkpoint['checkpoint']['sha256'] == previous['checkpoint_sha256']
    assert m['checkpoint'] == checkpoint['checkpoint']['path']
    assert m['hydra_config_sha256'] == checkpoint['configuration']['sha256']
    assert m['checkpoint_acceptance'] == cached.record(cached.CHECKPOINT_ACCEPTANCE)
    assert m['candidate_source'] == source and m['frozen_selection'] == frozen
    assert m['no_candidate_regeneration'] and m['validation_acceptance'] == frozen['validation_acceptance']
    assert m['beam_log_prob_weight'] == 1. and m['rerank_logit_weight'] == .1
    provenance = m['dataset_provenance']
    assert provenance['protocol'] == 'history-last-user-v1' and provenance['expected_samples'] == original.N
    assert provenance['source_compatibility'] is None
    assert provenance['training_preflight']['sha256'] == checkpoint['training_preflight']['sha256']
    assert provenance['initialization_checkpoint']['sha256'] == checkpoint['teacher_checkpoint']['sha256']
    verify_sources(m)
    assert m['cpu_budget']['allocated_cpus'] == 4 and m['cpu_budget']['library_threads'] == 1
    cpus = m['cpu_budget']['allowed_cpus']
    assert 0 < len(cpus) <= 4 and len(m['cpu_process_tree_samples']) >= 2
    for sample in m['cpu_process_tree_samples']:
        assert sample['allowed_cpus'] == cpus and sample['processes']
        assert all(p['threads_checked'] > 0 for p in sample['processes'])
    print('[sim-test-recovery] immutable source, sidecar, selection and provenance passed', flush=True)
    with h5py.File(output, 'r') as f, h5py.File(original.CACHE, 'r') as reference:
        hits = original.audit_predictions(f, reference, original.N)
    assert hits == json.loads(original.TEST_ACCEPTANCE.read_text())['models']['sid']['coverage_count_at_20']
    result = dict(status='passed', scope='full cached-candidate SIM test integrity; paired statistics pending',
                  checked_at=datetime.now().astimezone().isoformat(), num_samples=original.N, unique_users=original.N,
                  export_job_id=EXPORT_JOB, export_source_git_commit=EXPORT_COMMIT,
                  audit_job_id=os.environ['SLURM_JOB_ID'], audit_source_git_commit=os.environ['SOURCE_GIT_COMMIT'],
                  natural_beam_hits=hits, checkpoint_sha256=m['checkpoint_sha256'], output=str(output),
                  h5_sha256=e.file_sha256(output), sidecar_sha256=SIDECAR_SHA,
                  checkpoint_acceptance=cached.record(cached.CHECKPOINT_ACCEPTANCE),
                  candidate_source=source, frozen_selection=frozen, input_identity_alignment=True,
                  cached_beam_exact=True, teacher_regenerated=False, runtime_fusion_recomputed_exact=True,
                  export_cpu_budget=m['cpu_budget'], audit_cpu_affinity=sorted(os.sched_getaffinity(0)),
                  audit_script_sha256=e.file_sha256(Path(__file__)),
                  recovery=dict(original_exporter_sha256=EXPORTER_SHA,
                      fix='Convert JSON path strings to Path before hashing; bind export and audit jobs separately',
                      predictions_changed=False, sidecar_changed=False, gpu_inference_repeated=False))
    with receipt.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('[sim-test-recovery-acceptance] ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    accept(args.input, args.output)
