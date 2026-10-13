"""Verify SIM validation exports against accepted same-seed input identities."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import h5py
import numpy as np

from scripts.export_paired_evaluation import file_sha256


def check_candidates(arrays):
    n, k, h = arrays['beam_sid'].shape
    assert (k, h) == (20, 4)
    for name in ('target_sid', 'beam_sid', 'rerank_sid'):
        a = arrays[name]
        assert np.issubdtype(a.dtype, np.integer) and (a >= 0).all() and (a < 4096).all()
    for name in ('beam_score', 'raw_rerank_logit', 'rerank_score'):
        assert arrays[name].shape == (n, k) and np.isfinite(arrays[name]).all()
    assert (np.diff(arrays['beam_score'], axis=1) <= 1e-6).all()
    assert (np.diff(arrays['rerank_score'], axis=1) <= 1e-6).all()
    powers = np.array([4096**3, 4096**2, 4096, 1], dtype=np.int64)
    beam = np.sort((arrays['beam_sid'].astype(np.int64) * powers).sum(-1), axis=1)
    rerank = np.sort((arrays['rerank_sid'].astype(np.int64) * powers).sum(-1), axis=1)
    assert (np.diff(beam, axis=1) > 0).all(), 'Duplicate beam candidates'
    np.testing.assert_array_equal(beam, rerank)
    return int((arrays['beam_sid'] == arrays['target_sid'][:, None, :]).all(-1).any(-1).sum())


def accept(output, checkpoint_acceptance, reference_selection, smoke):
    m = json.loads(output.with_suffix('.json').read_text())
    checkpoint = json.loads(checkpoint_acceptance.read_text())
    reference = json.loads(reference_selection.read_text())
    expected = 64 if smoke else 347416
    assert checkpoint['status'] == 'passed'
    assert m['num_samples'] == expected and m['split'] == 'val' and m['schema_version'] == 3
    assert m['limits'] == {'max_batches': 2 if smoke else None, 'max_samples': None}
    assert m['amp'] == 'bf16' and m['batch_size'] == 32 and m['num_workers'] == 1
    assert m['checkpoint_sha256'] == checkpoint['checkpoint']['sha256']
    assert m['checkpoint'] == checkpoint['checkpoint']['path']
    assert m['hydra_config_sha256'] == checkpoint['configuration']['sha256']
    assert m['git_commit'] == os.environ['SOURCE_GIT_COMMIT']
    assert m['slurm_job_id'] == os.environ['SLURM_JOB_ID']
    provenance = m['dataset_provenance']
    assert provenance['protocol'] == 'history-last-user-v1' and provenance['expected_samples'] == 347416
    assert provenance['source_compatibility'] is None
    assert provenance['training_preflight']['sha256'] == checkpoint['training_preflight']['sha256']
    for entry in m['source_files'] + list(provenance['manifests'].values()):
        assert file_sha256(entry['path']) == entry['sha256'], entry['path']
    for path, digest in provenance['verified_training_sources'].items():
        assert file_sha256(path) == digest, path
    assert m['cpu_budget']['allocated_cpus'] == 4 and m['cpu_budget']['library_threads'] == 1
    cpus = m['cpu_budget']['allowed_cpus']
    assert 0 < len(cpus) <= 4 and len(m['cpu_process_tree_samples']) == 2
    for sample in m['cpu_process_tree_samples']:
        assert sample['allowed_cpus'] == cpus and sample['processes']
        assert all(p['threads_checked'] > 0 for p in sample['processes'])
    ref_path = Path(reference['input'])
    assert reference['input_split'] == 'val' and reference['num_examples'] == 347416
    assert file_sha256(ref_path) == reference['input_sha256']
    assert file_sha256(ref_path.with_suffix('.json')) == reference['input_sidecar_sha256']
    ref_meta = json.loads(ref_path.with_suffix('.json').read_text())
    assert ref_meta['dataset_provenance']['initialization_checkpoint']['sha256'] == checkpoint['teacher_checkpoint']['sha256']
    same_beam_rows, score_differences, hits = 0, 0, 0
    max_score_difference = 0.
    with h5py.File(output, 'r') as f, h5py.File(ref_path, 'r') as ref:
        shapes = dict(user_id=(expected,), example_hash=(expected, 2), target_sid=(expected, 4),
                      beam_sid=(expected, 20, 4), rerank_sid=(expected, 20, 4),
                      beam_score=(expected, 20), raw_rerank_logit=(expected, 20), rerank_score=(expected, 20))
        assert set(f.keys()) == set(shapes)
        assert all(f[k].shape == shape for k, shape in shapes.items())
        assert f['user_id'].dtype == np.int64 and f['example_hash'].dtype == np.uint64
        ids = f['user_id'][:]
        assert len(np.unique(ids)) == expected and len(np.unique(f['example_hash'][:], axis=0)) == expected
        ref_ids = ref['user_id'][:]
        assert len(np.unique(ref_ids)) == 347416
        order = np.argsort(ref_ids)
        positions = np.searchsorted(ref_ids[order], ids)
        assert (positions < len(ref_ids)).all()
        aligned = order[positions]
        for key in ('user_id', 'example_hash', 'target_sid'):
            np.testing.assert_array_equal(f[key][:], ref[key][:][aligned], err_msg=key)
        ref_beam, ref_score = ref['beam_sid'][:][aligned], ref['beam_score'][:][aligned]
        for start in range(0, expected, 8192):
            arrays = {key: f[key][start:start + 8192] for key in shapes}
            hits += check_candidates(arrays)
            same_beam_rows += int((arrays['beam_sid'] == ref_beam[start:start + 8192]).all((1, 2)).sum())
            diff = np.abs(arrays['beam_score'] - ref_score[start:start + 8192])
            score_differences += int(np.count_nonzero(diff))
            max_score_difference = max(max_score_difference, float(diff.max()))
    # Different-hardware numerical discrepancies must be reviewed, not silently accepted.
    assert same_beam_rows == expected and score_differences == 0, (
        f'Frozen teacher differs from accepted reference: equal_rows={same_beam_rows}/{expected}, '
        f'score_differences={score_differences}, max_abs={max_score_difference}')
    return dict(status='passed', checked_at=datetime.now().astimezone().isoformat(),
                scope='64-example export smoke' if smoke else 'full validation integrity; not test',
                job_id=os.environ['SLURM_JOB_ID'], source_git_commit=m['git_commit'],
                num_samples=expected, unique_users=expected, natural_beam_hits=hits,
                checkpoint_sha256=m['checkpoint_sha256'], output=str(output),
                h5_sha256=file_sha256(output), sidecar_sha256=file_sha256(output.with_suffix('.json')),
                checkpoint_acceptance_sha256=file_sha256(checkpoint_acceptance),
                reference_selection_sha256=file_sha256(reference_selection),
                reference=str(ref_path), reference_h5_sha256=reference['input_sha256'],
                input_identity_alignment=True, frozen_teacher_beam_exact=True,
                cpu_budget=m['cpu_budget'], audit_script_sha256=file_sha256(__file__))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--checkpoint-acceptance', type=Path, required=True)
    p.add_argument('--reference-selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    assert len(os.sched_getaffinity(0)) == 1 and not args.output.exists()
    result = accept(args.input, args.checkpoint_acceptance, args.reference_selection, args.smoke)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('[sim-validation] ' + json.dumps(result), flush=True)
