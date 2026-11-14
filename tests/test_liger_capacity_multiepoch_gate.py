import copy
import json
from pathlib import Path

import pytest

from scripts import probe_liger_capacity_multiepoch as probe


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/baselines/liger_capacity384_multiepoch_seed42_v1.json').read_text())


def test_diagnostic_census_never_mutates_formal_budget():
    c = config()
    before = copy.deepcopy(c)
    small = probe.diagnostic_config(c)
    assert c == before
    assert small['training']['prefixes_per_epoch'] == 6144
    assert small['validation']['records'] == 65
    assert small['training']['max_optimizer_steps'] == 269520
    assert c['training']['prefixes_per_epoch'] == 45996113


def test_synthetic_policy_checks_minimum_patience_and_cap():
    r = probe.policy_evidence(config())
    assert r['plateau_stops_at'] == 6 and r['improving_stops_at'] == 12
    assert r['patience'] == 5 and r['serialized_resume_preserves_policy']


@pytest.mark.parametrize('kind,gpus,limit', [('gate',2,'00:45:00'), ('train',1,'12:00:00'), ('val',2,'06:00:00')])
def test_launchers_keep_cpu4_no_job_chain_and_bound_gpu(kind,gpus,limit):
    p = Path(__file__).resolve().parents[1]/f'run_liger_capacity_multiepoch_{kind}_cpu4.sh'
    s = p.read_text()
    assert '#SBATCH --cpus-per-task=4' in s and '#SBATCH --ntasks=1' in s
    assert f'#SBATCH --gres=gpu:{gpus}' in s and f'#SBATCH --time={limit}' in s
    assert '#SBATCH --mem=24G' in s and '--cpu-bind=verbose,threads' in s
    assert 'LIGER_MULTIEPOCH_MANIFEST_SHA256' in s
    active = '\n'.join(line for line in s.splitlines() if not line.startswith('#'))
    assert 'sbatch' not in active and 'scancel' not in active
    if kind == 'gate':
        assert 'run_bound 2 -m scripts.probe_liger_capacity_multiepoch validation' in s
        assert 'run_bound 1 -m scripts.probe_liger_capacity_multiepoch cross_resumed' in s


def pair(cross=False):
    steps = [dict(attempt=i,rows=2048,microbatch=1024,microbatches=2,gradient_tensors=152,
        skipped=False,candidate_classes=34700,sid_loss=2.,dense_loss=3.,total_loss=5.)
        for i in ([4,5] if cross else [1,2,3])]
    cp = {'path':'diagnostic.pt','sha256':'a'*64,'full_state_sha256':'b'*64}
    a = dict(status='passed',steps=steps,final_progress={'prefix_count':10240 if cross else 6144},
        final_state_sha256='c'*64,initial_state_sha256='d'*64,checkpoint_binding={},gpus=[],clock={},
        manifest_sha256='e'*64,job_id='1',source_git_commit='f'*40,
        initial_progress={'epoch':1 if cross else 0,'next_batch':0,'updates':0,'attempts':0},
        step_one_checkpoint=cp)
    if cross:
        a.update(loaded_checkpoint=cp,restore_proof={'full_state_exact':True,'full_state_sha256':'b'*64})
        a['final_progress'].update(epoch=1,next_batch=2)
    b = copy.deepcopy(a)
    b.update(loaded_checkpoint=cp,restore_proof={'full_state_exact':True,'full_state_sha256':'b'*64})
    if not cross:
        b['steps'] = b['steps'][1:]
        b['initial_progress'].update(updates=1,attempts=1)
    return a,b


@pytest.mark.parametrize('cross',[False,True])
def test_exact_pair_and_timing_exclusion(cross):
    a,b=pair(cross)
    b['steps'][0]['seconds']=99.
    probe.verify_pair(a,b,cross=cross)


@pytest.mark.parametrize('cross',[False,True])
@pytest.mark.parametrize('field',['final_state_sha256','checkpoint_binding','job_id','manifest_sha256'])
def test_pair_rejects_state_or_source_drift(cross,field):
    a,b=pair(cross)
    b[field]='changed'
    with pytest.raises(AssertionError): probe.verify_pair(a,b,cross=cross)


@pytest.mark.parametrize('cross',[False,True])
def test_pair_rejects_changed_next_data_or_loss(cross):
    a,b=pair(cross)
    b['steps'][0]['sid_loss']=10.
    with pytest.raises(AssertionError): probe.verify_pair(a,b,cross=cross)
