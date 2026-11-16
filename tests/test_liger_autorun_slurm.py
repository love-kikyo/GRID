"""No real Slurm writes: uncertain submissions are never retried."""
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from scripts import liger_autorun_slurm as s


ROOT = Path('/project/GRID')
MANIFEST = ROOT/'logs/training_launch/formal/manifest.json'
PREFIX = 'GRID-L384AUTO-40c3c21b4fe7'
COMMIT, SHA = 'a'*40, 'b'*64


class Runner:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        expected, value = self.responses.pop(0)
        assert Path(args[0]).name == expected
        if isinstance(value, Exception):
            raise value
        if isinstance(value, str):
            return SimpleNamespace(returncode=0, stdout=value, stderr='')
        return SimpleNamespace(returncode=value[0], stdout=value[1], stderr=value[2])


def backend(responses=()):
    runner = Runner(responses)
    return s.Slurm(ROOT, 'MMReco2021', MANIFEST, COMMIT, SHA, PREFIX,
                   '2026-11-16T00:00:00', runner=runner), runner


def name(epoch=3, role='train', legacy=False):
    return s._LEGACY[role] if legacy else f'{PREFIX}-e{epoch:02d}-{role}'


def submitline(epoch=3, role='train', legacy=False, parent='100'):
    args = ['sbatch', '--parsable', '--export=ALL,SOURCE_GIT_COMMIT='+COMMIT+
            ',LIGER_MULTIEPOCH_MANIFEST_SHA256='+SHA]
    if role == 'val':
        args += ['--dependency=afterok:'+parent, '--kill-on-invalid-dep=yes']
    if not legacy:
        args += ['--job-name='+name(epoch, role), '--comment='+name(epoch, role)]
    args += [s._SCRIPTS[role], str(MANIFEST), str(epoch)]
    return shlex.join(args)


def accounting(identifier='101', epoch=3, role='train', state='COMPLETED',
               exit_code='0:0', legacy=False, **overrides):
    values = dict(identifier=identifier, state=state, exit_code=exit_code,
        name=name(epoch, role, legacy), comment='' if legacy else name(epoch, role),
        user='MMReco2021', workdir=str(ROOT), submit=submitline(epoch, role, legacy),
        allocated='4', requested='4', memory='24G', tres='cpu=4,mem=24G,node=1,gres/gpu='+('1' if role=='train' else '2'),
        limit='720' if role=='train' else '360', partition='rtx4090')
    values.update(overrides)
    return '|'.join(values.values())+'\n'


def control(identifier='101', epoch=3, role='train', state='RUNNING', legacy=False,
            **overrides):
    values = dict(JobId=identifier, JobName=name(epoch,role,legacy), UserId='MMReco2021(1018)',
        JobState=state, ExitCode='0:0', NumCPUs='4', MinMemoryNode='24G',
        TRES='cpu=4,mem=24G,gres/gpu='+('1' if role=='train' else '2'),
        TimeLimit='12:00:00' if role=='train' else '06:00:00', Partition='rtx4090',
        Command=f'{ROOT/s._SCRIPTS[role]} {MANIFEST} {epoch}', WorkDir=str(ROOT),
        Dependency='afterok:100(unfulfilled)' if role=='val' else '(null)',
        KillOInInvalidDependent='Yes' if role=='val' else 'No')
    if not legacy:
        values['Comment']=name(epoch,role)
    values.update(overrides)
    return ' '.join(k+'='+v for k,v in values.items())+'\n'


MISSING = (1, '', 'slurm_load_jobs error: Invalid job id specified\n')


def test_active_and_empty_queue():
    b,r=backend([('squeue', '101|other|RUNNING|(null)\n102_2|array|PENDING|note\n'), ('squeue','')])
    assert b.active()==[
        dict(job_id='101',name='other',job_name='other',state='RUNNING',comment=''),
        dict(job_id='102_2',name='array',job_name='array',state='PENDING',comment='note')]
    assert b.active()==[]
    assert '--user' in r.calls[0][0]


@pytest.mark.parametrize('out', ['bad\n', '1|n|RUNNING|x\n1|n|RUNNING|x\n'])
def test_malformed_queue_fails_closed(out):
    b,_=backend([('squeue',out)])
    with pytest.raises(s.SlurmError):b.active()


@pytest.mark.parametrize('response', [(1,'','private error'), subprocess.TimeoutExpired('squeue',30), OSError('missing')])
def test_query_errors_fail_closed_without_retry(response):
    b,r=backend([('squeue',response)])
    with pytest.raises(s.SlurmError):b.active()
    assert len(r.calls)==1


@pytest.mark.parametrize('role,state', [('train','RUNNING'),('val','PENDING')])
def test_live_legacy_attach_has_exact_command_and_resources(role,state):
    b,_=backend([('scontrol',control(epoch=2,role=role,state=state,legacy=True)),
                 ('sacct',accounting(epoch=2,role=role,state=state,legacy=True))])
    info=b.job('101')
    assert info['state']==state and info['memory_bytes']==24*2**30
    b.verify_job(info,2,role,expected_train_id='100' if role=='val' else None,allow_legacy=True)


@pytest.mark.parametrize('state,code', [('COMPLETED','0:0'),('FAILED','1:0'),
    ('CANCELLED by 1018','0:15'),('TIMEOUT','0:15'),('OUT_OF_MEMORY','0:125')])
def test_accounting_fallback_keeps_terminal_exit_codes(state,code):
    b,_=backend([('scontrol',MISSING),('sacct',accounting(state=state,exit_code=code))])
    info=b.job('101');assert info['state']==state.split()[0] and info['exit_code']==code
    b.verify_job(info,3,'train')


def test_missing_record_is_unknown_not_completed():
    b,_=backend([('scontrol',MISSING),('sacct','')])
    info=b.job('101');assert info['state']=='UNKNOWN'
    with pytest.raises(s.SlurmError):b.verify_job(info,3,'train')


@pytest.mark.parametrize('account', ['',accounting(state='RUNNING')])
def test_completed_waits_for_accounting_confirmation(account):
    b,_=backend([('scontrol',control(state='COMPLETED')),('sacct',account)])
    assert b.job('101')['state']=='UNKNOWN'


def test_live_without_accounting_can_be_verified():
    b,_=backend([('scontrol',control(epoch=2,role='val',state='PENDING',legacy=True)),('sacct','')])
    info=b.job('101');b.verify_job(info,2,'val',expected_train_id='100',allow_legacy=True)
    info['dependency']='afterany:100';
    with pytest.raises(s.SlurmError):b.verify_job(info,2,'val',expected_train_id='100',allow_legacy=True)


@pytest.mark.parametrize('overrides', [dict(user='other'),dict(memory='12G'),dict(requested='8'),
    dict(tres='cpu=4,mem=24G,gres/gpu=2'),dict(partition='other'),dict(limit='600')])
def test_control_accounting_disagreement_stops(overrides):
    b,_=backend([('scontrol',control()),('sacct',accounting(state='RUNNING',**overrides))])
    with pytest.raises(s.SlurmError):b.job('101')


@pytest.mark.parametrize('key,value', [('user','other'),('workdir','/other'),('requested_cpus',8),
    ('allocated_cpus',8),('memory_bytes',48*2**30),('gpus',2),('time_limit_minutes',1440),
    ('partition','other'),('name','different'),('comment','different')])
def test_wrong_job_identity_rejected(key,value):
    b,_=backend([('scontrol',MISSING),('sacct',accounting())])
    info=b.job('101');info[key]=value
    with pytest.raises(s.SlurmError):b.verify_job(info,3,'train')


@pytest.mark.parametrize('change', ['epoch','manifest','script','extra_argument','source','manifest_sha','missing_export'])
def test_command_and_export_binding_verified(change):
    b,_=backend([('scontrol',MISSING),('sacct',accounting())]);info=b.job('101')
    line=info['submit_line']
    if change=='epoch':line=line[:-1]+'4'
    elif change=='manifest':line=line.replace(str(MANIFEST),'/wrong/manifest.json')
    elif change=='script':line=line.replace(s._SCRIPTS['train'],'other.sh')
    elif change=='extra_argument':line+=' --resume-entry /wrong.json'
    elif change=='source':line=line.replace(COMMIT,'c'*40)
    elif change=='manifest_sha':line=line.replace(SHA,'d'*64)
    elif change=='missing_export':line=' '.join(x for x in line.split() if not x.startswith('--export='))
    info['submit_line']=line
    with pytest.raises(s.SlurmError):b.verify_job(info,3,'train')


@pytest.mark.parametrize('change', ['dependency','kill_invalid'])
def test_validation_dependency_is_verified_from_accounting(change):
    b,_=backend([('scontrol',MISSING),('sacct',accounting(role='val'))]);info=b.job('101')
    if change=='dependency':info['submit_line']=info['submit_line'].replace('afterok:100','afterany:100')
    else:info['submit_line']=info['submit_line'].replace('--kill-on-invalid-dep=yes','')
    with pytest.raises(s.SlurmError):b.verify_job(info,3,'val',expected_train_id='100')


def test_lookup_deduplicates_active_accounting_and_ignores_unrelated_cpu_steps():
    token=name();b,r=backend([
        ('squeue',f'101|{token}|RUNNING|{token}\n'),
        ('sacct',accounting(state='RUNNING')+accounting(identifier='22',name='cpu',comment='',submit='sbatch cpu.sh',tres='cpu=1',memory='',limit='')+
            accounting(identifier='101.batch')),
        ('scontrol',control()),('sacct',accounting(state='RUNNING'))])
    result=b.lookup(token);assert len(result)==1 and result[0]['job_id']=='101'
    b.verify_job(result[0],3,'train')


def test_lookup_retains_duplicate_submissions_for_controller_to_halt():
    b,_=backend([('squeue',''),('sacct',accounting()+accounting(identifier='102')),
        ('scontrol',MISSING),('sacct',accounting()),('scontrol',MISSING),('sacct',accounting(identifier='102'))])
    assert len(b.lookup(name()))==2


def test_lookup_does_not_create_missing_job():
    b,r=backend([('squeue',''),('sacct','')]);assert b.lookup(name())==[]
    assert len(r.calls)==2


def test_accounting_uses_saved_submit_line_when_comment_storage_is_disabled():
    b,_=backend([('scontrol',MISSING),('sacct',accounting(comment=''))])
    info=b.job('101');assert info['comment']==name()
    b.verify_job(info,3,'train')


def test_lookup_recovers_terminal_job_with_no_stored_comment():
    b,_=backend([('squeue',''),('sacct',accounting(comment='')),
        ('scontrol',MISSING),('sacct',accounting(comment=''))])
    found=b.lookup(name());assert len(found)==1 and found[0]['job_id']=='101'


def test_accounting_comment_conflict_fails_closed():
    b,_=backend([('scontrol',MISSING),('sacct',accounting(comment='conflicting'))])
    with pytest.raises(s.SlurmError):b.job('101')


def test_live_pending_dependency_cannot_be_changed_despite_saved_submit_line():
    b,_=backend([('scontrol',control(role='val',state='PENDING',Dependency='afterany:100')),
                 ('sacct',accounting(role='val',state='PENDING'))])
    info=b.job('101')
    with pytest.raises(s.SlurmError):b.verify_job(info,3,'val',expected_train_id='100')


def test_satisfied_pending_dependency_may_be_cleared_by_scheduler():
    b,_=backend([('scontrol',control(role='val',state='PENDING',Dependency='(null)')),
                 ('sacct',accounting(role='val',state='PENDING'))])
    b.verify_job(b.job('101'),3,'val',expected_train_id='100')


@pytest.mark.parametrize('line', [f'101|wrong|RUNNING|{name()}\n', f'101|{name()}|RUNNING|(null)\n'])
def test_lookup_collision_fails_closed(line):
    b,_=backend([('squeue',line),('sacct','')])
    with pytest.raises(s.SlurmError):b.lookup(name())


@pytest.mark.parametrize('role', ['train','val'])
def test_submit_is_one_explicit_bounded_command(role):
    b,r=backend([('sbatch','200\n')]);token=name(role=role)
    assert b.submit(3,role,token,train_job='100' if role=='val' else None)=='200'
    args,kw=r.calls[0];assert len(r.calls)==1 and args[0]=='/usr/bin/sbatch'
    assert '--comment='+token in args and '--job-name='+token in args
    assert '--cpus-per-task=4' in args and '--mem=24G' in args
    assert '--gres=gpu:'+('1' if role=='train' else '2') in args
    assert args[-3:]==[str(ROOT/s._SCRIPTS[role]),str(MANIFEST),'3']
    if role=='val':assert '--dependency=afterok:100' in args and '--kill-on-invalid-dep=yes' in args
    assert kw['timeout']==30 and kw['cwd']==str(ROOT) and 'shell' not in kw
    assert kw['env']['LC_ALL']=='C'


@pytest.mark.parametrize('response', [(1,'','some scheduler failure'), '123;cluster\n',
    '123\n124\n','',subprocess.TimeoutExpired('sbatch',30),OSError('missing')])
def test_uncertain_submit_never_retried(response):
    b,r=backend([('sbatch',response)])
    with pytest.raises(s.SubmissionUncertain):b.submit(3,'train',name())
    assert len(r.calls)==1


@pytest.mark.parametrize('epoch,role,token,parent', [(2,'train',name(2),None),
    (13,'train',name(12),None),(3,'train','wrong',None),(3,'val',name(role='val'),None),
    (3,'train',name(),'100'),(3,'val',name(role='val'),'1; touch x')])
def test_invalid_submit_has_no_external_call(epoch,role,token,parent):
    b,r=backend()
    with pytest.raises(s.SlurmError):b.submit(epoch,role,token,parent)
    assert not r.calls


@pytest.mark.parametrize('value,cpus,want', [('24G',4,24*2**30),('24576M',4,24*2**30),
    ('24Gn',4,24*2**30),('6Gc',4,24*2**30),('24576',4,24*2**30)])
def test_memory_units(value,cpus,want):
    assert s._memory(value,cpus)==want


def test_invalid_job_ids_never_query():
    b,r=backend()
    for value in ('1.batch','1,2','0','1;bad'):
        with pytest.raises(s.SlurmError):b.job(value)
    assert not r.calls


def test_submit_strips_inherited_sbatch_resource_overrides(monkeypatch):
    monkeypatch.setenv('SBATCH_GPUS','8')
    monkeypatch.setenv('SBATCH_EXCLUSIVE','1')
    b,r=backend([('sbatch','200\n')])
    b.submit(3,'train',name())
    assert not any(key.startswith('SBATCH_') for key in r.calls[0][1]['env'])


def test_submit_does_not_export_controller_gpu_mask_or_parent_slurm_allocation(monkeypatch):
    for key,value in {'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'none',
                      'ROCR_VISIBLE_DEVICES':'','SLURM_JOB_ID':'123',
                      'SLURM_CPUS_ON_NODE':'1','SOURCE_GIT_COMMIT':COMMIT,
                      'LIGER_MULTIEPOCH_MANIFEST_SHA256':SHA}.items():
        monkeypatch.setenv(key,value)
    b,r=backend([('sbatch','200\n')]);b.submit(3,'train',name())
    args,kwargs=r.calls[0];env=kwargs['env']
    assert not any(key.startswith('SLURM_') for key in env)
    assert all(key not in env for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
    assert env['SOURCE_GIT_COMMIT']==COMMIT and env['LIGER_MULTIEPOCH_MANIFEST_SHA256']==SHA
    assert '--export=ALL,SOURCE_GIT_COMMIT='+COMMIT+',LIGER_MULTIEPOCH_MANIFEST_SHA256='+SHA in args
