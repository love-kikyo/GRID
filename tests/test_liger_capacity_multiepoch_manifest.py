"""Twelve-epoch authority is fresh-only, exactly bound and gated independently."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import liger_capacity_multiepoch_manifest as contract


@pytest.fixture
def bound(tmp_path, monkeypatch):
    files = {}
    for name in ('old.py', 'new.py', 'upstream.py', 'meta.json', 'asset.npy',
                 'catalog.npy', 'keys.npy', 'rows.npy', 'receipt.json'):
        files[name] = tmp_path/name
        files[name].write_text(name)
    config, stats = tmp_path/'new_config.json', tmp_path/'stats.json'
    config.write_bytes(contract.CONFIG.read_bytes())
    stats.write_text(json.dumps(dict(stats_sha256='a'*64, seen_rows_sha256='b'*64)))
    item = lambda p: dict(path=str(p), sha256=contract.sha(p))
    old = dict(phase='capacity384_first_pass_formal', formal_training_ready=True,
        automatic_successor=False, test_records_read=False, dirty_at_submission=False,
        initialization='old one pass', source_git_commit='a'*40,
        execution_sources={str(files['old.py']):contract.sha(files['old.py'])},
        input_metadata={str(files['meta.json']):contract.sha(files['meta.json'])},
        upstream_files={str(files['upstream.py']):contract.sha(files['upstream.py'])},
        upstream_root=str(tmp_path), upstream_commit='c'*40, seed=42,
        config='/old/config.json', config_sha256='d'*64, formal_config={'path':'/old/config.json','sha256':'d'*64},
        capacity_config=item(config), standardization=item(stats),
        assets={'frozen_sid':item(files['asset.npy'])}, catalog=item(files['catalog.npy']),
        index_artifacts={'sid_keys':item(files['keys.npy']), 'asset_rows':item(files['rows.npy'])},
        dense_objective={'num_negatives':32768, 'seed':42},
        expected_initial_state_sha256='e'*64, expected_feature_cache_sha256='f'*64,
        train_files=['frozen_train'], validation_files=['frozen_val'],
        run_directory='/old/run', gate_acceptance={'path':'/old/gate','sha256':'d'*64})
    prior = tmp_path/'prior.json'
    prior.write_text(json.dumps(old))
    receipts = {'training':item(files['receipt.json']), 'validation':item(files['receipt.json'])}
    evidence = {str(prior):contract.sha(prior), str(files['receipt.json']):contract.sha(files['receipt.json'])}
    monkeypatch.setattr(contract, 'ROOT', tmp_path)
    monkeypatch.setattr(contract, 'CONFIG', config)
    monkeypatch.setattr(contract, 'PRIOR', prior)
    monkeypatch.setattr(contract, 'PRIOR_SHA', contract.sha(prior))
    monkeypatch.setattr(contract, 'SOURCES', ['new.py'])
    monkeypatch.setattr(contract, 'predecessor', lambda: (copy.deepcopy(old),copy.deepcopy(receipts),copy.deepcopy(evidence)))
    monkeypatch.setattr(contract, 'git', lambda root,*args: '' if args[0]=='status' else 'b'*40)
    monkeypatch.setenv('NUMPY_MADVISE_HUGEPAGE','0')
    path = tmp_path/'launch/gate/manifest.json'
    m = contract.prepare(path)
    return path,m,files


def gate_receipt(path, m, monkeypatch):
    directory = Path(m['run_directory'])
    directory.mkdir(parents=True)
    report = directory/'continuous.json'
    report_value = dict(status='passed', manifest_sha256=contract.sha(path), job_id='12345')
    report.write_text(json.dumps(report_value))
    refs = [dict(path=str(report),sha256=contract.sha(report))]
    calls = []
    def check(gate, config, root):
        calls.append((gate,config,root))
        assert root == directory and gate['phase'] == contract.GATE_PHASE
        assert config['early_stopping']['patience'] == 5
        assert json.loads(report.read_text()) == report_value
        return refs
    monkeypatch.setitem(sys.modules,'scripts.probe_liger_capacity_multiepoch',SimpleNamespace(verify_gate_reports=check))
    receipt = dict(status='passed_capacity384_multiepoch_gate',phase=contract.GATE_PHASE,
        gate_manifest=dict(path=str(path),sha256=contract.sha(path)), manifest_sha256=contract.sha(path),
        source_git_commit=m['source_git_commit'],formal_config_sha256=m['formal_config']['sha256'],
        execution_sources=m['execution_sources'],initial_state_sha256=m['expected_initial_state_sha256'],
        exact_fresh_process_replay=True,cross_epoch_exact_replay=True,two_gpu_validation_records=65,
        validation_model_weights_unchanged=True,early_stopping_policy_verified=True,
        formal_training_ready=False,formal_checkpoint_produced=False,test_records_read=False,
        automatic_successor=False,reports=refs,job_id='12345')
    target=directory/'gate_acceptance.json'
    target.write_text(json.dumps(receipt))
    return target,receipt,calls


def test_fresh_gate_uses_new_global_runtime_budget(bound):
    path,m,_=bound
    c=contract.verify(m)
    assert m['config']==m['formal_config']['path']
    assert m['config_sha256']==m['formal_config']['sha256']==contract.CONFIG_SHA
    assert c['training']['max_optimizer_steps']==269520
    assert c['early_stopping']['patience']==5
    assert m['phase']==contract.GATE_PHASE and m['gate_acceptance'] is None
    assert 'resume_checkpoint' not in m and 'initial_checkpoint' not in m
    binding=contract.checkpoint_binding(m)
    assert binding['purpose']==contract.GATE_PHASE and binding['training_budget']['max_epochs']==12
    assert binding['early_stopping']['patience']==5
    with pytest.raises(AssertionError):contract.prepare(path)


def test_formal_uses_actual_new_gate_checker_and_distinct_purpose(bound,monkeypatch):
    path,m,_=bound
    target,receipt,calls=gate_receipt(path,m,monkeypatch)
    formal=contract.prepare(path.parent.parent/'formal/manifest.json',phase='formal',gate_receipt=target)
    assert contract.verify(formal)['training']['max_epochs']==12 and calls
    assert formal['formal_training_ready'] is True and formal['phase']==contract.FORMAL_PHASE
    a,b=contract.checkpoint_binding(m),contract.checkpoint_binding(formal)
    assert a.pop('purpose')!=b.pop('purpose') and a==b
    assert formal['run_directory']!=m['run_directory']


@pytest.mark.parametrize('key,value',[
    ('phase','old'),('formal_training_ready',True),('formal_training_ready',0),
    ('automatic_successor',True),('test_records_read',True),('dirty_at_submission',True),
    ('initialization','load old checkpoint'),('source_git_commit','bad'),('seed',43),
    ('expected_initial_state_sha256','9'*64),('run_directory','/tmp/arbitrary'),
    ('gate_acceptance',{}),('config','/old/config.json'),('formal_config',{}),
    ('execution_sources',{}),('input_metadata',{}),('multiepoch_predecessor',{}),
    ('multiepoch_predecessor_acceptances',{}),('config_sha256','9'*64),
])
def test_reject_scope_config_source_or_budget_drift(bound,key,value):
    _,m,_=bound;m[key]=value
    with pytest.raises((AssertionError,ValueError,FileNotFoundError)):contract.verify(m)


@pytest.mark.parametrize('key',['resume_checkpoint','test_files','initial_checkpoint','extra_budget'])
def test_reject_unknown_authority_fields(bound,key):
    _,m,_=bound;m[key]='unauthorized'
    with pytest.raises(AssertionError):contract.verify(m)


@pytest.mark.parametrize('name',['old.py','new.py','upstream.py','meta.json','receipt.json'])
def test_bound_file_bytes_cannot_drift(bound,name):
    _,m,files=bound;files[name].write_text('changed')
    with pytest.raises((AssertionError,ValueError)):contract.verify(m)


@pytest.mark.parametrize('section,key,value',[
    ('training','max_epochs',13),('training','min_epochs',5),('training','max_attempts',22460),
    ('training','warmup_steps',13476),('training','physical_batch_size',512),
    ('training','drop_last',True),('early_stopping','patience',3),('early_stopping','min_delta',.001),
    ('early_stopping','monitor','test Recall@5'),('validation','records',65),
    ('validation','test_access',True),('resources','cpu_ceiling',8),('resources','memory_gb',24.),
])
def test_configuration_values_and_types_frozen(bound,section,key,value):
    _,m,_=bound;c=contract.verify(m);c[section][key]=value
    with pytest.raises(AssertionError):contract.check_config(c)


@pytest.mark.parametrize('key,value',[
    ('exact_fresh_process_replay',False),('cross_epoch_exact_replay',False),
    ('early_stopping_policy_verified',False),('two_gpu_validation_records',64),
    ('validation_model_weights_unchanged',False),('formal_training_ready',True),
    ('formal_checkpoint_produced',True),('test_records_read',True),('automatic_successor',True),
    ('formal_config_sha256','9'*64),('initial_state_sha256','9'*64),('execution_sources',{}),
    ('reports',[]),('job_id',''),('source_git_commit','9'*40),
    ('job_id','other-job'),
])
def test_gate_summary_cannot_bypass_new_proofs(bound,monkeypatch,key,value):
    path,m,_=bound;target,r,_=gate_receipt(path,m,monkeypatch);r[key]=value
    target.write_text(json.dumps(r))
    with pytest.raises((AssertionError,ValueError)):
        contract.prepare(path.parent.parent/'formal/manifest.json',phase='formal',gate_receipt=target)


@pytest.mark.parametrize('phase,receipt',[('formal',None),('gate','oldcheckpoint'),('resume',None)])
def test_prepare_never_infers_new_scope(bound,phase,receipt):
    path,_,_=bound
    with pytest.raises(AssertionError):contract.prepare(path.with_name('other.json'),phase,receipt)


def test_environment_binding_is_exact(bound,monkeypatch):
    path,m,_=bound
    monkeypatch.setenv('LIGER_MULTIEPOCH_MANIFEST_SHA256',contract.sha(path))
    monkeypatch.setenv('SOURCE_GIT_COMMIT',m['source_git_commit'])
    assert contract.binding(path)[0]==m
    monkeypatch.setenv('LIGER_MULTIEPOCH_MANIFEST_SHA256','a'*64)
    with pytest.raises(AssertionError):contract.binding(path)
