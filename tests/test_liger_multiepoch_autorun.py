import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import liger_multiepoch_autorun as ctl


class MemoryStore:
    def __init__(self):
        self.events = []
        self.snapshots = []

    def save(self, state, event, **details):
        self.events.append((event, details))
        self.snapshots.append(copy.deepcopy(state))


class FakeSlurm:
    def __init__(self):
        self.jobs = {}
        self.submissions = []
        self.queries = []
        self.fail_role = None
        self.extra_matches = []

    def token(self, epoch, role):
        return f'{ctl.PREFIX}-e{epoch:02d}-{role}'

    def put(self, job_id, epoch, role, state, *, legacy=False):
        self.jobs[job_id] = dict(job_id=job_id, state=state, exit_code='0:0',
            name=('GRID-LIGER-384MultiTrain42' if role == 'train' else 'GRID-LIGER-384MultiVal42')
                 if legacy else self.token(epoch, role),
            comment='' if legacy else self.token(epoch, role), epoch=epoch, role=role)

    def active(self):
        self.queries.append('active')
        return [j for j in self.jobs.values() if j['state'] in ctl.LIVE]

    def job(self, job_id):
        self.queries.append(('job', job_id))
        return self.jobs.get(job_id, dict(job_id=job_id, state='UNKNOWN'))

    def lookup(self, token):
        self.queries.append(('lookup', token))
        return [j for j in self.jobs.values() if j.get('comment') == token] + self.extra_matches

    def verify_job(self, info, epoch, role, expected_train_id=None, allow_legacy=False):
        assert info['epoch'] == epoch and info['role'] == role
        assert allow_legacy is (epoch == 2)
        if role == 'val':
            assert expected_train_id is not None

    def submit(self, epoch, role, token, train_job=None):
        from scripts.liger_autorun_slurm import SubmissionUncertain
        assert token == self.token(epoch, role)
        assert (train_job is not None) is (role == 'val')
        job_id = str(700000 + len(self.submissions))
        self.submissions.append((epoch, role, token, train_job))
        self.put(job_id, epoch, role, 'RUNNING' if role == 'train' else 'PENDING')
        if self.fail_role == role:
            raise SubmissionUncertain('Response lost after scheduler accepted job')
        return job_id


@pytest.fixture
def context(tmp_path):
    a = dict(token_prefix=ctl.PREFIX, manifest_sha256=ctl.MANIFEST_SHA)
    s = dict(version=1, authorization_sha256='auth', epoch=2, train_job='162259', val_job='162260',
             enabled=True, status='watching', pending_intent=None, continuation=None, history=[],
             latest_acceptance=None, stop_reason=None)
    b = FakeSlurm()
    b.put('162259', 2, 'train', 'RUNNING', legacy=True)
    b.put('162260', 2, 'val', 'PENDING', legacy=True)
    store = MemoryStore()
    artifact = tmp_path / 'accepted.json'
    artifact.write_text('{}')
    ref = dict(path=str(artifact), sha256=ctl.sha(artifact))

    def evidence(a, epoch, train_job, val_job, stopped=False):
        return dict(status='accepted', epoch=epoch, stopped=stopped,
            stop_reason='max_epochs' if epoch == 12 else ('patience' if stopped else None),
            next_epoch=None if stopped else epoch+1,
            policy=dict(completed_epochs=epoch, stopped=stopped), best=dict(epoch=epoch),
            checkpoint=ref, acceptance=ref, train_job=train_job, validation_job=val_job,
            manifest_sha256=ctl.MANIFEST_SHA)
    return a, s, b, store, evidence


def complete_pair(b):
    for j in b.jobs.values():
        j['state'] = 'COMPLETED'


def test_current_jobs_not_modified_or_duplicated(context):
    a, s, b, store, ev = context
    original = copy.deepcopy(b.jobs)
    for _ in range(3):
        ctl.advance_once(a, s, b, store, ev)
    assert b.jobs == original and not b.submissions
    assert s['epoch'] == 2 and s['status'] == 'watching'


def test_success_starts_exactly_one_train_and_dependent_validation(context):
    a, s, b, store, ev = context
    complete_pair(b)
    ctl.advance_once(a, s, b, store, ev)
    assert s['epoch'] == 3 and len(b.submissions) == 2
    assert b.submissions[0][:2] == (3, 'train')
    assert b.submissions[1][3] == s['train_job']
    assert len(b.active()) == 2 and s['pending_intent'] is None
    ctl.advance_once(a, s, b, store, ev)
    assert len(b.submissions) == 2
    ctl.validate_state(s, 'auth')


@pytest.mark.parametrize('waiting', ['PENDING', 'RUNNING', 'COMPLETING', 'UNKNOWN'])
def test_validation_must_finish_before_audit(context, waiting):
    a, s, b, store, _ = context
    b.jobs['162259']['state'] = 'COMPLETED'
    b.jobs['162260']['state'] = waiting
    def no_audit(*args):
        pytest.fail('Audit must not run before scheduler completion')
    ctl.advance_once(a, s, b, store, no_audit)
    assert not b.submissions


@pytest.mark.parametrize('failed', sorted(ctl.FAILED))
@pytest.mark.parametrize('role', ['train', 'val'])
def test_failed_jobs_never_retry_or_advance(context, failed, role):
    a, s, b, store, ev = context
    b.jobs[s[role+'_job']]['state'] = failed
    with pytest.raises(ctl.SafetyError):
        ctl.advance_once(a, s, b, store, ev)
    assert not b.submissions


def test_nonzero_exit_is_not_success(context):
    a, s, b, store, ev = context
    complete_pair(b)
    b.jobs['162260']['exit_code'] = '1:0'
    with pytest.raises(ctl.SafetyError):
        ctl.advance_once(a, s, b, store, ev)
    assert not b.submissions


def test_missing_evidence_fails_closed(context):
    a, s, b, store, _ = context
    complete_pair(b)
    def fail(*args):
        raise FileNotFoundError('acceptance missing')
    with pytest.raises(FileNotFoundError):
        ctl.advance_once(a, s, b, store, fail)
    assert not b.submissions


def move_epoch(a, s, b, ev, epoch):
    s['epoch'] = epoch
    s['continuation'] = ev(a, epoch-1, 'previous-train', 'previous-val')
    b.put('162259', epoch, 'train', 'COMPLETED')
    b.put('162260', epoch, 'val', 'COMPLETED')


@pytest.mark.parametrize('epoch', [6, 7, 11, 12])
def test_authorized_stop_submits_nothing(context, epoch):
    a, s, b, store, ev = context
    move_epoch(a, s, b, ev, epoch)
    ctl.advance_once(a, s, b, store, lambda *args: ev(*args, stopped=True))
    assert s['status'] == 'completed' and not b.submissions
    ctl.advance_once(a, s, b, store, ev)
    assert not b.submissions


def test_no_epoch13_even_if_bad_receipt_claims_continue(context):
    a, s, b, store, ev = context
    move_epoch(a, s, b, ev, 12)
    with pytest.raises(ctl.SafetyError, match='Budget'):
        ctl.advance_once(a, s, b, store, ev)
    assert not b.submissions


@pytest.mark.parametrize('epoch', [2, 3, 5])
def test_cannot_stop_before_min6(context, epoch):
    a, s, b, store, ev = context
    if epoch != 2:
        move_epoch(a, s, b, ev, epoch)
    else:
        complete_pair(b)
    with pytest.raises(ctl.SafetyError, match='Premature'):
        ctl.advance_once(a, s, b, store, lambda *args: ev(*args, stopped=True))
    assert not b.submissions


@pytest.mark.parametrize('role', ['train', 'val'])
def test_lost_submission_response_is_adopted_not_retried(context, role):
    a, s, b, store, ev = context
    complete_pair(b)
    b.fail_role = role
    ctl.advance_once(a, s, b, store, ev)
    assert s['status'] == 'submission_uncertain' and s['pending_intent']['role'] == role
    assert any(x['pending_intent'] is not None for x in store.snapshots)
    b.fail_role = None
    ctl.advance_once(a, s, b, store, ev)
    assert s['pending_intent'] is None and len(b.submissions) == 2
    assert len(b.active()) == 2
    ctl.advance_once(a, s, b, store, ev)
    assert len(b.submissions) == 2


def test_unknown_submission_outcome_never_resubmits(context):
    a, s, b, store, ev = context
    complete_pair(b)
    s.update(epoch=3, train_job=None, val_job=None, continuation=ev(a, 2, '162259', '162260'),
             pending_intent=dict(epoch=3, role='train', token=b.token(3, 'train')))
    for _ in range(3):
        ctl.advance_once(a, s, b, store, ev)
    assert s['status'] == 'submission_uncertain' and not b.submissions


def test_duplicate_matching_intent_halts(context):
    a, s, b, store, ev = context
    complete_pair(b)
    s.update(epoch=3, train_job=None, val_job=None, continuation=ev(a, 2, '162259', '162260'),
             pending_intent=dict(epoch=3, role='train', token=b.token(3, 'train')))
    b.put('9', 3, 'train', 'RUNNING')
    b.put('10', 3, 'train', 'RUNNING')
    with pytest.raises(ctl.SafetyError, match='Multiple jobs'):
        ctl.advance_once(a, s, b, store, ev)
    assert not b.submissions


def test_unexpected_project_job_blocks_new_submission(context):
    a, s, b, store, ev = context
    b.put('999', 4, 'train', 'RUNNING')
    with pytest.raises(ctl.SafetyError, match='Unexpected'):
        ctl.advance_once(a, s, b, store, ev)
    assert not b.submissions


def test_two_executing_jobs_fail_closed_without_cancelling(context):
    a, s, b, store, ev = context
    b.jobs['162260']['state'] = 'RUNNING'
    ctl.advance_once(a, s, b, store, ev)
    assert s['status'] == 'waiting_scheduler'
    assert len(b.active()) == 2 and not b.submissions


def test_validation_may_not_run_before_training_finished(context):
    a, s, b, store, ev = context
    b.jobs['162259']['state'] = 'PENDING'
    b.jobs['162260']['state'] = 'RUNNING'
    ctl.advance_once(a, s, b, store, ev)
    assert s['status'] == 'waiting_scheduler'
    assert not b.submissions


def test_legal_nonatomic_train_to_validation_transition_does_not_halt(context):
    a, s, b, store, ev = context
    b.jobs['162259']['state'] = 'COMPLETED'
    b.jobs['162260']['state'] = 'RUNNING'
    original = b.job
    reads = []
    def changing(job_id):
        if job_id == '162259' and not reads:
            reads.append(job_id)
            return dict(original(job_id), state='RUNNING')
        return original(job_id)
    b.job = changing
    ctl.advance_once(a, s, b, store, ev)
    assert s['status'] == 'watching' and not b.submissions


def test_unrelated_job_not_touched(context):
    a, s, b, store, ev = context
    b.jobs['999'] = dict(job_id='999', name='unrelated-project', state='RUNNING', comment='')
    ctl.advance_once(a, s, b, store, ev)
    assert b.jobs['999']['state'] == 'RUNNING' and not b.submissions


@pytest.mark.parametrize('status,enabled', [('halted', True), ('completed', True), ('watching', False)])
def test_terminal_or_paused_controller_makes_no_scheduler_calls(context, status, enabled):
    a, s, b, store, ev = context
    s.update(status=status, enabled=enabled)
    ctl.advance_once(a, s, b, store, ev)
    assert not b.queries and not b.submissions


def test_changed_continuation_checkpoint_prevents_submit(context):
    a, s, b, store, ev = context
    complete_pair(b)
    p = ev(a, 2, '162259', '162260')
    Path(p['checkpoint']['path']).write_text('changed')
    with pytest.raises(ctl.SafetyError, match='artifact changed'):
        ctl.advance_once(a, s, b, store, lambda *args: p)
    assert not b.submissions


@pytest.mark.parametrize('mutation', [dict(epoch=13), dict(epoch=1), dict(train_job='other'),
    dict(train_job=None), dict(authorization_sha256='wrong')])
def test_invalid_state_rejected(context, mutation):
    _, s, _, _, _ = context
    s.update(mutation)
    with pytest.raises(ctl.SafetyError):
        ctl.validate_state(s, 'auth')


def test_atomic_state_and_nonblocking_lock(tmp_path):
    with ctl.locked(tmp_path) as first:
        assert first
        with ctl.locked(tmp_path) as second:
            assert not second
        ctl.atomic_json(tmp_path/'state.json', {'n': 1})
        ctl.atomic_json(tmp_path/'state.json', {'n': 2})
        assert ctl.read(tmp_path/'state.json') == {'n': 2}
    with ctl.locked(tmp_path) as acquired:
        assert acquired


def test_changed_authorization_refused_before_read(tmp_path):
    path = tmp_path/'authorization.json'
    path.write_text('{}')
    with pytest.raises(ctl.SafetyError, match='authorization bytes changed'):
        ctl.verify_authorization(path, 'wrong')


def test_pause_does_not_cancel_existing_jobs(context, tmp_path, monkeypatch, capsys):
    a, s, b, _, _ = context
    auth = tmp_path/'authorization.json'
    auth.write_text('{}')
    ctl.atomic_json(tmp_path/'state.json', s)
    monkeypatch.setattr(ctl, 'verify_authorization', lambda *args: a)
    monkeypatch.setattr(ctl, 'backend', lambda a: b)
    ctl.tick(auth, 'auth', pause=True)
    assert ctl.read(tmp_path/'state.json')['enabled'] is False
    assert not b.queries and not b.submissions


def test_evidence_error_is_persisted_without_cancelling_jobs(context, tmp_path, monkeypatch):
    a, s, b, _, _ = context
    complete_pair(b)
    auth = tmp_path/'authorization.json'
    auth.write_text('{}')
    ctl.atomic_json(tmp_path/'state.json', s)
    monkeypatch.setattr(ctl, 'verify_authorization', lambda *args: a)
    monkeypatch.setattr(ctl, 'backend', lambda a: b)
    def fail(*args):
        raise ctl.SafetyError('broken evidence')
    monkeypatch.setattr(ctl, 'advance_once', fail)
    ctl.tick(auth, 'auth')
    assert ctl.read(tmp_path/'state.json')['status'] == 'halted'
    assert not b.submissions
