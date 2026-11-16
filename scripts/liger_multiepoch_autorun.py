"""External, fail-closed scheduler for the already frozen LIGER experiment.

This does not change the runners' historical automatic_successor=False fields:
the runners still never submit jobs. A separately authorized cron controller
waits for both successful jobs AND verified full-validation artifacts.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'logs/training_launch/liger_capacity384_multiepoch_seed42_2026-11-14_v1/manifest.json'
MANIFEST_SHA = '40c3c21b4fe7f61125dbccd736bab4b85bb930b21484818cc26415757f717e9d'
SOURCE_COMMIT = 'da8486eaba3a880321b6b848df770643d9c951cf'
PREFIX = 'GRID-L384AUTO-40c3c21b4fe7'
PYTHON = '/home/MMReco2021/.conda/envs/GRID/bin/python'
NEW_SOURCES = ('scripts/liger_multiepoch_autorun.py', 'scripts/liger_autorun_evidence.py',
               'scripts/liger_autorun_slurm.py', 'run_liger_multiepoch_autorun_cpu1.sh')
LEGACY_NAMES = {'GRID-LIGER-384MultiTrain42', 'GRID-LIGER-384MultiVal42'}
LIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING',
        'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'SIGNALING', 'STAGE_OUT'}
EXECUTING = {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'SIGNALING', 'STAGE_OUT'}
FAILED = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL',
          'DEADLINE', 'PREEMPTED', 'REVOKED', 'SPECIAL_EXIT'}


class SafetyError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise SafetyError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    """Runtime state, not a replacement for any experiment artifact."""
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def locked(directory):
    with (Path(directory) / 'controller.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / 'state.json'

    def save(self, state, event, **details):
        state['checked_at'] = now()
        state['last_event'] = event
        atomic_json(self.path, state)
        with (self.directory / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(at=state['checked_at'], event=event,
                epoch=state['epoch'], status=state['status'], **details), ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())


def verify_authorization(path, expected_sha):
    require(sys.flags.optimize == 0, 'Python optimization would disable inherited audit assertions')
    path = Path(path).resolve()
    require(sha(path) == expected_sha, 'External authorization bytes changed')
    a = read(path)
    require(a['version'] == 1 and a['root'] == str(ROOT), 'Wrong controller root/version')
    require(a['manifest'] == str(MANIFEST) and a['manifest_sha256'] == MANIFEST_SHA, 'Wrong experiment')
    require(a['source_commit'] == SOURCE_COMMIT and a['token_prefix'] == PREFIX, 'Wrong experiment identity')
    require(a['user'] == 'MMReco2021' and a['python'] == PYTHON, 'Wrong execution owner/interpreter')
    require(a['policy'] == dict(max_epochs=12, min_epochs=6, patience=5), 'Unauthorized stopping policy')
    require(a['initial_pair'] == dict(epoch=2, train_job='162259', val_job='162260'), 'Wrong attachment')
    require(a['no_test'] is True and a['no_other_seeds'] is True, 'Scope expansion forbidden')
    require(sha(MANIFEST) == MANIFEST_SHA, 'Frozen manifest changed')
    m = read(MANIFEST)
    expected = dict(m['execution_sources'])
    expected.update({str(ROOT / name): a['sources'].get(str(ROOT / name)) for name in NEW_SOURCES})
    require(a['sources'] == expected, 'Missing/extra controller source bindings')
    for filename, digest in a['sources'].items():
        require(isinstance(digest, str) and sha(filename) == digest, f'Execution source changed: {filename}')
    require(sha(m['formal_config']['path']) == m['formal_config']['sha256'], 'Training config changed')
    c = read(m['formal_config']['path'])
    require({k: c['early_stopping'][k] for k in ('max_epochs', 'min_epochs', 'patience')} == a['policy'], 'Policy drift')
    return a


def backend(a):
    from scripts.liger_autorun_slurm import Slurm
    return Slurm(ROOT, a['user'], Path(a['manifest']), a['source_commit'],
        a['manifest_sha256'], a['token_prefix'], a['since'])


def evaluate(a, epoch, train_job, val_job):
    env = dict(os.environ, SOURCE_GIT_COMMIT=a['source_commit'],
        LIGER_MULTIEPOCH_MANIFEST_SHA256=a['manifest_sha256'], NUMPY_MADVISE_HUGEPAGE='0',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='', USE_TF='0')
    result = subprocess.run([a['python'], '-m', 'scripts.liger_autorun_evidence', a['manifest'],
        '--epoch', str(epoch), '--train-job', train_job, '--validation-job', val_job],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600, check=False)
    require(result.returncode == 0, 'Validation evidence audit failed: ' + result.stderr[-3000:])
    proof = json.loads(result.stdout)
    require(proof['status'] == 'accepted' and proof['epoch'] == epoch, 'Wrong epoch evidence')
    require(proof['train_job'] == train_job and proof['validation_job'] == val_job, 'Evidence job mismatch')
    require(proof['manifest_sha256'] == a['manifest_sha256'], 'Evidence experiment mismatch')
    return proof


def scoped(active, a):
    return [j for j in active if j['name'] in LEGACY_NAMES or j['name'].startswith(a['token_prefix'])]


def validate_state(s, auth_sha):
    require(s['version'] == 1 and s['authorization_sha256'] == auth_sha, 'Controller state binding mismatch')
    require(type(s['enabled']) is bool, 'Invalid enable flag')
    require(s['status'] in {'watching', 'completed', 'halted', 'submission_uncertain',
            'submitting', 'waiting_scheduler', 'ready_next'}, 'Unknown controller status')
    require(type(s['epoch']) is int and 2 <= s['epoch'] <= 12, 'Epoch out of authorized range')
    for key in ('train_job', 'val_job'):
        require(s[key] is None or re.fullmatch(r'[1-9][0-9]*', s[key]), 'Invalid job ID')
    require(s['train_job'] is not None or s['val_job'] is None, 'Validation without training')
    if s['epoch'] == 2:
        require(s['train_job'] == '162259' and s['val_job'] == '162260', 'Existing pair must never be replaced')
    else:
        p = s['continuation']
        require(p is not None and p['status'] == 'accepted' and p['epoch'] == s['epoch'] - 1,
                'No accepted previous epoch')
        require(p['stopped'] is False and p['next_epoch'] == s['epoch'], 'Continuation past stopping boundary')


def validate_job(slurm, info, epoch, role, train_job):
    try:
        slurm.verify_job(info, epoch, role, expected_train_id=train_job if role == 'val' else None,
                        allow_legacy=epoch == 2)
    except Exception as error:
        raise SafetyError('Submitted job identity/resource mismatch: ' + str(error)) from error


def require_continuation(s):
    p = s['continuation']
    require(s['epoch'] >= 3 and p['epoch'] == s['epoch'] - 1 and p['stopped'] is False,
            'Submission without accepted continuation')
    for key in ('acceptance', 'checkpoint'):
        ref = p[key]
        require(sha(ref['path']) == ref['sha256'], 'Accepted continuation artifact changed')


def reconcile_intent(s, slurm, store):
    """An ambiguous sbatch is NEVER retried. Adopt only one exact matching job."""
    intent = s['pending_intent']
    if not intent:
        return True
    require(intent['epoch'] == s['epoch'] and intent['role'] in ('train', 'val'), 'Invalid submission intent')
    require(intent['token'] == slurm.token(s['epoch'], intent['role']), 'Submission token mismatch')
    found = slurm.lookup(intent['token'])
    require(len(found) <= 1, 'Multiple jobs match one submission intent; manual inspection required')
    if not found:
        s['status'] = 'submission_uncertain'
        store.save(s, 'awaiting_submission_reconciliation', token=intent['token'])
        return False
    job = found[0]
    if job['state'] == 'UNKNOWN':
        s['status'] = 'submission_uncertain'
        store.save(s, 'waiting_matched_submission_metadata', token=intent['token'])
        return False
    validate_job(slurm, job, s['epoch'], intent['role'], s['train_job'])
    key = intent['role'] + '_job'
    require(s[key] in (None, job['job_id']), 'Conflicting persisted job ID')
    s[key], s['pending_intent'], s['status'] = job['job_id'], None, 'watching'
    store.save(s, 'adopted_submission', role=intent['role'], job_id=job['job_id'])
    return True


def submit_one(a, s, slurm, store, role):
    from scripts.liger_autorun_slurm import SubmissionUncertain
    require(s[role + '_job'] is None and s['pending_intent'] is None, 'Duplicate submission forbidden')
    require_continuation(s)
    token = slurm.token(s['epoch'], role)
    require(not slurm.lookup(token), 'Submission token already exists without state; inspect before continuing')
    known = {j for j in (s['train_job'], s['val_job']) if j is not None}
    active = scoped(slurm.active(), a)
    require(all(j['job_id'] in known for j in active), 'Unexpected project job prevents submission')
    require(len(active) <= (0 if role == 'train' else 1), 'One-running/one-dependent queue bound exceeded')
    s['pending_intent'] = dict(epoch=s['epoch'], role=role, token=token, created_at=now())
    s['status'] = 'submitting'
    store.save(s, 'submission_intent', role=role, token=token)
    try:
        job_id = slurm.submit(s['epoch'], role, token, train_job=s['train_job'] if role == 'val' else None)
    except SubmissionUncertain as error:
        s['status'] = 'submission_uncertain'
        s['pending_intent']['error'] = str(error)
        store.save(s, 'submission_uncertain', message=str(error))
        return False
    require(isinstance(job_id, str) and re.fullmatch(r'[1-9][0-9]*', job_id), 'Invalid submission response')
    s[role + '_job'], s['pending_intent'], s['status'] = job_id, None, 'watching'
    store.save(s, 'submitted', role=role, job_id=job_id)
    return True


def start_pair(a, s, slurm, store):
    require(s['epoch'] >= 3, 'Never resubmit the attached second epoch')
    if s['train_job'] is None and not submit_one(a, s, slurm, store, 'train'):
        return
    if s['val_job'] is None:
        info = slurm.job(s['train_job'])
        if info['state'] == 'UNKNOWN':
            store.save(s, 'waiting_training_job_metadata')
            return
        validate_job(slurm, info, s['epoch'], 'train', None)
        require(info['state'] in LIVE | {'COMPLETED'}, 'New training job failed or has unknown state; no validation submitted')
        if info['state'] == 'COMPLETED':
            require(info['exit_code'] == '0:0', 'New training exited nonzero; no validation submitted')
        submit_one(a, s, slurm, store, 'val')


def advance_once(a, s, slurm, store, evaluator=evaluate):
    """At most one new pair per invocation. Pure orchestration, no training."""
    if not s['enabled'] or s['status'] in ('completed', 'halted'):
        return
    if not reconcile_intent(s, slurm, store):
        return
    known = {j for j in (s['train_job'], s['val_job']) if j is not None}
    active = scoped(slurm.active(), a)
    require(all(j['job_id'] in known for j in active), 'Unexpected project jobs; no automatic submissions')
    require(len(active) <= 2, 'More than one training/one dependent validation')
    if sum(j['state'] in EXECUTING for j in active) > 1:
        # Scheduler snapshots are not atomic across cleanup/dependency release.
        # Do not submit anything until a later tick confirms a single live job.
        s['status'] = 'waiting_scheduler'
        store.save(s, 'concurrency_guard_waiting_no_submission')
        return
    if s['train_job'] is None or s['val_job'] is None:
        start_pair(a, s, slurm, store)
        return
    infos = [(role, slurm.job(s[role + '_job'])) for role in ('train', 'val')]
    for role, info in infos:
        if info['state'] == 'UNKNOWN':
            s['status'] = 'waiting_scheduler'
            store.save(s, 'waiting_accounting', role=role)
            return
        validate_job(slurm, info, s['epoch'], role, s['train_job'])
        require(info['state'] not in FAILED, f'{role} job {info["job_id"]} ended {info["state"]}; no retry')
        require(info['state'] in LIVE | {'COMPLETED'}, 'Unrecognized scheduler state; manual review required')
        if info['state'] == 'COMPLETED':
            require(info['exit_code'] == '0:0', f'{role} exited nonzero')
    if infos[1][1]['state'] in EXECUTING:
        if infos[0][1]['state'] != 'COMPLETED':
            refreshed = slurm.job(s['train_job'])
            if refreshed['state'] != 'UNKNOWN':
                validate_job(slurm, refreshed, s['epoch'], 'train', None)
                require(refreshed['state'] not in FAILED, 'Training failed while validation is executing')
            if refreshed['state'] != 'COMPLETED':
                s['status'] = 'waiting_scheduler'
                store.save(s, 'waiting_nonatomic_dependency_transition')
                return
            require(refreshed['exit_code'] == '0:0', 'Training exited nonzero')
            infos[0] = ('train', refreshed)
    if active or any(info['state'] != 'COMPLETED' for _, info in infos):
        s['status'] = 'watching'
        store.save(s, 'waiting_jobs', jobs={role: info['state'] for role, info in infos})
        return
    proof = evaluator(a, s['epoch'], s['train_job'], s['val_job'])
    require(proof['status'] == 'accepted' and proof['epoch'] == s['epoch'], 'Invalid acceptance')
    require(proof['train_job'] == s['train_job'] and proof['validation_job'] == s['val_job'], 'Wrong accepted jobs')
    require(proof['manifest_sha256'] == a['manifest_sha256'], 'Wrong accepted experiment')
    require(proof['policy']['completed_epochs'] == s['epoch'], 'Policy epoch mismatch')
    require(proof['stopped'] == proof['policy']['stopped'], 'Policy stopping mismatch')
    s['history'].append(dict(epoch=s['epoch'], train_job=s['train_job'], val_job=s['val_job'],
                            acceptance=proof['acceptance'], stopped=proof['stopped']))
    s['latest_acceptance'] = proof
    if proof['stopped']:
        require(proof['next_epoch'] is None and 6 <= s['epoch'] <= 12, 'Premature metric stopping')
        s['status'], s['stop_reason'] = 'completed', proof['stop_reason']
        store.save(s, 'completed', stop_reason=s['stop_reason'], best=proof['best'])
        return
    require(s['epoch'] < 12 and proof['next_epoch'] == s['epoch'] + 1, 'Budget exceeded')
    s.update(epoch=proof['next_epoch'], train_job=None, val_job=None,
             continuation=proof, status='ready_next', pending_intent=None)
    store.save(s, 'accepted_previous_epoch')
    start_pair(a, s, slurm, store)


def initialize(directory):
    directory = Path(directory).resolve()
    require(directory == MANIFEST.parent / 'autorun', 'Controller state must stay inside this experiment')
    require(not directory.exists(), 'Controller already exists; never initialize twice')
    require(sha(MANIFEST) == MANIFEST_SHA, 'Wrong manifest')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Commit controller implementation before activation')
    m = read(MANIFEST)
    sources = dict(m['execution_sources'])
    for name in NEW_SOURCES:
        sources[str(ROOT / name)] = sha(ROOT / name)
    a = dict(version=1, root=str(ROOT), manifest=str(MANIFEST), manifest_sha256=MANIFEST_SHA,
        source_commit=SOURCE_COMMIT, controller_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        user='MMReco2021', python=PYTHON, token_prefix=PREFIX, since='2026-11-16T00:00:00',
        policy=dict(max_epochs=12, min_epochs=6, patience=5), sources=sources,
        initial_pair=dict(epoch=2, train_job='162259', val_job='162260'),
        no_test=True, no_other_seeds=True, created_at=now(),
        authorization='User approved external automatic serial continuation after accepted epoch2; existing jobs untouched')
    slurm = backend(a)
    active = scoped(slurm.active(), a)
    require({j['job_id'] for j in active} <= {'162259', '162260'}, 'Unexpected project jobs at activation')
    for role, job_id in (('train', '162259'), ('val', '162260')):
        info = slurm.job(job_id)
        require(info['state'] in LIVE | {'COMPLETED'}, 'Attached job unavailable or failed')
        validate_job(slurm, info, 2, role, '162259')
    directory.mkdir(mode=0o700)
    auth_path = directory / 'authorization.json'
    atomic_json(auth_path, a)
    digest = sha(auth_path)
    verify_authorization(auth_path, digest)
    state = dict(version=1, authorization_sha256=digest, epoch=2, train_job='162259', val_job='162260',
        enabled=True, status='watching', pending_intent=None, continuation=None, history=[],
        latest_acceptance=None, stop_reason=None, created_at=now())
    Store(directory).save(state, 'attached_existing_pair')
    print(json.dumps(dict(authorization=str(auth_path), authorization_sha256=digest, status='initialized')))


def tick(auth_path, expected_sha, *, pause=False):
    a = verify_authorization(auth_path, expected_sha)
    directory = Path(auth_path).resolve().parent
    with locked(directory) as acquired:
        if not acquired:
            print(json.dumps(dict(status='lock_busy_no_action')))
            return
        store = Store(directory)
        s = read(store.path)
        validate_state(s, expected_sha)
        if pause:
            s['enabled'] = False
            store.save(s, 'paused_no_jobs_cancelled')
        else:
            from scripts.liger_autorun_slurm import SlurmError
            try:
                advance_once(a, s, backend(a), store)
            except SlurmError as error:
                s['status'] = 'waiting_scheduler'
                store.save(s, 'scheduler_query_failed_no_submission', message=str(error))
            except Exception as error:
                s['status'], s['error'] = 'halted', f'{type(error).__name__}: {error}'
                store.save(s, 'halted_no_jobs_cancelled', message=s['error'])
        print(json.dumps({k: s.get(k) for k in ('status', 'enabled', 'epoch', 'train_job', 'val_job',
              'checked_at', 'last_event', 'stop_reason', 'error')}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('init').add_argument('directory', type=Path)
    for name in ('tick', 'pause', 'status'):
        q = sub.add_parser(name)
        q.add_argument('authorization', type=Path)
        q.add_argument('--authorization-sha256', required=True)
    args = p.parse_args()
    if args.command == 'init':
        initialize(args.directory)
    elif args.command == 'status':
        verify_authorization(args.authorization, args.authorization_sha256)
        print(json.dumps(read(args.authorization.parent/'state.json'), ensure_ascii=False, indent=2))
    else:
        tick(args.authorization, args.authorization_sha256, pause=args.command == 'pause')


if __name__ == '__main__':
    main()
