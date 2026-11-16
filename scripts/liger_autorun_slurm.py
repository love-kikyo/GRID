"""Read-only Slurm inspection and single-attempt, bounded LIGER submissions.

This module never cancels, releases, updates or retries a job. The caller must
persist a submission intent *before* calling submit and reconcile its unique
token after any uncertain response. Shell execution is never used.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess


class SlurmError(RuntimeError):
    """A read failed or a job identity cannot be verified safely."""


class SubmissionUncertain(SlurmError):
    """Submission may have reached Slurm; do not repeat it automatically."""


TERMINAL = frozenset({'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED',
    'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'TIMEOUT'})
_FIELDS = ('JobIDRaw', 'State', 'ExitCode', 'JobName%128', 'Comment%256', 'User',
    'WorkDir%1200', 'SubmitLine%6000', 'AllocCPUS', 'ReqCPUS', 'ReqMem',
    'ReqTRES%300', 'TimelimitRaw', 'Partition')
_SCRIPTS = {'train': 'run_liger_capacity_multiepoch_train_cpu4.sh',
            'val': 'run_liger_capacity_multiepoch_val_cpu4.sh'}
_LEGACY = {'train': 'GRID-LIGER-384MultiTrain42', 'val': 'GRID-LIGER-384MultiVal42'}


def _require(condition, message):
    if not condition:
        raise SlurmError(message)


def _job_id(value):
    value = str(value)
    _require(bool(re.fullmatch(r'[1-9][0-9]*', value)), 'Invalid numeric Slurm job ID')
    return value


def _state(value):
    # Accounting includes "CANCELLED by 1018"; terminal names are not truncated.
    value = value.strip().split(' ', 1)[0].rstrip('+')
    return value or 'UNKNOWN'


def _comment(value):
    return '' if value in ('', '(null)', 'None') else value


def _submitted_comment(submit_line):
    """This cluster stores SubmitLine but not the accounting Comment field."""
    try:
        values = [part.split('=', 1)[1] for part in shlex.split(submit_line)
                  if part.startswith('--comment=')]
    except ValueError as exc:
        raise SlurmError('Unparseable accounting submission line') from exc
    _require(len(values) <= 1, 'Ambiguous submission comment')
    return values[0] if values else ''


def _memory(value, cpus):
    found = re.fullmatch(r'([0-9]+)([KMGT]?)([cn]?)', value)
    _require(found is not None, 'Unknown Slurm memory format')
    count, unit, scope = found.groups()
    # Unqualified Slurm memory integers are MiB, not bytes.
    result = int(count) * {'': 2**20, 'K': 2**10, 'M': 2**20,
                           'G': 2**30, 'T': 2**40}[unit]
    return result * cpus if scope == 'c' else result


def _gpu_count(tres):
    values = dict(part.split('=', 1) for part in tres.split(',') if '=' in part)
    gpu_values = [int(v) for k, v in values.items()
                  if k == 'gres/gpu' or k.startswith('gres/gpu:')]
    _require(len(set(gpu_values)) <= 1, 'Ambiguous GPU request')
    return gpu_values[0] if gpu_values else 0


def _minutes(value):
    found = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', value)
    _require(found is not None, 'Unknown Slurm time limit')
    days, hours, minutes, seconds = found.groups()
    _require(int(seconds) == 0, 'Unexpected fractional-minute time limit')
    return int(days or 0) * 1440 + int(hours) * 60 + int(minutes)


def _control_fields(line):
    # Values such as Command and WorkDir may contain spaces. Match only field
    # boundaries, then shlex the command separately; never run the parsed text.
    matches = list(re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=', line.strip()))
    return {match.group(1): line.strip()[match.end():
        matches[i + 1].start() if i + 1 < len(matches) else len(line.strip())].strip()
        for i, match in enumerate(matches)}


class Slurm:
    def __init__(self, root, user, manifest, source_commit, manifest_sha,
                 token_prefix, since, *, runner=subprocess.run, timeout=30):
        self.root, self.manifest = Path(root).resolve(), Path(manifest).resolve()
        self.user, self.source_commit, self.manifest_sha = user, source_commit, manifest_sha
        self.token_prefix, self.since = token_prefix, since
        self.runner, self.timeout = runner, timeout
        _require(bool(re.fullmatch(r'[A-Za-z0-9_.-]+', user)), 'Invalid Slurm user')
        _require(bool(re.fullmatch(r'[0-9a-f]{40}', source_commit)), 'Invalid source commit')
        _require(bool(re.fullmatch(r'[0-9a-f]{64}', manifest_sha)), 'Invalid manifest hash')
        _require(bool(re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', token_prefix)), 'Invalid token prefix')
        _require(bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?', since)),
                 'Explicit accounting start date required')
        _require(type(timeout) in (int, float) and 0 < timeout <= 60, 'Invalid command timeout')
        _require(self.manifest.is_relative_to(self.root), 'Manifest outside project')

    def name(self, epoch, role):
        _require(type(epoch) is int and 1 <= epoch <= 12 and role in _SCRIPTS,
                 'Invalid bounded epoch or role')
        return f'{self.token_prefix}-e{epoch:02d}-{role}'

    token = name

    def _call(self, command, args):
        env = dict(os.environ, LC_ALL='C', LANG='C')
        if command == 'sbatch':
            # Neither inherited resource overrides nor the CPU controller's
            # empty CUDA mask may leak into a fresh Slurm allocation.
            hidden_devices = {'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'}
            env = {key: value for key, value in env.items()
                   if not key.startswith(('SBATCH_', 'SLURM_')) and key not in hidden_devices}
        try:
            result = self.runner([f'/usr/bin/{command}', *args], cwd=str(self.root),
                env=env, capture_output=True, text=True, timeout=self.timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SlurmError(f'{command} query unavailable ({type(exc).__name__})') from exc
        _require(isinstance(result.stdout, str) and isinstance(result.stderr, str),
                 'Invalid Slurm subprocess response')
        return result

    def active(self):
        result = self._call('squeue', ['--noheader', '--user', self.user,
                                     '--format=%i|%j|%T|%k'])
        _require(result.returncode == 0, f'squeue query failed ({result.returncode})')
        jobs = []
        for line in result.stdout.splitlines():
            fields = line.split('|')
            _require(len(fields) == 4, 'Malformed squeue record')
            identifier, name, state, comment = fields
            # Other users of the shared account may run array jobs. Preserve
            # their opaque IDs; our tracked IDs remain strictly numeric.
            _require(bool(identifier.strip()), 'Missing squeue job ID')
            jobs.append(dict(job_id=identifier.strip(), name=name, job_name=name,
                state=_state(state), comment=_comment(comment)))
        _require(len({row['job_id'] for row in jobs}) == len(jobs), 'Duplicate active Slurm ID')
        return jobs

    def _accounting(self, identifiers=None):
        args = ['--allocations', '--parsable2', '--noheader', '--user', self.user,
                '--starttime', self.since, '--format=' + ','.join(_FIELDS)]
        if identifiers:
            args.extend(['--jobs', ','.join(_job_id(x) for x in identifiers)])
        result = self._call('sacct', args)
        _require(result.returncode == 0, f'sacct query failed ({result.returncode})')
        jobs = []
        for line in result.stdout.splitlines():
            fields = line.split('|')
            _require(len(fields) == len(_FIELDS), 'Malformed sacct record')
            if not re.fullmatch(r'[1-9][0-9]*', fields[0]):
                continue  # Never adopt .batch/.extern/array-step records.
            (identifier, state, exit_code, name, comment, user, workdir, submit,
             allocated, requested, memory, tres, limit, partition) = fields
            stored_comment = _comment(comment)
            ours = (identifiers is not None or name.startswith(self.token_prefix) or
                    stored_comment.startswith(self.token_prefix) or
                    '--comment=' + self.token_prefix in submit)
            submitted_comment = _submitted_comment(submit) if ours else ''
            _require(not stored_comment or not submitted_comment or stored_comment == submitted_comment,
                     'Accounting comment disagrees with submission token')
            row = dict(job_id=identifier, name=name, job_name=name, state=_state(state),
                exit_code=exit_code, comment=stored_comment or submitted_comment, user=user, workdir=workdir,
                submit_line=submit, partition=partition, source='sacct')
            # The all-account lookup also contains unrelated CPU/array jobs.
            # Only tracked job details need the full resource normalization.
            if identifiers:
                try:
                    requested = int(requested or 0)
                    row.update(allocated_cpus=int(allocated or 0), requested_cpus=requested,
                        memory_bytes=_memory(memory, requested), gpus=_gpu_count(tres),
                        time_limit_minutes=int(limit))
                except (TypeError, ValueError) as exc:
                    raise SlurmError('Malformed accounting resources') from exc
            jobs.append(row)
        _require(len({row['job_id'] for row in jobs}) == len(jobs), 'Duplicate sacct job ID')
        return jobs

    def _control(self, identifier):
        result = self._call('scontrol', ['show', 'job', identifier, '-o'])
        if result.returncode:
            if 'Invalid job id specified' in result.stderr:
                return None
            raise SlurmError(f'scontrol query failed ({result.returncode})')
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        _require(len(lines) == 1, 'Ambiguous scontrol job response')
        values = _control_fields(lines[0])
        _require(values.get('JobId') == identifier, 'scontrol returned a different job')
        try:
            cpus = int(values['NumCPUs'])
            memory = values.get('MinMemoryNode') or values.get('MinMemoryCPU')
            if 'MinMemoryCPU' in values and 'MinMemoryNode' not in values:
                memory += 'c'
            return dict(job_id=identifier, name=values['JobName'], job_name=values['JobName'],
                state=_state(values['JobState']), exit_code=values['ExitCode'],
                comment=_comment(values.get('Comment', '')), user=values['UserId'].split('(')[0],
                workdir=values['WorkDir'], command=shlex.split(values['Command']),
                allocated_cpus=cpus, requested_cpus=cpus, memory_bytes=_memory(memory, cpus),
                gpus=_gpu_count(values['TRES']), time_limit_minutes=_minutes(values['TimeLimit']),
                partition=values['Partition'], dependency=values.get('Dependency', '(null)'),
                kill_invalid_dependency=values.get('KillOInInvalidDependent') == 'Yes',
                source='scontrol')
        except (KeyError, TypeError, ValueError) as exc:
            raise SlurmError('Incomplete scontrol job identity') from exc

    def job(self, job_id):
        identifier = _job_id(job_id)
        controlled = self._control(identifier)
        accounted = self._accounting([identifier])
        _require(len(accounted) <= 1, 'Ambiguous accounting identity')
        account = accounted[0] if accounted else None
        if account:
            _require(account['job_id'] == identifier, 'sacct returned a different job')
        if controlled is None:
            if account:
                return account
            return dict(job_id=identifier, name='', job_name='', state='UNKNOWN',
                        exit_code='', comment='', source='unavailable')
        if account:
            for key in ('job_id', 'name', 'user', 'workdir', 'requested_cpus',
                        'memory_bytes', 'gpus', 'time_limit_minutes', 'partition'):
                _require(controlled[key] == account[key], f'Slurm identity disagreement: {key}')
            # A recently assigned comment may not yet be mirrored to accounting.
            _require(not controlled['comment'] or not account['comment'] or
                     controlled['comment'] == account['comment'], 'Slurm comment disagreement')
            controlled['submit_line'] = account['submit_line']
            controlled['comment'] = controlled['comment'] or account['comment']
            controlled['accounting_state'] = account['state']
            controlled['accounting_exit_code'] = account['exit_code']
            if controlled['state'] in TERMINAL:
                if account['state'] not in TERMINAL:
                    controlled['state'] = 'UNKNOWN'
                else:
                    _require(controlled['state'] == account['state'] and
                             controlled['exit_code'] == account['exit_code'],
                             'Terminal Slurm state disagreement')
        elif controlled['state'] in TERMINAL:
            controlled['state'] = 'UNKNOWN'  # Require accounting confirmation before advancing.
        return controlled

    def lookup(self, token):
        pattern = rf'{re.escape(self.token_prefix)}-e(\d{{2}})-(train|val)'
        matched = re.fullmatch(pattern, token)
        _require(matched is not None, 'Lookup token outside controller namespace')
        epoch, role = int(matched.group(1)), matched.group(2)
        _require(self.token(epoch, role) == token, 'Noncanonical lookup token')
        active = self.active()
        accounted = self._accounting()
        candidates = set()
        for row in active + accounted:
            if row['comment'] == token or row['name'] == token:
                _require(row['name'] == token, 'Token collision with a different job name')
                # squeue comment is authoritative; accounting may briefly lag.
                if row['comment'] == token:
                    candidates.add(row['job_id'])
                else:
                    raise SlurmError('Matching job name lacks the submission token')
        jobs = [self.job(identifier) for identifier in sorted(candidates, key=int)]
        for row in jobs:
            _require(row['name'] == token and row['comment'] == token,
                     'Recovered job no longer matches its submission token')
        return jobs

    def _submit_command(self, epoch, role, token, train_job):
        _require(type(epoch) is int and 3 <= epoch <= 12, 'Automatic submission permits epochs 3–12 only')
        _require(token == self.token(epoch, role), 'Noncanonical submission token')
        _require((role == 'val') == (train_job is not None), 'Validation requires exactly one training dependency')
        args = ['--parsable', '--job-name=' + self.name(epoch, role), '--comment=' + token,
            '--chdir=' + str(self.root), '--partition=rtx4090', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=4', '--gres=gpu:' + ('1' if role == 'train' else '2'),
            '--mem=24G', '--time=' + ('12:00:00' if role == 'train' else '06:00:00'),
            '--export=ALL,SOURCE_GIT_COMMIT=' + self.source_commit +
            ',LIGER_MULTIEPOCH_MANIFEST_SHA256=' + self.manifest_sha]
        if role == 'val':
            args.extend(['--dependency=afterok:' + _job_id(train_job), '--kill-on-invalid-dep=yes'])
        args.extend([str(self.root/_SCRIPTS[role]), str(self.manifest), str(epoch)])
        return args

    def submit(self, epoch, role, token, train_job=None):
        args = self._submit_command(epoch, role, token, train_job)
        try:
            result = self._call('sbatch', args)
        except SlurmError as exc:
            raise SubmissionUncertain('sbatch response unavailable; reconcile token, never resubmit blindly') from exc
        if result.returncode != 0:
            raise SubmissionUncertain(f'sbatch returned {result.returncode}; reconcile token before any action')
        # A cluster-qualified result is intentionally rejected: this controller
        # does not claim authority over a remote/federated cluster.
        value = result.stdout.strip()
        if not re.fullmatch(r'[1-9][0-9]*', value):
            raise SubmissionUncertain('sbatch response ambiguous; reconcile the persisted token')
        return value

    def _submission_identity(self, info, epoch, role, expected_train_id):
        try:
            tokens = shlex.split(info['submit_line'])
        except ValueError as exc:
            raise SlurmError('Unparseable accounting submission line') from exc
        _require(tokens and Path(tokens[0]).name == 'sbatch', 'Unexpected submission program')
        script_indices = [i for i, value in enumerate(tokens)
                          if Path(value).name == _SCRIPTS[role]]
        _require(len(script_indices) == 1, 'Submission script is missing or ambiguous')
        at = script_indices[0]
        command = tokens[at:]
        self._verify_command(command, info['workdir'], epoch, role)
        options = tokens[1:at]
        if info['name'] != _LEGACY[role]:
            _require([value for value in options if value.startswith('--comment=')] ==
                     ['--comment=' + self.token(epoch, role)], 'Submission comment binding changed')
            _require([value for value in options if value.startswith('--job-name=')] ==
                     ['--job-name=' + self.name(epoch, role)], 'Submission name binding changed')
        exports = [value.split('=', 1)[1] for value in options if value.startswith('--export=')]
        _require(len(exports) == 1, 'Missing source export binding')
        env = dict(value.split('=', 1) for value in exports[0].split(',') if '=' in value)
        _require(env.get('SOURCE_GIT_COMMIT') == self.source_commit and
                 env.get('LIGER_MULTIEPOCH_MANIFEST_SHA256') == self.manifest_sha,
                 'Submission source or manifest binding changed')
        dependencies = [value.split('=', 1)[1] for value in options if value.startswith('--dependency=')]
        if role == 'val':
            _require(dependencies == ['afterok:' + _job_id(expected_train_id)],
                     'Validation dependency changed')
            _require('--kill-on-invalid-dep=yes' in options,
                     'Validation does not cancel an invalid dependency')
        else:
            _require(not dependencies, 'Unexpected training dependency')

    def _verify_command(self, command, workdir, epoch, role):
        _require(isinstance(command, list) and len(command) == 3, 'Unexpected training command arguments')
        path = Path(command[0])
        if not path.is_absolute():
            path = Path(workdir)/path
        _require(path.resolve() == self.root/_SCRIPTS[role] and
                 Path(command[1]).resolve() == self.manifest and command[2] == str(epoch),
                 'Job command does not match this run and epoch')

    def verify_job(self, info, epoch, role, expected_train_id=None, allow_legacy=False):
        _require(info.get('state') != 'UNKNOWN', 'Job identity is not yet available')
        _require(role in _SCRIPTS and type(epoch) is int and 1 <= epoch <= 12, 'Invalid job role or epoch')
        _require((role == 'val') == (expected_train_id is not None), 'Unexpected dependency argument')
        _job_id(info['job_id'])
        _require(info['user'] == self.user and Path(info['workdir']).resolve() == self.root,
                 'Job user or working directory changed')
        _require(info['requested_cpus'] == 4 and info['allocated_cpus'] in (0, 4) and
                 info['memory_bytes'] == 24 * 2**30 and info['gpus'] == (1 if role == 'train' else 2) and
                 info['time_limit_minutes'] == (720 if role == 'train' else 360) and
                 info['partition'] == 'rtx4090', 'Job resource allocation changed')
        if allow_legacy:
            _require(epoch == 2 and info['name'] == _LEGACY[role] and info['comment'] == '',
                     'Only the existing unmodified second-pass jobs may be attached')
        else:
            _require(info['name'] == self.name(epoch, role) and info['comment'] == self.token(epoch, role),
                     'Job token/name identity changed')
        if 'command' in info:
            self._verify_command(info['command'], info['workdir'], epoch, role)
            if role == 'val' and info['state'] == 'PENDING':
                dependency = re.sub(r'\([^)]*\)', '', info.get('dependency', ''))
                # Once afterok is satisfied Slurm may erase the dependency
                # while the validation job is still pending for resources.
                _require(dependency in ('afterok:' + _job_id(expected_train_id), '') and
                         info.get('kill_invalid_dependency') is True,
                         'Live validation dependency changed')
        if info.get('submit_line'):
            self._submission_identity(info, epoch, role, expected_train_id)
        else:
            _require(info.get('source') == 'scontrol' and 'command' in info,
                     'No live or accounted command identity')
