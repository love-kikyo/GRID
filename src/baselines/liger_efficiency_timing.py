"""Opt-in wall-clock probes without modifying training code or global functions.

No ML imports, CUDA calls, model hooks, RNG operations or array copies occur
here. Supply a synchronization callback explicitly when timing a device region.
Catalog CPU time may overlap queued GPU work, so subtracting it from step wall
time is NOT pure GPU time or a guaranteed GPU-time upper bound. Report any such
remainder only as unattributed wall time, with overlapping scopes disclosed.
"""
import math
import os
from pathlib import Path
import resource
import sys
import time


CATALOG_SCOPE = (
    'CPU catalog.__getitem__ wall time, including row gather and any lazy '
    'standardization; all forward/backward visits combined; excludes history '
    'packing, H2D transfer, projection and GPU synchronization; may overlap GPU work'
)


class TimedCatalog:
    """Transparent per-run RowCatalog proxy; counters are not thread-safe.

    Pass the original content separately to the unchanged training step. Only
    accesses through this proxy are counted, so history packing is not mixed
    into full-catalog visits. A complete streamed CE forward plus backward
    should visit 2 * len(catalog) rows, irrespective of the chunk tail size.
    Count rows from the returned shape, without gathering the index a second
    time. Returned arrays and .content/.rows retain their original identities.
    """

    def __init__(self, catalog, *, clock=None):
        if not hasattr(catalog, 'content') or not hasattr(catalog, 'rows'):
            raise ValueError('Expected a RowCatalog-compatible content/rows object')
        self._catalog = catalog
        self._clock = time.perf_counter if clock is None else clock
        self.reset()

    @property
    def content(self):
        return self._catalog.content

    @property
    def rows(self):
        return self._catalog.rows

    def __len__(self):
        return len(self._catalog)

    def __getitem__(self, selection):
        started = self._clock()
        self._calls += 1
        try:
            result = self._catalog[selection]
        except BaseException:
            self._failed_calls += 1
            raise
        finally:
            self._seconds += self._clock() - started
        shape = getattr(result, 'shape', None)
        if shape is None or len(shape) < 1:
            self._unknown_rows_calls += 1
        else:
            self._rows += math.prod(shape[:-1])  # Scalar row: [features] -> 1.
        return result

    def reset(self):
        """Reset only instrumentation, never the catalog or its content."""
        self._calls = self._rows = self._failed_calls = self._unknown_rows_calls = 0
        self._seconds = 0.0

    def snapshot(self):
        """Return an independent JSON-ready counter snapshot."""
        return dict(calls=int(self._calls), rows=int(self._rows),
                    failed_calls=int(self._failed_calls),
                    unknown_rows_calls=int(self._unknown_rows_calls),
                    cpu_fetch_wall_seconds=float(self._seconds), timing_scope=CATALOG_SCOPE)


def timed_call(function, *args, synchronize=None, clock=None, **kwargs):
    """Return (unchanged result, timing dictionary) for one successful call.

    Use ``timed_call(next, stream)`` for data wait, or supply a CUDA synchronize
    callback for whole-step / explicitly separated forward timings. The initial
    synchronize drains earlier work *before* starting the clock; the trailing
    synchronization is included. This is region wall time, not a CUDA event
    measurement or pure device time. Exceptions propagate unchanged and produce
    no successful timing record; synchronization after a failed call is left to
    the caller's existing error handling. No global function is monkeypatched.
    """
    timer = time.perf_counter if clock is None else clock
    if synchronize is not None:
        synchronize()
    started = timer()
    result = function(*args, **kwargs)
    if synchronize is not None:
        synchronize()
    elapsed = timer() - started
    scope = ('host wall time after a leading device drain, including callable and trailing '
             'synchronization; not pure device time' if synchronize is not None else
             'host callable wall time without device synchronization; may only measure enqueue time')
    return result, dict(wall_seconds=float(elapsed), synchronized=synchronize is not None,
                        timing_scope=scope)


class TimedModelForward:
    """Local evaluation.model_forward proxy; no module/global mutation.

    A provided synchronization callback drains earlier work before the measured
    region and waits for this call afterwards. The successful-call wall time
    therefore includes the original forward helper's CPU work and transfers,
    not just model kernels. Failed calls are counted but do not contribute a
    successful timing measurement. Counters are per-instance, not thread-safe.
    """

    def __init__(self, evaluation, *, synchronize=None, clock=None):
        if not callable(getattr(evaluation, 'model_forward', None)):
            raise ValueError('Expected an evaluation object with model_forward')
        self._evaluation, self._synchronize, self._clock = evaluation, synchronize, clock
        self.reset()

    def model_forward(self, *args, **kwargs):
        self._calls += 1
        try:
            result, measured = timed_call(self._evaluation.model_forward, *args,
                synchronize=self._synchronize, clock=self._clock, **kwargs)
        except BaseException:
            self._failed_calls += 1
            raise
        self._seconds += measured['wall_seconds']
        return result

    def reset(self):
        self._calls = self._failed_calls = 0
        self._seconds = 0.0

    def snapshot(self):
        return dict(calls=int(self._calls), failed_calls=int(self._failed_calls),
                    model_forward_wall_seconds=float(self._seconds),
                    synchronized=self._synchronize is not None,
                    timing_scope='successful evaluation.model_forward host wall time, '
                    'including helper CPU work and transfers; optional leading drain '
                    'excluded and trailing synchronization included; not pure GPU time')


def process_memory_snapshot(proc_root='/proc/self'):
    """Read Linux process-self memory and CPU counters without psutil.

    RSS/HWM are bytes, CPU counters seconds. ru_maxrss is the process lifetime
    high-water RSS (Linux KiB converted to bytes), not current RSS or an interval
    peak. Values exclude GPU allocations and are not a process-tree/job total.
    Missing /proc fields are None, with sanitized error classes rather than
    filesystem details. proc_root supports small synthetic fixtures in tests.
    """
    result = dict(schema='liger.process_self_resources.v1',
                  timing_scope='process self only; snapshot/lifetime counters, not GPU or job-tree totals',
                  monotonic_seconds=time.monotonic(), errors=[])
    mapping = {'VmRSS': 'rss_bytes', 'VmHWM': 'high_water_rss_bytes',
               'VmSize': 'virtual_memory_bytes', 'RssAnon': 'anonymous_rss_bytes',
               'RssFile': 'file_rss_bytes', 'RssShmem': 'shared_rss_bytes', 'VmSwap': 'swap_bytes'}
    result.update({value: None for value in mapping.values()})
    result['threads'] = None
    try:
        for line in (Path(proc_root) / 'status').read_text().splitlines():
            name, separator, raw = line.partition(':')
            if not separator or name not in {*mapping, 'Threads'}:
                continue
            fields = raw.split()
            if name == 'Threads':
                if len(fields) != 1 or not fields[0].isdigit():
                    raise ValueError('Invalid thread counter')
                result['threads'] = int(fields[0])
            else:
                if len(fields) != 2 or fields[1] != 'kB' or not fields[0].isdigit():
                    raise ValueError('Invalid memory counter')
                result[mapping[name]] = int(fields[0]) * 1024
    except (OSError, ValueError) as exc:
        result['errors'].append('proc_status:' + type(exc).__name__)
    result.update(proc_cpu_user_seconds=None, proc_cpu_system_seconds=None,
                  proc_cpu_total_seconds=None, clock_ticks_per_second=None)
    try:
        fields = (Path(proc_root) / 'stat').read_text().rpartition(')')[2].split()
        ticks = int(os.sysconf('SC_CLK_TCK'))
        if ticks <= 0 or len(fields) < 13:
            raise ValueError('Invalid process CPU counters')
        user_ticks, system_ticks = int(fields[11]), int(fields[12])
        if min(user_ticks, system_ticks) < 0:
            raise ValueError('Negative process CPU counters')
        result.update(proc_cpu_user_seconds=user_ticks / ticks,
                      proc_cpu_system_seconds=system_ticks / ticks,
                      proc_cpu_total_seconds=(user_ticks + system_ticks) / ticks,
                      clock_ticks_per_second=ticks)
    except (OSError, ValueError) as exc:
        result['errors'].append('proc_stat:' + type(exc).__name__)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(resource_cpu_user_seconds=float(usage.ru_utime),
                  resource_cpu_system_seconds=float(usage.ru_stime),
                  ru_maxrss_bytes=int(usage.ru_maxrss) * 1024 if sys.platform.startswith('linux') else None)
    if not sys.platform.startswith('linux'):
        result['errors'].append('ru_maxrss:unsupported_platform_units')
    return result
