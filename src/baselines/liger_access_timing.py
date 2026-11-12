"""Per-access CPU instrumentation without replacing standardization math.

These timers measure nested host calls, not isolated GPU work. Raw access is
only source.__getitem__: FP64 conversion belongs to the caller's standardized
overhead. Process I/O/fault counters are diagnostic evidence, not a complete
view of BeeGFS/network storage or proof that zero counters mean zero disk I/O.
"""
import math
import os
from pathlib import Path
import resource
import time

from src.baselines.liger_standardization import StandardizedContent


class _Counters:
    def __init__(self, scope, clock=None, cpu_clock=None):
        self.scope = scope
        self.clock = time.perf_counter if clock is None else clock
        self.cpu_clock = time.process_time if cpu_clock is None else cpu_clock
        self.reset()

    def reset(self):
        self.calls = self.rows = self.failed_calls = self.unknown_rows_calls = 0
        self.wall_seconds = self.cpu_seconds = 0.

    def call(self, function, selection):
        start_wall, start_cpu = self.clock(), self.cpu_clock()
        self.calls += 1
        try:
            result = function(selection)
        except BaseException:
            self.failed_calls += 1
            raise
        finally:
            self.cpu_seconds += self.cpu_clock() - start_cpu
            self.wall_seconds += self.clock() - start_wall
        shape = getattr(result, 'shape', None)
        if shape is None or len(shape) < 1:
            self.unknown_rows_calls += 1
        else:
            self.rows += math.prod(shape[:-1])
        return result

    def snapshot(self):
        return dict(calls=int(self.calls), rows=int(self.rows), failed_calls=int(self.failed_calls),
            unknown_rows_calls=int(self.unknown_rows_calls), wall_seconds=float(self.wall_seconds),
            cpu_seconds=float(self.cpu_seconds), timing_scope=self.scope)


class AccessTimer:
    """Transparent row-access proxy; returned objects and RNGs are untouched."""
    def __init__(self, original, *, clock=None, cpu_clock=None):
        if not callable(getattr(original, '__getitem__', None)) or not hasattr(original, '__len__'):
            raise ValueError('AccessTimer requires a row-addressable source')
        self.original = original
        self._timing = _Counters(
            'host source.__getitem__ only; raw gather excludes caller FP64 conversion; '
            'nested access times may overlap GPU execution', clock, cpu_clock)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'original'), name)

    def __len__(self):
        return len(self.original)

    def __getitem__(self, selection):
        return self._timing.call(self.original.__getitem__, selection)

    def reset(self):
        self._timing.reset()

    def snapshot(self):
        return self._timing.snapshot()


class TimedStandardizedContent(StandardizedContent):
    """Run the original standardized __getitem__ unchanged, timing its call.

This remains a StandardizedContent instance for the existing cache builder.
Its raw source can be an AccessTimer. Constructor validation is not counted.
"""
    def __init__(self, content, statistics, *, expected_content_sha256=None,
                 clock=None, cpu_clock=None):
        super().__init__(content, statistics, expected_content_sha256=expected_content_sha256)
        self._timing = _Counters(
            'host original StandardizedContent.__getitem__; includes validation, raw gather, '
            'FP64 conversion, finite checks, normalization and FP32 output cast', clock, cpu_clock)

    def __getitem__(self, selection):
        return self._timing.call(super().__getitem__, selection)

    def reset(self):
        self._timing.reset()

    def snapshot(self):
        return self._timing.snapshot()


class CatalogAccessView:
    """Keep history access instrumentation separate from sampled catalog reads.

The training step sees the same global content object as .content. Catalog
access delegates to the original direct catalog, so it never increments the
history-only AccessTimer. Underlying row/content identities must still match.
"""
    def __init__(self, base_catalog, history):
        if (not isinstance(history, AccessTimer) or not hasattr(base_catalog, 'rows')
                or not hasattr(base_catalog, 'content') or history.original is not base_catalog.content):
            raise ValueError('Catalog and history must share the identical underlying content')
        self.base_catalog, self.content = base_catalog, history

    @property
    def rows(self):
        return self.base_catalog.rows

    def __len__(self):
        return len(self.base_catalog)

    def __getitem__(self, selection):
        if self.content.original is not self.base_catalog.content:
            raise ValueError('Catalog/history content identity changed')
        return self.base_catalog[selection]


def summarize_access(raw, standardized, history, catalog_snapshot):
    """Summarize nested CPU scopes; no subtraction from a GPU/step timer."""
    raw, standardized, history = [dict(value.snapshot()) for value in (raw, standardized, history)]
    overhead = {}
    for key in ('wall_seconds', 'cpu_seconds'):
        value = standardized[key] - raw[key]
        if value < 0:
            raise ValueError('Raw counters are not nested within standardized counters; reset them together')
        overhead['standardization_overhead_' + key] = value
    return dict(raw=raw, standardized=standardized, history=history,
        catalog=dict(catalog_snapshot), **overhead,
        timing_scope='CPU nested-call difference includes FP64 conversion, validation, finite checks, '
            'normalization, output cast and instrumentation; not isolated arithmetic or GPU time; '
            'history and catalog scopes overlap raw/standardized scopes and must not be summed')


IO_KEYS = ('rchar', 'wchar', 'syscr', 'syscw', 'read_bytes', 'write_bytes', 'cancelled_write_bytes')
IO_SCOPE = ('process self only; /proc I/O and page faults may not capture BeeGFS/network traffic; '
            'zero read_bytes or major_faults does not prove zero storage access')


def io_snapshot(proc_root='/proc/self'):
    """Read process counters, keeping missing/invalid kernel fields explicit."""
    result = dict(schema='liger.process_access_io.v1', pid=os.getpid(),
        monotonic_seconds=time.monotonic(), process_cpu_seconds=time.process_time(),
        io={key: None for key in IO_KEYS}, errors=[], timing_scope=IO_SCOPE)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(minor_faults=int(usage.ru_minflt), major_faults=int(usage.ru_majflt))
    try:
        seen = set()
        for line in (Path(proc_root) / 'io').read_text().splitlines():
            key, separator, value = line.partition(':')
            if not separator or key not in IO_KEYS:
                continue
            if key in seen or not value.strip().isdigit():
                raise ValueError('Invalid or duplicate process I/O field')
            seen.add(key)
            result['io'][key] = int(value.strip())
    except (OSError, ValueError) as exc:
        result['errors'].append('proc_io:' + type(exc).__name__)
        # A malformed/partial read must not present some fields as verified.
        result['io'] = {key: None for key in IO_KEYS}
    result['unavailable_fields'] = [key for key in IO_KEYS if result['io'][key] is None]
    return result


def delta_io(before, after):
    """Return same-process deltas; unsupported/reset I/O fields stay unknown."""
    if (before.get('schema') != 'liger.process_access_io.v1'
            or after.get('schema') != before['schema'] or before.get('pid') != after.get('pid')):
        raise ValueError('I/O snapshots must come from the same process and schema')
    result = dict(schema='liger.process_access_io_delta.v1', pid=before['pid'],
        io={}, unavailable_fields=[], counter_reset_fields=[],
        errors=before.get('errors', []) + after.get('errors', []), timing_scope=IO_SCOPE)
    for source, output in (('monotonic_seconds', 'wall_seconds'),
                           ('process_cpu_seconds', 'cpu_seconds'),
                           ('minor_faults', 'minor_faults'), ('major_faults', 'major_faults')):
        values = (before[source], after[source])
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in values) or values[1] < values[0]:
            raise ValueError('Invalid nonmonotonic process counter')
        result[output] = values[1] - values[0]
    for key in IO_KEYS:
        start, stop = before['io'].get(key), after['io'].get(key)
        if (type(start) is not int or type(stop) is not int or start < 0 or stop < 0):
            result['io'][key] = None
            result['unavailable_fields'].append(key)
        elif stop < start:
            result['io'][key] = None
            result['counter_reset_fields'].append(key)
            result['unavailable_fields'].append(key)
        else:
            result['io'][key] = stop - start
    return result
