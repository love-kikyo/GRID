import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines import liger_efficiency_timing as timing


class Clock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class Catalog:
    def __init__(self):
        self.content = np.arange(60).reshape(12, 5)
        self.rows = np.array([0, 2, 3, 8, 11])
        self.selections = []
        self.last_result = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, selection):
        self.selections.append(selection)
        self.last_result = self.content[self.rows[selection]]
        return self.last_result


def test_catalog_proxy_preserves_identity_values_and_counts_two_complete_passes():
    original = Catalog()
    proxy = timing.TimedCatalog(original, clock=Clock(range(12)))
    assert proxy.content is original.content and proxy.rows is original.rows
    assert len(proxy) == len(original) == 5
    for _ in range(2):
        for start in (0, 2, 4):
            actual = proxy[start:min(start + 2, 5)]
            assert actual is original.last_result
            np.testing.assert_array_equal(actual, original.content[original.rows[start:start + 2]])
    snapshot = proxy.snapshot()
    assert snapshot['calls'] == 6 and snapshot['rows'] == 2 * len(original)
    assert snapshot['cpu_fetch_wall_seconds'] == 6.0
    assert snapshot['failed_calls'] == snapshot['unknown_rows_calls'] == 0
    assert 'all forward/backward visits combined' in snapshot['timing_scope']
    # History packing accesses original content, not the catalog proxy.
    original.content[[0, 1, 2]]
    assert proxy.snapshot() == snapshot
    assert len(original.selections) == 6  # No second gather for row counting.


def test_catalog_scalar_empty_and_array_selections_have_correct_row_counts():
    proxy = timing.TimedCatalog(Catalog(), clock=Clock(range(6)))
    assert proxy[0].shape == (5,)
    assert proxy[0:0].shape == (0, 5)
    assert proxy[np.array([1, 2])].shape == (2, 5)
    assert proxy.snapshot()['rows'] == 3
    assert proxy.snapshot()['calls'] == 3


def test_reset_and_snapshot_cannot_change_content_or_prior_snapshot():
    original = Catalog()
    proxy = timing.TimedCatalog(original, clock=Clock([1., 1.5]))
    proxy[:2]
    snapshot = proxy.snapshot()
    proxy.reset()
    assert snapshot['rows'] == 2 and snapshot['cpu_fetch_wall_seconds'] == .5
    assert proxy.snapshot()['calls'] == proxy.snapshot()['rows'] == 0
    assert proxy.snapshot()['cpu_fetch_wall_seconds'] == 0.
    assert proxy.content is original.content and proxy.rows is original.rows
    snapshot['calls'] = 999
    assert proxy.snapshot()['calls'] == 0


def test_catalog_error_is_not_hidden_and_failed_access_does_not_count_rows():
    proxy = timing.TimedCatalog(Catalog(), clock=Clock([3., 3.25]))
    with pytest.raises(IndexError):
        proxy[999]
    result = proxy.snapshot()
    assert result['calls'] == result['failed_calls'] == 1 and result['rows'] == 0
    assert result['cpu_fetch_wall_seconds'] == .25


def test_catalog_requires_explicit_content_and_rows_contract():
    with pytest.raises(ValueError, match='RowCatalog'):
        timing.TimedCatalog(object())


def test_timed_call_preserves_result_args_and_exact_synchronization_boundaries():
    events = []
    original = object()
    def sync(): events.append('sync')
    def clock():
        events.append('clock')
        return 10. if len(events) == 2 else 12.
    def function(arg, *, value):
        events.append('call')
        assert arg is original and value == 7
        return original
    value, result = timing.timed_call(function, original, value=7, synchronize=sync, clock=clock)
    assert value is original
    assert events == ['sync', 'clock', 'call', 'sync', 'clock']
    assert result['wall_seconds'] == 2. and result['synchronized'] is True
    assert 'not pure device time' in result['timing_scope']


def test_next_stream_timer_has_no_device_sync_and_does_not_consume_extra_element():
    stream = iter([10, 20])
    result, measured = timing.timed_call(next, stream, clock=Clock([1., 1.125]))
    assert result == 10 and next(stream) == 20
    assert measured['wall_seconds'] == .125 and measured['synchronized'] is False
    with pytest.raises(StopIteration):
        timing.timed_call(next, stream, clock=Clock([2.]))


def test_timed_call_propagates_same_exception_without_extra_call_or_trailing_sync():
    failure, events = RuntimeError('fixture failure'), []
    def function():
        events.append('call')
        raise failure
    with pytest.raises(RuntimeError) as captured:
        timing.timed_call(function, synchronize=lambda: events.append('sync'), clock=Clock([3.]))
    assert captured.value is failure and events == ['sync', 'call']


def test_model_forward_proxy_preserves_call_and_only_times_original_helper():
    value, events = object(), []
    def model_forward(model, batch, *, method):
        assert model is value and batch == 'batch' and method == 'method'
        events.append('forward')
        return value
    original = SimpleNamespace(model_forward=model_forward)
    proxy = timing.TimedModelForward(original, synchronize=lambda: events.append('sync'),
                                     clock=Clock([3., 3.5, 5., 5.75]))
    assert proxy.model_forward(value, 'batch', method='method') is value
    assert proxy.model_forward(value, 'batch', method='method') is value
    assert original.model_forward is model_forward
    assert events == ['sync', 'forward', 'sync'] * 2
    before = proxy.snapshot()
    assert before['calls'] == 2 and before['model_forward_wall_seconds'] == 1.25
    assert before['failed_calls'] == 0 and before['synchronized'] is True
    proxy.reset()
    assert proxy.snapshot()['calls'] == proxy.snapshot()['model_forward_wall_seconds'] == 0
    assert before['calls'] == 2


def test_model_forward_proxy_counts_failure_without_successful_timing_or_global_change():
    error = RuntimeError('same failure')
    def model_forward():
        raise error
    original = SimpleNamespace(model_forward=model_forward)
    proxy = timing.TimedModelForward(original, clock=Clock([1.]))
    with pytest.raises(RuntimeError) as captured:
        proxy.model_forward()
    assert captured.value is error and original.model_forward is model_forward
    result = proxy.snapshot()
    assert result['calls'] == result['failed_calls'] == 1
    assert result['model_forward_wall_seconds'] == 0. and result['synchronized'] is False
    with pytest.raises(ValueError, match='model_forward'):
        timing.TimedModelForward(object())


def test_instrumentation_does_not_advance_python_or_numpy_rng():
    random.seed(23)
    np.random.seed(23)
    before_python, before_numpy = random.getstate(), np.random.get_state()
    catalog = timing.TimedCatalog(Catalog(), clock=Clock([1., 2.]))
    timing.timed_call(lambda: catalog[:3], clock=Clock([0., 3.]))
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]


def resource_fixture(tmp_path, monkeypatch):
    (tmp_path / 'status').write_text(
        'Name:\tfixture\nVmRSS:\t123 kB\nVmHWM:\t150 kB\nVmSize:\t250 kB\n'
        'RssAnon:\t80 kB\nRssFile:\t40 kB\nRssShmem:\t3 kB\nVmSwap:\t0 kB\nThreads:\t4\n')
    fields = ['S'] + ['0'] * 20
    fields[11:13] = ['125', '50']
    (tmp_path / 'stat').write_text('123 (fixture with ) name) ' + ' '.join(fields))
    monkeypatch.setattr(timing.os, 'sysconf', lambda _: 100)
    monkeypatch.setattr(timing.resource, 'getrusage', lambda kind: SimpleNamespace(
        ru_maxrss=180, ru_utime=1.3, ru_stime=.6))
    monkeypatch.setattr(timing.time, 'monotonic', lambda: 7.)
    monkeypatch.setattr(timing.sys, 'platform', 'linux')


def test_memory_snapshot_reports_bytes_and_process_cpu_seconds_without_psutil(tmp_path, monkeypatch):
    resource_fixture(tmp_path, monkeypatch)
    result = timing.process_memory_snapshot(tmp_path)
    assert result['rss_bytes'] == 123 * 1024
    assert result['high_water_rss_bytes'] == 150 * 1024
    assert result['virtual_memory_bytes'] == 250 * 1024
    assert result['anonymous_rss_bytes'] == 80 * 1024
    assert result['file_rss_bytes'] == 40 * 1024 and result['shared_rss_bytes'] == 3 * 1024
    assert result['swap_bytes'] == 0 and result['threads'] == 4
    assert result['ru_maxrss_bytes'] == 180 * 1024  # Lifetime peak, not current RSS.
    assert result['proc_cpu_user_seconds'] == 1.25 and result['proc_cpu_system_seconds'] == .5
    assert result['proc_cpu_total_seconds'] == 1.75
    assert result['resource_cpu_user_seconds'] == 1.3 and result['resource_cpu_system_seconds'] == .6
    assert result['clock_ticks_per_second'] == 100 and result['monotonic_seconds'] == 7.
    assert result['errors'] == []
    assert 'not GPU or job-tree totals' in result['timing_scope']
    json.dumps(result, allow_nan=False)


def test_missing_proc_files_keep_unavailable_values_distinct_from_zero(tmp_path, monkeypatch):
    resource_fixture(tmp_path, monkeypatch)
    result = timing.process_memory_snapshot(tmp_path / 'missing')
    assert result['rss_bytes'] is result['proc_cpu_total_seconds'] is None
    assert result['ru_maxrss_bytes'] == 180 * 1024
    assert result['errors'] == ['proc_status:FileNotFoundError', 'proc_stat:FileNotFoundError']
    assert str(tmp_path) not in json.dumps(result)


def test_unknown_memory_units_and_bad_cpu_fields_are_reported_not_misconverted(tmp_path, monkeypatch):
    resource_fixture(tmp_path, monkeypatch)
    (tmp_path / 'status').write_text('VmRSS: 123 MB\n')
    (tmp_path / 'stat').write_text('123 (fixture) S 0')
    result = timing.process_memory_snapshot(tmp_path)
    assert result['rss_bytes'] is result['proc_cpu_total_seconds'] is None
    assert result['errors'] == ['proc_status:ValueError', 'proc_stat:ValueError']


def test_proxy_preserves_cpu_full_catalog_loss_dropout_gradients_and_torch_rng():
    torch = pytest.importorskip('torch')
    from src.baselines.liger_full_catalog import RowCatalog, full_catalog_cross_entropy
    torch.manual_seed(73)
    content = np.arange(35, dtype=np.float64).reshape(7, 5) / 20
    seen = np.arange(7, dtype=np.int64)
    original = RowCatalog(content, seen)
    projection = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Dropout(.2),
                                     torch.nn.Linear(4, 3)).double()
    query = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, 6])
    parameters = tuple(projection.parameters())
    before = torch.get_rng_state().clone()
    expected = full_catalog_cross_entropy(query, labels, projection, original, chunk_size=3)
    expected_gradients = torch.autograd.grad(expected, (query, *parameters))
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    proxy = timing.TimedCatalog(original, clock=Clock(range(12)))
    actual = full_catalog_cross_entropy(query, labels, projection, proxy, chunk_size=3)
    actual_gradients = torch.autograd.grad(actual, (query, *parameters))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert proxy.snapshot()['rows'] == 14 and proxy.snapshot()['calls'] == 6
    assert proxy.content is content and proxy.rows is seen
