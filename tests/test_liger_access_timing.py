import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.baselines.liger_access_timing import (
    AccessTimer, CatalogAccessView, IO_KEYS, TimedStandardizedContent,
    delta_io, io_snapshot, summarize_access,
)
from src.baselines.liger_data import Record
from src.baselines.liger_efficiency_timing import TimedCatalog
from src.baselines.liger_feature_cache import build_cached_seen, verify_cached_seen
from src.baselines.liger_full_catalog import RowCatalog
from src.baselines.liger_prefix_training import initial_progress, train_step
from src.baselines.liger_standardization import StandardizedContent, fit_standardization
from src.baselines.liger_state import BudgetSchedule, capture, tree_digest
from src.baselines.liger_supervision import expand_training_records


class Clock:
    def __init__(self, values): self.values = iter(values)
    def __call__(self): return next(self.values)


def source():
    raw = np.random.default_rng(9).integers(-100, 100, (11, 3), dtype=np.int8)
    seen = np.array([0, 2, 4, 6, 8], dtype=np.int32)
    stats = fit_standardization(raw, seen, content_sha256='a' * 64)
    binding = dict(standardization_stats_sha256=stats['stats_sha256'],
        seen_rows_sha256=stats['seen_rows_sha256'], source_sha256='a' * 64)
    return raw, seen, stats, binding


def wrapped():
    original, seen, stats, binding = source()
    raw = AccessTimer(original)
    standard = TimedStandardizedContent(raw, stats, expected_content_sha256='a' * 64)
    return raw, standard, seen, stats, binding


@pytest.mark.parametrize('selection', [0, slice(None), slice(0, 0), [1, 2, 1],
    np.array([[0, 2], [4, 8]]), np.array([], dtype=np.int64)])
def test_raw_proxy_preserves_values_shapes_attributes_without_extra_gather(selection):
    values, *_ = source()
    timer = AccessTimer(values, clock=Clock([1, 3]), cpu_clock=Clock([2, 2.5]))
    result = timer[selection]
    np.testing.assert_array_equal(result, values[selection])
    assert timer.original is values and timer.shape == values.shape
    assert timer.ndim == values.ndim and timer.dtype == values.dtype and len(timer) == len(values)
    s = timer.snapshot()
    assert s['calls'] == 1 and s['rows'] == int(np.prod(result.shape[:-1]))
    assert s['wall_seconds'] == 2 and s['cpu_seconds'] == .5
    assert s['failed_calls'] == s['unknown_rows_calls'] == 0


def test_reset_does_not_modify_source_or_previous_snapshot():
    raw, standard, *_ = wrapped()
    standard[[0, 2]]
    before = standard.snapshot()
    values = raw.original.copy()
    raw.reset()
    standard.reset()
    assert before['rows'] == 2 and standard.snapshot()['rows'] == raw.snapshot()['rows'] == 0
    before['calls'] = 999
    assert standard.snapshot()['calls'] == 0
    np.testing.assert_array_equal(raw.original, values)


def test_error_is_propagated_and_counted_without_fake_rows():
    timer = AccessTimer(np.zeros((2, 3)), clock=Clock([1, 2]), cpu_clock=Clock([3, 3.5]))
    with pytest.raises(IndexError): timer[99]
    assert timer.snapshot()['calls'] == timer.snapshot()['failed_calls'] == 1
    assert timer.snapshot()['rows'] == 0 and timer.snapshot()['wall_seconds'] == 1


def test_proxy_returns_original_object_and_unknown_shapes_are_explicit():
    marker = object()
    class Original:
        def __len__(self): return 1
        def __getitem__(self, selection): return marker
    timer = AccessTimer(Original())
    assert timer[0] is marker and timer.snapshot()['unknown_rows_calls'] == 1
    with pytest.raises(ValueError): AccessTimer(object())


@pytest.mark.parametrize('selection', [0, [0, 2, 0], [[4, 8], [2, 6]], slice(None),
                                      np.array([], dtype=np.int64)])
def test_timed_standardization_uses_original_bitwise_numeric_path(selection):
    raw, standard, seen, stats, _ = wrapped()
    baseline = StandardizedContent(raw.original, stats, expected_content_sha256='a' * 64)
    np.testing.assert_array_equal(standard[selection].view(np.uint32), baseline[selection].view(np.uint32))
    assert standard.snapshot()['calls'] == raw.snapshot()['calls'] == 1
    assert isinstance(standard, StandardizedContent)


def test_invalid_selection_is_rejected_before_raw_access_and_nonfinite_source_is_recorded():
    raw, standard, *_ = wrapped()
    with pytest.raises(ValueError): standard[-1]
    assert standard.snapshot()['failed_calls'] == 1 and raw.snapshot()['calls'] == 0
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    stats = fit_standardization(values, [0, 2])
    raw = AccessTimer(values)
    standard = TimedStandardizedContent(raw, stats)
    values[0, 0] = np.nan
    with pytest.raises(ValueError): standard[[0]]
    assert standard.snapshot()['failed_calls'] == 1
    assert raw.snapshot()['failed_calls'] == 0 and raw.snapshot()['calls'] == 1


def test_nested_overhead_labels_conversion_and_checks_not_pure_math():
    values, _, stats, _ = source()
    raw = AccessTimer(values, clock=Clock([1, 3]), cpu_clock=Clock([1, 2]))
    standard = TimedStandardizedContent(raw, stats, clock=Clock([0, 5]), cpu_clock=Clock([0, 3]))
    history = AccessTimer(standard, clock=Clock([0, 6]), cpu_clock=Clock([0, 4]))
    history[[0, 2]]
    result = summarize_access(raw, standard, history, {'calls': 0})
    assert result['standardization_overhead_wall_seconds'] == 3
    assert result['standardization_overhead_cpu_seconds'] == 2
    assert 'FP64 conversion' in result['timing_scope'] and 'must not be summed' in result['timing_scope']
    json.dumps(result, allow_nan=False)
    standard.reset()
    with pytest.raises(ValueError, match='reset them together'):
        summarize_access(raw, standard, history, {'calls': 0})


def test_cached_seen_builder_and_verifier_remain_compatible_with_timed_subclass():
    raw, standard, seen, _, binding = wrapped()
    content, direct, summary = build_cached_seen(standard, seen, expected_binding=binding, chunk_size=2)
    receipt = verify_cached_seen(content, direct, expected_binding=binding, chunk_size=2)
    assert receipt['feature_values_bitwise_equal'] and summary['cache_bytes'] == 5 * 3 * 4
    raw.reset()
    standard.reset()
    history = AccessTimer(content)
    catalog = TimedCatalog(CatalogAccessView(direct, history))
    history[[0, 2, 8]]
    catalog[[0, 1, 4]]
    assert raw.snapshot()['calls'] == standard.snapshot()['calls'] == 0
    assert history.snapshot()['calls'] == 1 and history.snapshot()['rows'] == 3
    assert catalog.snapshot()['calls'] == 1 and catalog.snapshot()['rows'] == 3
    assert catalog.content is history and catalog.rows is seen
    # Cold fallback remains exact and is measured, rather than silently cached.
    np.testing.assert_array_equal(history[[1]], standard[[1]])
    assert raw.snapshot()['calls'] == standard.snapshot()['calls'] == 2


def test_catalog_view_keeps_history_and_negative_counts_disjoint_and_rejects_identity_drift():
    raw, standard, seen, *_ = wrapped()
    direct = RowCatalog(standard, seen)
    history = AccessTimer(standard)
    catalog = TimedCatalog(CatalogAccessView(direct, history))
    history[[0, 4]]
    catalog[[0, 2, 3]]
    assert history.snapshot()['calls'] == catalog.snapshot()['calls'] == 1
    assert raw.snapshot()['calls'] == standard.snapshot()['calls'] == 2
    assert raw.snapshot()['rows'] == standard.snapshot()['rows'] == 5
    with pytest.raises(ValueError): CatalogAccessView(direct, AccessTimer(raw.original))
    view = CatalogAccessView(direct, history)
    direct.content = raw.original
    with pytest.raises(ValueError, match='identity changed'): view[:]


def test_access_instrumentation_does_not_advance_python_numpy_or_torch_rng():
    raw, standard, seen, *_ = wrapped()
    history = AccessTimer(standard)
    catalog = CatalogAccessView(RowCatalog(standard, seen), history)
    before = tree_digest((random.getstate(), np.random.get_state(), torch.get_rng_state()))
    history[[0, 2, 4]]
    catalog[[0, 2]]
    summarize_access(raw, standard, history, {'calls': 1})
    assert tree_digest((random.getstate(), np.random.get_state(), torch.get_rng_state())) == before


def joint_state(backend):
    torch.manual_seed(19)
    model = torch.nn.Module()
    model.emb_proj = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(3, 4))
    model.query = torch.nn.Parameter(torch.randn(3, 4))
    def forward(current, batch, device, *args):
        current.predicted_embedding = current.emb_proj(batch['input_embeddings'].mean(dim=1)) + current.query
        return SimpleNamespace(loss=current.predicted_embedding.square().mean()), None
    values, seen, stats, binding = source()
    raw = AccessTimer(values)
    standard = TimedStandardizedContent(raw, stats)
    plain = StandardizedContent(values, stats)
    content = plain if backend == 'plain' else standard
    direct = RowCatalog(content, seen)
    if backend == 'cached':
        content, direct, _ = build_cached_seen(standard, seen, expected_binding=binding, chunk_size=2)
    history = AccessTimer(content)
    catalog = TimedCatalog(CatalogAccessView(direct, history))
    raw.reset()
    standard.reset()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = BudgetSchedule(optimizer, total=10, warmup=2)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    progress = initial_progress(max_optimizer_steps=10, max_attempts=1)
    examples = expand_training_records([Record(1, (0, 2, 4), 'a' * 64), Record(2, (4, 6), 'b' * 64)])
    report = train_step(model, optimizer, scheduler, scaler, {}, SimpleNamespace(model_forward=forward),
        examples, np.arange(44).reshape(11, 4), history, seen, catalog, progress,
        dense_sampling={'num_negatives': 1, 'seed': 42}, expected_gradients=3)
    state = capture(model, optimizer, scheduler, scaler, progress, {'same': 'learning'})
    return report, tree_digest(state), raw.snapshot(), standard.snapshot(), history.snapshot(), catalog.snapshot()


@pytest.mark.parametrize('backend', ['timed_lazy', 'cached'])
def test_sampled_joint_step_and_all_learning_state_are_bitwise_equal_to_plain(backend):
    expected, digest, *_ = joint_state('plain')
    actual, actual_digest, raw, standard, history, catalog = joint_state(backend)
    assert actual == expected and actual_digest == digest
    assert history['calls'] == 3 and history['rows'] == 4
    assert catalog['calls'] == 1 and catalog['rows'] == actual['dense_candidate_classes'] == 4
    assert raw['calls'] == standard['calls'] == (0 if backend == 'cached' else 4)


def snapshots():
    before = dict(schema='liger.process_access_io.v1', pid=20, monotonic_seconds=10.,
        process_cpu_seconds=2., minor_faults=10, major_faults=1,
        io={key: 100 for key in IO_KEYS}, errors=[], unavailable_fields=[])
    after = copy.deepcopy(before)
    after.update(monotonic_seconds=12., process_cpu_seconds=2.5, minor_faults=12, major_faults=2)
    after['io'] = {key: 110 for key in IO_KEYS}
    return before, after


def test_proc_io_snapshot_and_delta_keep_logical_bytes_distinct_from_storage_evidence(tmp_path):
    (tmp_path / 'io').write_text('\n'.join(f'{key}: {i}' for i, key in enumerate(IO_KEYS)))
    s = io_snapshot(tmp_path)
    assert s['io'] == dict(zip(IO_KEYS, range(len(IO_KEYS)))) and not s['errors']
    assert s['unavailable_fields'] == [] and s['minor_faults'] >= 0
    result = delta_io(*snapshots())
    assert result['wall_seconds'] == 2 and result['cpu_seconds'] == .5
    assert result['minor_faults'] == 2 and result['major_faults'] == 1
    assert result['io'] == {key: 10 for key in IO_KEYS}
    assert not result['unavailable_fields'] and not result['counter_reset_fields']
    assert 'does not prove zero storage access' in result['timing_scope']
    json.dumps(s, allow_nan=False)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('payload', [None, 'read_bytes: nope', 'read_bytes: 1\nread_bytes: 2'])
def test_missing_or_malformed_io_is_unknown_not_a_zero_read_claim(tmp_path, payload):
    if payload is not None: (tmp_path / 'io').write_text(payload)
    s = io_snapshot(tmp_path)
    assert s['errors'] and s['unavailable_fields'] == list(IO_KEYS)
    assert all(value is None for value in s['io'].values())


def test_partial_kernel_io_fields_and_reset_counters_are_explicit(tmp_path):
    (tmp_path / 'io').write_text('rchar: 123\nunknown_kernel_field: 10\n')
    s = io_snapshot(tmp_path)
    assert s['io']['rchar'] == 123 and 'rchar' not in s['unavailable_fields']
    assert len(s['unavailable_fields']) == len(IO_KEYS) - 1 and not s['errors']
    a, b = snapshots()
    b['io']['read_bytes'] = None
    b['io']['rchar'] = 99
    result = delta_io(a, b)
    assert result['io']['read_bytes'] is result['io']['rchar'] is None
    assert result['counter_reset_fields'] == ['rchar']
    assert set(result['unavailable_fields']) == {'rchar', 'read_bytes'}


@pytest.mark.parametrize('field,value', [('pid', 21), ('schema', 'wrong'),
    ('monotonic_seconds', 9.), ('process_cpu_seconds', float('nan')),
    ('minor_faults', 9), ('major_faults', True)])
def test_io_delta_rejects_foreign_process_or_invalid_monotonic_counters(field, value):
    a, b = snapshots()
    b[field] = value
    with pytest.raises(ValueError): delta_io(a, b)
