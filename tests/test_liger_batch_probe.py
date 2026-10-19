import copy

import pytest

from src.baselines.liger_batch_probe import take_batch, summarize, verify_cases


def test_successive_batches_consume_distinct_real_records_and_keep_last_gt():
    stream = iter([(10, [0, 2, 4], b'a'), (20, [1, 6], b'b'), (30, [0, 3, 0], b'c')])
    seen = set()
    a, ma = take_batch(stream, 2, seen)
    b, mb = take_batch(stream, 1, seen)
    assert a == [[0, 2, 4], [1, 6]] and b == [[0, 3, 0]]
    assert ma['users'] == [10, 20] and mb['users'] == [30] and seen == {10, 20, 30}
    assert ma['serialized_sha256'] != mb['serialized_sha256']


@pytest.mark.parametrize('records', [[(1, [0, 1], b'a'), (1, [2, 3], b'b')],
                                    [(1, [0], b'a'), (2, [1, 2], b'b')]])
def test_duplicate_or_invalid_record_is_not_silently_skipped(records):
    with pytest.raises(ValueError):
        take_batch(iter(records), 2, set())


def case(size=2, offset=0):
    steps = [dict(step=i + 1, warmup=i == 0, optimizer_step_verified=True, gradient_tensors=152,
        data_seconds=1., forward_seconds=2., backward_update_seconds=3., total_seconds=6.,
        sid_loss=1., dense_loss=2., total_loss=3., grad_norm=1.,
        peak_allocated_bytes=10, peak_reserved_bytes=20) for i in range(4)]
    attempts = [dict(users=list(range(offset + i * size, offset + (i + 1) * size)),
                     records=size, serialized_sha256='a' * 64) for i in range(4)]
    return dict(batch_size=size, status='completed', steps=steps, attempts=attempts,
                initial_state_sha256='b' * 64, summary=summarize(steps, size, 100))


def test_warmup_excluded_from_throughput_but_included_in_peak():
    c = case()
    c['steps'][0]['total_seconds'] = 90.
    c['steps'][0]['peak_allocated_bytes'] = 19
    s = summarize(c['steps'], 2, 100)
    assert s['measured_samples'] == 6 and s['measured_seconds'] == 18
    assert s['samples_per_second'] == 1 / 3 and s['estimated_single_gpu_epoch_seconds'] == 300
    assert s['peak_allocated_bytes'] == 19


@pytest.mark.parametrize('corruption', [None, 'repeat', 'state', 'throughput', 'nan', 'step', 'time'])
def test_case_audit_checks_identity_arithmetic_and_actual_updates(corruption):
    cases = [case(), case(4, 8)]
    if corruption == 'repeat': cases[1]['attempts'][0]['users'][0] = 0
    if corruption == 'state': cases[1]['initial_state_sha256'] = 'c' * 64
    if corruption == 'throughput': cases[0]['summary']['samples_per_second'] *= 2
    if corruption == 'nan': cases[0]['steps'][0]['grad_norm'] = float('nan')
    if corruption == 'step': cases[0]['steps'][0]['optimizer_step_verified'] = False
    if corruption == 'time': cases[0]['steps'][0]['total_seconds'] = 7.
    if corruption:
        with pytest.raises(AssertionError):
            verify_cases(cases, [2, 4], 1, 3, 100, 1000)
    else:
        result = verify_cases(cases, [2, 4], 1, 3, 100, 1000)
        assert result['unique_consumed_users'] == 24 and not result['formal_batch_selected']


def test_oom_stops_larger_cases_and_partial_throughput_is_not_reported():
    failed = dict(batch_size=4, status='oom', error_type='OutOfMemoryError', error='fixture', steps=[], attempts=[])
    skipped = dict(batch_size=8, status='skipped_after_oom', steps=[], attempts=[])
    cases = [case(), failed, skipped]
    result = verify_cases(cases, [2, 4, 8], 1, 3, 100, 1000)
    assert result['stopped_after_oom'] and result['completed_batch_sizes'] == [2]
    bad = copy.deepcopy(cases)
    bad[1]['summary'] = {}
    with pytest.raises(AssertionError):
        verify_cases(bad, [2, 4, 8], 1, 3, 100, 1000)


def test_all_oom_is_not_an_accepted_capacity_result():
    failed = dict(batch_size=2, status='oom', error_type='OutOfMemoryError', error='fixture', steps=[], attempts=[])
    with pytest.raises(AssertionError):
        verify_cases([failed], [2], 1, 3, 100, 1000)
