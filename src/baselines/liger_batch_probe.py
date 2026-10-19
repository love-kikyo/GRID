"""Real-record batching and auditable capacity summaries; no model selection."""
import hashlib
import math
import statistics


def take_batch(records, size, seen_users):
    if size < 1:
        raise ValueError('Positive batch size required')
    sequences, users, digest = [], [], hashlib.sha256()
    for _ in range(size):
        user, sequence, serialized = next(records)
        if user in seen_users:
            raise ValueError('Repeated user in the capacity probe')
        if not 2 <= len(sequence) <= 1000:
            raise ValueError('Invalid history-last sequence length')
        seen_users.add(user)
        sequences.append(sequence)
        users.append(user)
        digest.update(serialized)
    return sequences, dict(users=users, records=len(users), serialized_sha256=digest.hexdigest())


def summarize(steps, batch_size, train_records):
    measured = [s for s in steps if not s['warmup']]
    if not measured:
        raise ValueError('No measured updates')
    elapsed = sum(s['total_seconds'] for s in measured)
    samples = batch_size * len(measured)
    return dict(measured_steps=len(measured), measured_samples=samples,
        measured_seconds=elapsed, samples_per_second=samples / elapsed,
        median_step_seconds=statistics.median(s['total_seconds'] for s in measured),
        median_data_seconds=statistics.median(s['data_seconds'] for s in measured),
        peak_allocated_bytes=max(s['peak_allocated_bytes'] for s in steps),
        peak_reserved_bytes=max(s['peak_reserved_bytes'] for s in steps),
        estimated_single_gpu_epoch_seconds=train_records * elapsed / samples)


def verify_cases(cases, batch_sizes, warmup_steps, measured_steps, train_records, gpu_bytes):
    assert [c['batch_size'] for c in cases] == batch_sizes
    stopped, all_users, initial_states, completed = False, [], [], []
    for case in cases:
        if stopped:
            assert case['status'] == 'skipped_after_oom' and not case['steps'] and not case['attempts']
            continue
        assert case['status'] in {'completed', 'oom'}
        assert len(case['attempts']) in {len(case['steps']), len(case['steps']) + 1}
        for attempt in case['attempts']:
            assert len(attempt['users']) == attempt['records'] == case['batch_size']
            assert len(attempt['serialized_sha256']) == 64
            assert all(c in '0123456789abcdef' for c in attempt['serialized_sha256'])
            all_users.extend(attempt['users'])
        for index, step in enumerate(case['steps']):
            assert step['step'] == index + 1 and step['warmup'] == (index < warmup_steps)
            assert step['optimizer_step_verified'] and step['gradient_tensors'] == 152
            for key in ('data_seconds', 'forward_seconds', 'backward_update_seconds', 'total_seconds',
                        'sid_loss', 'dense_loss', 'total_loss', 'grad_norm'):
                assert math.isfinite(step[key]) and step[key] > 0
            assert abs(step['data_seconds'] + step['forward_seconds'] + step['backward_update_seconds'] - step['total_seconds']) < 1e-6
            assert abs(step['sid_loss'] + step['dense_loss'] - step['total_loss']) < 1e-4
            assert 0 < step['peak_allocated_bytes'] <= step['peak_reserved_bytes'] <= gpu_bytes
        if case['status'] == 'completed':
            assert len(case['steps']) == len(case['attempts']) == warmup_steps + measured_steps
            assert case['summary'] == summarize(case['steps'], case['batch_size'], train_records)
            assert len(case['initial_state_sha256']) == 64
            initial_states.append(case['initial_state_sha256'])
            completed.append(case['batch_size'])
        else:
            assert case['error_type'] == 'OutOfMemoryError' and case['error']
            assert 'summary' not in case  # No throughput claim for partial execution.
            stopped = True
    assert completed and len(set(initial_states)) == 1
    assert len(all_users) == len(set(all_users))
    return dict(completed_batch_sizes=completed, unique_consumed_users=len(all_users),
                stopped_after_oom=stopped, largest_completed_batch=max(completed),
                formal_batch_selected=False)
