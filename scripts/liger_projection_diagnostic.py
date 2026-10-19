"""Bounded CUDA correctness and timing checks for projection execution kernels."""
import copy
import math
import statistics
import time

import torch

from src.baselines.liger_full_catalog import full_catalog_cross_entropy
from src.baselines.liger_projection import use_linear_residuals


def diagnose(projection):
    checks, timings = [], []
    # Diagnostics must not change the real run's weights or dropout RNG state.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for precision in ('float64', 'float32', 'amp_float16'):
            dtype = torch.float64 if precision == 'float64' else torch.float32
            for training in (False, True):
                a = copy.deepcopy(projection).to(dtype).train(training)
                b = copy.deepcopy(a)
                assert use_linear_residuals(b) == 3
                assert list(a.state_dict()) == list(b.state_dict())
                x = torch.randn(7, 128, dtype=dtype, device='cuda', requires_grad=True)
                cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
                with torch.autocast('cuda', dtype=torch.float16, enabled=precision == 'amp_float16'):
                    expected = a(x)
                after_cpu, after_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
                torch.set_rng_state(cpu)
                torch.cuda.set_rng_state(gpu)
                with torch.autocast('cuda', dtype=torch.float16, enabled=precision == 'amp_float16'):
                    actual = b(x)
                assert torch.equal(after_cpu, torch.get_rng_state()) and torch.equal(after_gpu, torch.cuda.get_rng_state())
                tol = {'float64': (1e-10, 1e-9), 'float32': (1e-5, 1e-4), 'amp_float16': (2e-3, 2e-2)}[precision]
                torch.testing.assert_close(actual, expected, atol=tol[0], rtol=tol[1])
                ga = torch.autograd.grad(expected.to(torch.float64).square().mean(), (x, *a.parameters()))
                gb = torch.autograd.grad(actual.to(torch.float64).square().mean(), (x, *b.parameters()))
                for p, q in zip(ga, gb):
                    torch.testing.assert_close(p, q, atol=tol[0], rtol=tol[1])
                checks.append(dict(precision=precision, training=training, parameter_tensors=len(ga) - 1,
                    output_max_abs_error=(actual - expected).abs().max().item(),
                    gradient_max_abs_error=max((p - q).abs().max().item() for p, q in zip(ga, gb)),
                    absolute_tolerance=tol[0], relative_tolerance=tol[1], rng_equal=True, passed=True))
                del a, b, x, actual, expected, ga, gb
        # One warm-up and three measurements per kernel/chunk size. GPU content
        # removes disk/host-transfer effects; this is NOT end-to-end training time.
        for chunk in (8192, 65536):
            raw = torch.randn(chunk, 128, device='cuda')
            q = torch.randn(32, 128, device='cuda', requires_grad=True)
            labels = torch.arange(32, device='cuda')
            for backend in ('conv1d', 'linear'):
                p = copy.deepcopy(projection).train()
                if backend == 'linear':
                    use_linear_residuals(p)
                values = []
                for iteration in range(4):
                    p.zero_grad(set_to_none=True)
                    q.grad = None
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                    loss = full_catalog_cross_entropy(q, labels, p, raw, chunk_size=chunk)
                    torch.cuda.synchronize()
                    forward = time.monotonic() - started
                    loss.backward()
                    torch.cuda.synchronize()
                    total = time.monotonic() - started
                    if iteration:
                        values.append(dict(forward_seconds=forward, total_seconds=total,
                            backward_recompute_seconds=total - forward,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                    assert torch.isfinite(loss).item() and all(torch.isfinite(t.grad).all().item() for t in p.parameters())
                    del loss
                timings.append(dict(backend=backend, chunk_size=chunk, measured=values,
                    median_forward_seconds=statistics.median(v['forward_seconds'] for v in values),
                    median_total_seconds=statistics.median(v['total_seconds'] for v in values)))
                del p
            del raw, q, labels
    return dict(checks=checks, timings=timings, measured_repeats=3, warmup_repeats=1,
                scope='Projection/full-catalog-loss kernel microbenchmark, GPU content, not model throughput')


def verify_diagnostic(report):
    assert [(x['precision'], x['training']) for x in report['checks']] == [
        (precision, training) for precision in ('float64', 'float32', 'amp_float16') for training in (False, True)]
    for check in report['checks']:
        assert check['passed'] and check['rng_equal'] and check['parameter_tensors'] == 17
        assert math.isfinite(check['output_max_abs_error']) and math.isfinite(check['gradient_max_abs_error'])
    assert report['measured_repeats'] == 3 and report['warmup_repeats'] == 1
    assert {(x['backend'], x['chunk_size']) for x in report['timings']} == {
        (backend, chunk) for backend in ('conv1d', 'linear') for chunk in (8192, 65536)}
    assert len(report['timings']) == 4
    for timing in report['timings']:
        assert len(timing['measured']) == 3
        for value in timing['measured']:
            assert all(math.isfinite(value[k]) and value[k] > 0 for k in
                       ('forward_seconds', 'total_seconds', 'backward_recompute_seconds', 'peak_allocated_bytes'))
            assert abs(value['total_seconds'] - value['forward_seconds'] - value['backward_recompute_seconds']) < 1e-6
        assert timing['median_forward_seconds'] == statistics.median(x['forward_seconds'] for x in timing['measured'])
        assert timing['median_total_seconds'] == statistics.median(x['total_seconds'] for x in timing['measured'])
