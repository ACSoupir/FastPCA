#!/usr/bin/env python
"""Which ops actually respond to torch.set_num_threads()?

Distinguishes the two thread pools in play on macOS:
  * ATen intra-op pool (OpenMP/pthreadpool) -- honours torch.set_num_threads().
    Used by elementwise kernels and reductions over large tensors.
  * Apple Accelerate BLAS -- used for matmul; sizes itself independently and
    may decline to thread skinny shapes at all.
"""
import argparse, json, resource, time

ap = argparse.ArgumentParser()
ap.add_argument("--requested", type=int, default=4)
args = ap.parse_args()

import torch  # noqa: E402


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def measure(fn, reps=3):
    fn()
    c0, t0 = cpu_seconds(), time.perf_counter()
    for _ in range(reps):
        fn()
    wall = (time.perf_counter() - t0) / reps
    cpu = (cpu_seconds() - c0) / reps
    return round(wall, 4), round(cpu, 4), round(cpu / wall, 2)


torch.set_num_threads(args.requested)
m, n = 978, 199112
x = torch.randn(m, n, dtype=torch.float64)

out = {"requested": args.requested, "get_num_threads": torch.get_num_threads()}

# 1. elementwise (ATen pool)
out["elementwise"] = measure(lambda: x.mul(2.0))
# 2. reduction (ATen pool)
out["reduction_sum"] = measure(lambda: x.sum())
# 3. skinny GEMM (Accelerate) -- the shape this benchmark actually uses
p = 17
omega = torch.randn(n, p, dtype=torch.float64)
out["skinny_gemm"] = measure(lambda: x @ omega)
# 4. large square GEMM (Accelerate, compute bound -> most likely to thread)
s = 2048
a = torch.randn(s, s, dtype=torch.float64)
b = torch.randn(s, s, dtype=torch.float64)
out["square_gemm_2048"] = measure(lambda: a @ b, reps=2)

print("PROBE2_JSON " + json.dumps(out))
