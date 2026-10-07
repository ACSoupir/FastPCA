#!/usr/bin/env python3
"""Can the single-core GEMM be avoided?

torch's macOS wheel is built against Apple Accelerate (BLAS_INFO=accelerate),
which runs these GEMMs on the AMX matrix unit from a single core.  This script
tests the two escape routes:

  1. use a different BLAS library (numpy in this env links one that threads)
  2. move the work to the Metal GPU (MPS)

Run:  KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/FastPCA/bin/python \
          TESTING/blas_backend_probe.py
"""
import contextlib, io, resource, time

import numpy as np
import torch


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def bench(fn, reps=3):
    """Return (best wall ms, cpu/wall).  Warm up first."""
    fn()
    best, par = float("inf"), float("nan")
    for _ in range(reps):
        c0, t0 = cpu_seconds(), time.perf_counter()
        fn()
        dt, dc = time.perf_counter() - t0, cpu_seconds() - c0
        if dt < best:
            best, par = dt, dc / dt
    return best * 1e3, par


def main():
    N = 3072
    print(f"torch {torch.__version__} | BLAS_INFO="
          f"{'accelerate' if 'BLAS_INFO=accelerate' in str(torch.__config__.show()) else '?'}"
          f" | numpy {np.__version__}")

    a64 = torch.randn(N, N, dtype=torch.float64)
    b64 = torch.randn(N, N, dtype=torch.float64)
    na, nb = a64.numpy(), b64.numpy()

    print(f"\n--- fp64 {N}x{N} GEMM ---")
    for th in (1, 4):
        torch.set_num_threads(th)
        wall, par = bench(lambda: a64 @ b64)
        print(f"  torch threads={th}       wall={wall:8.1f} ms   cpu/wall={par:.2f}x")
    wall, par = bench(lambda: na @ nb)
    print(f"  numpy (own BLAS)     wall={wall:8.1f} ms   cpu/wall={par:.2f}x")

    print(f"\n--- fp32 {N}x{N} GEMM ---")
    a32, b32 = a64.float(), b64.float()
    torch.set_num_threads(4)
    wall, par = bench(lambda: a32 @ b32)
    print(f"  torch CPU fp32       wall={wall:8.1f} ms   cpu/wall={par:.2f}x")

    if torch.backends.mps.is_available():
        am, bm = a32.to("mps"), b32.to("mps")

        def mps_gemm():
            am @ bm
            torch.mps.synchronize()

        wall, par = bench(mps_gemm)
        print(f"  MPS GPU fp32         wall={wall:8.1f} ms   cpu/wall={par:.2f}x")
        try:
            torch.randn(8, 8, dtype=torch.float64, device="mps")
            print("\nMPS float64: supported (unexpected)")
        except TypeError as e:
            print(f"\nMPS float64: REFUSED -- {e}")
    else:
        print("\nMPS not available on this machine")


if __name__ == "__main__":
    main()
