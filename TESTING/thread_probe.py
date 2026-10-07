#!/usr/bin/env python
"""Measure how many cores are ACTUALLY used, vs what was requested.

The benchmark scripts call torch.set_num_threads(N), but on macOS PyTorch's CPU
matmul is dispatched to Apple Accelerate (BLAS_INFO=accelerate, USE_MKL=OFF).
Accelerate sizes its own thread pool from VECLIB_MAXIMUM_THREADS and ignores
both torch.set_num_threads() and OMP_NUM_THREADS.

We therefore report the effective parallelism = CPU time / wall time, measured
with getrusage. ~1.0x means the op really ran on one core no matter what was
requested; ~N x means N cores were genuinely busy.

Run as a subprocess per setting so the env var is read before torch import:

    python thread_probe.py --requested 4
    VECLIB_MAXIMUM_THREADS=1 python thread_probe.py --requested 4
"""
import argparse, json, os, resource, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--requested", type=int, default=4,
                help="value passed to torch.set_num_threads()")
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()

import torch  # noqa: E402


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def measure(fn, reps):
    fn()  # warm up (allocation + first-touch)
    c0, t0 = cpu_seconds(), time.perf_counter()
    for _ in range(reps):
        fn()
    wall = (time.perf_counter() - t0) / reps
    cpu = (cpu_seconds() - c0) / reps
    return wall, cpu


def main():
    torch.set_num_threads(args.requested)
    m, n, p = 978, 199112, 17          # shapes from the real workload
    x = torch.randn(m, n, dtype=torch.float64)
    omega = torch.randn(n, p, dtype=torch.float64)

    gemm_wall, gemm_cpu = measure(lambda: x @ omega, args.reps)      # (m,n)@(n,p)
    gemv_wall, gemv_cpu = measure(lambda: x @ omega[:, 0], args.reps)  # (m,n)@(n,)

    out = {
        "requested": args.requested,
        "torch_get_num_threads": torch.get_num_threads(),
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "env_VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "gemm_wall_s": round(gemm_wall, 4),
        "gemm_cpu_s": round(gemm_cpu, 4),
        "gemm_parallelism": round(gemm_cpu / gemm_wall, 2),
        "gemv_wall_s": round(gemv_wall, 4),
        "gemv_cpu_s": round(gemv_cpu, 4),
        "gemv_parallelism": round(gemv_cpu / gemv_wall, 2),
    }
    print("PROBE_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
