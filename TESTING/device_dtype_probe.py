#!/usr/bin/env python
"""Probe the precision x device matrix for torch on this machine.

Answers, with evidence:
  1. which dtype/device combinations are legal (fp64 on MPS?),
  2. whether the iterative kernels' linalg ops exist on MPS,
  3. what each combination actually costs (wall + real cores used),
so a precision=/device= design can be grounded rather than guessed.

Run:  KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/FastPCA/bin/python \
        TESTING/device_dtype_probe.py
"""
import os, resource, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

def cpu_wall(fn):
    u0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    out = fn()
    wall = time.perf_counter() - t0
    u1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (u1.ru_utime - u0.ru_utime) + (u1.ru_stime - u0.ru_stime)
    return out, wall, cpu / max(wall, 1e-9)

def attempt(label, fn):
    try:
        r = fn()
        print(f"  OK   {label}: {r}")
    except Exception as e:
        msg = " ".join(str(e).split())
        print(f"  FAIL {label}: {type(e).__name__}: {msg[:150]}")

print("torch", torch.__version__)
cfg = dict(l.split(": ") for l in torch.__config__.show().splitlines() if ": " in l)
print("  BLAS:", cfg.get("BLAS_INFO"), "| LAPACK:", cfg.get("LAPACK_INFO"),
      "| MKL:", cfg.get("USE_MKL"), "| CUDA:", torch.cuda.is_available(),
      "| MPS built/runnable:", torch.backends.mps.is_built(), torch.backends.mps.is_available())
print("  default dtype:", torch.get_default_dtype())

m = torch.device("mps")
print("\n[1] legal dtype x device combinations (the hard constraint)")
attempt("float32 -> mps", lambda: torch.zeros(4).to(m).dtype)
attempt("float64 -> mps", lambda: torch.zeros(4, dtype=torch.float64).to(m))
attempt("float16 -> mps", lambda: torch.zeros(4, dtype=torch.float16).to(m))
attempt("bfloat16 -> mps", lambda: torch.zeros(4, dtype=torch.bfloat16).to(m))
attempt("mps fp32 matmul", lambda: tuple((torch.randn(64, 8).to(m) @ torch.randn(8, 4).to(m)).shape))

print("\n[2] linalg ops the iterative kernels need, on MPS")
attempt("linalg.svd(fp32)", lambda: tuple(torch.linalg.svd(torch.randn(20, 10).to(m), full_matrices=False)[1].shape))
attempt("linalg.svdvals(fp32)", lambda: tuple(torch.linalg.svdvals(torch.randn(20, 10).to(m)).shape))
attempt("linalg.qr(fp32)", lambda: tuple(torch.linalg.qr(torch.randn(20, 10).to(m))[0].shape))
attempt("linalg.eigh(fp32)", lambda: tuple(torch.linalg.eigh(torch.randn(10, 4) @ torch.randn(10, 4).T.to(m))[0].shape))

print("\n[3] cost of each legal combination, 3072x3072 GEMM")
N = 1536 if os.environ.get("QUICK") else 3072
for dtype in (torch.float64, torch.float32):
    for threads in (1, 4):
        A = torch.randn(N, N, dtype=dtype); B = torch.randn(N, N, dtype=dtype)
        _, wall, par = cpu_wall(lambda: A @ B)
        gflops = 2 * N ** 3 / wall / 1e9
        print(f"  cpu {str(dtype):>14} threads={threads}: {wall*1e3:8.1f} ms  "
              f"{gflops:6.0f} GFLOPS  cpu/wall={par:.2f}x")
        del A, B
for dtype in (torch.float32,):
    try:
        Am = torch.randn(N, N, dtype=dtype).to(m); Bm = torch.randn(N, N, dtype=dtype).to(m)
        def go():
            r = Am @ Bm
            torch.mps.synchronize()
            return r
        _, wall, par = cpu_wall(go)
        print(f"  mps {str(dtype):>14}              : {wall*1e3:8.1f} ms  "
              f"{2*N**3/wall/1e9:6.0f} GFLOPS  cpu/wall={par:.2f}x")
    except Exception as e:
        print("  mps fp32:", " ".join(str(e).split())[:120])

print("\n[4] what fp32 costs in accuracy (svd of a 60x17 B-like matrix)")
torch.manual_seed(0)
X = torch.randn(60, 17, dtype=torch.float64)
s64 = torch.linalg.svdvals(X.to(torch.device("cpu")))
for dev, dt in (("cpu", torch.float32), ("mps", torch.float32)):
    try:
        s = torch.linalg.svdvals(X.to(dtype=dt, device=torch.device(dev))).to(torch.float64)
        rel = ((s - s64).abs() / s64.abs()).max().item()
        print(f"  svdvals on {dev:>3} fp32 vs cpu fp64: max rel err {rel:.2e}")
    except Exception as e:
        print(f"  svdvals on {dev} fp32: {' '.join(str(e).split())[:100]}")
