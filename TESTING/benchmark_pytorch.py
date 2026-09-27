"""Benchmark and validate FastPCA randomized-SVD kernels.

Usage (from the repo root):
    KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/FastPCA/bin/python \
        TESTING/benchmark_pytorch.py --check
    KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/FastPCA/bin/python \
        TESTING/benchmark_pytorch.py --devices cpu,mps --sizes 4000x1000

Compares the in-package baseline kernel against the optimized variants in
TESTING/pytorch_rsvd.py on speed and reconstruction accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# torch 2.9 has no native MPS linalg.qr; let it fall back to CPU instead of error.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pytorch_rsvd import KERNELS, resolve_device  # noqa: E402


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def parse_sizes(text):
    sizes = []
    for part in text.split(","):
        m, n = part.lower().split("x")
        sizes.append((int(m), int(n)))
    return sizes


def make_matrix(m, n, seed=0, model="decay"):
    rng = np.random.default_rng(seed)
    if model == "shifted":
        # pathological rank-1-dominated matrix (stress test for low precision)
        return (rng.standard_normal((m, n)) + 10.0).astype(np.float64)
    # realistic PCA-like matrix: rank-r signal with geometrically decaying
    # spectrum + small dense noise (spectrum of a real log-normalised dataset)
    r = min(m, n, 120)
    U, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))
    s = 1000.0 * (0.85 ** np.arange(r))
    A = (U * s) @ V.T
    A += 0.1 * rng.standard_normal((m, n))
    return A


def reconstruction_error(A_np, U, S, Vh, k):
    Uk = np.asarray(U[:, :k], dtype=np.float64)
    Sk = np.asarray(S[:k], dtype=np.float64)
    Vhk = np.asarray(Vh[:k, :], dtype=np.float64)
    approx = (Uk * Sk) @ Vhk
    return float(np.linalg.norm(A_np - approx) / np.linalg.norm(A_np))


def reference_svals(A_t, k):
    return torch.linalg.svdvals(A_t)[:k].detach().to("cpu").numpy()


def time_kernel(fn, A_np, k, p, q_iter, device_name, dtype, cores,
                warmup=1, repeats=3):  # noqa: C901
    dev = resolve_device(device_name)
    times = []
    out = None
    for r in range(warmup + repeats):
        sync(dev)
        t0 = time.perf_counter()
        out = fn(A_np, k=k, p=p, q_iter=q_iter, device=device_name,
                 dtype=dtype, cores=cores)
        sync(dev)
        dt = time.perf_counter() - t0
        if r >= warmup:
            times.append(dt)
    return min(times), out


def run_check():
    print("=== correctness check (small matrices) ===")
    rng = np.random.default_rng(7)
    A64 = (rng.standard_normal((600, 250)) + 10.0)
    k, p, q = 20, 8, 2
    ref_s = np.linalg.svd(A64, compute_uv=False)[:k]
    for name in KERNELS:
        fn = KERNELS[name]
        U, S, Vh = fn(A64, k=k, p=p, q_iter=q, device="cpu",
                      dtype=torch.float64, cores=4)
        assert U.shape == (600, k + p), (name, U.shape)
        assert S.shape == (k + p,), (name, S.shape)
        assert Vh.shape == (k + p, 250), (name, Vh.shape)
        sv_err = np.max(np.abs(S[:k] - ref_s) / np.abs(ref_s))
        rec = reconstruction_error(A64, U, S, Vh, k)
        print(f"  {name:11s} shape_ok=True topk_sv_relerr={sv_err:.3e} "
              f"recon_relFro={rec:.4f}")
    print("  all variants produced the expected shapes")


def run_bench(args):
    sizes = parse_sizes(args.sizes)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    rows = []
    print(f"k={args.k} p={args.p} q_iter={args.q_iter} cores={args.cores} "
          f"model={args.model} warmup={args.warmup} repeats={args.repeats}")
    for (m, n) in sizes:
        A64 = make_matrix(m, n, seed=args.seed, model=args.model)
        for dev_name in devices:
            dev = resolve_device(dev_name)
            dtypes = []
            if args.dtypes == "auto":
                dtypes.append(torch.float64)
                if dev.type != "mps":
                    pass
                dtypes.append(torch.float32)
            else:
                for dt in args.dtypes.split(","):
                    dtypes.append(getattr(torch, "float" + dt.strip()))
            for dtype in dtypes:
                if dev.type == "mps" and dtype == torch.float64:
                    continue
                A = A64 if dtype == torch.float64 else A64.astype(np.float32)
                # accuracy reference
                try:
                    ref_s = np.linalg.svd(A, compute_uv=False)[:args.k]
                except Exception:
                    ref_s = None
                for name in variants:
                    fn = KERNELS[name]
                    try:
                        t, out = time_kernel(fn, A, args.k, args.p,
                                             args.q_iter, dev_name, dtype,
                                             args.cores, warmup=args.warmup,
                                             repeats=args.repeats)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {name} on {dev_name}/{dtype}: FAILED ({exc})")
                        continue
                    U, S, Vh = out
                    rec = reconstruction_error(
                        A.astype(np.float64), U, S, Vh, args.k)
                    sv_err = float("nan")
                    if ref_s is not None:
                        sv_err = float(np.max(
                            np.abs(S[:args.k] - ref_s) / np.abs(ref_s)))
                    rows.append(dict(m=m, n=n, device=str(dev), dtype=str(dtype),
                                     variant=name, seconds=t,
                                     recon_relFro=rec, topk_sv_relerr=sv_err))
                    print(f"  {m:>7d}x{n:<6d} {str(dev):5s} {str(dtype):15s} "
                          f"{name:11s} t={t*1e3:9.2f} ms  "
                          f"recon={rec:.4f} sv_relerr={sv_err:.2e}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print("wrote", args.json)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sizes", default="2000x500,6000x1000")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--q-iter", dest="q_iter", type=int, default=2)
    ap.add_argument("--variants", default="baseline,direct,triangular,lowrank")
    ap.add_argument("--devices", default="cpu")
    ap.add_argument("--dtypes", default="auto",
                    help='"auto" -> float64+float32, or e.g. "32", "64"')
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--model", default="decay",
                    choices=["decay", "shifted"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.check:
        run_check()
        return
    run_bench(args)


if __name__ == "__main__":
    main()
