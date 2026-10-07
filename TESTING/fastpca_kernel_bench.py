#!/usr/bin/env python
"""FastPCA's shipped randomized-SVD kernel, called directly from Python.

    python TESTING/fastpca_kernel_bench.py --x ... --m 978 --n 2000 \
        --k 10 --p 10 --q_iter 2 --threads 4

This is the same inst/python/torch_svd_func.py::randomized_svd_py that
FastPCA(backend = "pytorch") runs, but without reticulate and the R matrix
copy. It isolates how much of that backend's cost is the kernel itself versus
the R/Python boundary.
"""
import argparse
import importlib.util
import json
import os
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch        # noqa: E402

_T0 = time.perf_counter()


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def load_kernel(path):
    spec = importlib.util.spec_from_file_location("fastpca_torch_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bin(path, m, n):
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != m * n:
        raise SystemExit(f"{path}: expected {m*n} values, got {raw.size}")
    return raw.reshape((n, m), order="C").T.copy(order="F")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--q_iter", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--center", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--kernel",
                    default="inst/python/torch_svd_func.py")
    ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--save-bin", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    proj_gb = (args.m * args.n * 8) / 1e9
    if proj_gb > args.max_gb:
        out = {"impl": "fastpca_kernel", "tag": args.tag, "m": args.m,
               "n": args.n, "status": "skipped",
               "reason": f"input alone is {proj_gb:.3g} GB > cap "
                         f"{args.max_gb:g} GB"}
        print("RESULT_JSON " + json.dumps(out))
        return

    mod = load_kernel(args.kernel)

    t_load0 = time.perf_counter()
    A_np = load_bin(args.x, args.m, args.n)
    if args.center:
        A_np -= A_np.mean(axis=0)      # in place, same centring as the others
    t_load = time.perf_counter() - t_load0

    # warm-up so the first-call allocator/thread-pool cost is not charged
    _ = mod.randomized_svd_py(np.ascontiguousarray(A_np[:40, :12]), k=3, p=2,
                              q_iter=1, device="CPU", cores=args.threads)

    times = []
    for r in range(args.repeats):
        t0 = time.perf_counter()
        U, S, Vh = mod.randomized_svd_py(A_np, k=args.k, p=args.p,
                                         q_iter=args.q_iter, device="CPU",
                                         cores=args.threads)
        times.append(time.perf_counter() - t0)
        print(f"[fastpca_kernel] repeat {r+1}: {times[-1]:.3f}s", flush=True)

    best = min(times)
    S = np.asarray(S, dtype="<f8")
    Vh = np.asarray(Vh)
    out = {
        "impl": "fastpca_kernel", "tag": args.tag, "m": args.m, "n": args.n,
        "threads": args.threads, "k": args.k, "p_oversample": args.p,
        "q_iter": args.q_iter, "center": bool(args.center),
        "load_s": round(t_load, 3), "times_s": [round(t, 4) for t in times],
        "best_s": round(best, 4), "sv": [round(float(x), 8) for x in S],
        "eig": [round(float(x * x), 8) for x in S],
        "peak_rss_gb": round(peak_gb(), 3),
        "wall_total_s": round(time.perf_counter() - _T0, 2),
    }
    print(f"[fastpca_kernel] best={best:.3f}s peakRSS={out['peak_rss_gb']}GB "
          f"sv[:5]={np.round(S[:5], 4).tolist()}", flush=True)

    if args.save_bin:
        v = Vh.T                          # nvar x k
        np.asarray(S ** 2, dtype="<f8").tofile(args.save_bin + ".eig.f64")
        np.asarray(v, dtype="<f8").flatten(order="F").tofile(
            args.save_bin + ".loadings.f64")
        with open(args.save_bin + ".json", "w") as fh:
            json.dump({"nvar": int(v.shape[0]), "ncomp": int(v.shape[1])}, fh)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh)
    print("RESULT_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
