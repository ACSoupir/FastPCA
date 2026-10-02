"""Correctness + speed/memory benchmark for FastPCA matrix transforms.

Each backend runs in a fresh subprocess. Correctness (small shapes) and
performance/memory (large shapes, no reference buffers) are separate runs so the
reference does not pollute peak RSS.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "inst", "python"))

import numpy as np

CHECK_SHAPES = [(800, 300)]
PERF_SHAPES = [(2000, 500), (6000, 1500), (15000, 2500)]
CONFIGS = [dict(log2=1, transpose=0, scale=1), dict(log2=1, transpose=1, scale=1)]


def make_matrix(m, n, seed=0):
    rng = np.random.default_rng(seed)
    return (np.abs(rng.standard_normal((m, n))) * 20 + 1).astype(np.float64)


def reference(A, log2=0, transpose=0, scale=1):
    X = np.log2(A) if log2 else A.copy()
    if transpose:
        X = X.T
    if scale:
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)          # torch/tinygrad default correction=1
        X = (X - mean) / std
    return np.ascontiguousarray(X)


def backend_fn(backend):
    if backend == "torch":
        from torch_transform import transform_py
        return lambda A, cfg: transform_py(A, **cfg)
    if backend == "tinygrad":
        from tinygrad_transform_optim import transform_py_tg
        return lambda A, cfg: transform_py_tg(A, **cfg)
    if backend == "numpy":
        return lambda A, cfg: reference(A, **cfg)
    raise SystemExit("unknown backend " + backend)


def run_check(backend):
    fn = backend_fn(backend)
    rows = []
    for (m, n) in CHECK_SHAPES:
        A = make_matrix(m, n)
        for cfg in CONFIGS:
            ref = reference(A, **cfg)
            out = fn(A, cfg)
            err = float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-12))
            rows.append(dict(shape=[m, n], transpose=cfg["transpose"],
                             relerr=float(f"{err:.3e}"), shape_ok=list(out.shape) == list(ref.shape)))
    print("RESULT" + json.dumps(dict(backend=backend, rows=rows)))


def run_perf(backend):
    import time, resource
    fn = backend_fn(backend)
    rows = []
    for (m, n) in PERF_SHAPES:
        A = make_matrix(m, n)
        for cfg in CONFIGS:
            fn(A, cfg)                              # warm up / compile
            times = []
            for _ in range(3):
                t0 = time.perf_counter(); fn(A, cfg); times.append((time.perf_counter() - t0) * 1e3)
            rows.append(dict(shape=[m, n], transpose=cfg["transpose"], ms=round(sorted(times)[1], 2)))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    print("RESULT" + json.dumps(dict(backend=backend, peak_mb=round(peak / 1e6, 1), rows=rows)))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "drive"
    if mode == "--check":
        run_check(sys.argv[2])
    elif mode == "--perf":
        run_perf(sys.argv[2])
    else:
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", XDG_CACHE_HOME=os.path.join(HERE, ".cache"))
        def call(mode, backend):
            p = subprocess.run([sys.executable, os.path.abspath(__file__), mode, backend],
                               capture_output=True, text=True, env=env)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
            return json.loads(line[0][len("RESULT"):]) if line else None
        print("== correctness (relative error vs float64 NumPy) ==")
        for b in ("numpy", "torch", "tinygrad"):
            d = call("--check", b)
            if not d: print(f"  {b}: FAILED"); continue
            for r in d["rows"]:
                print(f"  {b:>9} {str(r['shape']):>13} transpose={r['transpose']} relerr={r['relerr']:.2e} shape_ok={r['shape_ok']}")
        print("\n== performance / peak RSS (whole process, no reference) ==")
        for b in ("numpy", "torch", "tinygrad"):
            d = call("--perf", b)
            if not d: print(f"  {b}: FAILED"); continue
            times = ", ".join(f"{r['shape']} t={r['transpose']}: {r['ms']:.1f} ms" for r in d["rows"])
            print(f"  {b:>9} peak_rss={d['peak_mb']:>6.1f} MB | {times}")
