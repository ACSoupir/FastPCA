"""Native-Metal tinygrad randomized SVD.

Approach
--------
tinygrad's built-in `Tensor.qr` materialises a full m x m Q (m=15000 -> ~0.9 GB
for float32 alone) and its `Tensor.svd` runs hundreds of Jacobi rounds, so both
are unusable for the range finder.

Instead this kernel uses **Newton-Schulz polar iteration**, which only ever
touches m x b and b x b tensors:

    orth(X) = X (X^T X)^{-1/2}     via the quintic iteration
    X <- a X + b (X G) + c (X G^2),  G = X^T X,  a,b,c = (15,-10,3)/8

Every O(m*n*b) operation therefore runs on Metal and no m x m tensor exists.
Newton-Schulz also *grows* small singular directions (by ~15/8 per step) rather
than losing them, which is what keeps fp32 stable; the classic direct power
scheme collapses in fp32.

The whole range finder is wrapped in `TinyJit`, which removes the per-op Python
dispatch that otherwise dominates (~220 ms -> ~6 ms steady state on 2000x500).
The only host-side step is the SVD of B = Q^T A (b x n, b ~= k+p).

Memory safeguard: every public entry point checks peak RSS and aborts past
`MEM_LIMIT_BYTES` (20 GB) so a bad shape cannot take the machine down.
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_HERE, ".cache"))

import numpy as np  # noqa: E402

try:
    from tinygrad import Tensor, TinyJit
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tinygrad required; run with TESTING/.venv-tg/bin/python") from exc

# Peak-memory safeguard. Override with FASTPCA_TG_MEM_GB if a machine has more
# headroom; the default keeps a runaway shape from taking 36 GB of RAM down.
MEM_LIMIT_GB = float(os.environ.get("FASTPCA_TG_MEM_GB", "20"))
MEM_LIMIT_BYTES = int(MEM_LIMIT_GB * (1024 ** 3))

# Quintic Newton-Schulz polar-iteration coefficients. a + b + c == 1, so after
# Frobenius normalisation every non-zero singular value converges to 1.
_NS_A, _NS_B, _NS_C = 15.0 / 8.0, -10.0 / 8.0, 3.0 / 8.0


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def check_mem(where: str = "") -> None:
    used = peak_rss_bytes()
    if used > MEM_LIMIT_BYTES:
        raise MemoryError(
            f"peak RSS {used / 1e9:.2f} GB exceeds the "
            f"{MEM_LIMIT_BYTES / 1e9:.0f} GB safeguard ({where})"
        )


def _estimate_bytes(m: int, n: int, b: int) -> int:
    """Upper-bound working set of one call (float32 buffers, in bytes).

    Counts the input A plus Omega/W/NS temporaries and Y/Q/B, all of which are
    real allocations; used to refuse a shape *before* it is allocated.
    """
    return 4 * (m * n + m * b * 3 + n * b * 10 + b * b * 8)


def check_shape(m: int, n: int, b: int) -> None:
    """Refuse a shape whose projected working set exceeds the memory limit."""
    est = _estimate_bytes(m, n, b)
    if est > MEM_LIMIT_BYTES:
        raise MemoryError(
            f"projected working set {est / 1e9:.2f} GB for {m}x{n}, b={b} "
            f"exceeds the {MEM_LIMIT_GB:.0f} GB safeguard "
            f"(set FASTPCA_TG_MEM_GB to raise)"
        )


def _ns_orth(X: Tensor, steps: int) -> Tensor:
    """Orthogonal polar factor of X (p x b), matmul-only.

    The quintic update is written with 3 matmuls instead of 4 by reusing
    T = X G:  X <- a X + b T + c (T G),  since (X G) G == X (G G).
    """
    X = X / (X.square().sum(axis=(-2, -1), keepdim=True).sqrt() + 1e-12)
    for _ in range(int(steps)):
        G = X.transpose(-2, -1) @ X          # b x b
        T = X @ G                            # p x b
        X = _NS_A * X + _NS_B * T + _NS_C * (T @ G)
    return X


def warmup(m, n, k=50, p=10, q_iter=2, ns_steps=16):
    """Pre-trace the JIT graph for one shape so later calls skip compilation."""
    b = min(int(k) + int(p), m, n)
    A = Tensor(np.zeros((m, n), dtype=np.float32), device="METAL").realize()
    Omega = Tensor(np.zeros((n, b), dtype=np.float32), device="METAL").realize()
    _get_jit(m, n, b, q_iter, ns_steps)(A, Omega)
    del A, Omega


_JIT_CACHE: dict = {}


def _get_jit(m, n, b, q_iter: int, ns_steps: int):
    """Return a TinyJit'd range finder specialised for one exact shape/config.

    tinygrad's TinyJit asserts on input-shape changes, so each (shape, config)
    needs its own jitted callable.
    """
    key = (int(m), int(n), int(b), int(q_iter), int(ns_steps))
    fn = _JIT_CACHE.get(key)
    if fn is None:
        qi, nst = int(q_iter), int(ns_steps)

        @TinyJit
        def _core(A, Omega):
            Y = A @ Omega
            for _ in range(qi):
                Z = _ns_orth(A.transpose(-2, -1) @ Y, nst)
                Y = A @ Z
            Q = _ns_orth(Y, nst)
            B = Q.transpose(-2, -1) @ A
            return B.realize(), Q.realize()
        fn = _JIT_CACHE[key] = _core
    return fn


def randomized_svd_tg(A_np, k, p=10, q_iter=2, seed=None, ns_steps=16,
                      device="METAL"):
    """Randomized SVD matching the package baseline's output contract.

    Returns numpy (U, S, Vh) with U m x b, S length b, Vh b x n and
    b = min(k + p, m, n) -- identical to inst/python/torch_svd_func.py.
    """
    check_mem("start")
    A_np = np.asarray(A_np)
    m, n = A_np.shape
    b = min(int(k) + int(p), m, n)
    if b <= 0:
        raise ValueError("k + p must be positive and no larger than min(m, n)")
    check_shape(m, n, b)

    A = Tensor(A_np.astype(np.float32), device=device).realize()
    Omega = Tensor(
        np.random.default_rng(seed).standard_normal((n, b)).astype(np.float32),
        device=device,
    ).realize()

    B_t, Q_t = _get_jit(m, n, b, q_iter, ns_steps)(A, Omega)
    check_mem("range-finder")

    B = B_t.numpy().astype(np.float64)          # b x n (small)
    Q = Q_t.numpy().astype(np.float64)          # m x b
    del A, Omega

    Uhat, S, Vh = np.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat
    check_mem("project")
    return U[:, :b], S[:b].astype(np.float64), Vh[:b, :]


def randomized_svd_py_tg(A_np, k, p=10, q_iter=2, device="CPU", cores=None):
    """Signature-compatible alias for the in-package tinygrad entry point.

    inst/python/tinygrad_svd_func.py exposes randomized_svd_py_tg with this
    exact signature, so R's call site in FastPCA.R keeps working if the loader is
    pointed at this module. `device`/`cores` are accepted for compatibility;
    the compute device is always tinygrad's Metal backend.
    """
    return randomized_svd_tg(A_np, k=k, p=p, q_iter=q_iter)


def make_matrix(m, n, seed=0):
    """Realistic PCA-like matrix: decaying spectrum + dense noise."""
    rng = np.random.default_rng(seed)
    r = min(m, n, 120)
    U, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))
    s = 1000.0 * (0.85 ** np.arange(r))
    return ((U * s) @ V.T + 0.1 * rng.standard_normal((m, n))).astype(np.float64)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2000x500")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--q-iter", dest="q_iter", type=int, default=2)
    ap.add_argument("--ns-steps", dest="ns_steps", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    for spec in args.sizes.split(","):
        m, n = (int(x) for x in spec.lower().split("x"))
        A = make_matrix(m, n)
        ref = np.linalg.svd(A, compute_uv=False)[:args.k]
        # first call includes TinyJit compilation
        t0 = time.perf_counter()
        U, S, Vh = randomized_svd_tg(A, args.k, args.p, args.q_iter,
                                     ns_steps=args.ns_steps)
        first = time.perf_counter() - t0
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            U, S, Vh = randomized_svd_tg(A, args.k, args.p, args.q_iter,
                                         ns_steps=args.ns_steps)
            ts.append(time.perf_counter() - t0)
        rec = np.linalg.norm(A - (U[:, :args.k] * S[:args.k]) @ Vh[:args.k, :])
        rec /= np.linalg.norm(A)
        se = float(np.max(np.abs(S[:args.k] - ref) / np.abs(ref)))
        print(f"{m}x{n}: first={first*1e3:8.2f} ms  steady={min(ts)*1e3:7.2f} ms"
              f"  recon={rec:.4f} sv_relerr={se:.2e}"
              f"  peakRSS={peak_rss_bytes()/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    _main()
