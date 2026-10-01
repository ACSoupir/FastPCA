"""FastPCA tinygrad SVD backend (Apple Metal).

randomized_svd_py_tg() mirrors inst/python/torch_svd_func.py::randomized_svd_py
and returns the same (U, S, Vh) contract. The O(m*n*b) range finder runs on
tinygrad's METAL device; the small b x n SVD of B is done on the host with
LAPACK (Gram + eigh), which measured ~4x faster than a direct SVD and exact.

This file is self-contained so reticulate can load it with py_run_file().
It mirrors TESTING/tinygrad_native.py (the development / benchmark copy).

Design notes
------------
* Newton-Schulz polar iteration replaces every QR -- matmul-only, so no m x m
  tensor is ever allocated: X <- aX + b(XG) + c((XG)G), G = X^T X.
  ns_steps=16 is required for the returned tail singular values to match torch;
  12 leaves them wrong (the leading subspace is already correct).
* The range finder is wrapped in TinyJit, which removes per-op Python dispatch
  (~220 ms -> ~10 ms steady state on 2000x500).
* Device: on Apple Silicon only METAL is functional (tinygrad's CPU codegen
  invokes a cross-compiler the toolchain rejects), so an R "CPU" request is
  bumped to METAL with a warning. Override with FASTPCA_TG_DEVICE.
* Peak RSS is capped at FASTPCA_TG_MEM_GB (default 20 GB) by a pre-flight
  estimate and staged checks, so a bad shape cannot take the machine down.
"""
from __future__ import annotations

import os
import resource
import sys
import tempfile
import warnings

# tinygrad needs a writable kernel cache; ~/.cache is not always writable.
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "fastpca_tinygrad_cache"),
)

import numpy as np  # noqa: E402

try:
    from tinygrad import Tensor, TinyJit
except ImportError as exc:  # pragma: no cover - guarded by R's check_backend()
    raise ImportError(
        "The 'tinygrad' backend requires the tinygrad Python module. "
        "Run FastPCA::setup_py_env(backend = 'tinygrad') then start_FastPCA_env()."
    ) from exc


def _cache_writable(path: str) -> bool:
    """True if we can actually create and write inside *path*."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".fastpca_write_test")
        with open(probe, "w") as handle:
            handle.write("1")
        os.remove(probe)
        return True
    except Exception:
        return False


def _ensure_writable_cache() -> None:
    """Repoint tinygrad's kernel cache if its location is not writable.

    tinygrad resolves the sqlite path once, at import. R checks availability with
    reticulate::py_module_available(), which imports tinygrad *before* this file
    runs, so the default (~/Library/Caches or ~/.cache) is already baked in. In
    containers, restricted HPC logins and sandboxes that path can be read-only,
    which surfaces as "attempt to write a readonly database" on the first
    realize(). Point it somewhere writable instead.
    """
    import tinygrad.helpers as tg_helpers

    if _cache_writable(tg_helpers.cache_dir):
        return
    fallback = os.path.join(tempfile.gettempdir(), "fastpca_tinygrad_cache")
    if not _cache_writable(fallback):
        raise RuntimeError(
            "the tinygrad kernel cache is not writable (%s) and no fallback "
            "could be created; set XDG_CACHE_HOME to a writable directory"
            % tg_helpers.cache_dir
        )
    tg_helpers.cache_dir = fallback
    tg_helpers.CACHEDB = os.path.join(fallback, "cache.db")
    if getattr(tg_helpers, "_db_connection", None) is not None:
        try:
            tg_helpers._db_connection.close()
        except Exception:
            pass
        tg_helpers._db_connection = None
    os.environ["XDG_CACHE_HOME"] = fallback


_ensure_writable_cache()

MEM_LIMIT_GB = float(os.environ.get("FASTPCA_TG_MEM_GB", "20"))
MEM_LIMIT_BYTES = int(MEM_LIMIT_GB * (1024 ** 3))

# Quintic Newton-Schulz polar coefficients; a + b + c == 1, so after Frobenius
# normalisation every non-zero singular value converges to 1.
_NS_A, _NS_B, _NS_C = 15.0 / 8.0, -10.0 / 8.0, 3.0 / 8.0

_DEVICE_OK: dict = {}
_WARNED_METAL_BUMP = False


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def check_mem(where: str = "") -> None:
    used = peak_rss_bytes()
    if used > MEM_LIMIT_BYTES:
        raise MemoryError(
            "peak RSS %.2f GB exceeds the %.0f GB safeguard (%s)"
            % (used / 1e9, MEM_LIMIT_GB, where)
        )


def _estimate_bytes(m: int, n: int, b: int) -> int:
    """Upper-bound working set of one call (float32 buffers, in bytes)."""
    return 4 * (m * n + m * b * 3 + n * b * 10 + b * b * 8)


def check_shape(m: int, n: int, b: int) -> None:
    """Refuse a shape whose projected working set exceeds the memory limit."""
    est = _estimate_bytes(m, n, b)
    if est > MEM_LIMIT_BYTES:
        raise MemoryError(
            "projected working set %.2f GB for %dx%d, b=%d exceeds the "
            "%.0f GB safeguard (set FASTPCA_TG_MEM_GB to raise)"
            % (est / 1e9, m, n, b, MEM_LIMIT_GB)
        )


def _device_works(dev: str) -> bool:
    """Probe a tinygrad device once with a trivial kernel."""
    if dev not in _DEVICE_OK:
        try:
            (Tensor([1.0, 2.0], device=dev) + 1).realize().numpy()
            _DEVICE_OK[dev] = True
        except Exception:
            _DEVICE_OK[dev] = False
    return _DEVICE_OK[dev]


def resolve_device(device=None) -> str:
    """Map an R-style device request ('CPU'/'GPU') to a working tinygrad device.

    Apple Silicon: only METAL works, so an explicit 'CPU' request is bumped to
    METAL (with a warning) rather than failing. Elsewhere 'CPU' is honoured.
    Override with FASTPCA_TG_DEVICE.
    """
    global _WARNED_METAL_BUMP
    req = str(device or os.environ.get("FASTPCA_TG_DEVICE", "")).upper().strip()
    if req in ("GPU", "CUDA"):
        candidates = ["METAL", "CUDA"]
    elif req == "CPU":
        if sys.platform == "darwin":
            candidates = ["METAL"]
            if not _WARNED_METAL_BUMP:
                warnings.warn(
                    "FastPCA tinygrad backend: Apple Silicon (arm64) has no "
                    "working tinygrad CPU codegen, so 'CPU' is using METAL.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _WARNED_METAL_BUMP = True
        else:
            candidates = ["CPU", "CUDA"]
    elif req == "METAL":
        candidates = ["METAL"]
    else:
        candidates = ["METAL", "CUDA", "CPU"]

    for cand in candidates:
        if _device_works(cand):
            return cand
    raise RuntimeError(
        "no working tinygrad device among %s; set FASTPCA_TG_DEVICE"
        % (candidates,)
    )


def _ns_orth(X: Tensor, steps: int) -> Tensor:
    """Orthogonal polar factor of X (p x b), matmul-only.

    The quintic update uses 3 matmuls instead of 4 by reusing T = X G:
    X <- aX + bT + c(TG), since (X G) G == X (G G).
    """
    X = X / (X.square().sum(axis=(-2, -1), keepdim=True).sqrt() + 1e-12)
    for _ in range(int(steps)):
        G = X.transpose(-2, -1) @ X
        T = X @ G
        X = _NS_A * X + _NS_B * T + _NS_C * (T @ G)
    return X


_JIT_CACHE: dict = {}


def _get_jit(device, m, n, b, q_iter: int, ns_steps: int):
    """TinyJit'd range finder for one exact (device, shape, config)."""
    key = (str(device), int(m), int(n), int(b), int(q_iter), int(ns_steps))
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


def warmup(m, n, k=50, p=10, q_iter=2, ns_steps=16, device=None):
    """Pre-trace the JIT graph for one shape so later calls skip compilation."""
    b = min(int(k) + int(p), m, n)
    dev = resolve_device(device)
    A = Tensor(np.zeros((m, n), dtype=np.float32), device=dev).realize()
    Omega = Tensor(np.zeros((n, b), dtype=np.float32), device=dev).realize()
    _get_jit(dev, m, n, b, q_iter, ns_steps)(A, Omega)
    del A, Omega


def _small_svd(B: np.ndarray, mode: str = "gram"):
    """SVD of the b x n matrix B on the host.

    'gram': C = B B^T (b x b), symmetric eigendecomposition, then
    Vh = diag(1/S) Uc^T B. ~4x faster than LAPACK's direct SVD at these shapes
    and exact to ~1e-15. Rows whose singular value is numerically zero are set
    to 0 rather than dividing.
    'svd': plain np.linalg.svd, kept as a reference/fallback.

    tinygrad's own Tensor.svd is not usable: it unrolls num*log2(num)*2 Jacobi
    rounds into one lazy graph (14 s for an 8x8, then RecursionError).
    """
    if mode == "svd":
        return np.linalg.svd(B, full_matrices=False)
    w, Uc = np.linalg.eigh(B @ B.T)
    idx = np.argsort(w)[::-1]
    w, Uc = w[idx], Uc[:, idx]
    S = np.sqrt(np.maximum(w, 0.0))
    scale = float(S[0]) if S.size else 1.0
    safe = S > np.finfo(B.dtype).eps * max(B.shape) * scale
    Vh = np.divide(Uc.T @ B, S[:, None], out=np.zeros_like(B),
                   where=safe[:, None])
    return Uc, S, Vh


def randomized_svd_tg(A_np, k, p=10, q_iter=2, seed=None, ns_steps=16,
                      device="METAL", final: str = "gram"):
    """Randomized SVD matching torch_svd_func.randomized_svd_py's contract.

    Returns numpy (U, S, Vh) with U m x b, S length b, Vh b x n and
    b = min(k + p, m, n).
    """
    check_mem("start")
    A_np = np.asarray(A_np)
    m, n = A_np.shape
    b = min(int(k) + int(p), m, n)
    if b <= 0:
        raise ValueError("k + p must be positive and no larger than min(m, n)")
    check_shape(m, n, b)

    dev = resolve_device(device)
    A = Tensor(A_np.astype(np.float32), device=dev).realize()
    Omega = Tensor(
        np.random.default_rng(seed).standard_normal((n, b)).astype(np.float32),
        device=dev,
    ).realize()

    B_t, Q_t = _get_jit(dev, m, n, b, q_iter, ns_steps)(A, Omega)
    check_mem("range-finder")

    B = B_t.numpy().astype(np.float64)
    Q = Q_t.numpy().astype(np.float64)
    del A, Omega

    Uhat, S, Vh = _small_svd(B, final)
    U = Q @ Uhat
    check_mem("project")
    return U[:, :b], S[:b].astype(np.float64), Vh[:b, :]


def randomized_svd_py_tg(A_np, k, p=10, q_iter=2, device="cpu", cores=2):
    """FastPCA R entry point (matches inst/python/torch_svd_func.py).

    Args:
        A_np: Features x Samples NumPy array (as passed by R).
        k, p, q_iter: randomized SVD parameters.
        device: "CPU" or "GPU"; on Apple Silicon both use METAL (see
            resolve_device). Override with FASTPCA_TG_DEVICE.
        cores: accepted for signature compatibility; tinygrad schedules its own
            GPU work, so there are no OpenMP threads to clamp.

    Returns:
        (U, S, Vh) NumPy arrays: U is m x b, S length b, Vh b x n with
        b = k + p (R slices to the first k).
    """
    return randomized_svd_tg(A_np, k=k, p=p, q_iter=q_iter, device=device)
