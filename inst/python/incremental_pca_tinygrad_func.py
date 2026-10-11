"""FastPCA incremental PCA backend (tinygrad, Apple Metal).

Same streaming contract as inst/python/incremental_pca_func.py -- partial_fit()
per pre-processed block, then finalize()/transform() -- but the per-block SVD is
a randomized range finder with Newton-Schulz orthonormalisation (matmul-only,
mirroring inst/python/tinygrad_svd_func.py) plus a host-side small SVD.

Why randomized: tinygrad's own Tensor.svd unrolls num*log2(num)*2 Jacobi rounds
into one lazy graph and is unusable at these shapes. The consequence is the same
accuracy caveat as the tinygrad SVD backend: components are approximate, exact to
float32 precision for well-separated spectra. The "r" and "pytorch" backends
compute an exact SVD of each stacked block; use one of those when the top
components must be reproduced bit-for-bit.

This file is self-contained so reticulate can load it with py_run_file().
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings

# tinygrad opens its kernel cache when it is first imported; ~/.cache is not
# always writable (mirrors the guard in tinygrad_svd_func.py).
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "fastpca_tinygrad_cache"),
)

import numpy as np  # noqa: E402

try:
    from tinygrad import Tensor
except ImportError as exc:  # pragma: no cover - guarded by R's check_backend()
    raise ImportError(
        "The 'tinygrad' backend requires the tinygrad Python module. "
        "Run FastPCA::setup_py_env(backend = 'tinygrad') then start_FastPCA_env()."
    ) from exc

# quintic Newton-Schulz coefficients: X <- aX + b(XG) + c((XG)G)
_NS_A, _NS_B, _NS_C = 15.0 / 8.0, -10.0 / 8.0, 3.0 / 8.0
_DEVICE_OK: dict = {}
_WARNED_METAL_BUMP = False


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
    METAL (with a warning) rather than failing. Override with FASTPCA_TG_DEVICE.
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
    """Orthogonal polar factor of X (p x b), matmul-only."""
    X = X / (X.square().sum(axis=(-2, -1), keepdim=True).sqrt() + 1e-12)
    for _ in range(int(steps)):
        G = X.transpose(-2, -1) @ X
        T = X @ G
        X = _NS_A * X + _NS_B * T + _NS_C * (T @ G)
    return X


def _small_svd(B: np.ndarray):
    """SVD of the b x n matrix B on the host via its b x b Gram.

    Returns (Uc, S, Vh); rows with a numerically zero singular value are set to
    0 rather than divided. Same helper as tinygrad_svd_func.py.
    """
    w, Uc = np.linalg.eigh(B @ B.T)
    idx = np.argsort(w)[::-1]
    w, Uc = w[idx], Uc[:, idx]
    S = np.sqrt(np.maximum(w, 0.0))
    scale = float(S[0]) if S.size else 1.0
    safe = S > np.finfo(B.dtype).eps * max(B.shape) * scale
    Vh = np.divide(Uc.T @ B, S[:, None], out=np.zeros_like(B),
                   where=safe[:, None])
    return Uc, S.astype(np.float64), Vh


class IncrementalPCAState:
    """Streaming PCA state for tinygrad: partial_fit() per block, then finalize()."""

    def __init__(self, n_features, ncomp, oversample=10, q_iter=2,
                 ns_steps=16, seed=0):
        self.n_features = int(n_features)
        self.ncomp = int(ncomp)
        self.oversample = max(0, int(oversample))
        self.q_iter = int(q_iter)
        self.ns_steps = int(ns_steps)
        self.device = resolve_device("GPU")
        # a fixed seed keeps repeated fits reproducible while still drawing a
        # fresh sketch for every block
        self.rng = np.random.default_rng(seed)
        self.SV: Tensor | None = None
        self.Vh_t: Tensor | None = None
        self.S = None
        self.Vh = None

    def partial_fit(self, X):
        A_np = np.asarray(X, dtype=np.float32)
        if A_np.ndim != 2 or int(A_np.shape[1]) != self.n_features:
            raise ValueError(
                "block has shape %s but the state was created for %d features"
                % (tuple(A_np.shape), self.n_features))
        A = Tensor(A_np, device=self.device)
        B = A if self.SV is None else Tensor.cat(self.SV, A, dim=0)

        m = int(B.shape[0])
        b = min(self.ncomp + self.oversample, m)
        if b < 1:
            raise ValueError("cannot sketch an empty block")
        Omega = Tensor(
            self.rng.standard_normal((self.n_features, b)).astype(np.float32),
            device=self.device,
        )
        Y = B @ Omega
        for _ in range(self.q_iter):
            Z = _ns_orth(B.transpose(-2, -1) @ Y, self.ns_steps)
            Y = B @ Z
        Q = _ns_orth(Y, self.ns_steps)
        # the small (b x p) factor comes back to the host for LAPACK
        Bp = Q.transpose(-2, -1) @ B
        _, S, Vh = _small_svd(Bp.realize().numpy())

        k = min(self.ncomp, int(S.size))
        self.S = S[:k]
        # re-orthonormalise the retained rows so the carried state stays a
        # proper factor even when the sketch leaves them slightly dependent
        Vh_k = Vh[:k, :]
        norms = np.linalg.norm(Vh_k, axis=1)
        Vh_k = np.divide(Vh_k, norms[:, None], out=np.zeros_like(Vh_k),
                         where=(norms > 0)[:, None])
        self.Vh = Vh_k.astype(np.float32)
        self.SV_np = (self.Vh * np.asarray(self.S, dtype=np.float32)[:, None])
        self.SV = Tensor(self.SV_np.astype(np.float32), device=self.device)
        self.Vh_t = Tensor(self.Vh, device=self.device).contiguous()
        return self

    def transform(self, X):
        """Scores of an already pre-processed block against the components."""
        if self.Vh_t is None:
            raise RuntimeError("transform() called before any partial_fit()")
        A = Tensor(np.asarray(X, dtype=np.float32), device=self.device)
        return (A @ self.Vh_t.transpose(-2, -1)).realize().numpy()

    def finalize(self, ncomp=None):
        """Components and singular values; ncomp drops the oversampling buffer."""
        if self.Vh is None:
            raise RuntimeError("finalize() called before any partial_fit()")
        if ncomp is not None:
            k = min(int(ncomp), int(np.asarray(self.S).shape[0]))
            self.S = np.asarray(self.S)[:k]
            self.Vh = self.Vh[:k, :].astype(np.float32)
            self.Vh_t = Tensor(self.Vh, device=self.device).contiguous()
        return {"S": np.asarray(self.S, dtype=np.float64), "Vh": self.Vh}
