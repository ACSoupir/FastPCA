"""Optimized randomized SVD on tinygrad (Metal).

Design notes
------------
tinygrad has no native QR/SVD, and its bundled Jacobi/Householder helpers are
Python-graph heavy (hundreds of tiny kernels), so they lose badly to LAPACK.
The *expensive* part of randomized SVD, however, is the tall/wide matmuls and
those are exactly what tinygrad/Metal does well here (measured ~4.6x faster than
torch-CPU on 15000x2000).

So this kernel runs the matmuls in tinygrad and the small (b x b / m x b)
linear algebra on CPU with numpy. It is a tinygrad kernel for the part that
matters.

Run:  TESTING/.venv-tg/bin/python -c "import sys;sys.path.insert(0,'TESTING');\
        from tinygrad_rsvd import randomized_svd_tg; ..."
"""
from __future__ import annotations

import os
import time

# tinygrad caches compiled kernels under XDG_CACHE_HOME; keep it in the workspace
# because ~/.cache is not writable under the DSH sandbox.
os.environ.setdefault(
    "XDG_CACHE_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
)

import numpy as np  # noqa: E402

try:
    from tinygrad import Tensor, dtypes
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tinygrad is required; use TESTING/.venv-tg/bin/python") from exc


def _tensor(x, device):
    return Tensor(np.ascontiguousarray(x), device=device)


def randomized_svd_tg(A_np, k, p=10, q_iter=2, device="METAL", seed=None,
                      dtype=np.float32):
    """Randomized SVD with tinygrad matmuls and CPU small linear algebra.

    Returns (U, S, Vh) as numpy arrays with the package contract: b = k + p
    columns (capped at min(m, n)).
    """
    A_np = np.asarray(A_np)
    m, n = A_np.shape
    b = min(int(k) + int(p), m, n)
    if b <= 0:
        raise ValueError("k + p must be positive and no larger than min(m, n)")

    if seed is not None:
        np.random.seed(int(seed))

    A = _tensor(A_np.astype(dtype), device)
    Omega = Tensor.randn(n, b, dtype=dtypes.float32).to(device)

    # range finder with the stable (small-QR) power scheme
    Y = A @ Omega
    for _ in range(int(q_iter)):
        Z, _ = np.linalg.qr((A.T @ Y).numpy())          # n x b on CPU
        Z = _tensor(Z.astype(dtype), device)
        Y = A @ Z

    # one final tall QR on CPU; then project
    Q, _ = np.linalg.qr(Y.numpy())                      # m x b
    B = (Q.T @ A_np.astype(np.float64))                 # small, exact projection
    Uhat, S, Vh = np.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat
    return (U[:, :b].astype(np.float64),
            S[:b].astype(np.float64),
            Vh[:b, :].astype(np.float64))


KERNELS = {"tg_hybrid": randomized_svd_tg}
