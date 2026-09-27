"""Optimized randomized SVD kernels for the FastPCA PyTorch backend.

Research sandbox -- see TESTING/README.md. Drop-in alternatives to
inst/python/torch_svd_func.randomized_svd_py so that speed and accuracy can be
compared directly.

Baseline pain points this module targets:
  * the package baseline performs a redundant QR of the tall m x b sketch on
    EVERY power iteration (only the small n x b QR is needed for stability);
  * it builds an autograd graph that is immediately discarded (no no_grad);
  * it only knows CPU/CUDA -- Apple MPS is unsupported (and requires float32).

All kernels return NumPy (U, S, Vh) with shapes (m, b), (b,), (b, n),
where b = min(k + p, m, n). That matches the package baseline contract (it
returns k + p columns and lets R slice down to k).
"""

from __future__ import annotations

import gc
import os
from typing import Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid libomp double-init
# torch 2.9 has no native MPS linalg.qr; fall back to CPU rather than erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

# A full gc.collect() costs ~12-15 ms on this machine and the package baseline
# calls it unconditionally at the end of every SVD. Reference counting already
# frees our temporaries, so collection is opt-in via FASTPCA_SVD_GC=1.
_COLLECT = os.environ.get("FASTPCA_SVD_GC", "0").lower() not in ("0", "", "false")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> torch.device:
    """Map user-facing device names to a torch.device with safe fallbacks."""
    d = (device or "cpu").strip().lower()
    if d in ("gpu", "cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d in ("mps", "metal"):
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def _prepare(A_np, device: torch.device, dtype) -> torch.Tensor:
    A = torch.as_tensor(np.asarray(A_np))
    if not A.is_floating_point():
        raise TypeError("input matrix must be floating point")
    # MPS has no float64 kernel support; silently use float32 there.
    if device.type == "mps" and A.dtype == torch.float64:
        A = A.to(torch.float32)
    td = _to_torch_dtype(dtype)
    if td is not None and A.dtype != td:
        A = A.to(td)
    return A.contiguous().to(device)


def _to_torch_dtype(dtype):
    """Accept None, torch.dtype, numpy dtype or a short string ("32"/"float64")."""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    s = str(dtype)
    if s.startswith("torch."):
        return getattr(torch, s.split(".", 1)[1])
    if hasattr(dtype, "name"):          # numpy dtype
        s = dtype.name
    if s in ("32", "float"):
        return torch.float32
    if s in ("64", "double"):
        return torch.float64
    return getattr(torch, s)


def _seed(seed) -> None:
    if seed is not None:
        torch.manual_seed(int(seed))


def _finish(U, S, Vh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (U.detach().to("cpu").numpy(),
            S.detach().to("cpu").numpy(),
            Vh.detach().to("cpu").numpy())


def _dims(A, k, p):
    m, n = A.shape
    b = min(int(k) + int(p), m, n)
    if b <= 0:
        raise ValueError("k + p must be positive and no larger than min(m, n)")
    return m, n, b


def _qr(Y):
    """Reduced QR of a (possibly tall) Y, portable across CPU/CUDA/MPS.

    torch 2.9 has no native MPS linalg.qr, so fall back to a Householder QR on
    the CPU and move the (small) factors back. Cholesky-QR was tried first but
    is not stable enough here: for a rank-1-dominated A the Gram matrix can be
    numerically indefinite (observed on 600x250 data with a +10 shift).
    """
    try:
        return torch.linalg.qr(Y, mode="reduced")
    except (NotImplementedError, RuntimeError):
        Q, R = torch.linalg.qr(Y.detach().to("cpu"), mode="reduced")
        return Q.to(Y.device), R.to(Y.device)


# ---------------------------------------------------------------------------
# baseline (imported from the package) -- used for apples-to-apples timing
# ---------------------------------------------------------------------------

def randomized_svd_baseline(A_np, k, p=10, q_iter=2, device="cpu", cores=None,
                            dtype=None):
    """Thin wrapper around the in-package baseline kernel.

    The package version only understands "cpu"/"gpu"; MPS requests are mapped to
    CPU here because the baseline cannot run on Metal.
    """
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    py_dir = str(repo / "inst" / "python")
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)
    from torch_svd_func import randomized_svd_py as _baseline

    dev = resolve_device(device)
    backend_name = "gpu" if dev.type == "cuda" else "cpu"
    A = np.asarray(A_np)
    td = _to_torch_dtype(dtype)
    if td is not None:
        A = np.ascontiguousarray(A, dtype=np.float32 if td == torch.float32
                                 else np.float64)
    return _baseline(A, k=k, p=p, q_iter=q_iter, device=backend_name,
                     cores=int(cores) if cores else 2)


# ---------------------------------------------------------------------------
# kernel 1: direct power scheme -- Algorithm 4.1 (Halko et al.)
#   Y = (A A^T)^q A Omega  -> single QR at the very end.
#   Saves q_iter tall QRs relative to the baseline orthogonal iteration.
# ---------------------------------------------------------------------------

@torch.no_grad()
def randomized_svd_direct(A_np, k, p=10, q_iter=2, device="cpu", dtype=None,
                          seed=None, cores=None, reorthogonalize=False):
    if cores:
        torch.set_num_threads(int(cores))
    dev = resolve_device(device)
    A = _prepare(A_np, dev, dtype)
    m, n, b = _dims(A, k, p)

    _seed(seed)
    Omega = torch.randn(n, b, device=dev, dtype=A.dtype)
    Y = A @ Omega
    for _ in range(int(q_iter)):
        if reorthogonalize:
            Qz, _ = _qr(Y)
            Y = A @ (A.T @ Qz)
        else:
            Y = A @ (A.T @ Y)

    Q, _ = _qr(Y)
    B = Q.T @ A
    Uhat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat

    del Omega, Y, Q, B
    if _COLLECT:
        gc.collect()
    return _finish(U, S, Vh)


# ---------------------------------------------------------------------------
# kernel 2: orthogonal iteration (Halko 4.3) -- same math as the baseline but
#   with no_grad and an explicit reduced QR. Kept to isolate the cost of those
#   two changes from the algorithmic change in kernel 1.
# ---------------------------------------------------------------------------

@torch.no_grad()
def randomized_svd_orth(A_np, k, p=10, q_iter=2, device="cpu", dtype=None,
                        seed=None, cores=None):
    if cores:
        torch.set_num_threads(int(cores))
    dev = resolve_device(device)
    A = _prepare(A_np, dev, dtype)
    m, n, b = _dims(A, k, p)

    _seed(seed)
    Omega = torch.randn(n, b, device=dev, dtype=A.dtype)
    Y = A @ Omega
    for _ in range(int(q_iter)):
        Qz, _ = _qr(A.T @ Y)
        Y, _ = _qr(A @ Qz)

    Q, _ = _qr(Y)
    B = Q.T @ A
    Uhat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat

    del Omega, Y, Q, B
    if _COLLECT:
        gc.collect()
    return _finish(U, S, Vh)


# ---------------------------------------------------------------------------
# kernel 2b: stable minimal orthogonal iteration.
#   The baseline orthonormalises A^T Y (small n x b) AND does a redundant tall
#   QR of A Q on every power iteration. Only the small QR is needed to keep the
#   subspace from collapsing in low precision; dropping the per-iteration tall
#   QR makes this stable *and* fast (direct's speed with triangular's accuracy).
# ---------------------------------------------------------------------------

@torch.no_grad()
def randomized_svd_stable(A_np, k, p=10, q_iter=2, device="cpu", dtype=None,
                          seed=None, cores=None):
    if cores:
        torch.set_num_threads(int(cores))
    dev = resolve_device(device)
    A = _prepare(A_np, dev, dtype)
    m, n, b = _dims(A, k, p)

    _seed(seed)
    Omega = torch.randn(n, b, device=dev, dtype=A.dtype)
    Y = A @ Omega
    for _ in range(int(q_iter)):
        Z, _ = _qr(A.T @ Y)          # small n x b QR: reorthonormalise row space
        Y = A @ Z                    # no tall QR here

    Q, _ = _qr(Y)
    B = Q.T @ A
    Uhat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat

    del Omega, Y, Q, B
    if _COLLECT:
        gc.collect()
    return _finish(U, S, Vh)


# ---------------------------------------------------------------------------
# kernel 3: triangular-solve formulation (builds on optim1).
#   Uses only R from the final QR and solves two triangular systems instead of
#   forming U = Q @ Uhat; lower peak memory for very tall-skinny A. (torch's QR
#   still computes Q internally, so this is a memory variant of `stable`.)
# ---------------------------------------------------------------------------

@torch.no_grad()
def randomized_svd_triangular(A_np, k, p=10, q_iter=2, device="cpu", dtype=None,
                              seed=None, cores=None):
    if cores:
        torch.set_num_threads(int(cores))
    dev = resolve_device(device)
    A = _prepare(A_np, dev, dtype)
    m, n, b = _dims(A, k, p)

    _seed(seed)
    Omega = torch.randn(n, b, device=dev, dtype=A.dtype)
    Y = A @ Omega
    for _ in range(int(q_iter)):
        Z, _ = _qr(A.T @ Y)
        Y = A @ Z

    _, R = _qr(Y)                                       # Y = Q R
    C = Y.T @ A                                        # b x n
    try:
        Bmat = torch.linalg.solve_triangular(R.T, C, upper=False)   # R^T B = C
        Uhat, S, Vh = torch.linalg.svd(Bmat, full_matrices=False)
        X = torch.linalg.solve_triangular(R, Uhat[:, :b], upper=True)
    except (RuntimeError, NotImplementedError):
        # MPS / older builds may lack solve_triangular.
        Bmat = torch.linalg.solve(R.T, C)
        Uhat, S, Vh = torch.linalg.svd(Bmat, full_matrices=False)
        X = torch.linalg.solve(R, Uhat[:, :b])
    U = Y @ X

    del Omega, Y, R, C, Bmat
    if _COLLECT:
        gc.collect()
    return _finish(U, S, Vh)


# ---------------------------------------------------------------------------
# kernel 4: PyTorch's built-in randomized range finder (svd_lowrank).
#   C++ implementation; uses its own subspace iteration. Kept as an
#   independent reference point / strong baseline.
# ---------------------------------------------------------------------------

@torch.no_grad()
def randomized_svd_lowrank(A_np, k, p=10, q_iter=2, device="cpu", dtype=None,
                           seed=None, cores=None):
    if cores:
        torch.set_num_threads(int(cores))
    dev = resolve_device(device)
    A = _prepare(A_np, dev, dtype)
    m, n, b = _dims(A, k, p)

    _seed(seed)
    U, S, V = torch.svd_lowrank(A, q=b, niter=int(q_iter))
    U = U[:, :b]
    S = S[:b]
    Vh = V.transpose(-1, -2)[:b, :]

    del A
    if _COLLECT:
        gc.collect()
    return _finish(U, S, Vh)


KERNELS = {
    "baseline": randomized_svd_baseline,
    "orth": randomized_svd_orth,
    "direct": randomized_svd_direct,
    "stable": randomized_svd_stable,
    "triangular": randomized_svd_triangular,
    "lowrank": randomized_svd_lowrank,
}
