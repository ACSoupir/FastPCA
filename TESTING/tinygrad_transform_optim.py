"""Optimized tinygrad matrix transform for FastPCA's prep_matrix().

Contract-compatible with inst/python/torch_transform.py::transform_py:
    transform(A) = scale(transpose(log2(A)))   (each step optional)
where "scale" mean-centres and divides by the *unbiased* standard deviation
along dim=0 of the (possibly transposed) matrix.

Why this is faster / smaller than the torch baseline
----------------------------------------------------
torch materialises every step (log2 -> copy-transpose -> mean/subtract ->
std/divide) in float64. Here:

* The transpose is never materialised as a copy: the statistics are taken along
  whichever axis becomes dim=0 after transposing (axis=1 when transpose=1), and
  the result is only transposed at the very end.
* log2 + subtract + divide are elementwise, so tinygrad's scheduler fuses them
  into a single kernel. Total traffic is ~3 reads + 1 write versus torch's
  ~7 reads + 4 writes.
* Buffers are float32 (Metal has no double), which halves the working set.

Precision: Metal cannot do float64, so this runs in float32 and is cast back to
the input dtype on return. That matches the torch baseline to ~1e-6 relative,
consistent with the SVD backend's float32 range finder.
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings

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

_DEVICE_OK: dict = {}
_WARNED_METAL_BUMP = False


def _cache_writable(path: str) -> bool:
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
    """Repoint tinygrad's kernel cache if its import-time location is read-only."""
    import tinygrad.helpers as tg_helpers

    if _cache_writable(tg_helpers.cache_dir):
        return
    fallback = os.path.join(tempfile.gettempdir(), "fastpca_tinygrad_cache")
    if not _cache_writable(fallback):
        raise RuntimeError(
            "the tinygrad kernel cache is not writable (%s); set XDG_CACHE_HOME"
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


def _device_works(dev: str) -> bool:
    if dev not in _DEVICE_OK:
        try:
            (Tensor([1.0, 2.0], device=dev) + 1).realize().numpy()
            _DEVICE_OK[dev] = True
        except Exception:
            _DEVICE_OK[dev] = False
    return _DEVICE_OK[dev]


def resolve_device(device=None) -> str:
    """Map an R-style device request ('CPU'/'GPU') to a working tinygrad device."""
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
        "no working tinygrad device among %s; set FASTPCA_TG_DEVICE" % (candidates,)
    )


def transform_py_tg(A_np: np.ndarray, log2: int = 0, transpose: int = 0,
                    scale: int = 1, cores: int = 2, device: str = "CPU") -> np.ndarray:
    """See module docstring. Returns a NumPy array in the input dtype."""
    A_np = np.asarray(A_np)
    if A_np.ndim != 2:
        raise ValueError("transform expects a 2-D matrix, got shape %r" % (A_np.shape,))
    m, n = A_np.shape
    dev = resolve_device(device)

    X = Tensor(A_np.astype(np.float32), device=dev)
    if int(log2) == 1:
        X = X.log2()

    # Statistics are taken along the axis that becomes dim=0 after an optional
    # transpose, so a transposed copy is never built.
    if int(scale) == 1:
        axis = 1 if int(transpose) == 1 else 0
        length = m if axis == 0 else n   # number of elements reduced over
        mean = X.mean(axis=axis, keepdim=True)
        var = (X - mean).square().sum(axis=axis, keepdim=True) / max(length - 1, 1)
        X = (X - mean) / var.sqrt()

    if int(transpose) == 1:
        X = X.transpose()

    out = np.ascontiguousarray(X.numpy())
    return out.astype(A_np.dtype, copy=False)
