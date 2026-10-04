"""tinygrad NIPALS for the FastPCA 'tinygrad' backend.

The matrix arrives already centred/scaled by NIPALS() in R with missing entries
as NaN, so this module only runs the iterative loop. The algorithm matches
nipals::nipals and inst/python/nipals_func.py: masked entries are held at zero
for every product, loadings/scores denominators sum over observed entries only,
and deflation subtracts the masked rank-one component.

Two things differ from the PyTorch module:

* everything is float32, because tinygrad's Metal backend has no float64;
* the inner loop is compiled once by TinyJit and replayed, unrolling 'block'
  iterations per replay so the host only synchronises once per block. Gram-
  Schmidt runs against fixed-width (nvar x ncomp) / (nobs x ncomp) buffers, so
  the compiled graph never changes shape.

Only two m x n arrays are ever allocated (the data and, when needed, the mask);
deflation is in place. A peak working-set guard refuses shapes above
FASTPCA_TG_MEM_GB (20 GB by default), mirroring getOption("FastPCA.memory_limit").
"""
import os
import sys
import tempfile

# A writable XDG cache is required before tinygrad's Metal runtime is imported,
# otherwise the first kernel aborts with "attempt to write a readonly database".
def _writable(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".fastpca-probe")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


_cache = os.environ.get("XDG_CACHE_HOME")
if not (_cache and _writable(_cache)):
    # NIPALS() normally points XDG_CACHE_HOME at a writable directory before the
    # first import; this keeps direct module use working too.
    _cache = os.path.join(tempfile.gettempdir(), "fastpca-tinygrad")
    if _writable(_cache):
        os.environ["XDG_CACHE_HOME"] = _cache

# tinygrad's CPU codegen shells out to 'clang'; conda toolchains shadow it and
# reject tinygrad's freestanding flags, so pin Apple clang when available.
if sys.platform == "darwin" and os.path.exists("/usr/bin/clang"):
    os.environ["CC"] = "/usr/bin/clang"

import numpy as np
from tinygrad import Tensor, TinyJit

MEM_LIMIT_GB = float(os.environ.get("FASTPCA_TG_MEM_GB", "20"))
_BLOCK_DEFAULT = 8


def _resolve_device(requested):
    """Pick a working tinygrad device for 'CPU' or 'GPU'.

    GPU requests try Metal then CUDA; CPU prefers the CPU backend but falls
    back to Metal where tinygrad's C codegen is unavailable.
    """
    req = str(requested or "CPU").upper()
    if req in ("GPU", "METAL", "CUDA"):
        candidates = ["METAL", "CUDA"]
    else:
        candidates = ["CPU", "METAL"]
    errors = []
    for cand in candidates:
        try:
            (Tensor([1.0], device=cand) + 1).realize().numpy()
            return cand
        except Exception as exc:                      # pragma: no cover - device probe
            errors.append("%s (%s)" % (cand, exc))
    raise RuntimeError(
        "the 'tinygrad' backend found no usable device: %s; install Apple "
        "Metal or CUDA, or use backend = 'pytorch'/'r'" % "; ".join(errors))


_JIT_CACHE = {}


def _get_core(dev, m, n, k, block, has_na, gramschmidt):
    """One TinyJit graph per (shape, block size, mask/GS configuration)."""
    key = (str(dev), int(m), int(n), int(k), int(block),
           bool(has_na), bool(gramschmidt))
    core = _JIT_CACHE.get(key)
    if core is not None:
        return core
    B, GS = int(block), bool(gramschmidt)

    if has_na:
        @TinyJit
        def core(x0, obs, PPp, TTp, th):
            ph = None
            for _ in range(B):
                denom_p = ((th * th) @ obs).maximum(1e-30)
                ph = (x0.transpose(-2, -1) @ th) / denom_p
                if GS:
                    ph = ph - PPp @ (PPp.transpose(-2, -1) @ ph)
                ph = ph / ((ph * ph).sum().sqrt() + 1e-30)
                denom_t = (obs @ (ph * ph)).maximum(1e-30)
                th = (x0 @ ph) / denom_t
                if GS:
                    th = th - TTp @ (TTp.transpose(-2, -1) @ th)
            return th.realize(), ph.realize()
    else:
        @TinyJit
        def core(x0, PPp, TTp, th):
            ph = None
            for _ in range(B):
                ph = (x0.transpose(-2, -1) @ th) / ((th * th).sum() + 1e-30)
                if GS:
                    ph = ph - PPp @ (PPp.transpose(-2, -1) @ ph)
                ph = ph / ((ph * ph).sum().sqrt() + 1e-30)
                th = (x0 @ ph) / ((ph * ph).sum() + 1e-30)
                if GS:
                    th = th - TTp @ (TTp.transpose(-2, -1) @ th)
            return th.realize(), ph.realize()

    _JIT_CACHE[key] = core
    return core


def nipals_py(X, ncomp=10, tol=1e-6, maxit=500, gramschmidt=True,
              device="CPU", cores=1, block=None):
    """Run float32 NIPALS on a (nobs x nvar) array that may contain NaN.

    Returns a dict with eig, scores (native T), loadings, iters and converged.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    nobs, nvar = X.shape
    block = _BLOCK_DEFAULT if not block else int(block)
    if ncomp < 1 or nobs < 1 or nvar < 1:
        raise ValueError("ncomp, nobs and nvar must all be positive")

    nan_mask = ~np.isfinite(X)
    has_na = bool(nan_mask.any())

    # fp32 working set: the data plus, when needed, one mask of the same shape
    est = 4 * nobs * nvar * (2 if has_na else 1)
    limit = MEM_LIMIT_GB * (1024 ** 3)
    if est > limit:
        raise MemoryError(
            "the 'tinygrad' backend needs about %.1f GB of device memory for a "
            "%d x %d matrix, above the %.0f GB FASTPCA_TG_MEM_GB limit; raise "
            "that variable or use backend = 'pytorch'/'r'" %
            (est / 1e9, nobs, nvar, MEM_LIMIT_GB))

    x0_np = np.ascontiguousarray(X, dtype=np.float32)
    obs_np = None
    if has_na:
        x0_np[nan_mask] = 0.0            # masked entries stay out of products
        obs_np = np.ascontiguousarray((~nan_mask).astype(np.float32))
    del nan_mask

    dev = _resolve_device(device)
    x0 = Tensor(x0_np).to(dev).realize()
    obs = Tensor(obs_np).to(dev).realize() if has_na else None
    del x0_np, obs_np

    # numpy zeros, not Tensor.zeros: the latter is an expanded view whose
    # strides change once real values are written, which TinyJit rejects
    PPp = Tensor(np.zeros((nvar, ncomp), dtype=np.float32)).to(dev).realize()
    TTp = Tensor(np.zeros((nobs, ncomp), dtype=np.float32)).to(dev).realize()
    core = _get_core(dev, nobs, nvar, ncomp, block, has_na, bool(gramschmidt))

    eig = np.zeros(ncomp)
    loadings = np.zeros((nvar, ncomp))
    scores = np.zeros((nobs, ncomp))
    iters, converged = [], []

    for h in range(ncomp):
        # start from the column with the largest observed absolute sum
        if has_na:
            colscore = (x0.abs() * obs).sum(axis=0).numpy()
        else:
            colscore = x0.abs().sum(axis=0).numpy()
        scol = int(np.argmax(colscore))
        th = x0[:, scol].contiguous().realize()

        pciter, conv = 0, False
        while True:
            th_old = th.numpy()
            if has_na:
                th, ph = core(x0, obs, PPp, TTp, th)
            else:
                th, ph = core(x0, PPp, TTp, th)
            pciter += block
            new = th.numpy()
            nn = float(np.sqrt((new * new).sum()))
            no = float(np.sqrt((th_old * th_old).sum()))
            if nn > 0 and no > 0:
                d = new / nn - th_old / no
                if float((d * d).sum()) < tol:
                    conv = True
                    break
            if pciter >= maxit:
                print("WARNING: stopping after %d iterations for PC %d"
                      % (pciter, h + 1), file=sys.stderr)
                break

        # masked rank-one deflation, in place: no second m x n allocation
        outer = th.reshape(-1, 1) @ ph.reshape(1, -1)
        if has_na:
            x0.assign(((x0 - outer) * obs)).realize()
        else:
            x0.assign(x0 - outer).realize()

        loadings[:, h] = ph.numpy()
        scores[:, h] = new
        eig[h] = float((th * th).sum().item())
        iters.append(pciter)
        converged.append(bool(conv))

        if gramschmidt:
            # NOTE: this tinygrad version silently ignores .assign() on a slice
            # view, so the fixed-width buffers are updated functionally instead.
            col = Tensor((np.arange(ncomp) == h).astype(np.float32),
                         device=dev).reshape(1, -1)
            PPp.assign(Tensor.where(col > 0.5, ph.reshape(-1, 1), PPp)).realize()
            tvec = (th / float(np.sqrt(max(eig[h], 1e-300)))).reshape(-1, 1)
            TTp.assign(Tensor.where(col > 0.5, tvec, TTp)).realize()

    return {
        "eig": np.asarray(eig, dtype=np.float64),
        "scores": np.ascontiguousarray(scores, dtype=np.float64),
        "loadings": np.ascontiguousarray(loadings, dtype=np.float64),
        "iters": iters,
        "converged": converged,
    }
