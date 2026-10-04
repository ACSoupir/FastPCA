#!/usr/bin/env python
"""Single-core tinygrad NIPALS (Metal or CPU), a faithful port of nipals_torch.py.

    python TESTING/nipals_tinygrad.py --x TESTING/_tmp/X_F2000_na05.f64 \
        --mask TESTING/_tmp/M_F2000_na05.u8 --m 978 --n 2000 --ncomp 10

Algorithm is identical to TESTING/nipals_torch.py (and so to the vendored R
package): observed-only centring/scaling, masked products with observed-only
denominators, Gram-Schmidt against the stored factors, convergence on the
normalised score vector. What differs is the execution engine:

  * float32 throughout, because tinygrad's Metal backend has no fp64;
  * the whole iterative loop is one TinyJit graph. tinygrad's eager dispatch
    costs ~200 ms per op on this machine, so the Python-level loop has to be
    unrolled B steps at a time (`--block`, default 8) and replayed as one graph;
    the host only syncs once per block to test convergence. `--block=1` restores
    one-sync-per-iteration for exact iteration counts.
  * only two m x n device buffers are ever live (x0 and, when needed, the mask).
    The input is centred/scaled in place on the host, masked immediately and
    freed before upload; no separate unmasked copy of X is ever kept.
  * a single fixed-width (nvar x ncomp) Gram-Schmidt buffer is reused for every
    component, so the JIT graph never changes shape.

Peak RSS is capped at 20 GB (FASTPCA_TG_MEM_GB to override), matching the
safeguard in tinygrad_native.py.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_HERE, ".cache"))
# tinygrad's CPU backend shells out to clang. The anaconda toolchain wrapper
# (arm64-apple-darwin20.0.0-cc) shadows Apple clang on PATH and is also what CC
# points at, so fix both before importing tinygrad.
os.environ["PATH"] = "/usr/bin:/bin:" + os.environ.get("PATH", "")
os.environ["CC"] = os.environ.get("FASTPCA_TG_CC", "/usr/bin/clang")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

try:
    from tinygrad import Tensor, TinyJit
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tinygrad required; run with TESTING/.venv-tg/bin/python") from exc

MEM_LIMIT_GB = float(os.environ.get("FASTPCA_TG_MEM_GB", "20"))
MEM_LIMIT_BYTES = int(MEM_LIMIT_GB * (1024 ** 3))
_T0 = time.perf_counter()


def peak_gb() -> float:
    # macOS reports ru_maxrss in bytes (Linux would be KiB).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (rss if sys.platform == "darwin" else rss * 1024) / 1e9


def check_mem(where: str = "") -> None:
    if peak_gb() * 1e9 > MEM_LIMIT_BYTES:
        raise MemoryError(f"peak RSS {peak_gb():.2f} GB exceeds the "
                          f"{MEM_LIMIT_GB:.0f} GB safeguard ({where})")


def check_shape(m: int, n: int) -> None:
    """Refuse a shape whose fp32 working set (x0 + mask + deflation) is too big."""
    est = 4 * m * n * 3
    if est > MEM_LIMIT_BYTES:
        raise MemoryError(f"projected fp32 working set {est / 1e9:.2f} GB for "
                          f"{m}x{n} exceeds the {MEM_LIMIT_GB:.0f} GB safeguard")


def load_bin(path, m, n):
    """Column-major float64 file -> (m, n) view of the raw buffer.

    No transpose copy is made: reshape((n, m)) is already C-contiguous in
    column-major order and .T presents it as samples x features. Saves a full
    156 MB on the 20k tier and 1.5 GB at full size.
    """
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != m * n:
        raise SystemExit(f"{path}: expected {m*n} values, got {raw.size}")
    return raw.reshape((n, m), order="C").T


def load_mask(path, m, n):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != m * n:
        raise SystemExit(f"{path}: expected {m*n} bytes, got {raw.size}")
    return raw.reshape((n, m), order="C").T.copy(order="F").astype(bool)


_DEVICE_OK: dict = {}


def _device_works(dev: str) -> bool:
    if dev not in _DEVICE_OK:
        try:
            (Tensor([1.0, 2.0], device=dev) + 1).realize().numpy()
            _DEVICE_OK[dev] = True
        except Exception:
            _DEVICE_OK[dev] = False
    return _DEVICE_OK[dev]


def resolve_device(device=None) -> str:
    req = str(device or os.environ.get("FASTPCA_TG_DEVICE", "METAL")).upper().strip()
    order = {"CPU": ["CPU"], "GPU": ["METAL", "CUDA"],
             "METAL": ["METAL"]}.get(req, ["METAL", "CPU"])
    for cand in order:
        if _device_works(cand):
            return cand
    raise RuntimeError(f"no working tinygrad device among {order}")


def prepare(X, mask=None, center=True, scale=False):
    """Observed-only centring/scaling; X is modified in place.

    Returns (x0, obs) as contiguous float32 arrays. x0 already has missing
    entries zeroed and is the only copy of the data kept; obs is None when there
    is no missingness, so the complete-data path allocates no mask at all.
    """
    has_na = bool(mask is not None and mask.any())
    if has_na:
        obs_b = ~mask
        cnt = obs_b.sum(0)
        cmeans = np.einsum("ij,ij->j", X, obs_b) / cnt
    else:
        cmeans = X.mean(0)
    if center:
        X -= cmeans
    if scale:
        mu2 = (np.einsum("ij,ij->j", X * X, obs_b) / np.maximum(cnt - 1.0, 1.0)
               if has_na else X.var(0, ddof=1))
        csds = np.sqrt(mu2)
        if not np.all(csds > 0):
            raise SystemExit("at least one column has zero standard deviation")
        X /= csds
    x0 = np.ascontiguousarray(X.astype(np.float32))
    obs_out = None
    if has_na:
        obs_out = np.ascontiguousarray(obs_b.astype(np.float32))
        x0 *= obs_out
    return x0, obs_out, has_na


_JIT_CACHE: dict = {}


def _get_jit(dev, m, n, k, block, has_na, gramschmidt):
    """One TinyJit graph for (shape, block size, mask/GS configuration).

    Unrolling `block` NIPALS steps into a single graph is what makes tinygrad
    competitive: eager dispatch costs ~200 ms/op here, a replayed graph does not.
    Gram-Schmidt always runs against fixed-width (nvar x ncomp) factor buffers so
    the graph shape is independent of which component we are on; unused columns
    are zero and contribute nothing to the projection.
    """
    key = (str(dev), int(m), int(n), int(k), int(block),
           bool(has_na), bool(gramschmidt))
    fn = _JIT_CACHE.get(key)
    if fn is not None:
        return fn
    B, GS = int(block), bool(gramschmidt)

    if has_na:
        @TinyJit
        def _core(x0, obs, PPp, TTp, th):
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
        def _core(x0, PPp, TTp, th):
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

    _JIT_CACHE[key] = fn = _core
    return fn


def run_prepared(x0_np, obs_np=None, has_na=False, ncomp=10, maxiter=500,
                 tol=1e-6, gramschmidt=True, block=8, device=None, verbose=False):
    """NIPALS on an already prepared (x0 fp32, optional mask) pair."""
    nobs, nvar = x0_np.shape
    check_shape(nobs, nvar)
    dev = resolve_device(device)

    x0 = Tensor(x0_np).to(dev).realize()
    obs = Tensor(obs_np).to(dev).realize() if has_na else None

    PPp = Tensor.zeros(nvar, ncomp).to(dev).realize()
    TTp = Tensor.zeros(nobs, ncomp).to(dev).realize()
    core = _get_jit(dev, nobs, nvar, ncomp, block, has_na, gramschmidt)

    eig = np.zeros(ncomp)
    loadings = np.zeros((nvar, ncomp))
    scores = np.zeros((nobs, ncomp))
    iters, pc_times = [], []

    for h in range(ncomp):
        tp = time.perf_counter()
        if has_na:
            colscore = (x0.abs() * obs).sum(axis=0).numpy()
        else:
            colscore = x0.abs().sum(axis=0).numpy()
        scol = int(np.argmax(colscore))
        th = x0[:, scol].contiguous().realize()

        n_iter, conv = 0, False
        while True:
            th_old = th.numpy()
            if has_na:
                th, ph = core(x0, obs, PPp, TTp, th)
            else:
                # the complete-data variant has no mask argument
                th, ph = core(x0, PPp, TTp, th)
            n_iter += block
            new = th.numpy()
            n_new = float(np.sqrt((new * new).sum()))
            n_old = float(np.sqrt((th_old * th_old).sum()))
            if n_new > 0 and n_old > 0:
                d = new / n_new - th_old / n_old
                if float((d * d).sum()) < tol:
                    conv = True
                    break
            if n_iter >= maxiter:
                print(f"WARNING: stopping after {n_iter} iterations for PC {h+1}",
                      file=sys.stderr)
                break

        # masked rank-one deflation, in place: no second m x n allocation
        outer = th.reshape(-1, 1) @ ph.reshape(1, -1)
        if has_na:
            x0.assign(((x0 - outer) * obs)).realize()
        else:
            x0.assign(x0 - outer).realize()

        eig[h] = float((th * th).sum().item())
        loadings[:, h] = ph.numpy()
        scores[:, h] = new
        iters.append(n_iter)
        pc_times.append(time.perf_counter() - tp)

        if gramschmidt:
            PPp[:, h].assign(ph).realize()
            TTp[:, h].assign(th / float(np.sqrt(max(eig[h], 1e-300)))).realize()
        if verbose:
            print(f"  PC{h+1}: iter={n_iter} eig={eig[h]:.4f}", flush=True)
        check_mem(f"PC{h+1}")

    sv = np.sqrt(np.maximum(eig, 0.0))
    return {
        "eig": eig, "sv": sv,
        "scores": scores / np.where(sv > 0, sv, 1.0), "loadings": loadings,
        "iter": iters, "pc_times": pc_times,
    }


def nipals_tg(X, mask=None, ncomp=10, center=True, scale=False, maxiter=500,
              tol=1e-6, gramschmidt=True, block=8, device=None, verbose=False):
    """Convenience wrapper: prepare (centres X in place) then run."""
    t0 = time.perf_counter()
    x0_np, obs_np, has_na = prepare(X, mask=mask, center=center, scale=scale)
    prep_s = time.perf_counter() - t0
    out = run_prepared(x0_np, obs_np=obs_np, has_na=has_na, ncomp=ncomp,
                       maxiter=maxiter, tol=tol, gramschmidt=gramschmidt,
                       block=block, device=device, verbose=verbose)
    out["prep_s"] = prep_s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ncomp", type=int, default=10)
    ap.add_argument("--center", type=int, default=1)
    ap.add_argument("--scale", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--gramschmidt", type=int, default=1)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--device", default="METAL")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    t_load0 = time.perf_counter()
    Xnp = load_bin(args.x, args.m, args.n)
    mask_np = load_mask(args.mask, args.m, args.n) if args.mask else None
    t_load = time.perf_counter() - t_load0

    # centre/scale on the host, then drop the float64 inputs before uploading
    t_prep0 = time.perf_counter()
    x0_np, obs_np, has_na = prepare(Xnp, mask=mask_np,
                                    center=bool(args.center),
                                    scale=bool(args.scale))
    t_prep = time.perf_counter() - t_prep0
    has_mask = bool(mask_np is not None and mask_np.any())
    del Xnp, mask_np
    print(f"[tinygrad] loaded {args.m}x{args.n} in {t_load:.2f}s "
          f"peakRSS={peak_gb():.3f}GB", flush=True)

    times, res = [], None
    for r in range(args.repeats):
        t0 = time.perf_counter()
        res = run_prepared(x0_np, obs_np=obs_np, has_na=has_na,
                           ncomp=args.ncomp, maxiter=args.maxiter,
                           tol=args.tol, gramschmidt=bool(args.gramschmidt),
                           block=args.block, device=args.device)
        times.append(time.perf_counter() - t0)
        print(f"[tinygrad] repeat {r+1}: {times[-1]:.2f}s iters={res['iter']}", flush=True)

    best = min(times)
    out = {
        "impl": "tinygrad",
        "device": resolve_device(args.device).lower(),
        "tag": args.tag, "m": args.m, "n": args.n,
        "dtype": "fp32", "threads": 1, "block": args.block,
        "ncomp": args.ncomp, "center": bool(args.center), "scale": bool(args.scale),
        "tol": args.tol, "maxiter": args.maxiter,
        "has_mask": has_mask,
        "load_s": round(t_load, 3), "prep_s": round(t_prep, 3),
        "times_s": [round(t, 4) for t in times], "best_s": round(best, 4),
        "iters": res["iter"],
        "pc_times_s": [round(t, 4) for t in res["pc_times"]],
        "eig": [round(float(v), 8) for v in res["eig"]],
        "peak_rss_gb": round(peak_gb(), 3),
        "wall_total_s": round(time.perf_counter() - _T0, 2),
    }
    print(f"[tinygrad] best={best:.3f}s peakRSS={out['peak_rss_gb']}GB eig[:5]="
          f"{np.round(res['eig'][:5], 4).tolist()}", flush=True)

    if args.save_npz:
        np.savez(args.save_npz, eig=res["eig"], sv=res["sv"],
                 scores=res["scores"], loadings=res["loadings"])
        print(f"[tinygrad] wrote {args.save_npz}", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh)
    print("RESULT_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
