#!/usr/bin/env python
"""Single-core pytorch NIPALS, a faithful port of the vendored R package.

    python TESTING/nipals_torch.py --x TESTING/_tmp/X_F2000_na05.f64 \
        --mask TESTING/_tmp/M_F2000_na05.u8 --ncomp 10

The algorithm mirrors TESTING/vendor/nipals/R/nipals.R line for line so that
eigenvalues, iteration counts and convergence behaviour are directly comparable:

  * centre / scale use observed-only means and sd (unbiased, n-1);
  * with missing values the loadings/scores denominators are the observed sums
    of squares, computed as vector matvecs (the R package instead materialises a
    full nobs x nvar matrix per iteration -- see README);
  * Gram-Schmidt is accumulated from the stored factors, so it costs O(ncomp)
    rather than R's tcrossprod of length nobs.

Unlike the R package this never allocates an O(nobs^2) / O(nvar^2) matrix, so it
runs on the full 978 x 199112 example.

Run one implementation per process so peak RSS is not polluted by the others.
"""
import argparse
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
    # macOS reports ru_maxrss in bytes (Linux would be KiB).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def load_bin(path, m, n):
    """Column-major float64 file -> F-contiguous array of shape (m, n)."""
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != m * n:
        raise SystemExit(f"{path}: expected {m*n} values, got {raw.size}")
    return raw.reshape((n, m), order="C").T.copy(order="F")


def load_mask(path, m, n):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != m * n:
        raise SystemExit(f"{path}: expected {m*n} bytes, got {raw.size}")
    return raw.reshape((n, m), order="C").T.copy(order="F").astype(bool)


def nipals(X, mask=None, ncomp=10, center=True, scale=False,
           maxiter=500, tol=1e-6, gramschmidt=True, verbose=False):
    """X: (nobs x nvar) float64 tensor. mask: bool tensor, True = missing."""
    nobs, nvar = X.shape
    has_na = mask is not None and bool(mask.any())

    obs_f = (~mask).to(torch.float64) if has_na else None

    # --- centre / scale with observed-only statistics ----------------------
    if has_na:
        cnt = obs_f.sum(0)
        cmeans = (X * obs_f).sum(0) / cnt
    else:
        cmeans = X.mean(0)
    if center:
        X.sub_(cmeans)          # in place: the caller owns this buffer
    csds = None
    if scale:
        if has_na:
            mu2 = ((X * X) * obs_f).sum(0) / (cnt - 1.0)
        else:
            mu2 = X.var(0, unbiased=True)
        csds = torch.sqrt(mu2)
        if bool((csds == 0).any()):
            raise SystemExit("At least one column has zero standard deviation")
        X.div_(csds)

    TotalSS = float(((X * X) * obs_f).sum()) if has_na else float((X * X).sum())
    # masked entries must contribute 0 to the matvecs (the R package sets them to 0).
    # x0 is a reusable buffer, updated in place after each deflation.
    x0 = (X * obs_f) if has_na else X

    eig = torch.zeros(ncomp, dtype=torch.float64)
    loadings = torch.empty((nvar, ncomp), dtype=torch.float64)
    scores = torch.empty((nobs, ncomp), dtype=torch.float64)
    iters = []
    pc_times = []

    PPp = torch.zeros((nvar, ncomp), dtype=torch.float64)  # only first h-1 used
    TTp = torch.zeros((nobs, ncomp), dtype=torch.float64)

    for h in range(ncomp):
        tp = time.perf_counter()
        # start column: max absolute sum (matches nipals::nipals, na.rm = TRUE)
        if has_na:
            colscore = (x0.abs() * obs_f).sum(0)
        else:
            colscore = X.abs().sum(0)
        scol = int(torch.argmax(colscore))
        th = x0[:, scol].clone()

        pciter = 1
        while True:
            # loadings p = X't / t't (observed denominator when there is NA)
            if has_na:
                denom_p = ((th * th) @ obs_f).clamp_min(1e-300)
                ph = (x0.T @ th) / denom_p
            else:
                ph = (X.T @ th) / float(th @ th)

            if gramschmidt and h > 0:
                ph = ph - PPp[:, :h] @ (PPp[:, :h].T @ ph)
            nrm = float(torch.sqrt((ph * ph).sum()))
            if nrm == 0.0:
                raise SystemExit(f"PC {h+1}: zero-norm loading; data is degenerate")
            ph = ph / nrm

            th_old = th
            if has_na:
                denom_t = (obs_f @ (ph * ph)).clamp_min(1e-300)
                th = (x0 @ ph) / denom_t
            else:
                th = (X @ ph) / float(ph @ ph)

            if gramschmidt and h > 0:
                th = th - TTp[:, :h] @ (TTp[:, :h].T @ th)

            # convergence on normalised scores (as in the R package)
            n_new = float(torch.sqrt((th * th).sum()))
            n_old = float(torch.sqrt((th_old * th_old).sum()))
            if n_new > 0 and n_old > 0:
                d = (th / n_new) - (th_old / n_old)
                if float((d * d).sum()) < tol:
                    break
            if pciter == maxiter:
                print(f"WARNING: stopping after {maxiter} iterations for PC {h+1}", file=sys.stderr)
                break
            pciter += 1

        # deflate in place (rank-1 update, BLAS dger) -- no m x n temporary
        X.addr_(th, ph, alpha=-1)
        if has_na:
            x0.copy_(X).mul_(obs_f)

        loadings[:, h] = ph
        scores[:, h] = th
        eig[h] = float((th * th).sum())
        iters.append(pciter)
        pc_times.append(time.perf_counter() - tp)

        if gramschmidt:
            PPp[:, h] = ph
            TTp[:, h] = th / float(torch.sqrt(eig[h]))
        if verbose:
            print(f"  PC{h+1}: iter={pciter} eig={float(eig[h]):.4f}", flush=True)

    sv = torch.sqrt(eig)
    scores = scores / sv
    return {
        "eig": eig.numpy(), "sv": sv.numpy(),
        "scores": scores.numpy(), "loadings": loadings.numpy(),
        "iter": iters, "pc_times": pc_times,
    }


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
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--dtype", choices=["fp64", "fp32"], default="fp64")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    t_load0 = time.perf_counter()
    Xnp = load_bin(args.x, args.m, args.n)
    mask_np = load_mask(args.mask, args.m, args.n) if args.mask else None
    X = torch.from_numpy(Xnp)
    mask = torch.from_numpy(mask_np) if mask_np is not None else None
    del Xnp, mask_np          # torch tensors keep the buffers alive; drop our refs
    if args.dtype == "fp32":
        X = X.float()
    t_load = time.perf_counter() - t_load0
    print(f"[torch] loaded {args.m}x{args.n} dtype={args.dtype} in {t_load:.2f}s "
          f"peakRSS={peak_gb():.3f}GB", flush=True)

    times, res = [], None
    for r in range(args.repeats):
        t0 = time.perf_counter()
        # nipals() deflates in place, so only copy when we will run again
        res = nipals(X if args.repeats == 1 else X.clone(), mask=mask, ncomp=args.ncomp,
                     center=bool(args.center), scale=bool(args.scale),
                     maxiter=args.maxiter, tol=args.tol,
                     gramschmidt=bool(args.gramschmidt))
        times.append(time.perf_counter() - t0)
        print(f"[torch] repeat {r+1}: {times[-1]:.2f}s iters={res['iter']}", flush=True)

    best = min(times)
    out = {
        "impl": "torch", "tag": args.tag, "m": args.m, "n": args.n,
        "dtype": args.dtype, "threads": args.threads,
        "ncomp": args.ncomp, "center": bool(args.center), "scale": bool(args.scale),
        "tol": args.tol, "maxiter": args.maxiter,
        "has_mask": mask is not None and bool(mask.any()),
        "load_s": round(t_load, 3),
        "times_s": [round(t, 4) for t in times],
        "best_s": round(best, 4),
        "iters": res["iter"],
        "pc_times_s": [round(t, 4) for t in res["pc_times"]],
        "eig": [round(float(v), 8) for v in res["eig"]],
        "peak_rss_gb": round(peak_gb(), 3),
        "wall_total_s": round(time.perf_counter() - _T0, 2),
    }
    print(f"[torch] best={best:.3f}s peakRSS={out['peak_rss_gb']}GB eig[:5]="
          f"{np.round(res['eig'][:5], 4).tolist()}", flush=True)

    if args.save_npz:
        np.savez(args.save_npz, eig=res["eig"], sv=res["sv"],
                 scores=res["scores"], loadings=res["loadings"])
        print(f"[torch] wrote {args.save_npz}", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh)
    print("RESULT_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
