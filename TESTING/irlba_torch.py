#!/usr/bin/env python
"""PyTorch port of irlba's own iterative algorithm (Lanczos bidiagonalization
with implicit restart), benchmarked against irlba::irlba and FastPCA-pytorch.

    python TESTING/irlba_torch.py --x TESTING/_tmp/X_F2000_na00.f64 \
        --m 978 --n 2000 --nv 10 [--work 17] [--threads 4]

This is a faithful transcription of the *pure-R fallback* in the installed
irlba 2.3.5.1 (reached with fastpath = FALSE): same start-vector handling,
classical Gram-Schmidt reorthogonalization (irlba::orthog), the same
bidiagonal B built incrementally, svd(B) for the small SVD, and the same
convtests() stopping rule (residual < tol*Smax AND singular-value change <
svtol). The only differences are dtype/device (torch) and that it never calls
the C path, so the algorithm is directly comparable to irlba(fastpath = FALSE).

Pass --v0 <file> (a raw little-endian float64 vector of length n) to make the
starting vector identical to an R run; otherwise torch.randn is used.

--precision double|single (aliases fp64/fp32) and --device CPU|GPU select
precision and compute device.  They are deliberately independent: fp64 is exact
but CPU-only on Apple hardware (MPS cannot hold float64), while --precision single
unlocks the GPU.  MPS has no native
SVD/QR, so the small work x work svd(B) is computed on a host copy and moved
back -- see the note in irlba_torch().

Output: RESULT_JSON on stdout, optional --save-npz / --save-bin for validation.
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


def to_np(t):
    """Tensor -> numpy, moving off an accelerator device first."""
    return t.detach().cpu().numpy()


def peak_gb():
    # macOS reports ru_maxrss in bytes (Linux would be KiB).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def load_bin(path, m, n):
    """Column-major float64 file -> F-contiguous array of shape (m, n)."""
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != m * n:
        raise SystemExit(f"{path}: expected {m*n} values, got {raw.size}")
    return raw.reshape((n, m), order="C").T.copy(order="F")


# --- helpers mirroring irlba's R internals ----------------------------------
def norm2(x):
    """irlba::norm2 -- Euclidean norm of a vector."""
    return float(torch.linalg.vector_norm(x))


def orthog(Y, X):
    """irlba::orthog -- Y - X (X'Y), classical projection.

    X is either 1-D (a single direction) or a matrix whose columns span the
    subspace; Y is 1-D. Matches R's cross()/crossprod orientation exactly.
    """
    if X.dim() == 1:
        return Y - X * torch.dot(X, Y)
    # Empty subspace (j == 0): the projection is identically zero.  Returning Y
    # early also avoids an MPS internal assert on a (m, 0) @ (0,) product.
    if X.numel() == 0:
        return Y
    coef = X.transpose(0, 1) @ Y           # cross(X, Y)
    return Y - X @ coef


def _convtests(k_org, Bsvd_d, residuals, k, Smax, lastsv, svtol,
               maxritz, work, S, tol):
    """irlba::convtests -- the stopping rule used by both code paths."""
    subspace_converged = residuals[:k_org] < tol * Smax
    if lastsv is None:
        lastsv = torch.zeros_like(Bsvd_d)
    denom = Bsvd_d[:k_org]
    safe = torch.where(denom == 0, torch.ones_like(denom), denom)
    delta_converged = (torch.abs(denom - lastsv[:k_org]) / safe) < svtol
    len_res = int((subspace_converged & delta_converged).sum().item())
    if len_res >= k_org:
        return True, k
    if S == 0:
        return True, k
    augment = min(int(subspace_converged.sum().item()), maxritz)
    k = int(min(max(k, k_org + augment), work - 1))
    return False, k


def _b_extend(B, S, R_, j, dev=None, dtype=torch.float64):
    """One incremental bidiagonal update.

    Replicates the two branches of irlba's fallback:
      if (j + 1 <= work) B <- rbind(cbind(B, 0), c(rep(0, ncol(B)-1), S, R))
      else               B <- rbind(B, c(rep(0, j-1), S))
    Here j is the 0-based loop index (R's j - 1); R_ is None in the else branch.
    """
    if B is None:
        return torch.tensor([[S, R_]], dtype=dtype, device=dev)
    ncol = B.shape[1]
    if R_ is not None:
        top = torch.cat([B, torch.zeros(B.shape[0], 1, dtype=B.dtype,
                                        device=B.device)], dim=1)
        row = torch.cat([torch.zeros(1, ncol - 1, dtype=B.dtype,
                                     device=B.device),
                         torch.tensor([[S, R_]], dtype=B.dtype,
                                      device=B.device)], dim=1)
        return torch.cat([top, row], dim=0)
    row = torch.cat([torch.zeros(1, j, dtype=B.dtype, device=B.device),
                     torch.tensor([[S]], dtype=B.dtype,
                                  device=B.device)], dim=1)
    return torch.cat([B, row], dim=0)


def irlba_torch(A, nv=5, work=None, maxit=1000, tol=1e-5, svtol=None,
                reorth=True, v0=None, maxritz=3, verbose=False):
    """Leading SVD of A (m x n) by irlba's Lanczos + implicit-restart iteration.

    Returns dict(d, u, v, iter, mprod, converged). Mirrors
    irlba::irlba(fastpath = FALSE) with smallest=FALSE, right_only=FALSE and
    no scale/shift/deflate.
    """
    if svtol is None:
        svtol = tol
    m, n = A.shape
    dev, dt = A.device, A.dtype
    eps = float(torch.finfo(dt).eps)
    k_org = int(nv)

    # argument clamps copied from irlba()
    if work is None:
        work = min(nv + 7, min(m, n))
    if k_org > min(m - 1, n - 1):
        raise SystemExit("max(nu, nv) must be strictly less than min(dim(A))")
    if work <= 1:
        raise SystemExit("work must be greater than 1")
    if work <= k_org:
        work = k_org + 1
    if work >= min(n, m):
        work = min(n, m)
        if work <= k_org:
            k_org = work - 1
    w_dim = int(work)

    V = torch.zeros(n, work, dtype=dt, device=dev)
    if v0 is None:
        V[:, 0] = torch.randn(n, dtype=A.dtype)
    else:
        vv = v0 if torch.is_tensor(v0) else torch.as_tensor(np.asarray(v0))
        V[:vv.numel(), 0] = vv.to(dtype=dt, device=dev)

    W = torch.zeros(m, w_dim, dtype=dt, device=dev)
    B = None
    eps2 = eps ** (4.0 / 5.0)
    eps23 = eps ** (2.0 / 3.0)

    iter_ = 1
    mprod = 0
    lastsv = None
    Smax, Smin = 1.0, None
    k = int(k_org)
    converged = False

    while iter_ <= maxit:
        j = 0
        if iter_ == 1:
            V[:, 0] = V[:, 0] / norm2(V[:, 0])
        else:
            j = k                       # R: j <- k + 1 (1-based)
        VJ = V[:, j]
        W[:, j] = A @ VJ                # w_dim > 1 here
        mprod += 1

        if iter_ != 1 and w_dim > 1 and reorth:
            W[:, j] = orthog(W[:, j], W[:, :j])
        S = norm2(W[:, j])
        if np.isnan(S) or (S < eps2 and j == 0):
            raise SystemExit("starting vector near the null space")
        if S < eps2:
            W[:, j] = torch.randn(m, dtype=dt, device=dev)
            if w_dim > 1:
                W[:, j] = orthog(W[:, j], W[:, :j])
            W[:, j] = W[:, j] / norm2(W[:, j])
            S = 0.0
        else:
            W[:, j] = W[:, j] / S

        while j < work:
            # F <- t(W[,j] %*% A);  F <- F - S * V[,j];  orthog(F, V[,1:j])
            F = A.transpose(0, 1) @ W[:, j]
            mprod += 1
            F = F - S * V[:, j]
            F = orthog(F, V[:, :j + 1])

            if (j + 1) < work:          # R: j+1 <= work
                R_ = norm2(F)
                if R_ < eps2:
                    Fr = torch.randn(n, dtype=dt, device=dev)
                    Fr = orthog(Fr, V[:, :j + 1])
                    V[:, j + 1] = Fr / norm2(Fr)
                    R_ = 0.0
                else:
                    V[:, j + 1] = F / R_
                B = _b_extend(B, S, R_, j, dev=dev, dtype=dt)

                W_old = W[:, j].clone()
                VJP1 = V[:, j + 1]
                W[:, j + 1] = A @ VJP1
                mprod += 1
                W[:, j + 1] = W[:, j + 1] - R_ * W_old
                if reorth and w_dim > 1:
                    W[:, j + 1] = orthog(W[:, j + 1], W[:, :j])
                S = norm2(W[:, j + 1])
                if S < eps2:
                    W[:, j + 1] = torch.randn(m, dtype=dt, device=dev)
                    if w_dim > 1:
                        W[:, j + 1] = orthog(W[:, j + 1], W[:, :j])
                    W[:, j + 1] = W[:, j + 1] / norm2(W[:, j + 1])
                    S = 0.0
                else:
                    W[:, j + 1] = W[:, j + 1] / S
            else:
                B = _b_extend(B, S, None, j, dev=dev, dtype=dt)
            j += 1

        Bsz = B.shape[0]
        R_F = norm2(F)
        F = F / R_F
        # B is only work x work (~17), so on a device without native SVD the
        # host copy is free next to the m x n products.  torch.linalg.svd would
        # silently fall back to the CPU on MPS anyway, with a warning per call.
        if dev.type == "mps":
            U_b, d_b, Vh_b = (t.to(dev) for t in torch.linalg.svd(B.cpu()))
        else:
            U_b, d_b, Vh_b = torch.linalg.svd(B)
        V_b = Vh_b.transpose(0, 1)

        if iter_ == 1:
            Smax = float(d_b[0])
            Smin = float(d_b[Bsz - 1])
        else:
            Smax = max(Smax, float(d_b[0]))
            Smin = min(Smin, float(d_b[Bsz - 1]))
        Smax = max(eps23, Smax)

        if not reorth and Smin / Smax < np.sqrt(eps):
            reorth = True

        Rvec = R_F * U_b[Bsz - 1, :]           # length Bsz
        convd, k = _convtests(k_org, d_b, torch.abs(Rvec), k, Smax,
                              lastsv, svtol, maxritz, work, S, tol)
        if verbose:
            print(f"  iter={iter_} mprod={mprod} sv[{k_org}]="
                  f"{float(d_b[k_org-1]):.3e} k={k}", flush=True)
        lastsv = d_b.clone()
        if convd:
            converged = True
            break
        if iter_ >= maxit:
            break

        # implicit restart: V[,1:(k+1)] <- cbind(V[,1:Bsz] %*% v[,1:k], F)
        newV = torch.empty(n, k + 1, dtype=dt, device=dev)
        newV[:, :k] = V[:, :Bsz] @ V_b[:, :k]
        newV[:, k] = F
        V[:, :k + 1] = newV
        B = torch.cat([torch.diag(d_b[:k]),
                       Rvec[:k].reshape(-1, 1)], dim=1)   # k x (k+1)
        W[:, :k] = W[:, :Bsz] @ U_b[:, :k]
        iter_ += 1

    if not converged:
        print("WARNING: did not converge -- try increasing maxit or work",
              file=sys.stderr)
    d = d_b[:k_org]
    u = W[:, :U_b.shape[0]] @ U_b[:, :k_org]
    v = V[:, :V_b.shape[0]] @ V_b[:, :k_org]
    return {"d": d, "u": u, "v": v, "iter": iter_, "mprod": mprod,
            "converged": converged}


def _resolve_device(requested, dtype_name):
    """Map an R-style device request onto a concrete torch device.

    "CPU" and "GPU" follow FastPCA's contract (R/validate.R, .fastpca_devices)
    exactly: a GPU request means CUDA when it is available and otherwise falls
    back to the CPU with a message -- never an error, and never silently.  The
    package has no Apple value in .fastpca_devices, so "MPS" is accepted here as
    an explicit benchmark-only extra that keeps the Apple GPU reachable; it has
    no float64 support, so an fp64 request is served on the CPU and *reported*.
    """
    req = str(requested).strip().upper()
    if req == "MPS":
        if not torch.backends.mps.is_available():
            print("[torch_irlba] MPS supplied as device but not available to "
                  "pytorch;\nfalling back to the CPU.", flush=True)
            return torch.device("cpu"), "no-mps-fallback"
        if dtype_name == "fp64":
            print("[torch_irlba] MPS has no float64 support; running fp64 on "
                  "the CPU (use --precision single for the GPU).", flush=True)
            return torch.device("cpu"), "mps-fp64-downgrade"
        return torch.device("mps"), "requested"
    if req != "GPU":
        return torch.device("cpu"), "requested"          # CPU, the default
    if torch.cuda.is_available():
        return torch.device("cuda"), "requested"
    print("CUDA supplied as device but not available to pytorch;\n"
          "falling back to the CPU.", flush=True)
    return torch.device("cpu"), "no-gpu-fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--nv", type=int, default=10)
    ap.add_argument("--work", type=int, default=None)
    ap.add_argument("--maxit", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--svtol", type=float, default=None)
    ap.add_argument("--center", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--precision", "--dtype", dest="dtype",
                    choices=["double", "single", "fp64", "fp32"],
                    default="double",
                    help="numeric precision; R-style 'double'/'single', with "
                         "fp64/fp32 accepted as aliases")
    ap.add_argument("--device", default="CPU",
                    help="'CPU' (default) or 'GPU'. Matches FastPCA's contract: "
                         "GPU means CUDA when available, otherwise the CPU with a "
                         "message. 'MPS' is an explicit extra for Apple GPUs.")
    ap.add_argument("--v0", default=None, help="raw float64 start vector (length n)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42,
                    help="torch RNG seed for the start vector (ignored with --v0)")
    ap.add_argument("--max-gb", type=float, default=20.0,
                    help="refuse before allocating if the input exceeds this")
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--save-bin", default=None,
                    help="prefix for .eig.f64/.loadings.f64/.json")
    ap.add_argument("--v0-out", default=None,
                    help="write the start vector actually used")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    # accept R-style precision names as well as the explicit fp64/fp32 aliases
    args.dtype = {"double": "fp64", "single": "fp32"}.get(
        str(args.dtype).lower(), args.dtype)
    # FastPCA vocabulary is upper-case CPU/GPU (R/validate.R .fastpca_devices)
    args.device = str(args.device).strip().upper()

    torch.set_num_threads(args.threads)

    # pre-flight memory refusal: input alone (8 or 4 bytes/element)
    bpe = 8 if args.dtype == "fp64" else 4
    proj_gb = (args.m * args.n * bpe) / 1e9
    if proj_gb > args.max_gb:
        out = {"impl": "torch_irlba", "tag": args.tag, "m": args.m, "n": args.n,
               "status": "skipped", "reason":
               f"input alone is {proj_gb:.3g} GB > cap {args.max_gb:g} GB"}
        print("RESULT_JSON " + json.dumps(out))
        return

    t_load0 = time.perf_counter()
    Xnp = load_bin(args.x, args.m, args.n)
    A = torch.from_numpy(Xnp)
    del Xnp
    if args.dtype == "fp32":
        A = A.float()
    t_load = time.perf_counter() - t_load0

    dev, device_status = _resolve_device(args.device, args.dtype)
    t_dev0 = time.perf_counter()
    A = A.to(dev)                      # no-op on CPU (same storage)
    if dev.type == "mps":
        torch.mps.synchronize()
    t_dev = time.perf_counter() - t_dev0

    # column-centre, exactly as the R runners do (rows = samples)
    if args.center:
        A.sub_(A.mean(dim=0))

    v0 = None
    if args.v0:
        raw_v0 = np.fromfile(args.v0, dtype="<f8")
        if raw_v0.size != args.n:
            raise SystemExit(f"{args.v0}: expected {args.n} values, got {raw_v0.size}")
        v0 = torch.from_numpy(raw_v0.copy())
    if args.v0_out:
        src = v0 if v0 is not None else torch.randn(args.n, dtype=torch.float64)
        np.asarray(src.detach().cpu().numpy(), dtype="<f8").tofile(args.v0_out)

    print(f"[torch_irlba] loaded {args.m}x{args.n} dtype={args.dtype} "
          f"device={dev.type} ({device_status}) in {t_load:.2f}s "
          f"+{t_dev:.2f}s transfer peakRSS={peak_gb():.3f}GB", flush=True)

    times, res = [], None
    for r in range(args.repeats):
        if v0 is None:
            torch.manual_seed(args.seed)   # identical start vector every repeat
        t0 = time.perf_counter()
        res = irlba_torch(A, nv=args.nv, work=args.work, maxit=args.maxit,
                          tol=args.tol, svtol=args.svtol, v0=v0)
        times.append(time.perf_counter() - t0)
        print(f"[torch_irlba] repeat {r+1}: {times[-1]:.3f}s iter={res['iter']} "
              f"mprod={res['mprod']}", flush=True)
        # ru_maxrss does not see Metal allocations, so the RSS guard alone
        # cannot bound an MPS run.  Check the driver-side total here and abort,
        # which makes memguard's kill (or this exit) the cap for GPU runs too.
        if dev.type == "mps":
            used = torch.mps.driver_allocated_memory() / 1e9
            print(f"[torch_irlba] MPS driver allocated {used:.3f}GB", flush=True)
            if used > args.max_gb:
                raise SystemExit(
                    f"MPS allocation {used:.2f}GB exceeds --max-gb "
                    f"{args.max_gb:g}GB")

    best = min(times)
    d = to_np(res["d"])
    out = {
        "impl": "torch_irlba", "tag": args.tag, "m": args.m, "n": args.n,
        "dtype": args.dtype, "threads": args.threads, "nv": args.nv,
        "device_requested": args.device, "device_resolved": dev.type,
        "device_status": device_status,
        "load_s": round(t_load, 3), "to_device_s": round(t_dev, 3),
        "torch_threads_effective": torch.get_num_threads(),
        "work": int(args.work) if args.work else None, "tol": args.tol,
        "svtol": args.svtol if args.svtol is not None else args.tol,
        "maxit": args.maxit, "center": bool(args.center),
        "converged": bool(res["converged"]), "iterations": int(res["iter"]),
        "mprod": int(res["mprod"]),
        "load_s": round(t_load, 3), "times_s": [round(t, 4) for t in times],
        "best_s": round(best, 4),
        "eig": [round(float(x), 8) for x in (d ** 2)],
        "sv": [round(float(x), 8) for x in d],
        "peak_rss_gb": round(peak_gb(), 3),
        "mps_driver_gb": (round(torch.mps.driver_allocated_memory() / 1e9, 3)
                          if dev.type == "mps" else None),
        "wall_total_s": round(time.perf_counter() - _T0, 2),
    }
    print(f"[torch_irlba] best={best:.3f}s iter={res['iter']} "
          f"mprod={res['mprod']} peakRSS={out['peak_rss_gb']}GB sv[:5]="
          f"{np.round(d[:5], 4).tolist()}", flush=True)

    if args.save_npz:
        np.savez(args.save_npz, d=d, eig=d ** 2,
                 u=to_np(res["u"]), v=to_np(res["v"]))
        print(f"[torch_irlba] wrote {args.save_npz}", flush=True)
    if args.save_bin:
        v = to_np(res["v"])
        np.asarray(d ** 2, dtype="<f8").tofile(args.save_bin + ".eig.f64")
        # R's writeBin(as.numeric(v)) is column-major; ndarray.tofile writes
        # C-order, so flatten in F order explicitly to stay byte-compatible.
        np.asarray(v, dtype="<f8").flatten(order="F").tofile(
            args.save_bin + ".loadings.f64")
        with open(args.save_bin + ".json", "w") as fh:
            json.dump({"nvar": int(v.shape[0]), "ncomp": int(v.shape[1])}, fh)
        print(f"[torch_irlba] wrote {args.save_bin}.*", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh)
    print("RESULT_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
