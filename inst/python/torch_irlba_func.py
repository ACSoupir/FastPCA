import gc
import numpy as np
import torch


def _resolve_device(device, precision):
    """Map the R-style device/precision request onto a concrete torch device.

    "CPU" is always CPU.  "GPU" means CUDA when it is available, otherwise the
    Apple Metal (MPS) accelerator on arm64 Macs.  MPS cannot hold float64, so a
    double-precision GPU request is served on the CPU and reported; this is why
    precision and device are separate arguments.  Mirrors FastPCA's contract that
    an unavailable accelerator falls back to the CPU with a message, never an
    error.
    """
    req = str(device).strip().upper()
    if req != "GPU":
        return torch.device("cpu"), "requested"
    if torch.cuda.is_available():
        return torch.device("cuda"), "requested"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        if precision == "single":
            return torch.device("mps"), "requested"
        print("[torch_irlba] Apple Metal (MPS) has no float64 support; running "
              "double precision on the CPU. Use precision = 'single' for the GPU.",
              flush=True)
        return torch.device("cpu"), "mps-fp64-downgrade"
    print("GPU supplied as device but no CUDA or Metal accelerator is available "
          "to pytorch; falling back to the CPU for computation", flush=True)
    return torch.device("cpu"), "no-gpu-fallback"


def _to_np(t):
    """Tensor -> numpy, moving off an accelerator device first."""
    return t.detach().cpu().numpy()


def _norm2(x):
    """irlba::norm2 -- Euclidean norm of a vector."""
    return float(torch.linalg.vector_norm(x))


def _orthog(Y, X):
    """irlba::orthog -- Y - X (X'Y), classical projection.

    X is either 1-D or a matrix whose columns span the subspace. The empty
    (j == 0) case returns Y directly, which also avoids an MPS internal assert
    on a (m, 0) @ (0,) product.
    """
    if X.dim() == 1:
        return Y - X * torch.dot(X, Y)
    if X.numel() == 0:
        return Y
    coef = X.transpose(0, 1) @ Y
    return Y - X @ coef


def _convtests(k_org, Bsvd_d, residuals, k, Smax, lastsv, svtol,
               maxritz, work, S, tol):
    """irlba::convtests -- the stopping rule used by both irlba code paths."""
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
    """One incremental bidiagonal update (irlba's fallback branches)."""
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


def _irlba_torch(A, nv=5, work=None, maxit=1000, tol=1e-5, svtol=None,
                 reorth=True, v0=None, maxritz=3):
    """Leading SVD of A (m x n) by irlba's Lanczos + implicit-restart iteration.

    Faithful transcription of the pure-R fallback in irlba (fastpath = FALSE):
    same start-vector handling, classical Gram-Schmidt reorthogonalisation
    (irlba::orthog), the same bidiagonal B, svd(B) for the small SVD, and the
    same convtests() stopping rule (residual < tol * Smax AND singular-value
    change < svtol).
    """
    if svtol is None:
        svtol = tol
    m, n = A.shape
    dev, dt = A.device, A.dtype
    eps = float(torch.finfo(dt).eps)
    k_org = int(nv)

    if work is None:
        work = min(nv + 7, min(m, n))
    if k_org > min(m - 1, n - 1):
        raise ValueError("max(nu, nv) must be strictly less than min(dim(A))")
    if work <= 1:
        raise ValueError("work must be greater than 1")
    if work <= k_org:
        work = k_org + 1
    if work >= min(n, m):
        work = min(n, m)
        if work <= k_org:
            k_org = work - 1
    w_dim = int(work)

    V = torch.zeros(n, w_dim, dtype=dt, device=dev)
    if v0 is None:
        V[:, 0] = torch.randn(n, dtype=dt)
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
            V[:, 0] = V[:, 0] / _norm2(V[:, 0])
        else:
            j = k
        VJ = V[:, j]
        W[:, j] = A @ VJ
        mprod += 1

        if iter_ != 1 and w_dim > 1 and reorth:
            W[:, j] = _orthog(W[:, j], W[:, :j])
        S = _norm2(W[:, j])
        if np.isnan(S) or (S < eps2 and j == 0):
            raise ValueError("starting vector near the null space")
        if S < eps2:
            W[:, j] = torch.randn(m, dtype=dt, device=dev)
            if w_dim > 1:
                W[:, j] = _orthog(W[:, j], W[:, :j])
            W[:, j] = W[:, j] / _norm2(W[:, j])
            S = 0.0
        else:
            W[:, j] = W[:, j] / S

        while j < work:
            F = A.transpose(0, 1) @ W[:, j]
            mprod += 1
            F = F - S * V[:, j]
            F = _orthog(F, V[:, :j + 1])

            if (j + 1) < work:
                R_ = _norm2(F)
                if R_ < eps2:
                    Fr = torch.randn(n, dtype=dt, device=dev)
                    Fr = _orthog(Fr, V[:, :j + 1])
                    V[:, j + 1] = Fr / _norm2(Fr)
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
                    W[:, j + 1] = _orthog(W[:, j + 1], W[:, :j])
                S = _norm2(W[:, j + 1])
                if S < eps2:
                    W[:, j + 1] = torch.randn(m, dtype=dt, device=dev)
                    if w_dim > 1:
                        W[:, j + 1] = _orthog(W[:, j + 1], W[:, :j])
                    W[:, j + 1] = W[:, j + 1] / _norm2(W[:, j + 1])
                    S = 0.0
                else:
                    W[:, j + 1] = W[:, j + 1] / S
            else:
                B = _b_extend(B, S, None, j, dev=dev, dtype=dt)
            j += 1

        Bsz = B.shape[0]
        R_F = _norm2(F)
        F = F / R_F
        # B is only work x work (~nv + 7), so on a device without native SVD the
        # host copy is free next to the m x n products; torch.linalg.svd would
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

        Rvec = R_F * U_b[Bsz - 1, :]
        convd, k = _convtests(k_org, d_b, torch.abs(Rvec), k, Smax,
                              lastsv, svtol, maxritz, work, S, tol)
        lastsv = d_b.clone()
        if convd:
            converged = True
            break
        if iter_ >= maxit:
            break

        newV = torch.empty(n, k + 1, dtype=dt, device=dev)
        newV[:, :k] = V[:, :Bsz] @ V_b[:, :k]
        newV[:, k] = F
        V[:, :k + 1] = newV
        B = torch.cat([torch.diag(d_b[:k]),
                       Rvec[:k].reshape(-1, 1)], dim=1)
        W[:, :k] = W[:, :Bsz] @ U_b[:, :k]
        iter_ += 1

    d = d_b[:k_org]
    u = W[:, :U_b.shape[0]] @ U_b[:, :k_org]
    v = V[:, :V_b.shape[0]] @ V_b[:, :k_org]
    return {"d": d, "u": u, "v": v, "iter": iter_, "mprod": mprod,
            "converged": converged}


def torch_irlba_py(A_np, nv=5, work=None, maxit=1000, tol=1e-5, svtol=None,
                   reorth=True, v0=None, center=False, scale_=False,
                   precision="double", device="CPU", cores=1):
    """Reticulate entry point: irlba's Lanczos SVD on a PyTorch backend.

    Rows of A_np are observations (samples/cells) and columns are features,
    matching FastPCA's convention. center/scale_ transform the columns before
    decomposing, as irlba's own arguments do. Returns a dict with d, u and v
    (numpy), the iteration count, matrix-product count and convergence flag.
    """
    precision = str(precision).strip().lower()
    if precision not in ("double", "single"):
        raise ValueError("precision must be 'double' or 'single'")
    dtype = torch.float64 if precision == "double" else torch.float32

    dev, device_status = _resolve_device(device, precision)
    torch.set_num_threads(int(cores))

    # Reticulate hands us a read-only numpy view. The algorithm never writes
    # into A (centring/scaling are out-of-place), so mark it writable instead of
    # copying; only fall back to a copy if numpy refuses.
    arr = np.asarray(A_np)
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64, copy=True)
    elif not arr.flags.writeable:
        try:
            arr.setflags(write=True)
        except ValueError:
            arr = np.array(arr, copy=True)

    A = torch.as_tensor(arr)
    if not A.is_contiguous():
        A = A.contiguous()
    A = A.to(dtype)

    if bool(center):
        A = A - A.mean(dim=0, keepdim=True)
    if bool(scale_):
        sd = A.std(dim=0, unbiased=True).clamp_min(torch.finfo(A.dtype).tiny)
        A = A / sd

    if dev.type != "cpu":
        A = A.to(dev)

    start = None
    if v0 is not None:
        arr = np.asarray(v0, dtype=np.float64).ravel()
        if arr.size != A.shape[1]:
            raise ValueError("the starting vector must have one entry per column")
        start = torch.as_tensor(arr).to(dtype=dtype, device=dev)

    res = _irlba_torch(A, nv=int(nv), work=(int(work) if work else None),
                       maxit=int(maxit), tol=float(tol),
                       svtol=(None if svtol is None else float(svtol)),
                       reorth=bool(reorth), v0=start)

    out = {
        "d": _to_np(res["d"]),
        "u": _to_np(res["u"]),
        "v": _to_np(res["v"]),
        "iterations": int(res["iter"]),
        "mprod": int(res["mprod"]),
        "converged": bool(res["converged"]),
        "device_resolved": dev.type,
        "device_status": device_status,
    }
    del A, res
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return out
