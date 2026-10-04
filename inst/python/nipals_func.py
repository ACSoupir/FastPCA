"""PyTorch NIPALS for the FastPCA 'pytorch' backend.

The matrix arrives already centred/scaled by NIPALS() in R, so this module only
runs the iterative loop. Missing values arrive as NaN and are detected with
~isfinite, which means no mask matrix crosses the reticulate boundary.

The algorithm matches nipals::nipals: masked entries are held at zero for every
product and the loadings/scores denominators sum over observed entries only.
Deflation subtracts the masked rank-one component. Everything is float64 (the
package never computes in single precision).
"""
import gc
import sys
import warnings

import numpy as np
import torch


def nipals_py(X, ncomp=10, tol=1e-6, maxit=500, gramschmidt=True,
              device="CPU", cores=1):
    """Run NIPALS on a (nobs x nvar) array that may contain NaN.

    Returns a dict with eig, scores (native T), loadings, iters and converged.
    """
    X = np.asarray(X, dtype=np.float64)

    if device is not None and str(device).lower() == "gpu":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            print("CUDA supplied as device but not available to pytorch;\n"
                  "Falling back to CPU for computation")
            dev = torch.device("cpu")
    else:
        dev = torch.device("cpu")
    if dev.type == "cpu":
        torch.set_num_threads(int(cores))

    with warnings.catch_warnings():
        # reticulate hands over a read-only NumPy view; the tensor is only ever
        # read (nan_to_num returns a new tensor), so silence torch's notice
        warnings.filterwarnings("ignore",
                                message="The given NumPy array is not writable")
        Xt = torch.as_tensor(X, dtype=torch.float64).to(dev)
    mask = ~torch.isfinite(Xt)          # True where missing
    has_na = bool(mask.any())
    if has_na:
        obs_f = (~mask).to(torch.float64)
        Xt = torch.nan_to_num(Xt, nan=0.0)   # masked entries stay out of products
    else:
        obs_f = None

    nobs, nvar = Xt.shape
    eig = torch.zeros(ncomp, dtype=torch.float64)
    loadings = torch.empty((nvar, ncomp), dtype=torch.float64)
    scores = torch.empty((nobs, ncomp), dtype=torch.float64)
    iters = []
    converged = []

    PPp = torch.zeros((nvar, ncomp), dtype=torch.float64)
    TTp = torch.zeros((nobs, ncomp), dtype=torch.float64)

    for h in range(ncomp):
        # start from the column with the largest observed absolute sum
        scol = int(torch.argmax(Xt.abs().sum(0)))
        th = Xt[:, scol].clone()

        pciter = 1
        conv = False
        while True:
            if has_na:
                denom_p = (obs_f.T @ (th * th)).clamp_min(1e-300)
                ph = (Xt.T @ th) / denom_p
            else:
                ph = (Xt.T @ th) / float(th @ th)
            if gramschmidt and h > 0:
                ph = ph - PPp[:, :h] @ (PPp[:, :h].T @ ph)
            nrm = float(torch.sqrt((ph * ph).sum()))
            if not np.isfinite(nrm) or nrm == 0.0:
                raise RuntimeError(
                    "NIPALS failed at component %d: zero-norm loading vector" % (h + 1))
            ph = ph / nrm

            th_old = th
            if has_na:
                denom_t = (obs_f @ (ph * ph)).clamp_min(1e-300)
                th = (Xt @ ph) / denom_t
            else:
                th = (Xt @ ph) / float(ph @ ph)
            if gramschmidt and h > 0:
                th = th - TTp[:, :h] @ (TTp[:, :h].T @ th)

            # convergence on the normalised score vector, as in nipals::nipals
            nn = float(torch.sqrt((th * th).sum()))
            no = float(torch.sqrt((th_old * th_old).sum()))
            if nn > 0 and no > 0:
                d = (th / nn) - (th_old / no)
                if float((d * d).sum()) < tol:
                    conv = True
                    break
            if pciter >= maxit:
                print("WARNING: stopping after %d iterations for PC %d"
                      % (maxit, h + 1), file=sys.stderr)
                break
            pciter += 1

        # masked rank-one deflation, then re-zero the missing entries
        Xt = Xt - torch.outer(th, ph)
        if has_na:
            Xt = Xt * obs_f

        loadings[:, h] = ph
        scores[:, h] = th
        eig[h] = float((th * th).sum())
        iters.append(pciter)
        converged.append(conv)

        if gramschmidt:
            PPp[:, h] = ph
            TTp[:, h] = th / float(torch.sqrt(eig[h]))

    out = {
        "eig": eig.to("cpu").detach().numpy(),
        "scores": scores.to("cpu").detach().numpy(),
        "loadings": loadings.to("cpu").detach().numpy(),
        "iters": iters,
        "converged": converged,
    }
    del Xt, PPp, TTp
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return out
