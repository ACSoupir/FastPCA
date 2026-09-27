"""Validate native tinygrad SVD against the in-package torch baseline.

Both libraries are visible in TESTING/.venv-tg (created with
--system-site-packages), so the comparison runs in one process on identical
matrices.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "inst", "python"))

import numpy as np  # noqa: E402
from torch_svd_func import randomized_svd_py as baseline_torch  # noqa: E402
from tinygrad_native import randomized_svd_tg  # noqa: E402


def decay(m, n, seed):
    rng = np.random.default_rng(seed)
    r = min(m, n, 120)
    U, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return ((U * (1000 * 0.85 ** np.arange(r))) @ V.T
            + 0.1 * rng.standard_normal((m, n)))


def shifted(m, n, seed):
    return np.random.default_rng(seed).standard_normal((m, n)) + 10.0


def angles(A, B):
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def main():
    m, n, k, p, q = 2000, 500, 50, 10, 2
    for name, gen in (("decay", decay), ("shifted", shifted)):
        seeds = (0, 1) if name == "decay" else (0,)
        for seed in seeds:
            A = gen(m, n, seed)
            ref = np.linalg.svd(A, compute_uv=False)[:k]
            Ub, Sb, Vhb = baseline_torch(A, k=k, p=p, q_iter=q,
                                         device="cpu", cores=4)
            Ut, St, Vht = randomized_svd_tg(A, k=k, p=p, q_iter=q, seed=0)
            rb = np.linalg.norm(A - (Ub[:, :k] * Sb[:k]) @ Vhb[:k, :])
            rt = np.linalg.norm(A - (Ut[:, :k] * St[:k]) @ Vht[:k, :])
            rb /= np.linalg.norm(A); rt /= np.linalg.norm(A)
            print(f"{name} seed={seed}: recon base={rb:.5f} tg={rt:.5f} | "
                  f"top10 sv_relerr base="
                  f"{np.max(np.abs(Sb[:10]-ref[:10])/ref[:10]):.1e} tg="
                  f"{np.max(np.abs(St[:10]-ref[:10])/ref[:10]):.1e} | "
                  f"U angle(top10)="
                  f"{angles(Ub[:, :10], Ut[:, :10]).max():.4f} deg", flush=True)


if __name__ == "__main__":
    main()
