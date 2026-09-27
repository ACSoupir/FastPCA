"""Generate deterministic test matrices into TESTING/.cache (gitignored)."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
os.makedirs(CACHE, exist_ok=True)

SIZES = os.environ.get("TG_SIZES", "2000x500,4000x1000,8000x3000,15000x2000")


def make_matrix(m, n, seed=0):
    rng = np.random.default_rng(seed)
    r = min(m, n, 120)
    U, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return ((U * (1000 * 0.85 ** np.arange(r))) @ V.T
            + 0.1 * rng.standard_normal((m, n))).astype(np.float64)


if __name__ == "__main__":
    for spec in SIZES.split(","):
        m, n = (int(x) for x in spec.lower().split("x"))
        path = os.path.join(CACHE, f"A_{m}x{n}.npy")
        if not os.path.exists(path):
            np.save(path, make_matrix(m, n))
        print(f"ready {path} {os.path.getsize(path) // 10**6} MB")
