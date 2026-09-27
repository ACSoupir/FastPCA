"""Peak-RSS + timing probe from a pre-saved matrix.

Run one backend per process so peak RSS is not polluted by the other:

    TESTING/.venv-tg/bin/python -u TESTING/tg_mem_probe.py baseline 8000x3000
    TESTING/.venv-tg/bin/python -u TESTING/tg_mem_probe.py tinygrad 8000x3000
"""
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CACHE = os.path.join(HERE, ".cache")
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402


def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


mode = sys.argv[1]
m, n = (int(x) for x in sys.argv[2].lower().split("x"))
k, p, q = 50, 10, 2

path = os.path.join(CACHE, f"A_{m}x{n}.npy")
if not os.path.exists(path):
    raise SystemExit(f"missing {path}; run TESTING/_gen_matrices.py first")
A = np.load(path)
print(f"[{mode}] after load peak={peak():.3f}", flush=True)

if mode == "baseline":
    sys.path.insert(0, os.path.join(REPO, "inst", "python"))
    from torch_svd_func import randomized_svd_py as _b

    def call():
        return _b(A, k=k, p=p, q_iter=q, device="cpu", cores=4)
else:
    from tinygrad_native import randomized_svd_tg as _t

    def call():
        return _t(A, k=k, p=p, q_iter=q)


t0 = time.perf_counter(); call(); first = time.perf_counter() - t0
print(f"[{mode}] after 1st SVD peak={peak():.3f} first={first*1e3:.2f}ms", flush=True)
ts = []
for _ in range(5):
    t0 = time.perf_counter(); call(); ts.append(time.perf_counter() - t0)
print(f"[{mode}] after 6 SVDs peak={peak():.3f} steady={min(ts)*1e3:.2f}ms", flush=True)
