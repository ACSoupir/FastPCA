"""Drop-in replacement for inst/python/tinygrad_svd_func.py.

Exposes the exact signature R calls through .globals$tinygrad_random_svd:

    randomized_svd_py_tg(A_np, k, p=10, q_iter=2, device="cpu", cores=2)

and returns the same (U, S, Vh) NumPy contract as torch_svd_func.randomized_svd_py
(U: m x b, S: length b, Vh: b x n with b = k + p; R slices to the first k).

The heavy range finder runs on tinygrad's Metal backend via
TESTING/tinygrad_native.py; the small b x n SVD of B stays on the host (Gram +
eigh).  `device` accepts R's "CPU"/"GPU"; see resolve_device() for how they map,
and `cores` is accepted only for signature compatibility (tinygrad schedules its
own GPU work -- there are no OpenMP threads to clamp).
"""
import os
import sys

_HERE = (os.path.dirname(os.path.abspath(__file__))
         if "__file__" in globals() else os.getcwd())
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tinygrad_native import randomized_svd_tg  # noqa: E402


def randomized_svd_py_tg(A_np, k, p=10, q_iter=2, device="cpu", cores=2):
    """See module docstring. `cores` is ignored (kept for R compatibility)."""
    return randomized_svd_tg(A_np, k=k, p=p, q_iter=q_iter, device=device)
