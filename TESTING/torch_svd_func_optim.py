"""Drop-in optimised replacement for inst/python/torch_svd_func.py.

This file keeps the exact call signature of the in-package baseline so it can be
swapped in at the R layer (or copied to inst/python/) without touching any R
code:

    from torch_svd_func_optim import randomized_svd_py

It simply forwards to the best kernel from TESTING/pytorch_rsvd.py. The default
("stable") is ~2x faster than the baseline in float64 and ~3-5x faster when
float32 is acceptable, while matching the baseline's accuracy.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pytorch_rsvd import randomized_svd_stable as _kernel  # noqa: E402


def randomized_svd_py(A_np, k, p=10, q_iter=2, device="cpu", cores=2):
    """API-compatible with inst/python/torch_svd_func.randomized_svd_py."""
    return _kernel(A_np, k=k, p=p, q_iter=q_iter,
                   device=device, cores=cores)
