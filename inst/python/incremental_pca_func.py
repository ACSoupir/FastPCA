"""FastPCA incremental PCA backend (PyTorch).

IncrementalPCAState keeps the factorization between blocks, so only one block
crosses the R/Python boundary per partial_fit() call. The update is the standard
incremental SVD (the one scikit-learn's IncrementalPCA uses): with Xc the
already pre-processed block,

    [ diag(S) @ Vh ]  = svd -> U, S_new, Vh_new   (keep the top ncomp)
    [      Xc      ]

and only S and Vh are carried forward. R owns all pre-processing (log2,
centring, scaling), so every backend decomposes identical blocks.

Device handling mirrors inst/python/torch_irlba_func.py: "GPU" means CUDA when
it is available, otherwise Apple Metal (MPS) -- and MPS cannot hold float64, so a
double-precision GPU request runs on the CPU with a message instead of failing.
MPS also has no kernel for torch.linalg.svd, so a single-precision Metal request
decomposes each block on the CPU through PyTorch's own fallback (only the matrix
products stay on Metal); that is reported too, and the tinygrad backend is the one
with a real accelerator decomposition.
"""

import numpy as np
import torch


def _resolve_device(device, precision):
    """Map the R-style device/precision request onto a concrete torch device."""
    if str(device).strip().upper() != "GPU":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        if str(precision).lower() == "single":
            return torch.device("mps")
        print("[incremental_pca] Apple Metal (MPS) has no float64 support; "
              "running double precision on the CPU. Use precision = 'single' "
              "for the GPU.", flush=True)
        return torch.device("cpu")
    print("[incremental_pca] GPU requested but no CUDA or Metal accelerator is "
          "available to pytorch; falling back to the CPU.", flush=True)
    return torch.device("cpu")


class IncrementalPCAState:
    """Streaming PCA state: partial_fit() per block, then finalize()."""

    def __init__(self, n_features, ncomp, device="CPU", cores=1,
                 precision="double"):
        self.n_features = int(n_features)
        self.ncomp = int(ncomp)
        precision = str(precision).lower()
        self.dtype = torch.float32 if precision == "single" else torch.float64
        self.device = _resolve_device(device, precision)
        if self.device.type == "mps":
            # torch.linalg.svd raises nothing on Metal: PyTorch prints its own
            # UserWarning and quietly runs the decomposition on the CPU, so say it
            # once here rather than let "device = GPU" look like a Metal speedup.
            print("[incremental_pca] Apple Metal (MPS) has no torch.linalg.svd "
                  "kernel, so each block is decomposed on the CPU through "
                  "pytorch's fallback; use backend = 'tinygrad' for a Metal "
                  "decomposition.", flush=True)
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, int(cores)))
        self.SV = None    # diag(S) @ Vh of everything fitted so far
        self.S = None
        self.Vh = None

    def partial_fit(self, X):
        A = torch.as_tensor(np.array(X), dtype=self.dtype, device=self.device)
        if A.ndim != 2 or int(A.shape[1]) != self.n_features:
            raise ValueError(
                "block has shape %s but the state was created for %d features"
                % (tuple(A.shape), self.n_features))
        B = A if self.SV is None else torch.cat([self.SV, A], dim=0)
        try:
            _, S, Vh = torch.linalg.svd(B, full_matrices=False)
        except (RuntimeError, NotImplementedError):
            # an accelerator may lack the kernel for this dtype; finish the
            # block on the host rather than fail the whole fit
            _, S, Vh = torch.linalg.svd(B.detach().cpu(), full_matrices=False)
            S, Vh = S.to(self.device), Vh.to(self.device)

        k = min(self.ncomp, int(S.numel()))
        self.S = S[:k]
        # contiguous(): the sliced Vh is a view, and SV must not keep B alive
        self.Vh = Vh[:k, :].contiguous()
        self.SV = (self.Vh * self.S[:, None]).detach().contiguous()
        return self

    def transform(self, X):
        """Scores of an already pre-processed block against the components."""
        if self.Vh is None:
            raise RuntimeError("transform() called before any partial_fit()")
        A = torch.as_tensor(np.array(X), dtype=self.dtype, device=self.device)
        return (A @ self.Vh.T).detach().cpu().numpy()

    def finalize(self, ncomp=None):
        """Components and singular values; ncomp drops the oversampling buffer."""
        if self.Vh is None:
            raise RuntimeError("finalize() called before any partial_fit()")
        if ncomp is not None:
            k = min(int(ncomp), int(self.S.numel()))
            self.S = self.S[:k]
            # contiguous(): the slice must not keep the buffered Vh alive
            self.Vh = self.Vh[:k, :].contiguous()
        return {
            "S": self.S.detach().cpu().numpy(),
            "Vh": self.Vh.detach().cpu().numpy(),
        }
