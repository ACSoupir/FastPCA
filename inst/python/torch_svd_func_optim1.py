import torch
from contextlib import nullcontext
import numpy as np

# note: call torch.set_float32_matmul_precision("high") once at init if you like
# and optionally set torch.backends.cuda.matmul.allow_tf32 = True on Ampere+ GPUs.

@torch.no_grad()
def randomized_svd_tsolve_stream(
    A_np: np.ndarray,
    k: int,
    p: int = 10,
    q_iter: int = 2,
    block_rows: int = 32_000,     # stream rows of A/Y for C = Y^T @ A
    use_amp: bool = False,         # bf16/fp16 matmuls with fp32 accumulate
    #device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    return_u: bool = True,
    return_v: bool = True,
    device: str = "CPU",
    cores: int = 4
):
    """
    Randomized SVD using:
      - Y = A @ Omega
      - thin-QR(Y) -> R   (do NOT materialize Q)
      - C = Y^T @ A  (streamed over rows)
      - B = R^{-T} @ C      [solve_triangular]
      - U_tilde, S, Vh = svd(B)
      - U = Y @ (R^{-1} @ U_tilde)   [solve_triangular]
    Avoids large Q and reduces peak memory for tall-skinny A (m >> n).

    Args:
      A: (m x n) tensor on CPU or GPU
      k: target rank
      p: oversampling
      q_iter: power iterations (0,1,2...)
      block_rows: streaming block size for row-wise accumulation of C
      use_amp: autocast matmuls where safe
      device/dtype: optional overrides
    """
    A = torch.tensor(A_np)
    
    #check device from user
    compute_device = device
    if compute_device.lower() == "gpu":
      if torch.cuda.is_available():
        device = torch.device("cuda")
      else:
        print("CUDA supplied as device but not available to pytorch;\nFalling back to CPU for computation")
        device = torch.device("cpu")
        #set number of cores to user given
        torch.set_num_threads(cores)
    else:
      device = torch.device("cpu")
      #set number of cores to user given
      torch.set_num_threads(cores)
    #dtyping
    if dtype is None:
        dtype = A.dtype
    
    A = A.to(device, dtype, non_blocking=True)

    m, n = A.shape
    b = k + p
    assert b <= min(m, n), "k+p must be <= min(m, n) for stability"

    # autocast only for matmuls; qr/svd stay in fp32 for stability
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else nullcontext

    # ---- range finder: Y = A @ Omega (and power iterations) ----
    with amp_ctx():
        # Omega on-the-fly (no need to store huge n x b if you don't want to).
        # Here we just allocate once; if you want zero-allocation, replace with a custom Triton kernel.
        Omega = torch.randn(n, b, device=device, dtype=dtype)
        Y = A @ Omega

    # Optional power iterations (stabilize subspace)
    for _ in range(q_iter):
        with amp_ctx():
            # Z = A^T @ Y   (n x b)
            Z = A.T @ Y
            # orthonormalize Z cheaply
            Z, _ = torch.linalg.qr(Z, mode="reduced")  # small (n x b)
            # Y = A @ Z
            Y = A @ Z

    # ---- thin QR: keep only R; never materialize Q ----
    # operate in fp32 for stability
    if use_amp:
      Y_32 = Y.float()
    else:
      Y_32 = Y #not changed
    # Y_32 = Q R  -> we save only R (b x b)
    _, R = torch.linalg.qr(Y_32, mode="reduced")

    # ---- C = Y^T @ A (stream over rows to limit peak mem) ----
    C = torch.zeros((b, n), device=device, dtype=Y_32.dtype)
    for i0 in range(0, m, block_rows):
        i1 = min(i0 + block_rows, m)
        Yi = Y_32[i0:i1]        # (rows x b)
        Ai = A[i0:i1] if not use_amp else A[i0:i1].float() # (rows x n) in fp32 for accuracy
        C.add_(Yi.T @ Ai)      # accumulate

    # ---- B = R^{-T} @ C  via triangular solve ----
    # Solve R^T X = C  => X = B
    B = torch.linalg.solve_triangular(R.T, C, upper=False)  # (b x n)

    # ---- SVD on small B ----
    # full_matrices=False gives U_tilde: (b x b_hat), Vh: (b_hat x n)
    U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
    # trim to k
    rk = min(k, U_tilde.shape[1])
    U_tilde = U_tilde[:, :rk]
    S = S[:rk]
    Vh = Vh[:rk, :]

    # ---- U = Y @ (R^{-1} @ U_tilde)  via triangular solve ----
    # Solve R X = U_tilde  => X = R^{-1} U_tilde
    X = torch.linalg.solve_triangular(R, U_tilde, upper=True)  # (b x rk)
    U = None
    if return_u:
        # multiply in fp32 then cast back to A.dtype
        U = (Y_32 @ X).to(dtype)

    if not return_v:
        return U, S

    return U.detach().numpy(), S.detach().numpy(), Vh.to(dtype).detach().numpy()
