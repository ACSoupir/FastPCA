import tinygrad
import numpy as np
import gc
import os
import math
from tinygrad.tensor import Tensor
from tinygrad import dtypes

def randomized_svd_py_tg(A_np: np.ndarray, k: int, p: int = 10, q_iter: int = 2, device: str = "cpu", cores: int = 2):
    """
    Performs scaling and centering, and transposing the matrix.

    Args:
        A_np: Input NumPy array. Assumed to be Features x Samples based on original script's transpose logic.
              (e.g., if R passes Samples x Features, R side should transpose it before sending here).
        k:    
        p:    
        q_iter: 
        device: 
        core: 

    Returns:
        NumPy arrays for the transformed data.
    """
    device = device.upper()
    if device == "CPU":
      os.environ["OMP_NUM_THREADS"] = str(cores)
      os.environ["MKL_NUM_THREADS"] = str(cores)
      os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
      os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores)
    
    #becuase for some reason Apple doesn't support 64bit with metal?
    if device != "CPU":
      tinygrad.device.Device.DEFAULT = "GPU"
      if "METAL" in tinygrad.device.Device._devices:
        A = Tensor(A_np, dtype = tinygrad.dtypes.float32).to("GPU")
        print("WARNING: Apple requires values to be 32-bit for GPU. Precision will be lost.")
      else:
        A = Tensor(A_np).to("GPU")
    else:
      A = Tensor(A_np).to("CPU")
      tinygrad.device.Device.DEFAULT = "CPU"
      
    #A is now on the device
    #prep power calculations
    m, n = A.shape
    device = A.device
    print("tinygrad!")
    dtype = A.dtype
    #random projection
    Omega = tinygrad.Tensor.randn((n, k+p), device = device, dtype = dtype)
    Y = A @ Omega
    # for _ in range(q_iter):
    #     Y = A @ (A.T @ Y)
    for i in range(q_iter):
        Q, _ = qr_acs(A.T @ Y)
        Y, _ = qr_acs(A @ Q)
    #orthonormalize the subspace to get a stable basis Q
    gc.collect()
    Y = Y.realize()
    Q, _ = qr_acs(Y)
    #project A onto the smaller subspace
    Q = Q.realize()
    B = Q.T @ A
    B = B.realize()
    #fallback to numpy because the svd for tidygrad doesn't seem to work right?
    #also, matrix multiplications in the for loop are insanely slow
    U_tilde, S, Vh = svd_acs(B, full_matrices = False)
    #recover left singular vectors for the original matrix A
    U = Q @ U_tilde
    
    #clean up the environment
    del A, Omega, Y, Q, U_tilde, B
    gc.collect()
    
    #return
    return U.numpy(), S.numpy(), Vh.numpy()

def qr_acs(self) -> tuple[Tensor, Tensor]:
  assert self.ndim > 1, f"expected two or more dimensions, got {self.ndim}"
  R = self.clone()
  b_shape, m, n = self.shape[0:self.ndim - 2], int(R.shape[-2]), int(R.shape[-1])
  Q = Tensor.eye(m, dtype = self.dtype, device = self.device).reshape((1,) * (len(self.shape) - 2) + 2 * (m,)).expand(b_shape + 2 * (m,)).contiguous()
  for i in range(int(min(m, n))):
    x = R[..., i:m, i]
    s = -x[..., 0].sign()
    u1 = x[..., 0] - s * x.square().sum(-1).sqrt()
    w = x.unsqueeze(-1) / u1.reshape(b_shape + 2 * (1,))
    w[..., 0, 0] = 1
    tau = (-s * u1 / x.square().sum(-1).sqrt()).reshape(b_shape + 2 * (1,)).expand(w.shape)
    R[..., i:m, :] = R[..., i:m, :] - (w * tau) @ (w.transpose(-2, -1) @ R[..., i:m, :])
    Q[..., :, i:m] = Q[..., :, i:m] - (Q[..., :, i:m] @ w) @ (tau.transpose(-2, -1) * w.transpose(-2, -1))
  return Q,R

def svd_acs(self, full_matrices = True) -> tuple[Tensor, Tensor, Tensor]:
  import math
  #partial implementation of https://www.netlib.org/lapack/lawnspdf/lawn169.pdf , pg 26
  assert self.ndim > 1, f"expected two or more dimensions, got {self.ndim}"
  b_shape, m, n = self.shape[:-2], int(self.shape[-2]), int(self.shape[-1])
  #preprocess the matrix
  Q, R = (qr_acs(self) if m >= n else qr_acs(self.transpose(-2, -1)))
  num, q_num = int(min(m, n)), int(max(m, n))
  U = R.shrink(tuple([(0, self.shape[i]) for i in range(self.ndim - 2)] + [(0, num), (0, num)])).contiguous()
  V = Tensor.eye(num, dtype = self.dtype, device = self.device).reshape((1,) * (self.ndim - 2) + (num, num)).expand(b_shape + 2 * (num,)).contiguous()
  #prepare round robin pairing
  permute, inverse_permute = Tensor.arange(0, num, dtype = dtypes.int, device = self.device), Tensor.zeros(num, dtype = dtypes.int, device = self.device).contiguous()
  permute[num//2:num] = permute[num//2:num].flip(0)
  inverse_permute[permute] = Tensor.arange(num, dtype = dtypes.int, device = self.device)
  def one_round_jacobi(U, V,permute,inverse_permute):
    #pair all the columns
    V_permuted, runoff_V = (V[..., permute].split(num - 1, -1)) if num % 2 == 1 else (V[..., permute], None)
    V_left, V_right = V_permuted.split(num//2, -1)
    U_permuted, runoff_U = (U[..., permute].split(num - 1, -1)) if num % 2 == 1 else (U[..., permute], None)
    U_left, U_right = U_permuted.split(num//2, -1)
    #compute the jacobi rotations for each pairing
    gamma = (U_left * U_right).sum(-2).reshape(b_shape + (1, num//2))
    alpha, beta = U_permuted.square().sum(-2).unsqueeze(-2).split(num//2, -1)
    tau = (beta - alpha) / (2 * gamma)
    t = tau.sign() / (tau.abs() + (1 + tau.square()).sqrt())
    c = 1 / (1 + t.square()).sqrt()
    s = c * t
    #apply the rotations
    U_left, U_right = c * U_left - s * U_right, s * U_left + c * U_right
    U = U_left.cat(U_right.cat(runoff_U, dim = -1) if num % 2 == 1 else U_right, dim = -1)[..., inverse_permute]
    V_left, V_right = c * V_left - s * V_right, s * V_left + c * V_right
    V = V_left.cat(V_right.cat(runoff_V, dim = -1) if num % 2 == 1 else V_right, dim = -1)[..., inverse_permute]
    #prepare the next round robin pairings
    if num % 2 == 1: permute = ((permute - 1) % num)
    else: permute = permute[0].reshape(1).cat(((permute[1:num] - 2) % (num - 1)) + 1)
    inverse_permute = inverse_permute.scatter(0,permute,Tensor.arange(num,dtype=dtypes.int32, device = self.device))
    return U.realize(), V.realize(), permute.realize(), inverse_permute.realize()
  max_iterations, iterations_per_round = 1, int((num) * math.log2(num) * 2 + 2)#sorta heuristic, most use num*log2(num)
  for _ in range(max_iterations * iterations_per_round): U, V, permute, inverse_permute = one_round_jacobi(U, V, permute, inverse_permute)
  #extract singular values and sort. construct U from Q
  S, indices = U.square().sum(-2).sqrt().sort(dim = -1, descending=True)
  new_indices = Tensor.arange(num, device = U.device).reshape((1,) * (self.ndim - 1) + (num,)).expand(b_shape + 2 * (num,)).contiguous()
  new_indices[..., :num] = indices.reshape(b_shape + (1,) + (U.shape[0],)).expand(b_shape + 2 * (num,))
  U,V = U.gather(-1, new_indices[...,0:num,0:num]) / S.unsqueeze(-2), V.gather(-1, new_indices[..., 0:num, 0:num])
  
  padded_u = Tensor.eye(q_num, dtype = U.dtype, device = U.device).reshape((1,) * (self.ndim - 2) + 2 * (q_num,)).expand(b_shape + 2 * (q_num,)).contiguous()
  padded_u[..., 0:num, 0:num] = U
  U = Q @ padded_u
  if not full_matrices: U, V = U[..., 0:num], V[..., 0:num]
  V = V.realize()
  U = U.realize()
  S = S.realize()
  return (U, S, V.transpose(-2,-1)) if m >= n else (V, S, U.transpose(-2, -1))
