import torch
import pandas as pd
#import gc
#import time

#import
df = pd.read_csv("20230411_smalley_slide27213_all_regions-nonorm_norm_filtered.txt", sep=None, engine='python', header=0, index_col=0)

#convert and transpose
tensor = torch.tensor(df.values).T
#log transpose
tensor_log = torch.log2(tensor)

#center at 0 by subtracting mean
mean_vec = tensor_log.mean(dim=0) #is the mean of the columns for some reason with 0
tensor_centered = tensor_log - mean_vec

#varaince
std_vec = tensor_centered.std(dim=0)
tensor_scaled = tensor_centered / std_vec
#clean up 
del df, tensor, tensor_log, mean_vec, tensor_centered, std_vec
gc.collect()

#detect and send
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_scaled = tensor_scaled.to(device)
#watch -n0.1 nvidia-smi

def randomized_svd(A, k, p=10, q_iter = 2):
    m, n = A.shape
    device = A.device
    dtype = A.dtype
    #create random projection
    Omega = torch.randn((n, k + p), device = device, dtype = dtype)
    #subspace identification with optional power iterations for accuracy
    Y = A @ Omega
    for _ in range(q_iter):
        Y = A @ (A.T @ Y)
    #orthonormalize the subspace to get a stable basis Q
    Q, _ = torch.linalg.qr(Y)
    #project A onto the smaller subspace
    B = Q.T @ A
    #compute the SVD of the smaller matrix B
    U_tilde, S, Vh = torch.linalg.svd(B, full_matrices = False)
    #recover left singular vectors for the original matrix A
    U = Q @ U_tilde
    #return results
    U = U.to("cpu")
    S = S.to("cpu")
    Vh = Vh.to("cpu")
    del m, n, device, dtype, Omega, Y, A, Q, B, U_tilde
    torch.cuda.empty_cache()
    return U[:, :k], S[:k], Vh[:k, :].T

start_time = time.time()
U, S, Vh = randomized_svd(tensor_scaled, k = 100, p = 10, q_iter = 2)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

tensor_scaled = tensor_scaled.to("cpu")
torch.cuda.empty_cache()
sos = torch.sum(tensor_scaled.square())

#percent variance explained
pve = S.square()/sos
pve
#cummulative variance
pve.cumsum(0)
#have to save out/conver to command line tool/write in R to with reticulate

##svd
import tinygrad
import numpy as np
import gc
import os
import pandas as pd
df = pd.read_csv("~/Downloads/test.txt", sep = None, engine = 'python', header = 0, index_col = 0)
A_np = df.to_numpy()
k = 10
p = 10 #oversampling
q_iter = 2 #power iterations
device = "GPU" #device
cores = 2
if device != "CPU":
  if "METAL" in tinygrad.device.Device._devices:
    A = tinygrad.tensor.Tensor(A_np, dtype = tinygrad.dtypes.float32).to("METAL")
    tinygrad.device.Device.DEFAULT = "METAL"
    print("WARNING: Apple requires values to be 32-bit for GPU. Precision will be lost.")
  else:
    A = tinygrad.tensor.Tensor(A_np).to("GPU")
    tinygrad.device.Device.DEFAULT = "GPU"
else:
  A = tinygrad.tensor.Tensor(A_np).to("CPU")
  tinygrad.device.Device.DEFAULT = "CPU"

m, n = A.shape
device = A.device
dtype = A.dtype
#random projection
Omega = tinygrad.Tensor.randn((n, k+p), device = device, dtype = dtype)
Y = A @ Omega
# for _ in range(q_iter):
#     Y = A @ (A.T @ Y)
for i in range(q_iter):
    Q, _ = tinygrad.Tensor.qr(A.T @ Y)
    Y, _ = tinygrad.Tensor.qr(A @ Q)
#orthonormalize the subspace to get a stable basis Q
Q, _ = tinygrad.Tensor.qr(Y)
#project A onto the smaller subspace
B = Q.T @ A


A_np = np.random.rand(100, 50).astype(np.float64) * 100
A_np[A_np < 1] = 1.0
A_np = A_np.transpose()
k = 10
p = 10 #oversampling
q_iter = 2 #power iterations
device = "GPU" #device
cores = 2

if device == "CPU":
  os.environ["OMP_NUM_THREADS"] = str(cores)
  os.environ["MKL_NUM_THREADS"] = str(cores)

#becuase for some reason Apple doesn't support 64bit with metal?
if device != "CPU":
  tinygrad.device.Device.DEFAULT = "GPU"
  if "METAL" in tinygrad.device.Device._devices:
    A = tinygrad.tensor.Tensor(A_np, dtype = tinygrad.dtypes.float32).to("GPU")
    print("WARNING: Apple requires values to be 32-bit for GPU. Precision will be lost.")
  else:
    A = tinygrad.tensor.Tensor(A_np).to("GPU")
else:
  A = tinygrad.tensor.Tensor(A_np).to("CPU")
  tinygrad.device.Device.DEFAULT = "CPU"
  
#A is now on the device
#prep power calculations
m, n = A.shape
device = A.device
dtype = A.dtype
#random projection
Omega = tinygrad.Tensor.randn((n, k+p), device = device, dtype = dtype)
Y = A @ Omega
# for _ in range(q_iter):
#     Y = A @ (A.T @ Y)
for i in range(q_iter):
    Q, _ = tinygrad.Tensor.qr(A.T @ Y)
    Y, _ = tinygrad.Tensor.qr(A @ Q)
#orthonormalize the subspace to get a stable basis Q
Q, _ = tinygrad.Tensor.qr(Y)
#project A onto the smaller subspace
B = Q.T @ A


U_tilde, S, Vh = np.linalg.svd(B.numpy(), full_matrices = False)
#recover left singular vectors for the original matrix A
Q = Q.numpy()
U = Q @ U_tilde
