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

