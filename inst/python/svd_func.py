import torch
import numpy as np
import gc

def randomized_svd_py(A_np: np.ndarray, k: int, p: int = 10, q_iter: int = 2):
    """
    Performs Randomized SVD on a NumPy array using PyTorch.

    Args:
        A_np: Input NumPy array. Assumed to be Features x Samples based on original script's transpose logic.
              (e.g., if R passes Samples x Features, R side should transpose it before sending here).
        k: Number of singular values/vectors to compute.
        p: Oversampling parameter.
        q_iter: Number of power iterations.

    Returns:
        Tuple of (U, S, Vh) as NumPy arrays.
        U: Left singular vectors (Features x k).
        S: Singular values (k).
        Vh: Transpose of right singular vectors (Samples x k).
    """
    A = torch.tensor(A_np)

    # Detect and send tensor to CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = A.to(device)

    print(f"Python (randomized_svd_py): Input tensor shape: {A.shape}, device: {A.device}")

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
    U = U.to("cpu").detach().numpy()
    S = S.to("cpu").detach().numpy()
    Vh = Vh.to("cpu").detach().numpy()
    # Clean up GPU memory
    del A, Omega, Y, Q, B, U_tilde
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    #final return
    return U, S, Vh

if __name__ == "__main__":
    import time
    # This block allows you to run and test the Python script directly
    # It will not be executed when imported by reticulate
    print("Running direct Python script test...")
    # Create a dummy matrix: 100 features, 20 samples (Features x Samples)
    # Use float64 to match R's default numeric type, reticulate will send it this way
    test_data_np = np.random.rand(100, 20).astype(np.float64) * 100
    # Add a small value to avoid log(0) if you were testing preprocessing here
    # test_data_np[test_data_np < 1] = 1.0

    start_time = time.time() # You'll need to import time again for this block if you want it
    U_test, S_test, Vh_test = randomized_svd_py(test_data_np, k=5, p=5, q_iter=1)
    end_time = time.time()
    # print(f"Elapsed time: {end_time - start_time:.4f} seconds")

    print(f"Test U shape: {U_test.shape}")
    print(f"Test S shape: {S_test.shape}")
    print(f"Test Vh shape: {Vh_test.shape}")

    # You can also compute PVE if you have the original scaled data
    # original_sum_of_squares = np.sum(test_data_np**2)
    # pve_test = S_test**2 / original_sum_of_squares
    # print("PVE:", pve_test)
