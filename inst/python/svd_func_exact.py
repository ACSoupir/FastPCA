import torch
import numpy as np
import gc

def exact_svd_py(A_np: np.ndarray):
    """
    Performs Exact SVD on a NumPy array using PyTorch.

    Args:
        A_np: Input NumPy array. Assumed to be Features x Samples based on original script's transpose logic.
              (e.g., if R passes Samples x Features, R side should transpose it before sending here).

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

    
    #compute the SVD of the smaller matrix B
    U, S, Vh = torch.linalg.svd(A, full_matrices = False)
    
    #return results
    U = U.to("cpu").detach().numpy()
    S = S.to("cpu").detach().numpy()
    Vh = Vh.to("cpu").detach().numpy()
    # Clean up GPU memory
    del A
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
