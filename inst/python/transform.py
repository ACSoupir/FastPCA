import torch
import numpy as np
import gc

def transform_py(A_np: np.ndarray, log2: int = 0, transpose: int = 0, scale: int = 1):
    """
    Performs scaling and centering, and transposing the matrix.

    Args:
        A_np: Input NumPy array. Assumed to be Features x Samples based on original script's transpose logic.
              (e.g., if R passes Samples x Features, R side should transpose it before sending here).
        log2: integer of 1 or 0 for whether to log transform the data
        transpose: integer of 1 or 0 for whether to tranpose the matrix.
        scale: integer of 1 or 0 for whether to mean center and unit variance transform the data. For whether to scale columns

    Returns:
        NumPy arrays for the transformed data.
    """
    A = torch.tensor(A_np)
    
    if log2 == 1:
      A = torch.log2(A)
      
    if transpose == 1:
      A = torch.tensor(A).T
    
    if scale == 1:
      #mean center
      mean_vec = A.mean(dim=0)
      A = A - mean_vec
      #scale
      std_vec = A.std(dim=0)
      A = A / std_vec
    
    A = A.detach().numpy()

    return A

if __name__ == "__main__":
    import time
    print("Running direct Python script test...")
    test_data_np = np.random.rand(100, 20).astype(np.float64) * 100
    # Add a small value to avoid log(0) if you were testing preprocessing here
    test_data_np[test_data_np < 1] = 1.0

    start_time = time.time() # You'll need to import time again for this block if you want it
    out_data = transform_py(test_data_np, log2 = 1, transpose = 1, scale = 1)
    end_time = time.time()
    # print(f"Elapsed time: {end_time - start_time:.4f} seconds")

    # You can also compute PVE if you have the original scaled data
    # original_sum_of_squares = np.sum(test_data_np**2)
    # pve_test = S_test**2 / original_sum_of_squares
    # print("PVE:", pve_test)
