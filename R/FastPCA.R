#' Perform Randomized SVD with PyTorch backend
#'
#' This function uses PyTorch implementation of Randomized SVD to compute
#' the singular value decomposition.
#'
#' @param input_r_matrix A numeric R matrix. It's assumed that rows are observations
#'   (e.g., samples) and columns are features (e.g., genes). The function will
#'   transpose this for the PyTorch SVD based on the original Python script's logic.
#' @param k Integer. The number of singular values/vectors to compute.
#' @param p Integer. Oversampling parameter (default: 10).
#' @param q_iter Integer. Number of power iterations (default: 2).
#'
#' @return A list containing:
#'   \item{U}{The left singular vectors (R matrix). Dimensions: Features x k.}
#'   \item{S}{The singular values (R numeric vector). Length: k.}
#'   \item{Vh}{The transpose of the right singular vectors (R matrix). Dimensions: Samples x k.}
#'   All results are moved to CPU by the Python script and returned as R objects.
#' @export
#' @examples
#' \dontrun{
#'   #need to have environment first
#'   FastPCA::setup_py_env()
#'
#'   # Create a sample R matrix (e.g., 20 samples, 100 features)
#'   # Ensure values are positive for log transform.
#'   set.seed(123)
#'   test_data <- matrix(runif(20 * 100, min=1, max=100), nrow = 20, ncol = 100)
#'   colnames(test_data) <- paste0("Feature", 1:ncol(test_data))
#'   rownames(test_data) <- paste0("Sample", 1:nrow(test_data))
#'
#'   print(paste("Original R matrix dimensions:", paste(dim(test_data), collapse = "x")))
#'
#'   # Perform Randomized SVD
#'   svd_results <- FastPCA::FastPCA(test_data, k = 5)
#' }
FastPCA <- function(input_r_matrix, k, p = 10, q_iter = 2) {
  #make sure input is actualy matrix
  if (!is.matrix(input_r_matrix) || !is.numeric(input_r_matrix)) {
    stop("Input must be a numeric R matrix.")
  }
  if (!is.numeric(k) || length(k) != 1 || k <= 0 || k > min(dim(input_r_matrix))) {
    stop("k must be a single positive integer not exceeding the smaller dimension of the matrix.")
  }
  k <- as.integer(k)
  p <- as.integer(p)
  q_iter <- as.integer(q_iter)

  #make sure environment is initialized and the script is loaded.
  if (is.null(.globals$svd_python_module)) {
    if (!reticulate::py_available()) {
      stop("Python environment not initialized. Please run `FastPCA::setup_py_env()` first.")
    }
    # Check if 'torch' and 'numpy' are available in the currently active Python environment
    if (!reticulate::py_module_available("torch") || !reticulate::py_module_available("numpy")) {
      stop("PyTorch 'torch' and/or 'numpy' are not available in the current Python environment.
           Please run `FastPCA::setup_py_env()` to install them.")
    }

    script_path <- get_python_script_path()
    if (!file.exists(script_path)) {
      stop(paste("Python SVD script not found at:", script_path,
                 "\nIs the package installed correctly?"))
    }
    .globals$svd_python_module <- reticulate::py_run_file(script_path)
    message(paste("Python SVD script '", basename(script_path), "' loaded.", sep=""))
  }

  # Call the Python function.
  #reticulate automatically converts input_r_matrix (R matrix) to a NumPy array.
  py_results <- .globals$svd_python_module$randomized_svd_py(
    input_r_matrix,
    k = k,
    p = p,
    q_iter = q_iter
  )

  # py_results is a list (U_np, S_np, Vh_np) from Python.
  # reticulate automatically converts NumPy arrays back to R matrices/vectors.
  U_r <- py_results[[1]][,1:k]  # (Features x k)
  S_r <- py_results[[2]][1:k]  # (k,) vector
  Vh_r <- py_results[[3]][1:k,] # (Samples x k)

  message("Received SVD results from Python. Returning as R objects.")

  # Optionally, set row/column names if you want to preserve them
  # U matrix columns are typically principal components
  colnames(U_r) <- paste0("PC", 1:ncol(U_r))
  # U matrix rows correspond to features
  if (!is.null(colnames(input_r_matrix))) {
    rownames(U_r) <- colnames(input_r_matrix)
  }

  # Vh matrix columns are also principal components
  rownames(Vh_r) <- paste0("PC", 1:nrow(Vh_r))
  # Vh matrix rows correspond to samples
  if (!is.null(rownames(input_r_matrix))) {
    rownames(Vh_r) <- rownames(input_r_matrix)
  }

  # S is a vector, no names needed typically

  return(list(U = U_r, S = S_r, Vh = Vh_r))
}
