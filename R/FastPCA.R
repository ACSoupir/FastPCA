#' Perform SVD
#'
#' This function will perform either Randomized SVD or exact SVD to compute
#' the singullar value decomposition of large matrices.
#'
#' @param input_r_matrix A numeric R matrix. It's assumed that rows are observations
#'   (e.g., samples) and columns are features (e.g., genes). The function will
#'   transpose this for the PyTorch or Tinygrad SVD based on the original Python script's logic.
#' @param k Integer. The number of singular values/vectors to compute.
#' @param p Integer. Oversampling parameter (default: 10).
#' @param q_iter Integer. Number of power iterations (default: 2).
#' @param exact Boolean. Whether to compute the exact matrix or not.
#' @param backend Character. which backend to use, either torch or tinygrad. **Only pytorch is currently implemented. Waiting on Tidygrad maturation**
#' @param cores Integer. number of CPU cores to use with the backend
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
FastPCA <- function(input_r_matrix, k = 100, p = 10, q_iter = 2, exact = FALSE, backend = c("tinygrad", "pytorch"), device = c("cpu","gpu"), cores = 4) {
  if(backend == "tinygrad") message("Falling back to pytorch - current implementations of SVD in tinygrad are slow and memory hungry.")
  backend = "pytorch"#match.arg(backend)
  device = match.arg(device)
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
  cores = as.integer(cores)

  check_backend(backend)
  #create new environment with the functions
  .globals = python_functions()

  #call python
  if(backend == "pytorch"){
    if(exact){
      py_results <- .globals$torch_exact_svd$exact_svd_py(input_r_matrix, device = device, cores = cores)
      U_r <- py_results[[1]]  # (Features x k)
      S_r <- py_results[[2]]  # (k,) vector
      Vh_r <- py_results[[3]] # (Samples x k)
    } else {
      py_results <- .globals$torch_random_svd$randomized_svd_py(input_r_matrix, k = k, p = p, q_iter = q_iter, device = device, cores = cores)
      U_r <- py_results[[1]][,1:k]  # (Features x k)
      S_r <- py_results[[2]][1:k]  # (k,) vector
      Vh_r <- py_results[[3]][1:k,] # (Samples x k)
    }
  } else {
    py_results <- .globals$tinygrad_random_svd$randomized_svd_py_tg(input_r_matrix, k = k, p = p, q_iter = q_iter, device = device, cores = cores)
    U_r <- py_results[[1]][,1:k]  # (Features x k)
    S_r <- py_results[[2]][1:k]  # (k,) vector
    Vh_r <- py_results[[3]][1:k,] # (Samples x k)
  }

  message("Received SVD results from Python. Returning as R objects.")

  # Optionally, set row/column names if you want to preserve them
  # U matrix columns are typically principal components
  colnames(U_r) <- paste0("PC", 1:ncol(U_r))
  #U_r is samples x PCs
  if (!is.null(rownames(input_r_matrix))) {
    rownames(U_r) <- rownames(input_r_matrix)
  }

  # Vh matrix columns are also principal components
  rownames(Vh_r) <- paste0("PC", 1:nrow(Vh_r))
  #Vh is features x PCs
  if (!is.null(colnames(input_r_matrix))) {
    colnames(Vh_r) <- colnames(input_r_matrix)
  }

  #clearn environment
  rm(.globals)
  invisible(gc(full=TRUE))

  # S is a vector, no names needed typically
  return(list(U = U_r, S = S_r, Vh = Vh_r))
}


