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
#' @param exact Boolean. Whether to compute the exact matrix or not. Only works with pytorch backend
#' @param backend Character. which backend to use, either r, rtorch, pytorch, or irlba. **Tinygrad is not implemented. Waiting on tinygrad maturation**
#' See details for informaiotn about backends.
#' @param cores Integer. number of CPU cores to use with the backend
#' @param ... other parameters to pass to irlba when `backend` is either `'r'` or `'irlba'`
#'
#' @return A list containing:
#'   \item{U}{The left singular vectors (R matrix). Dimensions: Features x k.}
#'   \item{S}{The singular values (R numeric vector). Length: k.}
#'   \item{Vh}{The transpose of the right singular vectors (R matrix). Dimensions: Samples x k.}
#'   All results are moved to CPU by the Python script and returned as R objects.
#'
#' @details
#' Depending on the backend chosen, the session may need to be reset with `rstudioapi::restartSession()`.
#' Mainly, this is due to conflicts between some underlying system level variables with 'rtorch' and 'pytorch'.
#' Once one is used in a session, the other will fail. Even testing using 'rtorch' and then starting the conda environment
#' resulted in the environment being loaded by the python libraries/modules are not available. Unless absolutely needed
#' and for testing, would stick with 'rtorch'.
#'
#' @export
#' @examples
#' \dontrun{
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
FastPCA <- function(input_r_matrix,
                    k = 100,
                    p = 10,
                    q_iter = 2,
                    exact = FALSE,
                    backend = c("r", "rtorch", "pytorch", "irlba"), #, "tinygrad"
                    device = c("CPU","GPU"), cores = 4,
                    ...) {
  dots = list(...)
  #tranformation backends
  backend = match.arg(backend)
  #devices
  device = toupper(device)
  device = match.arg(device)
  #validation
  backend_device = validate_backend(backend, device)
  k <- as.integer(k)
  p <- as.integer(p)
  q_iter <- as.integer(q_iter)
  cores = as.integer(cores)

  if(is.data.frame(input_r_matrix)) stop("Input data must be in matrix format.")

  #call python
  if(backend == "pytorch"){
    .globals = python_functions()
    if(exact){
      py_results <- .globals$torch_exact_svd$exact_svd_py(input_r_matrix, device = device, cores = cores)
      U_r <- py_results[[1]]  # (Features x k)
      S_r <- py_results[[2]]  # (k,) vector
      Vh_r <- py_results[[3]] # (Samples x k)
      message("Received SVD results from Python. Returning as R objects.")
      rm(.globals)
    } else {
      py_results <- .globals$torch_random_svd$randomized_svd_py(input_r_matrix, k = k, p = p, q_iter = q_iter, device = device, cores = cores)
      U_r <- py_results[[1]][,1:k]  # (Features x k)
      S_r <- py_results[[2]][1:k]  # (k,) vector
      Vh_r <- py_results[[3]][1:k,] # (Samples x k)
      message("Received SVD results from Python. Returning as R objects.")
      rm(.globals)
    }
  } else if(backend == "tinygrad") {
    .globals = python_functions()
    py_results <- .globals$tinygrad_random_svd$randomized_svd_py_tg(input_r_matrix, k = k, p = p, q_iter = q_iter, device = device, cores = cores)
    U_r <- py_results[[1]][,1:k]  # (Features x k)
    S_r <- py_results[[2]][1:k]  # (k,) vector
    Vh_r <- py_results[[3]][1:k,] # (Samples x k)
    message("Received SVD results from Python. Returning as R objects.")
    rm(.globals)
  } else if(backend == "rtorch"){
    results = rtorch_randomized_svd(input_r_matrix, k = k, p = p, q_iter = q_iter, device = device, cores = cores)
    U_r <- results[[1]]  # (Features x k)
    S_r <- results[[2]]  # (k,) vector
    Vh_r <- results[[3]] # (Samples x k)
  } else if(backend %in% c("r", "irlba")){
    irlba_vars = names(formals(irlba::irlba))
    explicit_vars = list(A = input_r_matrix,
                         nv = k,
                         work = k + p)
    irlba_params = dots[names(dots) %in% irlba_vars]
    irlba_params = irlba_params[!(names(irlba_params) %in% names(explicit_vars))]
    irlba_params = c(explicit_vars, irlba_params)
    results = do.call(irlba::irlba,
                      irlba_params, quote = TRUE)
    U_r = results$u
    S_r = results$d
    Vh_r = t(results$v)
  }

  # Optionally, set row/column names if you want to preserve them
  # U matrix columns are typically principal components
  colnames(U_r) <- paste0("dim", 1:ncol(U_r))
  #U_r is samples x PCs
  if (!is.null(rownames(input_r_matrix))) {
    rownames(U_r) <- rownames(input_r_matrix)
  }

  # Vh matrix columns are also principal components
  rownames(Vh_r) <- paste0("dim", 1:nrow(Vh_r))
  #Vh is features x PCs
  if (!is.null(colnames(input_r_matrix))) {
    colnames(Vh_r) <- colnames(input_r_matrix)
  }

  #clearn environment
  invisible(gc(full=TRUE))

  # S is a vector, no names needed typically
  return(list(U = U_r, S = S_r, Vh = Vh_r))
}


