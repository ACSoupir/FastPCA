#' Perform transpose and scaling of matrix
#'
#' This function uses PyTorch implementation of Randomized SVD to compute
#' the singular value decomposition.
#'
#' @param mat A numeric R matrix.
#' @param transpose Boolean. Whether the matrix needs to be transposed. If starting with samples as columns, set to `TRUE`. Default is `FALSE`
#' @param scale Boolean. Whether to center and scale the matrix. Default to `TRUE`
#'
#' @return A matrix that has been rotated (transposed) and scaled if needed
#' @export
#' @examples
#' \dontrun{
#'   #need to have environment first
#'   reticulate::use_condaenv("FastPCA")
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
#'   prepped_matrix <- FastPCA::prep_matrix(test_data, log2 = TRUE,
#'                                   transpose = FALSE, scale = TRUE)
#' }
prep_matrix <- function(mat, log2 = TRUE, transpose = FALSE, scale = TRUE) {
  #make sure input is actualy matrix
  if (!is.matrix(mat) || !is.numeric(mat)) {
    stop("Input must be a numeric R matrix.")
  }

  #make sure environment is initialized and the script is loaded.
  if (is.null(.globals$transform_python_module)) {
    if (!reticulate::py_available()) {
      stop("Python environment not initialized. Please run `FastPCA::setup_py_env()` first.")
    }
    # Check if 'torch' and 'numpy' are available in the currently active Python environment
    if (!reticulate::py_module_available("torch") || !reticulate::py_module_available("numpy")) {
      stop("PyTorch 'torch' and/or 'numpy' are not available in the current Python environment.
           Please run `FastPCA::setup_py_env()` to install them.")
    }

    script_path <- get_transform_script_path()
    if (!file.exists(script_path)) {
      stop(paste("Python tranforamtion script not found at:", script_path,
                 "\nIs the package installed correctly?"))
    }
    .globals$transform_python_module <- reticulate::py_run_file(script_path)
    message(paste("Python transformation script '", basename(script_path), "' loaded.", sep=""))
  }

  # Call the Python function.
  #reticulate automatically converts mat (R matrix) to a NumPy array.
  py_results <- .globals$transform_python_module$transform_py(
    mat,
    log2 = ifelse(log2 == FALSE, 0, 1),
    transpose = ifelse(transpose == FALSE, 0, 1),
    scale = ifelse(scale == FALSE, 0, 1)
  )

  # py_results is a list (U_np, S_np, Vh_np) from Python.
  # reticulate automatically converts NumPy arrays back to R matrices/vectors.
  return(py_results)


}
