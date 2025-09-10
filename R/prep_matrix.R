#' Perform transpose and scaling of matrix
#'
#' This function uses python to perform transformations of the input matrix
#'
#' @param mat A numeric R matrix.
#' @param transpose Boolean. Whether the matrix needs to be transposed. If starting with samples as columns, set to `TRUE`. Default is `FALSE`
#' @param scale Boolean. Whether to center and scale the matrix. Default to `TRUE`
#' @param backend Character. The backend which to use for performing transformations. Default is "pytorch"
#' @param cores Numeric. The number of cores to use.
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
prep_matrix <- function(mat,
                        log2 = TRUE,
                        transpose = FALSE,
                        scale = TRUE,
                        backend = c("pytorch"), #, "tinygrad"
                        cores = 2,
                        device = c("CPU", "GPU")) {
  backend = validate_backend(backend)
  device = match.arg(device)
  check_backend(backend)
  #convert cores to integer
  cores = as.integer(cores)

  #make sure input is actualy matrix
  if (!is.matrix(mat) || !is.numeric(mat)) {
    stop("Input must be a numeric R matrix.")
  }

  #make sure environment is initialized and the script is loaded.
  if (!reticulate::py_available(initialize = FALSE)) {
    stop("Python environment not initialized. Please run `FastPCA::start_FastPCA_env()` first.")
  }
  #create new environment with the functions
  .globals = python_functions()

  if(backend=="tinygrad"){
    avail_devices = .globals$tinygrad_devices$list_available_devices()
    if(device == "GPU"){
      device = match.arg(avail_devices, c("HIP", "METAL", "CUDA"), several.ok = TRUE)
      if(length(device) == 0){
        stop("GPU device not available")
      }
    }
  }


  if(backend == "pytorch"){
    # Call the Python function.
    #reticulate automatically converts mat (R matrix) to a NumPy array.
    py_results <- .globals$torch_tranformation$transform_py(
      mat,
      log2 = ifelse(log2 == FALSE, 0, 1),
      transpose = ifelse(transpose == FALSE, 0, 1),
      scale = ifelse(scale == FALSE, 0, 1),
      cores = cores
    )
    rownames(py_results) = colnames(mat)
    colnames(py_results) = row.names(mat)
  } else {
    py_results <- .globals$tinygrad_tranformation$transform_py_tg(
      mat,
      log2 = ifelse(log2 == FALSE, 0, 1),
      transpose = ifelse(transpose == FALSE, 0, 1),
      scale = ifelse(scale == FALSE, 0, 1),
      cores = cores,
      device = device
    )
    rownames(py_results) = colnames(mat)
    colnames(py_results) = row.names(mat)
  }

  #clearn environment
  rm(.globals)
  invisible(gc(full=TRUE))
  # py_results is a list (U_np, S_np, Vh_np) from Python.
  # reticulate automatically converts NumPy arrays back to R matrices/vectors.
  return(py_results)


}
