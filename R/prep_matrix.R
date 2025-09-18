#' Perform transpose and scaling of matrix
#'
#' This function uses python to perform transformations of the input matrix
#'
#' @param mat A numeric R matrix.
#' @param transpose Boolean. Whether the matrix needs to be transposed. If starting with samples as columns, set to `TRUE`. Default is `FALSE`
#' @param scale Boolean. Whether to center and scale the matrix. Default to `TRUE`
#' @param backend Character. The backend which to use for performing transformations. Default is "rtorch". Options are "r", "rtorch", or "pytorch". See details for information about the pytorch/conda environments
#' @param cores Numeric. The number of cores to use.
#'
#' @return A matrix that has been rotated (transposed) and scaled if needed
#'
#' @details
#' Depending on the backend chosen, the session may need to be reset with `rstudioapi::restartSession()`.
#' Mainly, this is due to conflicts between some underlying system level variables with 'rtorch' and 'pytorch'.
#' Once one is used in a session, the other will fail. Even testing using 'rtorch' and then starting the conda environment
#' resulted in the environment being loaded by the python libraries/modules are not available. Unless absolutely needed
#' and for testing, would stick with 'rtorch'
#'
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
                        backend = c("r", "rtorch", "pytorch"), #, "tinygrad"
                        cores = 2,
                        device = c("CPU", "GPU")) {

  #tranformation backends
  backend = match.arg(backend)
  #devices
  device = match.arg(device)
  #validation
  backend_device = validate_backend(backend, device)

  #convert cores to integer
  cores = as.integer(cores)


  if(backend == "pytorch"){
    .globals = python_functions()
    # Call the Python function.
    #reticulate automatically converts mat (R matrix) to a numpy array.
    out <- .globals$torch_tranformation$transform_py(
      mat,
      log2 = ifelse(log2 == FALSE, 0, 1),
      transpose = ifelse(transpose == FALSE, 0, 1),
      scale = ifelse(scale == FALSE, 0, 1),
      cores = cores
    )
    rownames(out) = colnames(mat)
    colnames(out) = row.names(mat)#clearn environment
    rm(.globals)
  } else if(backend == "tinygrad") {
    out <- .globals$tinygrad_tranformation$transform_py_tg(
      mat,
      log2 = ifelse(log2 == FALSE, 0, 1),
      transpose = ifelse(transpose == FALSE, 0, 1),
      scale = ifelse(scale == FALSE, 0, 1),
      cores = cores,
      device = device
    )
    rownames(out) = colnames(mat)
    colnames(out) = row.names(mat)
    #clearn environment
    rm(.globals)
  } else if(backend == "rtorch"){
    torch::torch_set_num_threads(cores)
    mat = torch::torch_tensor(mat, dtype = torch::torch_double())
    if(log2) mat = torch::torch_log2(mat)
    if(transpose) mat = torch::torch_t(mat)
    if(scale){
      mean_vec = torch::torch_mean(mat, dim = 1)
      std_vec = torch::torch_std(mat, dim = 1)
      out = as.matrix((mat - mean_vec)/std_vec)
    }
  } else if(backend == "r"){
    if(log2) mat = log2(mat)
    if(transpose) mat = t(mat)
    if(scale) mat = scale(mat)
    out = mat
    rm(mat)
  }

  #clearn environment
  invisible(gc(full=TRUE))
  return(out)


}
