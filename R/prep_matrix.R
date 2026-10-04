#' Perform transpose and scaling of matrix
#'
#' This function transforms an input matrix: optionally log2-transforming,
#' transposing and/or centre-scaling it. The base R backend needs no Python; the
#' PyTorch, rtorch and tinygrad backends are optional.
#'
#' @param mat A dense numeric R matrix (rows as features and columns as samples
#'   by default). Sparse \code{Matrix} input is not supported here; convert with
#'   \code{as.matrix()} first.
#' @param log2 Logical. Whether to apply a log2 transformation. Default
#'   \code{TRUE}. The input must be positive or the transform yields -Inf/NaN.
#' @param transpose Logical. Whether the matrix needs to be transposed. If
#'   starting with samples as columns, set to \code{TRUE}. Default is
#'   \code{FALSE}.
#' @param scale Logical. Whether to centre and scale the matrix to unit variance.
#'   Default is \code{TRUE}.
#' @param backend Character. The backend to use for the transformations: "r"
#'   (default, no Python), "rtorch", "pytorch" or "tinygrad". The "tinygrad"
#'   backend runs on Apple Metal (float32) and matches the others to ~1e-6. See
#'   details for information about the pytorch/conda environments.
#' @param cores Numeric. The number of cores to use. Clamped to 1 for this call
#'   when an OpenMP runtime conflict is detected in the session.
#' @param device Character, either "CPU" (default) or "GPU". Only the tinygrad
#'   backend acts on this directly; Python backends also fall back to CPU when a
#'   GPU is unavailable.
#'
#' @return A matrix that has been transposed and scaled if requested, with
#'   row/column names carried over from the input.
#'
#' @details
#' Depending on the backend chosen, the session may need to be reset with \code{rstudioapi::restartSession()}.
#' Mainly, this is due to conflicts between some underlying system level variables with 'rtorch' and 'pytorch'.
#' Once one is used in a session, the other will fail. Even testing using 'rtorch' and then starting the conda environment
#' resulted in the environment being loaded by the python libraries/modules are not available. Unless absolutely needed
#' and for testing, would stick with 'rtorch'
#'
#' @section Downstream spectra:
#' \code{prep_matrix()} itself is a deterministic transform, but centre-scaling data
#' with little structure can leave the result with nearly equal top singular
#' values. The randomized SVD backends are ill-conditioned in that case, so use
#' \code{backend = "r"} or \code{"irlba"} in [FastPCA()] when the top components must be
#' exact.
#'
#' @export
#' @examples
#' set.seed(123)
#' test_data <- matrix(runif(200, min = 1, max = 50), nrow = 10, ncol = 20)
#' rownames(test_data) <- paste0("Feature", seq_len(nrow(test_data)))
#' colnames(test_data) <- paste0("Sample", seq_len(ncol(test_data)))
#'
#' prepped_matrix <- prep_matrix(test_data, log2 = TRUE,
#'                               transpose = FALSE, scale = TRUE)
#' dim(prepped_matrix)
prep_matrix <- function(mat,
                        log2 = TRUE,
                        transpose = FALSE,
                        scale = TRUE,
                        backend = c("r", "rtorch", "pytorch", "tinygrad"),
                        cores = 2,
                        device = c("CPU", "GPU")) {

  # resolve backend/device once and use the returned values (R/validate.R)
  resolved <- resolve_backend_device(backend, device,
                                     allowed = .prep_matrix_backends)
  backend <- resolved$backend
  device <- resolved$device

  # prep_matrix always works on a dense base matrix (R/validate.R)
  .check_matrix_input(mat, name = "mat", allow_sparse = FALSE)

  # clamp to a single thread when an OpenMP runtime conflict is present (R/openmp.R)
  cores <- .resolve_cores(cores, openmp = .backend_uses_openmp(backend))

  if (isTRUE(log2) && any(mat <= 0)) {
    warning("'log2 = TRUE' with non-positive values produces -Inf/NaN; ",
            "shift the input before transforming.", call. = FALSE)
  }

  # keep only the names so dimnames can be restored without copying the matrix
  mat_names <- dimnames(mat)

  if(backend == "pytorch"){
    .globals = python_functions(scripts = "torch_tranformation")
    # Call the Python function.
    #reticulate automatically converts mat (R matrix) to a numpy array.
    out <- .globals$torch_tranformation$transform_py(
      mat,
      log2 = ifelse(log2 == FALSE, 0, 1),
      transpose = ifelse(transpose == FALSE, 0, 1),
      scale = ifelse(scale == FALSE, 0, 1),
      cores = cores
    )
    out <- .prep_dimnames(out, mat_names, transpose)
    rm(.globals)
  } else if(backend == "tinygrad") {
    .globals = python_functions("tinygrad")
    out <- .globals$tinygrad_tranformation$transform_py_tg(
      mat,
      log2 = ifelse(log2 == FALSE, 0, 1),
      transpose = ifelse(transpose == FALSE, 0, 1),
      scale = ifelse(scale == FALSE, 0, 1),
      cores = cores,
      device = device
    )
    out <- .prep_dimnames(out, mat_names, transpose)
    rm(.globals)
  } else if(backend == "rtorch"){
    torch::torch_set_num_threads(cores)
    mat_t = torch::torch_tensor(mat, dtype = torch::torch_double())
    if(log2) mat_t = torch::torch_log2(mat_t)
    if(transpose) mat_t = torch::torch_t(mat_t)
    if(scale){
      mean_vec = torch::torch_mean(mat_t, dim = 1)
      std_vec = torch::torch_std(mat_t, dim = 1)
      out = as.matrix((mat_t - mean_vec)/std_vec)
    } else {
      # previously out was only assigned when scale = TRUE
      out = as.matrix(mat_t)
    }
    dimnames(out) <- NULL
    out <- .prep_dimnames(out, mat_names, transpose)
  } else if(backend == "r"){
    if(log2) mat = log2(mat)
    if(transpose) mat = t(mat)
    if(scale) mat = scale(mat)
    out = mat
  }

  #clean environment
  invisible(gc(full=TRUE))
  return(out)
}

# Restore row/column names after a transform, accounting for the transpose.
.prep_dimnames <- function(out, mat_names, transpose) {
  if (isTRUE(transpose)) {
    rownames(out) <- mat_names[[2]]
    colnames(out) <- mat_names[[1]]
  } else {
    rownames(out) <- mat_names[[1]]
    colnames(out) <- mat_names[[2]]
  }
  out
}
