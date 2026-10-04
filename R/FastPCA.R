#' Perform SVD
#'
#' This function performs either a randomized or an exact singular value
#' decomposition (SVD) of large matrices.
#'
#' @param input_r_matrix A numeric matrix, with rows as observations (e.g.,
#'   samples or cells) and columns as features (e.g., genes). Sparse
#'   \code{Matrix} objects are accepted by the base R and \code{"irlba"}
#'   backends; all other backends require a dense numeric matrix (convert with
#'   \code{as.matrix()}).
#' @param k Integer. The number of singular values/vectors to compute.
#' @param p Integer. Oversampling parameter (default: 10). \code{k + p} must
#'   not exceed the smallest matrix dimension. The base R and \code{"irlba"}
#'   backends use at least 7 extra Lanczos vectors regardless of \code{p}, so a
#'   small \code{p} still converges.
#' @param q_iter Integer >= 1. Number of power iterations (default: 2).
#'   Randomized SVD requires at least one power iteration.
#' @param exact Logical. Whether to compute the full (exact) SVD instead of a
#'   randomized one. Only implemented for \code{backend = "pytorch"}; other
#'   backends compute a randomized decomposition and warn.
#' @param backend Character. Which backend to use: \code{"r"}, \code{"rtorch"},
#'   \code{"pytorch"}, \code{"irlba"} or \code{"tinygrad"}. The
#'   \code{"r"} and \code{"irlba"} backends need no Python. The
#'   \code{"tinygrad"} backend runs the randomized SVD on Apple Metal; note
#'   that on arm64 Macs a \code{device = "CPU"} request still runs on the GPU
#'   because tinygrad's CPU codegen is unavailable there. See Details for
#'   information about backends.
#' @param device Character, either \code{"CPU"} (default) or \code{"GPU"}.
#'   Backends fall back to the CPU with a message when the requested device is
#'   unavailable.
#' @param cores Integer. Number of CPU cores to use with the backend. Clamped
#'   to 1 for this call when an OpenMP runtime conflict is detected in the
#'   session (for example after loading \pkg{data.table} or Seurat).
#' @param store_variance Logical. When \code{TRUE} (default) the total variance
#'   of the input is stored on the result so that [pve()] can report percent
#'   variation explained. Set to \code{FALSE} for very large inputs if the
#'   extra pass is not wanted.
#' @param ... Other parameters passed to [irlba::irlba()] when \code{backend}
#'   is either \code{"r"} or \code{"irlba"}.
#'
#' @return An object of class \code{"fastpca"}, a list with elements:
#'   \item{U}{The left singular vectors (rows of the input x \code{k}).}
#'   \item{S}{The singular values (numeric vector of length \code{k}).}
#'   \item{Vh}{The transpose of the right singular vectors (\code{k} x columns
#'     of the input).}
#'   \code{print()}, \code{summary()} and [pve()] methods are available, as is
#'   [get_pc_scores()].
#'
#' @details
#' Depending on the backend chosen, the session may need to be reset with \code{rstudioapi::restartSession()}.
#' Mainly, this is due to conflicts between some underlying system level variables with 'rtorch' and 'pytorch'.
#' Once one is used in a session, the other will fail. Even testing using 'rtorch' and then starting the conda environment
#' resulted in the environment being loaded by the python libraries/modules are not available. Unless absolutely needed
#' and for testing, would stick with 'rtorch'.
#'
#' @section Near-degenerate (flat) spectra:
#' The randomized backends (\code{"pytorch"} and \code{"tinygrad"}) estimate the leading
#' subspace from a random sketch. When the top singular values are nearly equal --
#' as can happen after [prep_matrix()] centre-scales data with little structure --
#' that subspace is ill-conditioned, and *any* randomized method (including the
#' PyTorch baseline) can be well away from exact. On a flat spectrum we measured
#' roughly 7% error in the top singular values for both. Use \code{backend = "r"} or
#' \code{"irlba"}, which are near-exact, when the top components must be precise.
#'
#' @export
#' @examples
#' set.seed(123)
#' test_data <- matrix(runif(30 * 100, min = 1, max = 50), nrow = 30, ncol = 100)
#' rownames(test_data) <- paste0("Sample", seq_len(nrow(test_data)))
#' colnames(test_data) <- paste0("Feature", seq_len(ncol(test_data)))
#'
#' # base R backend: no Python required
#' svd_results <- FastPCA(test_data, k = 5, p = 2, backend = "r")
#' svd_results
#' summary(svd_results)
#'
#' \dontrun{
#'   # optional Python backend (requires setup_py_env()/start_FastPCA_env())
#'   svd_torch <- FastPCA(test_data, k = 5, p = 2, backend = "pytorch")
#' }
FastPCA <- function(input_r_matrix,
                    k = 100,
                    p = 10,
                    q_iter = 2,
                    exact = FALSE,
                    backend = c("r", "rtorch", "pytorch", "irlba", "tinygrad"),
                    device = c("CPU","GPU"), cores = 4,
                    store_variance = TRUE,
                    ...) {
  dots <- list(...)

  # resolve the backend/device pair once and use the returned values: a request
  # may be downgraded (e.g. rtorch -> base R without torch), and dispatching on
  # the original values would call a backend that cannot run (R/validate.R)
  resolved <- resolve_backend_device(backend, device)
  backend <- resolved$backend
  device <- resolved$device

  # shared input and argument validation (R/validate.R)
  .check_matrix_input(input_r_matrix,
                      allow_sparse = backend %in% c("r", "irlba"))
  svd_args <- .check_svd_args(input_r_matrix, k = k, p = p, q_iter = q_iter)
  k <- svd_args$k
  p <- svd_args$p
  q_iter <- svd_args$q_iter

  if (isTRUE(exact) && !identical(backend, "pytorch")) {
    warning("'exact = TRUE' is only implemented for backend = 'pytorch'; ",
            "computing a randomized SVD with the '", backend, "' backend.",
            call. = FALSE)
  }

  # clamp to a single thread when an OpenMP runtime conflict is present (R/openmp.R)
  cores <- .resolve_cores(cores, openmp = .backend_uses_openmp(backend))

  # total variance (sum of squares) for pve(); one cheap pass vs the decomposition
  total_variance <- if (isTRUE(store_variance)) {
    .fastpca_total_variance(input_r_matrix)
  } else NA_real_

  #call python
  if(backend == "pytorch"){
    .globals = python_functions(
      scripts = if (exact) "torch_exact_svd" else "torch_random_svd")
    if(exact){
      py_results <- .globals$torch_exact_svd$exact_svd_py(input_r_matrix, device = device, cores = cores)
      U_r <- py_results[[1]]  # (rows of input x k)
      S_r <- py_results[[2]]  # singular values
      Vh_r <- py_results[[3]] # (k x columns of input)
      message("Received SVD results from Python. Returning as R objects.")
      rm(.globals)
    } else {
      py_results <- .globals$torch_random_svd$randomized_svd_py(input_r_matrix, k = k, p = p, q_iter = q_iter, device = device, cores = cores)
      U_r <- py_results[[1]][,1:k]  # (rows of input x k)
      S_r <- py_results[[2]][1:k]  # (k,) vector
      Vh_r <- py_results[[3]][1:k,] # (k x columns of input)
      message("Received SVD results from Python. Returning as R objects.")
      rm(.globals)
    }
  } else if(backend == "tinygrad") {
    .globals = python_functions("tinygrad")
    py_results <- .globals$tinygrad_random_svd$randomized_svd_py_tg(input_r_matrix, k = k, p = p, q_iter = q_iter, device = device, cores = cores)
    U_r <- py_results[[1]][,1:k]
    S_r <- py_results[[2]][1:k]
    Vh_r <- py_results[[3]][1:k,]
    message("Received SVD results from Python. Returning as R objects.")
    rm(.globals)
  } else if(backend == "rtorch"){
    results = rtorch_randomized_svd(input_r_matrix, k = k, p = p, q_iter = q_iter, device = device, cores = cores)
    U_r <- results[[1]]
    S_r <- results[[2]]
    Vh_r <- results[[3]]
  } else if(backend %in% c("r", "irlba")){
    # irlba is a Lanczos solver, so 'work' needs headroom beyond nv or it may
    # not converge. Give it at least 7 extra vectors while still honoring a
    # larger user-supplied oversampling 'p' (and never exceed the matrix size).
    smallest = min(dim(input_r_matrix))
    work = min(k + max(p, 7L), smallest)
    irlba_vars = names(formals(irlba::irlba))
    explicit_vars = list(A = input_r_matrix,
                         nv = k,
                         work = work)
    irlba_params = dots[names(dots) %in% irlba_vars]
    irlba_params = irlba_params[!(names(irlba_params) %in% names(explicit_vars))]
    irlba_params = c(explicit_vars, irlba_params)
    results = do.call(irlba::irlba,
                      irlba_params, quote = TRUE)
    U_r = results$u
    S_r = results$d
    Vh_r = t(results$v)
  }

  # U rows are observations; columns are the components
  colnames(U_r) <- paste0("dim", seq_len(ncol(U_r)))
  if (!is.null(rownames(input_r_matrix))) {
    rownames(U_r) <- rownames(input_r_matrix)
  }

  # Vh rows are the components; columns follow the input features
  rownames(Vh_r) <- paste0("dim", seq_len(nrow(Vh_r)))
  if (!is.null(colnames(input_r_matrix))) {
    colnames(Vh_r) <- colnames(input_r_matrix)
  }

  #clean environment
  invisible(gc(full=TRUE))

  new_fastpca(
    U = U_r, S = as.numeric(S_r), Vh = Vh_r,
    k = length(S_r),
    p = p, q_iter = q_iter, exact = isTRUE(exact),
    backend = backend, device = device,
    n_obs = nrow(input_r_matrix), n_features = ncol(input_r_matrix),
    total_variance = total_variance
  )
}
