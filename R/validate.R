# Shared argument handling for every FastPCA entry point ----------------------
#
# The allowed backend/device sets are kept as data so future functions can reuse
# the same choices and error messages instead of re-declaring them.

.fastpca_backends <- c("r", "rtorch", "pytorch", "irlba", "tinygrad")
.prep_matrix_backends <- c("r", "rtorch", "pytorch", "tinygrad")
.fastpca_devices <- c("CPU", "GPU")

# Resolve a backend/device pair exactly once per call.
#
# validate_backend() owns the platform- and dependency-specific rules (torch
# availability, CUDA presence, tinygrad on Apple Metal). Callers *must* use the
# returned values: a request can be silently downgraded (for example rtorch to
# base R when torch is missing), and dispatching on the original values would
# then call into a backend that cannot run.
resolve_backend_device <- function(backend, device,
                                   allowed = .fastpca_backends) {
  backend <- match.arg(backend, allowed)
  device <- toupper(as.character(device))
  device <- match.arg(device, .fastpca_devices)
  validate_backend(backend = backend, device = device)
}

# Single positive-integer validator shared by the exported entry points.
.check_scalar_int <- function(x, name, min = 1L) {
  value <- suppressWarnings(as.integer(x))
  if (length(value) != 1L || is.na(value) || value < min) {
    stop(sprintf("`%s` must be a single integer greater than or equal to %d.",
                 name, min), call. = FALSE)
  }
  value
}

# Numeric-matrix input check shared by FastPCA() and prep_matrix(). Sparse Matrix
# objects are only accepted where the backend can consume them.
.check_matrix_input <- function(x, name = "input_r_matrix", allow_sparse = FALSE) {
  sparse <- inherits(x, "Matrix")
  if (sparse && !allow_sparse) {
    stop("`", name, "` must be a dense numeric matrix for this backend; ",
         "convert sparse input with `as.matrix()`.", call. = FALSE)
  }
  if (!sparse && !is.matrix(x)) {
    stop("`", name, "` must be a numeric matrix",
         if (is.data.frame(x)) "; convert the data frame with `as.matrix()`." else ".",
         call. = FALSE)
  }
  if (!sparse && !is.numeric(x)) {
    stop("`", name, "` must be numeric.", call. = FALSE)
  }
  if (anyNA(x)) {
    stop("`", name, "` contains missing values; FastPCA requires complete data.",
         call. = FALSE)
  }
  invisible(TRUE)
}

# k / p / q_iter validation plus the sketch-size constraint that every backend
# shares. Returns the coerced integers so callers do not re-coerce them.
.check_svd_args <- function(A, k = 100L, p = 10L, q_iter = 2L) {
  dims <- dim(A)
  if (length(dims) != 2L || anyNA(dims)) {
    stop("`input_r_matrix` must be two-dimensional.", call. = FALSE)
  }
  smallest <- min(dims)
  k <- .check_scalar_int(k, "k", 1L)
  p <- .check_scalar_int(p, "p", 0L)
  q_iter <- .check_scalar_int(q_iter, "q_iter", 1L)
  if (k > smallest) {
    stop(sprintf("`k` (%d) cannot exceed the smallest matrix dimension (%d).",
                 k, smallest), call. = FALSE)
  }
  if (k + p > smallest) {
    stop(sprintf(paste0("`k + p` (%d) cannot exceed the smallest matrix ",
                        "dimension (%d); reduce `p` or `k`."),
                 k + p, smallest), call. = FALSE)
  }
  list(k = k, p = p, q_iter = q_iter)
}

# Total variance (sum of squared entries) used for percent-variation-explained.
# Works without densifying sparse input and avoids an R-level copy for matrices
# by using the Frobenius norm.
.fastpca_total_variance <- function(A) {
  if (inherits(A, "sparseMatrix")) return(sum(as.numeric(A@x)^2))
  if (is.matrix(A)) {
    value <- tryCatch(norm(A, type = "F")^2, error = function(e) NA_real_)
    if (length(value) == 1L && is.finite(value)) return(as.numeric(value))
  }
  value <- tryCatch(sum(as.vector(A)^2), error = function(e) NA_real_)
  if (length(value) == 1L && is.finite(value)) as.numeric(value) else NA_real_
}
