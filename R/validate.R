# Shared argument handling for every FastPCA entry point ----------------------
#
# The allowed backend/device sets are kept as data so future functions can reuse
# the same choices and error messages instead of re-declaring them.

.fastpca_backends <- c("r", "rtorch", "pytorch", "irlba", "tinygrad")
.prep_matrix_backends <- c("r", "rtorch", "pytorch", "tinygrad")
# NIPALS has no sketch, so it gets its own backend vector: irlba is absent (it
# needs complete data) and "nipals" names the CRAN reference package.
.nipals_backends <- c("r", "rtorch", "pytorch", "nipals", "tinygrad")
# IncrementalPCA() streams blocks from files, so it has its own entry point and
# backend set: the base-R reference kernel plus the two Python kernels. There is
# no rtorch kernel yet (see ROADMAP.md), and "irlba"/"nipals" need the whole
# matrix in memory, which is exactly what this function exists to avoid.
.incremental_pca_backends <- c("r", "pytorch", "tinygrad")
.fastpca_devices <- c("CPU", "GPU")
# Numeric precision shared by the PyTorch/tinygrad backends. Only torch_irlba()
# exposes it so far; "double" is the historical behaviour (reticulate hands R
# doubles to Python unchanged) and "single" is the float32 escape hatch.
.fastpca_precisions <- c("double", "single")

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

# Single numeric-scalar validator (tolerance and proportion arguments).
.check_scalar_num <- function(x, name, min = -Inf, max = Inf) {
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || is.na(value) || value < min || value > max) {
    stop(sprintf("`%s` must be a single number between %s and %s.",
                 name, format(min), format(max)), call. = FALSE)
  }
  value
}

# Numeric-matrix input check shared by FastPCA(), NIPALS() and prep_matrix().
# Sparse Matrix objects are only accepted where the backend can consume them,
# and missing values only where the solver supports them (NIPALS).
.check_matrix_input <- function(x, name = "input_r_matrix", allow_sparse = FALSE,
                                allow_na = FALSE) {
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
  if (anyNA(x) && !allow_na) {
    stop("`", name, "` contains missing values; FastPCA requires complete data. ",
         "Use NIPALS() for matrices with missing values.", call. = FALSE)
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

# ncomp / tol / maxit / max_na_prop validation for NIPALS(). Returns coerced
# values so the caller does not re-coerce them. There is no sketch, so unlike
# FastPCA() the only size constraint is that ncomp cannot exceed the smallest
# dimension.
.check_nipals_args <- function(A, ncomp = 10L, tol = 1e-6,
                               maxit = 500L, max_na_prop = 1) {
  dims <- dim(A)
  if (length(dims) != 2L || anyNA(dims)) {
    stop("`input_r_matrix` must be two-dimensional.", call. = FALSE)
  }
  smallest <- min(dims)
  ncomp <- .check_scalar_int(ncomp, "ncomp", 1L)
  if (ncomp > smallest) {
    stop(sprintf("`ncomp` (%d) cannot exceed the smallest matrix dimension (%d).",
                 ncomp, smallest), call. = FALSE)
  }
  tol <- .check_scalar_num(tol, "tol", min = 0)
  if (tol <= 0) stop("`tol` must be greater than 0.", call. = FALSE)
  maxit <- .check_scalar_int(maxit, "maxit", 1L)
  max_na_prop <- .check_scalar_num(max_na_prop, "max_na_prop", min = 0, max = 1)
  list(ncomp = ncomp, tol = tol, maxit = maxit, max_na_prop = max_na_prop)
}

# Precision validation shared by the torch/tinygrad entry points.
.resolve_precision <- function(precision) {
  precision <- tolower(as.character(precision))
  match.arg(precision, .fastpca_precisions)
}

# nv / work / tol / svtol / maxit validation for torch_irlba(). irlba requires
# max(nu, nv) < min(dim(A)); work defaults to nv + 7 (as in irlba) and is capped
# at the smallest dimension, exactly as the PyTorch port clamps it.
.check_irlba_args <- function(A, nv = 5L, work = NULL, tol = 1e-5,
                              svtol = NULL, maxit = 1000L) {
  dims <- dim(A)
  if (length(dims) != 2L || anyNA(dims)) {
    stop("`input_r_matrix` must be two-dimensional.", call. = FALSE)
  }
  smallest <- min(dims)
  nv <- .check_scalar_int(nv, "nv", 1L)
  if (nv > smallest - 1L) {
    stop(sprintf(paste0("`nv` (%d) must be strictly less than the smallest ",
                        "matrix dimension (%d)."), nv, smallest), call. = FALSE)
  }
  if (is.null(work)) work <- min(nv + 7L, smallest)
  work <- .check_scalar_int(work, "work", 2L)
  if (work < nv + 1L) work <- min(nv + 1L, smallest)
  if (work > smallest) work <- smallest
  tol <- .check_scalar_num(tol, "tol", min = 0)
  if (tol <= 0) stop("`tol` must be greater than 0.", call. = FALSE)
  if (is.null(svtol)) svtol <- tol
  svtol <- .check_scalar_num(svtol, "svtol", min = 0)
  if (svtol <= 0) stop("`svtol` must be greater than 0.", call. = FALSE)
  maxit <- .check_scalar_int(maxit, "maxit", 1L)
  list(nv = nv, work = as.integer(work), tol = tol, svtol = svtol,
       maxit = maxit)
}

# files / ncomp / chunk_rows and the flag arguments for IncrementalPCA(). The
# paths are checked up front so a typo fails before any backend or Python setup,
# and the coerced values are returned for the caller to use.
.check_incremental_args <- function(files, ncomp = 10L, oversample = 0L,
                                    chunk_rows = NULL,
                                    center = TRUE, scale = FALSE, log2 = FALSE,
                                    store_scores = FALSE, store_variance = TRUE,
                                    header = TRUE, row.names = NULL, sep = ",",
                                    reader = NULL) {
  if (!is.character(files) || length(files) < 1L || anyNA(files)) {
    stop("`files` must be a character vector with at least one chunk file path.",
         call. = FALSE)
  }
  files <- as.character(files)
  if (any(!nzchar(files))) {
    stop("`files` contains an empty path.", call. = FALSE)
  }
  missing_files <- files[!file.exists(files)]
  if (length(missing_files) > 0L) {
    stop("Chunk file(s) not found: ",
         paste(utils::head(missing_files, 3L), collapse = ", "),
         if (length(missing_files) > 3L)
           paste0(" and ", length(missing_files) - 3L, " more") else "",
         ".", call. = FALSE)
  }

  ncomp <- .check_scalar_int(ncomp, "ncomp", 1L)
  oversample <- .check_scalar_int(oversample, "oversample", 0L)
  chunk_rows <- if (is.null(chunk_rows)) NULL else
    .check_scalar_int(chunk_rows, "chunk_rows", 1L)

  flags <- list(center = center, scale = scale, log2 = log2,
                store_scores = store_scores, store_variance = store_variance,
                header = header)
  for (nm in names(flags)) {
    value <- flags[[nm]]
    if (!is.logical(value) || length(value) != 1L || is.na(value)) {
      stop("`", nm, "` must be TRUE or FALSE.", call. = FALSE)
    }
  }

  if (!is.null(row.names) && !(length(row.names) == 1L &&
                               (is.character(row.names) || is.numeric(row.names)))) {
    stop("`row.names` must be NULL, a column name or a single column index.",
         call. = FALSE)
  }
  if (!is.null(reader) && !is.function(reader)) {
    stop("`reader` must be NULL or a function taking one file path and ",
         "returning that file as a numeric matrix or data frame (observation ",
         "identifiers in the row names).", call. = FALSE)
  }
  if (!is.character(sep) || length(sep) != 1L || is.na(sep)) {
    stop("`sep` must be a single character string.", call. = FALSE)
  }

  list(files = files, ncomp = ncomp, oversample = as.integer(oversample),
       chunk_rows = chunk_rows)
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
