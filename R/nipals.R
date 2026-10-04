# NIPALS: principal component analysis that tolerates missing values -----------
#
# Shared pre-processing (centring/scaling, degenerate columns) happens once in R
# so every backend iterates on the same working matrix; only the inner NIPALS
# loop differs. The result is a "fastpca" object with method = "nipals".

#' PCA by NIPALS, including for matrices with missing values
#'
#' \code{NIPALS()} extracts principal components one at a time by iterative
#' least squares, so unlike [FastPCA()] it can be run directly on a matrix that
#' contains \code{NA} (missing quantifications are common in proteomics and
#' metabolomics). The result is a \code{"fastpca"} object, so [print()],
#' \code{summary()}, [pve()], [get_pc_scores()] and \code{plot()} all work as
#' they do for an SVD result.
#'
#' @param input_r_matrix A dense numeric matrix, with rows as observations
#'   (samples or cells) and columns as features. Missing values are allowed.
#' @param ncomp Integer. Number of principal components to extract. Plays the
#'   role that \code{k} plays in [FastPCA()]; NIPALS has no random sketch, so
#'   there is no \code{p} or \code{q_iter}.
#' @param tol Numeric > 0. Convergence tolerance, compared against the squared
#'   change in the normalised score vector between iterations (default 1e-6).
#' @param maxit Integer >= 1. Maximum number of NIPALS iterations per component
#'   (default 500). Components that hit it are reported as not converged.
#' @param center Logical. Subtract the observed-only column means before
#'   decomposing (default \code{TRUE}).
#' @param scale Logical. Divide by the observed-only column standard deviations
#'   (unbiased, \code{n - 1}) after centring. Default \code{FALSE}; unlike
#'   [FastPCA()], NIPALS does not centre-scale by default.
#' @param gramschmidt Logical. Re-orthogonalise each loading against the
#'   previously extracted ones (default \code{TRUE}, as in the reference
#'   implementation). This costs one projection per component.
#' @param backend Character. One of \code{"r"} (base R, no Python),
#'   \code{"rtorch"} (\pkg{torch}), \code{"pytorch"} (PyTorch through
#'   \pkg{reticulate}), \code{"nipals"} (the CRAN \pkg{nipals} package) or
#'   \code{"tinygrad"}. The default is the base R implementation.
#'
#'   Note that \code{backend = "nipals"} names a *package*: it runs
#'   \code{nipals::nipals()} as the reference implementation. It is accurate but
#'   allocates an \code{O(max(nrow, ncol)^2)} Gram-Schmidt matrix, so it refuses
#'   matrices that would exceed \code{getOption("FastPCA.memory_limit")} (20 GB
#'   by default) and cannot handle all-NA rows or zero-variance columns. Use the
#'   \code{"r"}, \code{"rtorch"} or \code{"pytorch"} backends for large data.
#'
#'   The \code{"tinygrad"} backend is accepted but not implemented yet; it stops
#'   with an explicit message.
#' @param device Character, either \code{"CPU"} (default) or \code{"GPU"}. Only
#'   the torch backends act on it; \code{"r"} and \code{"nipals"} are always CPU
#'   (a GPU request warns and falls back).
#' @param cores Integer. Number of CPU threads for the torch backends, clamped
#'   to 1 when an OpenMP runtime conflict is detected (see [FastPCA()]).
#' @param max_na_prop Numeric in \code{[0, 1]}. Maximum allowed proportion of
#'   missing values per column; the default \code{1} never rejects a column.
#' @param verbose Logical. Print per-component progress (default \code{FALSE}).
#'
#' @return An object of class \code{"fastpca"} with elements:
#'   \item{U}{Left vectors (\code{nrow(input) x ncomp}); for NIPALS these are
#'     the unit-norm score vectors and, unlike an SVD result, they are not
#'     orthogonal to one another.}
#'   \item{S}{Singular values (the square roots of the per-component sums of
#'     squares).}
#'   \item{Vh}{Loadings, transposed (\code{ncomp x ncol(input)}). Columns that
#'     were dropped for being entirely missing are \code{NA}.}
#'   \item{scores}{The native score matrix, equal to
#'     \code{U \%*\% diag(S)}.}
#'
#'   Attributes describe the fit: \code{method = "nipals"}, the backend and
#'   device, \code{ncomp}, \code{tol}, \code{maxit}, per-component
#'   \code{iterations} and \code{converged}, \code{missing_fraction},
#'   \code{dropped_features}, the stored centring (\code{center}) and scaling
#'   vectors, and \code{total_variance} (the sum of squared observed entries,
#'   used by [pve()]).
#'
#' @details
#' The algorithm follows the reference NIPALS formulation: missing entries are
#' set to zero for every matrix product, and each denominator is the sum over
#' *observed* entries only. Centring uses observed-only means and scaling the
#' unbiased (\code{n - 1}) standard deviation, matching \pkg{nipals}. Deflation
#' subtracts the masked rank-one component (\code{X <- X - t p}).
#'
#' Signs are not identifiable in NIPALS, so each component is flipped to make
#' its largest-magnitude loading positive. Results are therefore stable across
#' backends and runs; the SVD path is deliberately left unchanged.
#'
#' Columns that are entirely missing are dropped before fitting (with a
#' warning) and reported in \code{attr(x, "dropped_features")}; their loadings
#' stay \code{NA} so the feature axis remains aligned with the input. A column
#' whose standard deviation is zero when \code{scale = TRUE} keeps its values
#' and uses a divisor of 1. Rows with no observed entries are kept; their scores
#' are \code{NA}.
#'
#' [fitted()] reconstructs the input, undoing centring and scaling; for missing
#' entries this is an imputation.
#'
#' @seealso [FastPCA()] for the SVD backends and \code{\link{fitted.fastpca}}.
#'
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(200), nrow = 20, ncol = 10,
#'             dimnames = list(paste0("s", 1:20), paste0("g", 1:10)))
#' x[sample(length(x), 20)] <- NA
#'
#' fit <- NIPALS(x, ncomp = 3, backend = "r")
#' fit
#' summary(fit)
#' head(get_pc_scores(fit))
#'
#' # impute the missing entries from three components
#' filled <- fitted(fit)
#' max(abs(filled[!is.na(x)] - x[!is.na(x)]))
#'
#' @export
NIPALS <- function(input_r_matrix,
                   ncomp = 10L,
                   tol = 1e-6,
                   maxit = 500L,
                   center = TRUE,
                   scale = FALSE,
                   gramschmidt = TRUE,
                   backend = c("r", "rtorch", "pytorch", "nipals", "tinygrad"),
                   device = c("CPU", "GPU"),
                   cores = 1L,
                   max_na_prop = 1,
                   verbose = FALSE) {
  # tinygrad is a placeholder and must be reported as such before backend
  # resolution, which would otherwise fail on the missing Python module first
  backend <- match.arg(backend, .nipals_backends)
  if (identical(backend, "tinygrad")) {
    stop("The 'tinygrad' backend is not implemented for NIPALS yet. ",
         "Use backend = 'pytorch', 'rtorch' or 'r'.", call. = FALSE)
  }

  # resolve the backend/device pair once and use the returned values: a request
  # may be downgraded (for example nipals -> r) and dispatching on the original
  # values would call a backend that cannot run (R/validate.R, R/utils.R)
  resolved <- resolve_backend_device(backend, device,
                                     allowed = .nipals_backends)
  backend <- resolved$backend
  device <- resolved$device

  .check_matrix_input(input_r_matrix, allow_sparse = FALSE, allow_na = TRUE)
  args <- .check_nipals_args(input_r_matrix, ncomp = ncomp, tol = tol,
                             maxit = maxit, max_na_prop = max_na_prop)
  ncomp <- args$ncomp
  tol <- args$tol
  maxit <- args$maxit
  max_na_prop <- args$max_na_prop

  # clamp to a single thread when an OpenMP runtime conflict is present (R/openmp.R)
  cores <- .resolve_cores(cores, openmp = .backend_uses_openmp(backend))

  prepared <- .nipals_prepare(input_r_matrix, center = isTRUE(center),
                              scale = isTRUE(scale), max_na_prop = max_na_prop)
  if (ncomp > length(prepared$keep)) {
    stop(sprintf(paste0("'ncomp' (%d) exceeds the number of usable features ",
                        "(%d); columns that are entirely missing cannot be used."),
                 ncomp, length(prepared$keep)), call. = FALSE)
  }

  if (isTRUE(verbose)) {
    message(sprintf("FastPCA NIPALS: %d x %d input, %.2f%% missing, ncomp = %d (%s backend)",
                    prepared$n_obs, prepared$n_features,
                    100 * prepared$missing_fraction, ncomp, backend))
  }

  fit <- switch(backend,
    r = .nipals_r(prepared$X, ncomp = ncomp, tol = tol, maxit = maxit,
                  gramschmidt = isTRUE(gramschmidt), verbose = isTRUE(verbose)),
    rtorch = .nipals_rtorch(prepared$X, ncomp = ncomp, tol = tol,
                            maxit = maxit, gramschmidt = isTRUE(gramschmidt),
                            device = device, cores = cores,
                            verbose = isTRUE(verbose)),
    pytorch = .nipals_pytorch(prepared$X, ncomp = ncomp, tol = tol,
                              maxit = maxit, gramschmidt = isTRUE(gramschmidt),
                              device = device, cores = cores),
    nipals = .nipals_package(prepared$X, ncomp = ncomp, tol = tol,
                             maxit = maxit, gramschmidt = isTRUE(gramschmidt))
  )

  .nipals_result(fit, prepared, backend = backend, device = device,
                 tol = tol, maxit = maxit)
}

# Prepare the working matrix once for every backend --------------------------
#
# Dense only. Returns the centred/scaled matrix, the information needed to
# align loadings back with the original features and to undo the transform in
# fitted(), plus total variance for pve().
.nipals_prepare <- function(X, center = TRUE, scale = FALSE, max_na_prop = 1) {
  X <- as.matrix(X)
  if (!is.double(X)) storage.mode(X) <- "double"
  n_obs <- nrow(X)
  n_features <- ncol(X)
  feature_names <- colnames(X)

  na_mask <- is.na(X)
  missing_fraction <- if (length(na_mask)) mean(na_mask) else 0
  observed_counts <- n_obs - colSums(na_mask)

  if (max_na_prop < 1) {
    na_prop <- colSums(na_mask) / n_obs
    too_sparse <- which(na_prop > max_na_prop)
    if (length(too_sparse)) {
      stop(sprintf(paste0("'max_na_prop' (%.2f) is exceeded by %d column%s (%s); ",
                          "up to %.0f%% of their entries are missing."),
                   max_na_prop, length(too_sparse),
                   if (length(too_sparse) == 1L) "" else "s",
                   paste(.nipals_feature_labels(X, too_sparse), collapse = ", "),
                   100 * max(na_prop[too_sparse])),
           call. = FALSE)
    }
  }

  # columns with no observed entries cannot be centred or used; keep the feature
  # axis aligned by leaving NA loadings for them in .nipals_result()
  all_na <- which(observed_counts == 0L)
  keep <- setdiff(seq_len(n_features), all_na)
  if (!length(keep)) {
    stop("Every column is entirely missing; there is nothing to decompose.",
         call. = FALSE)
  }
  dropped_features <- if (length(all_na)) .nipals_feature_labels(X, all_na) else NULL
  if (length(all_na)) {
    warning(sprintf("Dropped %d column%s with no observed values: %s.",
                    length(all_na), if (length(all_na) == 1L) "" else "s",
                    paste(dropped_features, collapse = ", ")), call. = FALSE)
  }

  X <- X[, keep, drop = FALSE]
  na_mask <- na_mask[, keep, drop = FALSE]
  observed_counts <- observed_counts[keep]
  has_na <- any(na_mask)

  # centre with observed-only means (matches nipals::nipals)
  cmeans <- if (has_na) colSums(X, na.rm = TRUE) / observed_counts else colMeans(X)
  if (center) X <- sweep(X, 2L, cmeans, "-")

  scaling <- NULL
  if (scale) {
    sumsq <- colSums(X * X, na.rm = TRUE)
    variance <- ifelse(observed_counts > 1L, sumsq / (observed_counts - 1), NA_real_)
    scaling <- sqrt(pmax(variance, 0))
    degenerate <- which(!is.finite(scaling) | scaling == 0)
    if (length(degenerate)) {
      warning(sprintf(paste0("Column%s %s zero variance; the standard ",
                             "deviation is treated as 1."),
                      if (length(degenerate) == 1L) "" else "s",
                      paste(.nipals_feature_labels(X, degenerate),
                            if (length(degenerate) == 1L) "has" else "have",
                            collapse = ", ")),
              call. = FALSE)
      scaling[degenerate] <- 1
    }
    X <- sweep(X, 2L, scaling, "/")
  }

  total_variance <- sum(X * X, na.rm = TRUE)

  # full-length transform vectors (NA for dropped features) so fitted() can
  # rebuild a matrix with the same feature axis as the input
  center_full <- NULL
  if (center) {
    center_full <- rep(NA_real_, n_features)
    center_full[keep] <- cmeans
  }
  scaling_full <- NULL
  if (scale) {
    scaling_full <- rep(NA_real_, n_features)
    scaling_full[keep] <- scaling
  }

  list(X = X, keep = keep, n_obs = n_obs, n_features = n_features,
       feature_names = feature_names, dropped_features = dropped_features,
       center_full = center_full, scaling_full = scaling_full,
       missing_fraction = as.numeric(missing_fraction),
       total_variance = as.numeric(total_variance))
}

# Column names when present, positions otherwise.
.nipals_feature_labels <- function(X, idx) {
  nm <- colnames(X)
  if (!is.null(nm)) nm[idx] else as.character(idx)
}

# Base R NIPALS: the reference implementation, no Python and no O(n^2) matrices.
.nipals_r <- function(X, ncomp, tol, maxit, gramschmidt = TRUE,
                      verbose = FALSE) {
  nobs <- nrow(X)
  nvar <- ncol(X)
  na_mask <- is.na(X)
  has_na <- any(na_mask)

  if (has_na) {
    # numeric observation mask: one BLAS matvec per denominator instead of an
    # nobs x nvar temporary, and the masked entries are zero for every product
    obsd <- matrix(as.numeric(!na_mask), nobs, nvar)
    x0 <- X
    x0[na_mask] <- 0
    empty_rows <- rowSums(obsd) == 0L
  } else {
    obsd <- NULL
    x0 <- X
    empty_rows <- rep(FALSE, nobs)
  }

  eig <- numeric(ncomp)
  loadings <- matrix(0, nvar, ncomp,
                     dimnames = list(colnames(X), .pc_names(ncomp)))
  scores <- matrix(0, nobs, ncomp,
                   dimnames = list(rownames(X), .pc_names(ncomp)))
  iterations <- integer(ncomp)
  converged <- logical(ncomp)

  for (h in seq_len(ncomp)) {
    # start from the column with the largest observed absolute sum (as nipals)
    colscore <- if (has_na) colSums(obsd * abs(x0)) else colSums(abs(X))
    th <- x0[, which.max(colscore)]

    pciter <- 1L
    conv <- FALSE
    repeat {
      # loadings p = X't / t't, with an observed-only denominator when there is NA
      if (has_na) {
        denom_p <- as.vector(crossprod(obsd, th * th))
        # a column whose observed entries all have zero score contributes
        # nothing to this component: keep its loading at 0 instead of NaN
        denom_p[denom_p == 0] <- .Machine$double.xmin
        ph <- as.vector(crossprod(x0, th)) / denom_p
      } else {
        ph <- as.vector(crossprod(X, th)) / sum(th * th)
      }
      if (gramschmidt && h > 1L) {
        P <- loadings[, seq_len(h - 1L), drop = FALSE]
        # as.vector: %*% would otherwise return a one-column matrix, which the
        # torch backends turn into a rank-2 tensor and break their shapes with
        ph <- as.vector(ph - P %*% crossprod(P, ph))
      }
      nrm <- sqrt(sum(ph * ph))
      if (!is.finite(nrm) || nrm == 0) {
        stop(sprintf(paste0("NIPALS failed at component %d: the loading vector ",
                            "has zero norm; the data may be rank-deficient."), h),
             call. = FALSE)
      }
      ph <- ph / nrm

      th_old <- th
      if (has_na) {
        denom_t <- as.vector(obsd %*% (ph * ph))
        # rows with no observed entries have a zero denominator; keep them at
        # exactly 0 (not NaN) so Gram-Schmidt stays finite. They are marked NA
        # once the loop is done.
        denom_t[denom_t == 0] <- .Machine$double.xmin
        th <- as.vector(x0 %*% ph) / denom_t
      } else {
        th <- as.vector(X %*% ph) / sum(ph * ph)
      }
      if (gramschmidt && h > 1L) {
        Tprev <- scores[, seq_len(h - 1L), drop = FALSE]
        sv_prev <- sqrt(eig[seq_len(h - 1L)])
        Tn <- sweep(Tprev, 2L, sv_prev, "/")
        th <- as.vector(th - Tn %*% crossprod(Tn, th))
      }
      if (has_na) th[empty_rows] <- 0

      # convergence on the normalised score vector (as in nipals::nipals)
      nn <- sqrt(sum(th * th))
      no <- sqrt(sum(th_old * th_old))
      if (nn > 0 && no > 0) {
        d <- th / nn - th_old / no
        if (sum(d * d) < tol) {
          conv <- TRUE
          break
        }
      }
      if (pciter >= maxit) {
        warning(sprintf(paste0("NIPALS: component %d stopped after %d iterations ",
                               "without reaching tol = %g."), h, maxit, tol),
                call. = FALSE)
        break
      }
      pciter <- pciter + 1L
    }

    # masked rank-one deflation (decision D3); x0 keeps the mask applied
    X <- X - tcrossprod(th, ph)
    if (has_na) {
      x0 <- X
      x0[na_mask] <- 0
    }

    loadings[, h] <- ph
    scores[, h] <- th
    eig[h] <- sum(th * th)
    iterations[h] <- pciter
    converged[h] <- conv

    if (verbose) {
      message(sprintf("  PC%d: %d iterations, eigenvalue %.4f", h, pciter, eig[h]))
    }
  }

  if (has_na && any(empty_rows)) scores[empty_rows, ] <- NA_real_

  list(eig = eig, loadings = loadings, scores = scores,
       iterations = iterations, converged = converged)
}

# Assemble a "fastpca" object from any backend's raw fit ----------------------
.nipals_result <- function(fit, prepared, backend, device, tol, maxit) {
  eig <- as.numeric(fit$eig)
  sv <- sqrt(pmax(eig, 0))
  if (any(!is.finite(sv) | sv <= 0)) {
    warning("NIPALS: one or more components have zero variance; their vectors ",
            "are reported as NA.", call. = FALSE)
    sv[!is.finite(sv) | sv <= 0] <- NA_real_
  }

  scores <- as.matrix(fit$scores)
  rownames(scores) <- rownames(prepared$X)
  colnames(scores) <- .pc_names(length(sv))

  # loadings keep the feature axis of the input; dropped features stay NA
  Vh <- matrix(NA_real_, nrow = length(sv), ncol = prepared$n_features,
               dimnames = list(.pc_names(length(sv)), prepared$feature_names))
  Vh[, prepared$keep] <- t(as.matrix(fit$loadings))

  # deterministic sign convention (decision D7): flip each component so its
  # largest-magnitude loading is positive, carrying the same flip into scores.
  for (h in seq_len(length(sv))) {
    ph <- Vh[h, prepared$keep]
    idx <- which.max(abs(ph))
    if (length(idx) && is.finite(ph[idx]) && ph[idx] < 0) {
      Vh[h, prepared$keep] <- -ph
      scores[, h] <- -scores[, h]
    }
  }

  U <- sweep(scores, 2L, sv, "/")

  new_fastpca(
    U = U, S = sv, Vh = Vh, scores = scores,
    method = "nipals",
    k = NULL, p = NULL, q_iter = NULL, exact = FALSE,
    backend = backend, device = device,
    n_obs = prepared$n_obs, n_features = prepared$n_features,
    total_variance = prepared$total_variance,
    center = prepared$center_full, scaling = prepared$scaling_full,
    ncomp = length(sv), tol = tol, maxit = as.integer(maxit),
    iterations = fit$iterations, converged = fit$converged,
    missing_fraction = prepared$missing_fraction,
    dropped_features = prepared$dropped_features
  )
}

# PyTorch through reticulate --------------------------------------------------
.nipals_pytorch <- function(X, ncomp, tol, maxit, gramschmidt = TRUE,
                            device = "CPU", cores = 1L) {
  .globals <- python_functions(backend = "nipals")
  on.exit(rm(.globals), add = TRUE)
  out <- .globals$nipals_func$nipals_py(
    X,
    ncomp = as.integer(ncomp),
    tol = as.numeric(tol),
    maxit = as.integer(maxit),
    gramschmidt = isTRUE(gramschmidt),
    device = toupper(as.character(device)),
    cores = as.integer(cores)
  )
  scores <- as.matrix(out$scores)
  loadings <- as.matrix(out$loadings)
  rownames(scores) <- rownames(X)
  colnames(scores) <- .pc_names(ncol(scores))
  rownames(loadings) <- colnames(X)
  list(eig = as.numeric(out$eig),
       loadings = loadings,
       scores = scores,
       iterations = as.integer(unlist(out$iters)),
       converged = as.logical(unlist(out$converged)))
}

# The CRAN 'nipals' package, used as the reference implementation -------------
.nipals_package <- function(X, ncomp, tol, maxit, gramschmidt = TRUE) {
  if (!requireNamespace("nipals", quietly = TRUE)) {
    stop("backend = 'nipals' requires the 'nipals' package; install it with ",
         "install.packages('nipals') or use backend = 'r'.", call. = FALSE)
  }

  # the package allocates O(max(nrow, ncol)^2) Gram-Schmidt matrices; refuse a
  # shape that would blow past the configured ceiling rather than crash R
  worst_gb <- max(dim(X))^2 * 8 / 1024^3
  limit_gb <- getOption("FastPCA.memory_limit", 20)
  if (worst_gb > limit_gb) {
    stop(sprintf(paste0("backend = 'nipals' (the CRAN nipals package) would ",
                        "allocate about %.0f GB for a %d x %d matrix, above the ",
                        "%.0f GB memory limit. Use backend = 'pytorch', 'rtorch' ",
                        "or 'r' for large matrices."),
                 worst_gb, nrow(X), ncol(X), limit_gb), call. = FALSE)
  }
  if (any(rowSums(!is.na(X)) == 0L)) {
    stop("backend = 'nipals' (the CRAN nipals package) cannot handle rows that ",
         "are entirely missing; remove them or use backend = 'r'.", call. = FALSE)
  }
  if (nrow(X) > 1L && any(apply(X, 2L, stats::var, na.rm = TRUE) == 0)) {
    stop("backend = 'nipals' (the CRAN nipals package) cannot handle ",
         "zero-variance columns; remove them or use backend = 'r'.",
         call. = FALSE)
  }

  # X is already centred/scaled, so the package must not transform it again
  fit <- nipals::nipals(X, ncomp = as.integer(ncomp), center = FALSE,
                        scale = FALSE, maxiter = as.integer(maxit),
                        tol = as.numeric(tol), gramschmidt = isTRUE(gramschmidt))

  # nipals() returns eig already as singular values and normalised scores
  sv <- as.numeric(fit$eig)
  loadings <- as.matrix(fit$loadings)
  scores <- sweep(as.matrix(fit$scores), 2L, sv, "*")
  rownames(scores) <- rownames(X)
  colnames(scores) <- .pc_names(ncol(scores))
  rownames(loadings) <- colnames(X)
  iterations <- as.integer(fit$iter)

  list(eig = sv^2, loadings = loadings, scores = scores,
       iterations = iterations,
       converged = is.finite(iterations) & iterations < as.integer(maxit))
}
