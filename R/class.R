# The FastPCA result object ---------------------------------------------------
#
# A "fastpca" is a list with elements U, S, Vh and scores plus attributes
# describing how it was computed. Keeping the list shape means existing code
# that indexes $U/$S/$Vh keeps working, while the attributes let print(),
# summary(), pve() and both solver families (randomized/exact SVD and NIPALS)
# share one contract.
#
# Attribute groups:
#   payload elements      U, S, Vh, scores                         always
#   identity attributes   method, backend, device, n_obs,
#                         n_features                              always
#   SVD-only attributes   k, p, q_iter, exact                     NA for NIPALS
#   NIPALS-only           ncomp, tol, maxit, iterations,
#                         converged, missing_fraction             NA for SVD
#   fitting attributes    total_variance; center/scaling vectors
#                         and dropped_features (NIPALS)
#
# scores is always populated, so callers never have to ask whether a result
# carries them. For SVD it is U %*% diag(S); for NIPALS it is the native score
# vector and U is defined as its unit-norm version, so scores == U %*% diag(S)
# holds for both methods. NIPALS' left vectors are not orthogonal, which is the
# real (documented) difference.

.pc_names <- function(k) paste0("PC", seq_len(k))

# PC scores of an SVD result: the left singular vectors scaled by the singular
# values. Kept in one place so new_fastpca() and .validate_fastpca() agree.
.fastpca_scores_from_udv <- function(U, S) {
  U %*% diag(as.numeric(S), nrow = length(S))
}

# Give a scores matrix its canonical column names (PC1, PC2, ...) and inherit
# row names from the left vectors when they are missing.
.as_scores_matrix <- function(scores, U = NULL) {
  scores <- as.matrix(scores)
  if (is.null(colnames(scores))) colnames(scores) <- .pc_names(ncol(scores))
  # FastPCA() labels the SVD columns "dim1", ...; report scores as PCs
  colnames(scores) <- sub("^dim", "PC", colnames(scores))
  if (is.null(rownames(scores)) && !is.null(U) && !is.null(rownames(U))) {
    rownames(scores) <- rownames(U)
  }
  scores
}

new_fastpca <- function(U, S, Vh, scores = NULL,
                        method = "svd",
                        k = length(S), p = NULL, q_iter = NULL,
                        exact = FALSE, backend = NA_character_,
                        device = NA_character_, n_obs = NULL, n_features = NULL,
                        total_variance = NA_real_,
                        center = NULL, scaling = NULL,
                        ncomp = NULL, tol = NA_real_, maxit = NA_integer_,
                        iterations = NULL, converged = NULL,
                        missing_fraction = NA_real_, dropped_features = NULL,
                        precision = NA_character_, work = NULL,
                        n_iter = NULL, mprod = NULL) {
  method <- match.arg(method, c("svd", "nipals", "incremental"))
  S <- as.numeric(S)
  # An out-of-core IncrementalPCA() result may carry neither U nor scores: both
  # are only derivable when the scoring pass was requested, which is what keeps a
  # constant-memory fit possible.
  if (is.null(scores) && !is.null(U)) scores <- .fastpca_scores_from_udv(U, S)
  if (!is.null(scores)) scores <- .as_scores_matrix(scores, U)
  if (is.null(n_obs)) {
    n_obs <- if (!is.null(U)) nrow(U) else
      (if (!is.null(scores)) nrow(scores) else NA_integer_)
  }

  structure(
    list(U = U, S = S, Vh = Vh, scores = scores),
    class = "fastpca",
    method = method,
    k = if (is.null(k)) NA_integer_ else as.integer(k)[1],
    p = if (is.null(p)) NA_integer_ else as.integer(p)[1],
    q_iter = if (is.null(q_iter)) NA_integer_ else as.integer(q_iter)[1],
    exact = isTRUE(exact),
    backend = if (length(backend)) as.character(backend)[1] else NA_character_,
    device = if (length(device)) as.character(device)[1] else NA_character_,
    n_obs = as.integer(n_obs)[1],
    n_features = if (is.null(n_features)) ncol(Vh) else as.integer(n_features)[1],
    total_variance = if (length(total_variance)) as.numeric(total_variance)[1] else NA_real_,
    center = if (is.null(center)) NULL else as.numeric(center),
    scaling = if (is.null(scaling)) NULL else as.numeric(scaling),
    ncomp = if (is.null(ncomp)) NA_integer_ else as.integer(ncomp)[1],
    tol = if (length(tol)) as.numeric(tol)[1] else NA_real_,
    maxit = if (length(maxit)) as.integer(maxit)[1] else NA_integer_,
    iterations = if (is.null(iterations)) NULL else as.integer(iterations),
    converged = if (is.null(converged)) NULL else as.logical(converged),
    missing_fraction = if (length(missing_fraction)) as.numeric(missing_fraction)[1] else NA_real_,
    dropped_features = if (is.null(dropped_features)) NULL else as.character(dropped_features),
    precision = if (length(precision) && !is.na(precision[1])) as.character(precision)[1] else NA_character_,
    work = if (is.null(work)) NULL else as.integer(work)[1],
    n_iter = if (is.null(n_iter)) NULL else as.integer(n_iter)[1],
    mprod = if (is.null(mprod)) NULL else as.integer(mprod)[1]
  )
}

# Invariant check shared by every accessor and print/summary method.
#
# A three-element list (an object saved before the scores element existed) is
# accepted: scores are derived from U and S so old results keep loading. Returns
# the object, possibly completed.
.validate_fastpca <- function(x) {
  if (!inherits(x, "fastpca") || !is.list(x)) {
    stop("'x' must be a 'fastpca' object (the result of FastPCA(), NIPALS() or ",
         "IncrementalPCA()).", call. = FALSE)
  }
  if (is.null(x$S) || is.null(x$Vh)) {
    stop("A 'fastpca' object must contain the elements S and Vh.", call. = FALSE)
  }
  # U is optional for an IncrementalPCA() result whose scores were not stored;
  # every other method always carries it
  if (is.null(x$U) && !identical(attr(x, "method"), "incremental")) {
    stop("A 'fastpca' object must contain the elements U, S and Vh.", call. = FALSE)
  }
  k <- length(x$S)
  if (k < 1L) stop("A 'fastpca' object must contain at least one component.", call. = FALSE)
  if (nrow(x$Vh) != k) {
    stop(sprintf("Malformed 'fastpca' object: length(S) is %d but Vh has %d rows.",
                 k, nrow(x$Vh)), call. = FALSE)
  }
  if (any(!is.finite(as.numeric(x$S)))) {
    stop("Malformed 'fastpca' object: S must be finite.", call. = FALSE)
  }
  n_obs <- attr(x, "n_obs")
  if (length(n_obs) && is.finite(n_obs) && !is.null(x$U) &&
      nrow(x$U) != as.integer(n_obs)) {
    stop(sprintf("Malformed 'fastpca' object: nrow(U) is %d but n_obs is %d.",
                 nrow(x$U), as.integer(n_obs)), call. = FALSE)
  }
  method <- attr(x, "method")
  if (is.null(method)) {
    # objects created before the method attribute existed are all SVD results
    attr(x, "method") <- "svd"
  } else if (!method %in% c("svd", "nipals", "incremental")) {
    stop(sprintf("Malformed 'fastpca' object: unknown method '%s'.", method),
         call. = FALSE)
  }
  if (is.null(x$scores) && !is.null(x$U)) {
    x$scores <- .as_scores_matrix(.fastpca_scores_from_udv(x$U, x$S), x$U)
  }
  x
}

# Human-readable solver label used by print()/summary().
.fastpca_method_label <- function(x) {
  if (identical(attr(x, "method"), "nipals")) return("NIPALS (missing-value)")
  if (identical(attr(x, "method"), "incremental")) return("incremental PCA")
  if (identical(attr(x, "backend"), "torch_irlba")) return("Lanczos SVD (IRLBA)")
  if (isTRUE(attr(x, "exact"))) return("exact SVD")
  "randomized SVD"
}

#' Print a FastPCA result
#'
#' @param x A \code{"fastpca"} object.
#' @param ... Ignored.
#'
#' @return \code{x}, invisibly.
#' @export
print.fastpca <- function(x, ...) {
  x <- .validate_fastpca(x)
  k <- length(x$S)
  cat(sprintf("<fastpca> %s x %s input; k = %d\n",
              format(attr(x, "n_obs")), format(attr(x, "n_features")), k))
  precision <- attr(x, "precision")
  cat(sprintf("  method: %s | backend: %s | device: %s%s\n",
              .fastpca_method_label(x), attr(x, "backend"), attr(x, "device"),
              if (length(precision) && !is.na(precision)) {
                paste0(" | precision: ", precision)
              } else ""))

  if (identical(attr(x, "method"), "incremental")) {
    count_or_na <- function(value) {
      if (is.null(value) || length(value) == 0L) NA_integer_ else as.integer(value)[1]
    }
    cat(sprintf("  out-of-core: %d file(s), %s block(s) read in %s pass(es)\n",
                count_or_na(length(attr(x, "files"))),
                format(count_or_na(attr(x, "n_blocks"))),
                format(count_or_na(attr(x, "passes")))))
    if (!isTRUE(attr(x, "scores_stored"))) {
      cat("  scores: not stored; re-run with store_scores = TRUE to obtain them\n")
    }
  } else if (identical(attr(x, "method"), "nipals")) {
    missing_fraction <- attr(x, "missing_fraction")
    if (length(missing_fraction) && is.finite(missing_fraction)) {
      cat(sprintf("  missing: %.1f%% of entries (masked)\n", 100 * missing_fraction))
    }
    iterations <- attr(x, "iterations")
    if (length(iterations)) {
      converged <- attr(x, "converged")
      n_converged <- if (length(converged)) sum(converged %in% TRUE) else NA_integer_
      cat(sprintf("  converged in median %s iterations (tol = %s, maxit = %d)",
                  format(stats::median(iterations)), format(attr(x, "tol")),
                  as.integer(attr(x, "maxit"))))
      if (!is.na(n_converged) && n_converged < length(iterations)) {
        cat(sprintf("; %d of %d hit maxit", length(iterations) - n_converged,
                    length(iterations)))
      }
      cat("\n")
    }
  } else if (identical(attr(x, "backend"), "torch_irlba")) {
    cat(sprintf("  solver: Lanczos (IRLBA), nv = %d, work = %s\n",
                k, format(attr(x, "work"))))
    n_iter <- attr(x, "n_iter")
    if (length(n_iter) && !is.na(n_iter)) {
      mprod <- attr(x, "mprod")
      cat(sprintf("  converged in %d iteration%s (%s matrix products)\n",
                  n_iter, if (identical(as.integer(n_iter), 1L)) "" else "s",
                  format(mprod)))
    }
  } else {
    cat(sprintf("  solver: %s (p = %s, q_iter = %s)\n",
                if (isTRUE(attr(x, "exact"))) "exact SVD" else "randomized SVD",
                format(attr(x, "p")), format(attr(x, "q_iter"))))
  }

  pct <- tryCatch(100 * pve(x), error = function(e) NULL)
  if (!is.null(pct)) {
    shown <- utils::head(sprintf("PC%d %.1f%%", seq_along(pct), pct), 3L)
    cat(sprintf("  PVE: %s%s | cumulative (%d PCs) %.1f%%\n",
                paste(shown, collapse = ", "),
                if (length(pct) > length(shown)) ", ..." else "",
                k, sum(pct)))
  } else {
    top <- utils::head(format(signif(as.numeric(x$S), 4)), 6L)
    cat("  singular values: ", paste(top, collapse = ", "),
        if (k > length(top)) ", ..." else "", "\n", sep = "")
  }
  invisible(x)
}

#' Summarise a FastPCA result
#'
#' Returns one row per component with the singular value and, when total
#' variance is available (see \code{store_variance} in [FastPCA()]), the
#' percent and cumulative percent variation explained. For NIPALS results the
#' \code{iterations} column records how many iterations each component needed
#' and \code{converged} whether it reached the tolerance before \code{maxit};
#' both are \code{NA} for SVD results.
#'
#' @param object A \code{"fastpca"} object.
#' @param ... Ignored.
#'
#' @return A \code{data.frame} with columns \code{PC},
#'   \code{singular_value}, \code{percent_variance},
#'   \code{cumulative_percent_variance}, \code{iterations} and
#'   \code{converged}. The solver (\code{"svd"} or \code{"nipals"}) and backend
#'   are carried as attributes.
#' @export
summary.fastpca <- function(object, ...) {
  object <- .validate_fastpca(object)
  k <- length(object$S)
  pct <- tryCatch(100 * pve(object), error = function(e) rep(NA_real_, k))
  iterations <- attr(object, "iterations")
  converged <- attr(object, "converged")
  if (is.null(iterations)) iterations <- rep(NA_integer_, k)
  if (is.null(converged)) converged <- rep(NA, k)
  out <- data.frame(
    PC = seq_len(k),
    singular_value = as.numeric(object$S),
    percent_variance = pct,
    cumulative_percent_variance = cumsum(pct),
    iterations = as.integer(iterations),
    converged = as.logical(converged)
  )
  attr(out, "method") <- attr(object, "method")
  attr(out, "backend") <- attr(object, "backend")
  out
}

#' Percent variation explained
#'
#' For NIPALS results the denominator is the sum of squared entries over the
#' observed (non-missing) cells only, so it is not directly comparable to a sum
#' of squares computed after imputation.
#'
#' @param x A \code{"fastpca"} object returned by [FastPCA()] or [NIPALS()].
#' @param total Optional numeric scalar: the sum of squared entries of the input
#'   matrix. Defaults to the total variance stored by [FastPCA()] or [NIPALS()].
#'
#' @return A named numeric vector of per-component fractions (\code{PC1}, ...).
#'   Percent values are \code{100 * pve(x)}.
#' @export
pve <- function(x, total = NULL) {
  x <- .validate_fastpca(x)
  if (is.null(total)) total <- attr(x, "total_variance")
  if (!length(total) || !is.finite(total)) {
    stop("Total variance is not available; supply 'total' (the sum of squared ",
         "entries of the input matrix) or call FastPCA(store_variance = TRUE).",
         call. = FALSE)
  }
  value <- (as.numeric(x$S)^2) / as.numeric(total)[1]
  names(value) <- .pc_names(length(value))
  value
}

#' PC scores from a FastPCA result
#'
#' A convenience wrapper around [get_pc_scores()].
#'
#' @param x A \code{"fastpca"} object returned by [FastPCA()] or [NIPALS()].
#'
#' @return A numeric matrix of scores, with rows as observations.
#' @export
scores <- function(x) {
  get_pc_scores(x)
}

#' Reconstruct the input matrix from a NIPALS result
#'
#' Returns \code{scores %*% Vh}, undoing the centring and scaling stored on
#' the object. This is how missing entries are imputed: observed cells are
#' reproduced up to convergence error and masked cells receive the component
#' estimate.
#'
#' The reconstruction is a dense \code{n_obs x n_features} matrix, which for the
#' package example (978 x 199112) is about 1.5 GB, so it is computed on demand
#' rather than stored.
#'
#' @param object A \code{"fastpca"} object returned by [NIPALS()].
#' @param ... Ignored.
#'
#' @return A numeric matrix with the dimensions and dimnames of the input that
#'   was passed to [NIPALS()].
#' @export
fitted.fastpca <- function(object, ...) {
  object <- .validate_fastpca(object)
  if (!identical(attr(object, "method"), "nipals")) {
    stop("fitted() is only available for NIPALS results; SVD results do not ",
         "store the centring/scaling needed to undo the decomposition.",
         call. = FALSE)
  }
  xhat <- object$scores %*% object$Vh
  scaling <- attr(object, "scaling")
  center <- attr(object, "center")
  if (!is.null(scaling)) xhat <- sweep(xhat, 2L, scaling, "*")
  if (!is.null(center)) xhat <- sweep(xhat, 2L, center, "+")
  colnames(xhat) <- colnames(object$Vh)
  rownames(xhat) <- rownames(object$scores)
  xhat
}

#' Plot a FastPCA result
#'
#' Draws either the scree plot of singular values or, when total variance is
#' stored (see \code{store_variance} in [FastPCA()]), the cumulative percent
#' variation explained.
#'
#' @param x A \code{"fastpca"} object.
#' @param type Either "scree" (default) or "cumulative".
#' @param ... Passed to \code{graphics::plot()}.
#'
#' @return \code{x}, invisibly.
#' @export
plot.fastpca <- function(x, type = c("scree", "cumulative"), ...) {
  x <- .validate_fastpca(x)
  type <- match.arg(type)
  pct <- tryCatch(pve(x), error = function(e) NULL)
  if (type == "cumulative" && is.null(pct)) {
    stop("Cumulative variance requires total variance; use ",
         "FastPCA(store_variance = TRUE).", call. = FALSE)
  }
  y <- if (is.null(pct)) as.numeric(x$S) else pct * 100
  if (type == "cumulative") y <- cumsum(y)
  graphics::plot(seq_along(y), y, type = "b", xlab = "Component",
                 ylab = if (is.null(pct)) "Singular value" else "Percent variance",
                 ...)
  invisible(x)
}
