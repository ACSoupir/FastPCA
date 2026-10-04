# The FastPCA result object ---------------------------------------------------
#
# A "fastpca" is a list with the same U/S/Vh elements FastPCA() has always
# returned, plus attributes describing how it was computed. Keeping the list
# shape means existing code that indexes $U/$S/$Vh keeps working, while the
# attributes let print()/summary()/pve() and future solvers share one contract.

new_fastpca <- function(U, S, Vh,
                        k = length(S), p = NULL, q_iter = NULL,
                        exact = FALSE, backend = NA_character_,
                        device = NA_character_, n_obs = NULL, n_features = NULL,
                        total_variance = NA_real_) {
  structure(
    list(U = U, S = as.numeric(S), Vh = Vh),
    class = "fastpca",
    k = as.integer(k)[1],
    p = if (is.null(p)) NA_integer_ else as.integer(p)[1],
    q_iter = if (is.null(q_iter)) NA_integer_ else as.integer(q_iter)[1],
    exact = isTRUE(exact),
    backend = if (length(backend)) as.character(backend)[1] else NA_character_,
    device = if (length(device)) as.character(device)[1] else NA_character_,
    n_obs = if (is.null(n_obs)) nrow(U) else as.integer(n_obs)[1],
    n_features = if (is.null(n_features)) ncol(Vh) else as.integer(n_features)[1],
    total_variance = if (length(total_variance)) as.numeric(total_variance)[1] else NA_real_
  )
}

#' Print a FastPCA result
#'
#' @param x A \code{"fastpca"} object.
#' @param ... Ignored.
#'
#' @return \code{x}, invisibly.
#' @export
print.fastpca <- function(x, ...) {
  k <- length(x$S)
  cat(sprintf("<fastpca> %s x %s input; k = %d\n",
              format(attr(x, "n_obs")), format(attr(x, "n_features")), k))
  cat(sprintf("  backend: %s | device: %s\n",
              attr(x, "backend"), attr(x, "device")))
  top <- utils::head(format(signif(as.numeric(x$S), 4)), 6L)
  cat("  singular values: ", paste(top, collapse = ", "),
      if (k > length(top)) ", ..." else "", "\n", sep = "")
  invisible(x)
}

#' Summarise a FastPCA result
#'
#' Returns one row per component with the singular value and, when total
#' variance is available (see \code{store_variance} in [FastPCA()]), the
#' percent and cumulative percent variation explained.
#'
#' @param object A \code{"fastpca"} object.
#' @param ... Ignored.
#'
#' @return A \code{data.frame} with columns \code{PC},
#'   \code{singular_value}, \code{percent_variance} and
#'   \code{cumulative_percent_variance}.
#' @export
summary.fastpca <- function(object, ...) {
  pct <- tryCatch(100 * pve(object), error = function(e) rep(NA_real_, length(object$S)))
  data.frame(
    PC = seq_along(object$S),
    singular_value = as.numeric(object$S),
    percent_variance = pct,
    cumulative_percent_variance = cumsum(pct)
  )
}

#' Percent variation explained
#'
#' @param x A \code{"fastpca"} object returned by [FastPCA()].
#' @param total Optional numeric scalar: the sum of squared entries of the input
#'   matrix. Defaults to the total variance stored by [FastPCA()].
#'
#' @return A named numeric vector of per-component fractions (\code{PC1}, ...).
#'   Percent values are \code{100 * pve(x)}.
#' @export
pve <- function(x, total = NULL) {
  if (!inherits(x, "fastpca")) {
    stop("'x' must be a 'fastpca' object (the result of FastPCA()).", call. = FALSE)
  }
  if (is.null(total)) total <- attr(x, "total_variance")
  if (!length(total) || !is.finite(total)) {
    stop("Total variance is not available; supply 'total' (the sum of squared ",
         "entries of the input matrix) or call FastPCA(store_variance = TRUE).",
         call. = FALSE)
  }
  value <- (as.numeric(x$S)^2) / as.numeric(total)[1]
  names(value) <- paste0("PC", seq_along(value))
  value
}

#' PC scores from a FastPCA result
#'
#' A convenience wrapper around [get_pc_scores()].
#'
#' @param x A \code{"fastpca"} object returned by [FastPCA()].
#'
#' @return A numeric matrix of scores, with rows as observations.
#' @export
scores <- function(x) {
  if (!inherits(x, "fastpca")) {
    stop("'x' must be a 'fastpca' object (the result of FastPCA()).", call. = FALSE)
  }
  get_pc_scores(x)
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
