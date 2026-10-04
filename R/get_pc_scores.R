#' Calculate the PC scores
#'
#' @param fastpca_out A \code{"fastpca"} object, i.e. the result of
#'   [FastPCA()], with elements \code{U}, \code{S} and \code{Vh}.
#'
#' @returns Matrix: contains the scores associated with each of the PCs
#' @export
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(60), nrow = 10, ncol = 6)
#' res <- FastPCA(x, k = 3, p = 1, backend = "r")
#' get_pc_scores(res)
get_pc_scores <- function(fastpca_out) {
  if (!is.list(fastpca_out) || is.null(fastpca_out$U) ||
      is.null(fastpca_out$S)) {
    stop("'fastpca_out' must be the result of FastPCA() (a list with elements ",
         "U, S and Vh).", call. = FALSE)
  }
  singular_values <- as.numeric(fastpca_out$S)
  # multiply the left singular vectors by the singular values
  scores <- fastpca_out$U %*% diag(singular_values, nrow = length(singular_values))
  # set back the correct column names
  colnames(scores) <- gsub("^dim", "PC", colnames(fastpca_out$U))
  scores
}
