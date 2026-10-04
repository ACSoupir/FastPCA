#' Calculate the PC scores
#'
#' Returns the stored \code{scores} element of a \code{"fastpca"} object, which
#' is always populated. For SVD results that element equals
#' \code{U \%*\% diag(S)}; NIPALS stores its native score vectors, and \code{U}
#' is defined as their unit-norm version so the same identity holds. A bare list
#' with \code{U} and \code{S} is still accepted, as are a Seurat object or
#' \code{DimReduc}, in which case the cell embeddings are returned.
#'
#' @param fastpca_out A \code{"fastpca"} object (the result of [FastPCA()] or
#'   [NIPALS()]), a list with \code{U} and \code{S}, or a Seurat object /
#'   \code{DimReduc}.
#' @param reduction For a Seurat object, the reduction to read. Defaults to
#'   \code{"pca"}.
#'
#' @returns Matrix: contains the scores associated with each of the PCs
#' @export
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(60), nrow = 10, ncol = 6)
#' res <- FastPCA(x, k = 3, p = 1, backend = "r")
#' get_pc_scores(res)
get_pc_scores <- function(fastpca_out, reduction = "pca") {
  # Seurat integration: hand back the embeddings of the requested reduction
  if (inherits(fastpca_out, "DimReduc")) {
    return(as.matrix(SeuratObject::Embeddings(fastpca_out)))
  }
  if (inherits(fastpca_out, "Seurat")) {
    if (!requireNamespace("SeuratObject", quietly = TRUE)) {
      stop("Package 'SeuratObject' is required to read scores from a Seurat object.",
           call. = FALSE)
    }
    reductions <- SeuratObject::Reductions(fastpca_out)
    if (!reduction %in% reductions) {
      stop("Seurat object has no reduction named '", reduction, "'; available: ",
           paste(reductions, collapse = ", "), ".", call. = FALSE)
    }
    return(as.matrix(SeuratObject::Embeddings(fastpca_out[[reduction]])))
  }
  if (!is.list(fastpca_out) || is.null(fastpca_out$U)) {
    stop("'fastpca_out' must be a 'fastpca' object (or a list with elements ",
         "U, S and Vh), or a Seurat object / DimReduc.", call. = FALSE)
  }

  # a fastpca result is validated (and gains derived scores) before use
  if (inherits(fastpca_out, "fastpca")) {
    fastpca_out <- .validate_fastpca(fastpca_out)
  }

  if (!is.null(fastpca_out$scores)) {
    scores <- as.matrix(fastpca_out$scores)
  } else {
    if (is.null(fastpca_out$S)) {
      stop("'fastpca_out' must contain either 'scores', or both 'U' and 'S'.",
           call. = FALSE)
    }
    singular_values <- as.numeric(fastpca_out$S)
    # multiply the left singular vectors by the singular values
    scores <- fastpca_out$U %*% diag(singular_values, nrow = length(singular_values))
  }
  # set back the correct column names
  colnames(scores) <- gsub("^dim", "PC", colnames(scores))
  scores
}
