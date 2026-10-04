# NIPALS <-> Seurat integration

# Peak-memory overhead applied to the raw dense size for NIPALS: R copy + mask +
# working matrix handed to Python/torch. One more than the SVD path because
# NIPALS also holds an observation mask.
.fastpca_nipals_dense_overhead <- 4

#' Run NIPALS on a Seurat object
#'
#' Computes principal components with [NIPALS()] -- which tolerates \code{NA},
#' as produced by scaling data with missing quantifications -- and stores them
#' as a Seurat \code{DimReduc}, so downstream steps such as
#' [Seurat::RunUMAP()] work unchanged.
#'
#' The scaled data is passed to NIPALS with cells as rows (the observations), so
#' the native score vectors are the cell embeddings and \code{Vh} carries the
#' feature loadings. This is why it returns cell embeddings directly rather than
#' deriving them from the singular vectors as [RunFastPCA()] does.
#'
#' @param object A Seurat object whose assay has a \code{scale.data} layer for
#'   the requested features (see [Seurat::ScaleData()]). Missing values in that
#'   layer are allowed.
#' @param assay Assay to use. Defaults to the object's default assay.
#' @param features Features used for the PCA. Defaults to
#'   \code{VariableFeatures(object)}; if none are set, all assay features.
#' @param npcs Number of principal components to compute and store. Default 50.
#' @param reduction.name Name under which the reduction is stored. Default
#'   \code{"nipals"}, so it can coexist with a [RunFastPCA()] reduction named
#'   \code{"pca"}.
#' @param reduction.key Key used for the DimReduc columns. Default \code{"NP_"}.
#' @param backend,device Passed to [NIPALS()]. The default is the base R
#'   backend on CPU, which needs no Python.
#' @param cores Number of CPU threads requested; automatically clamped to 1 when
#'   an OpenMP runtime conflict is detected (see [FastPCA()]).
#' @param tol,maxit Convergence tolerance and per-component iteration cap passed
#'   to [NIPALS()].
#' @param memory_limit Numeric GB ceiling for the call. Defaults to
#'   \code{getOption("FastPCA.memory_limit", 20)}. The dense conversion cost is
#'   estimated up front and the call stops before exceeding the limit.
#' @param verbose Print progress messages. Default \code{TRUE}.
#'
#' @return The Seurat object with the NIPALS reduction stored under
#'   \code{reduction.name}.
#'
#' @seealso [NIPALS()], [RunFastPCA()]
#' @export
#' @examples
#' \dontrun{
#'   obj <- Seurat::ScaleData(Seurat::FindVariableFeatures(obj))
#'   obj <- RunNIPALS(obj, npcs = 30)
#'   obj <- Seurat::RunUMAP(obj, dims = 1:30, reduction = "nipals")
#' }
RunNIPALS <- function(object,
                      assay = NULL,
                      features = NULL,
                      npcs = 50L,
                      reduction.name = "nipals",
                      reduction.key = "NP_",
                      backend = c("r", "rtorch", "pytorch"),
                      device = c("CPU", "GPU"),
                      cores = 1L,
                      tol = 1e-6,
                      maxit = 500L,
                      memory_limit = getOption("FastPCA.memory_limit", 20),
                      verbose = TRUE) {
  if (!inherits(object, "Seurat")) stop("'object' must be a Seurat object.", call. = FALSE)
  for (pkg in c("Seurat", "SeuratObject")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop("Package '", pkg, "' is required for RunNIPALS().", call. = FALSE)
    }
  }
  backend <- match.arg(backend)
  device <- match.arg(device)
  npcs <- as.integer(npcs)
  if (is.na(npcs) || npcs < 1L) stop("'npcs' must be a positive integer.", call. = FALSE)

  assay <- .fastpca_null_default(assay, SeuratObject::DefaultAssay(object))
  if (is.null(features)) features <- SeuratObject::VariableFeatures(object, assay = assay)
  if (length(features) == 0L) {
    features <- rownames(object[[assay]])
    if (verbose) message("No variable features set; using all ", length(features), " features.")
  }

  if (!"scale.data" %in% SeuratObject::Layers(object[[assay]])) {
    stop("Assay '", assay, "' has no 'scale.data' layer. ",
         "Run Seurat::ScaleData() before RunNIPALS().", call. = FALSE)
  }
  mat <- SeuratObject::LayerData(object[[assay]], layer = "scale.data")
  features <- intersect(features, rownames(mat))
  if (length(features) < npcs) {
    stop("Need at least 'npcs' scaled features; found ", length(features), ".", call. = FALSE)
  }
  if (!identical(features, rownames(mat))) mat <- mat[features, , drop = FALSE]
  if (!is.matrix(mat)) mat <- as.matrix(mat)

  dense_gb <- .fastpca_dense_gb(nrow(mat), ncol(mat))
  current_gb <- .fastpca_rss_gb()
  if (!is.na(current_gb) &&
      (current_gb + dense_gb * .fastpca_nipals_dense_overhead) > memory_limit) {
    stop(sprintf(paste0("RunNIPALS() would need about %.1f GB on top of the current ",
                        "%.1f GB, exceeding the %.0f GB memory limit. Reduce 'features' ",
                        "or raise 'memory_limit'."),
                 dense_gb * .fastpca_nipals_dense_overhead, current_gb, memory_limit),
         call. = FALSE)
  }
  if (verbose) {
    message(sprintf("NIPALS: %d features x %d cells (~%.2f GB dense, %.0f GB limit).",
                    nrow(mat), ncol(mat), dense_gb, memory_limit))
  }

  # NIPALS decomposes observations by variables: pass cells as rows so the score
  # vectors are cell embeddings and Vh holds feature loadings.
  res <- NIPALS(t(mat), ncomp = npcs, tol = tol, maxit = as.integer(maxit),
                backend = backend, device = device, cores = cores,
                verbose = FALSE)

  embeddings <- res$scores
  loadings <- t(res$Vh)
  # guard against a single cell, where the sample standard deviation is undefined
  stdev <- as.numeric(res$S) / sqrt(max(ncol(object) - 1, 1))
  pc_names <- paste0("PC_", seq_len(length(res$S)))
  rownames(embeddings) <- colnames(object)
  rownames(loadings) <- features
  colnames(embeddings) <- pc_names
  colnames(loadings) <- pc_names

  dr <- SeuratObject::CreateDimReducObject(
    embeddings = embeddings, loadings = loadings, assay = assay,
    stdev = stdev, key = reduction.key)
  object[[reduction.name]] <- dr

  if (verbose) {
    message("Stored NIPALS reduction '", reduction.name, "' with ", length(res$S), " PCs.")
  }
  object
}
