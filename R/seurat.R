# FastPCA <-> Seurat integration

.fastpca_null_default <- function(x, y) if (is.null(x)) y else x

# Peak-memory overhead applied to the raw dense size when handing a matrix to
# Python/torch: R copy + numpy conversion + torch tensor.
.fastpca_dense_overhead <- 3

# Best available estimate of the current process footprint in GB. Falls back to
# R's heap (which undercounts native/Python memory) when the OS value is blocked.
.fastpca_rss_gb <- function() {
  if (.Platform$OS.type == "unix") {
    rss <- suppressWarnings(tryCatch(
      system2("ps", c("-o", "rss=", "-p", Sys.getpid()),
              stdout = TRUE, stderr = TRUE),
      error = function(e) character()))
    value <- suppressWarnings(as.numeric(rss)[1])
    if (length(value) == 1L && !is.na(value)) return(value / 1024^2)

    status <- "/proc/self/status"
    if (file.exists(status)) {
      line <- grep("^VmRSS:", readLines(status, warn = FALSE), value = TRUE)
      if (length(line) > 0L) {
        kb <- suppressWarnings(as.numeric(sub("[^0-9]*([0-9]+).*", "\\1", line[1])))
        if (length(kb) == 1L && !is.na(kb)) return(kb / 1024^2)
      }
    }
  }
  round(sum(gc()[, 2]) / 1024, 3)
}

# Raw size (GB) of a dense double features x cells matrix.
.fastpca_dense_gb <- function(n_features, n_cells) {
  as.numeric(n_features) * as.numeric(n_cells) * 8 / 1024^3
}

#' Run FastPCA on a Seurat object
#'
#' Computes principal components with [FastPCA()] and stores them as a Seurat
#' code{DimReduc} so that downstream steps such as [Seurat::RunUMAP()] work
#' unchanged. The scaled data for the requested features is passed to FastPCA in
#' its native code{features x cells} orientation; the returned singular vectors
#' are converted into Seurat's code{cell embeddings}, code{feature loadings}
#' and code{stdev} conventions.
#'
#' @param object A Seurat object whose assay has a code{scale.data} layer for
#'   the requested features (see [Seurat::ScaleData()]).
#' @param assay Assay to use. Defaults to the object's default assay.
#' @param features Features used for the PCA. Defaults to
#'   code{VariableFeatures(object)}; if none are set, all assay features.
#' @param npcs Number of principal components to compute and store. Default 50.
#' @param p Oversampling parameter passed to [FastPCA()]. Default 10.
#' @param q_iter Number of power iterations passed to [FastPCA()]. Default 2.
#' @param reduction.name Name under which the reduction is stored. Default "pca".
#' @param backend,device Passed to [FastPCA()]. Default pytorch/CPU.
#' @param cores Number of CPU cores requested; automatically clamped to 1 when an
#'   OpenMP runtime conflict is detected (see code{?FastPCA}).
#' @param memory_limit Numeric GB ceiling for the call. Defaults to
#'   code{getOption("FastPCA.memory_limit", 20)}. The dense conversion cost is
#'   estimated up front and the call stops before exceeding the limit.
#' @param verbose Print progress messages. Default code{TRUE}.
#'
#' @return The Seurat object with the FastPCA reduction stored under
#'   code{reduction.name}.
#'
#' @export
#' @examples
#' \dontrun{
#'   obj <- Seurat::ScaleData(Seurat::FindVariableFeatures(obj))
#'   obj <- RunFastPCA(obj, npcs = 30)
#'   obj <- Seurat::RunUMAP(obj, dims = 1:30)
#' }
RunFastPCA <- function(object,
                       assay = NULL,
                       features = NULL,
                       npcs = 50L,
                       p = 10L,
                       q_iter = 2L,
                       reduction.name = "pca",
                       backend = c("pytorch", "rtorch", "r"),
                       device = c("CPU", "GPU"),
                       cores = 4L,
                       memory_limit = getOption("FastPCA.memory_limit", 20),
                       verbose = TRUE) {
  if (!inherits(object, "Seurat")) stop("`object` must be a Seurat object.", call. = FALSE)
  for (pkg in c("Seurat", "SeuratObject")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop("Package '", pkg, "' is required for RunFastPCA().", call. = FALSE)
    }
  }
  backend <- match.arg(backend)
  device <- match.arg(device)
  npcs <- as.integer(npcs)
  if (is.na(npcs) || npcs < 1L) stop("`npcs` must be a positive integer.", call. = FALSE)

  assay <- .fastpca_null_default(assay, SeuratObject::DefaultAssay(object))
  if (is.null(features)) features <- SeuratObject::VariableFeatures(object, assay = assay)
  if (length(features) == 0L) {
    features <- rownames(object[[assay]])
    if (verbose) message("No variable features set; using all ", length(features), " features.")
  }

  if (!"scale.data" %in% SeuratObject::Layers(object[[assay]])) {
    stop("Assay '", assay, "' has no 'scale.data' layer. ",
         "Run Seurat::ScaleData() before RunFastPCA().", call. = FALSE)
  }
  mat <- SeuratObject::LayerData(object[[assay]], layer = "scale.data")
  features <- intersect(features, rownames(mat))
  if (length(features) < npcs) {
    stop("Need at least `npcs` scaled features; found ", length(features), ".", call. = FALSE)
  }
  if (!identical(features, rownames(mat))) mat <- mat[features, , drop = FALSE]
  if (!is.matrix(mat)) mat <- as.matrix(mat)

  dense_gb <- .fastpca_dense_gb(nrow(mat), ncol(mat))
  current_gb <- .fastpca_rss_gb()
  if (!is.na(current_gb) && (current_gb + dense_gb * .fastpca_dense_overhead) > memory_limit) {
    stop(sprintf(paste0("RunFastPCA() would need about %.1f GB on top of the current ",
                        "%.1f GB, exceeding the %.0f GB memory limit. Reduce `features` ",
                        "or raise `memory_limit`."),
                 dense_gb * 3, current_gb, memory_limit), call. = FALSE)
  }
  if (verbose) {
    message(sprintf("FastPCA: %d features x %d cells (~%.2f GB dense, %.0f GB limit).",
                    nrow(mat), ncol(mat), dense_gb, memory_limit))
  }

  res <- FastPCA(mat,
                 k = npcs, p = as.integer(p), q_iter = as.integer(q_iter),
                 backend = backend, device = device, cores = cores)

  # mat rows are features: U holds feature loadings (features x k) and Vh is
  # components x cells, so the cell embeddings are t(Vh) scaled by singular values.
  loadings <- res$U
  emb <- t(res$Vh) %*% diag(res$S)
  stdev <- as.numeric(res$S) / sqrt(ncol(object) - 1)
  pc_names <- paste0("PC_", seq_len(length(res$S)))
  rownames(emb) <- colnames(object)
  rownames(loadings) <- features
  colnames(emb) <- pc_names
  colnames(loadings) <- pc_names

  dr <- SeuratObject::CreateDimReducObject(
    embeddings = emb, loadings = loadings, assay = assay,
    stdev = stdev, key = "PC_")
  object[[reduction.name]] <- dr

  if (verbose) {
    message("Stored FastPCA reduction '", reduction.name, "' with ", length(res$S), " PCs.")
  }
  object
}
