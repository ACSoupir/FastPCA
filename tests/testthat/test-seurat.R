skip_if_not_installed("Seurat")
skip_if_not_installed("SeuratObject")

make_scaled_seurat <- function(n_features = 30, n_cells = 40) {
  set.seed(1)
  counts <- matrix(rpois(n_features * n_cells, lambda = 5),
                   nrow = n_features, ncol = n_cells)
  rownames(counts) <- paste0("g", seq_len(n_features))
  colnames(counts) <- paste0("c", seq_len(n_cells))
  obj <- Seurat::CreateSeuratObject(counts = counts)
  obj <- suppressWarnings(Seurat::NormalizeData(obj, verbose = FALSE))
  obj <- Seurat::FindVariableFeatures(obj, nfeatures = 20, verbose = FALSE)
  obj <- suppressWarnings(Seurat::ScaleData(
    obj, features = SeuratObject::VariableFeatures(obj), verbose = FALSE))
  obj
}

test_that("RunFastPCA stores a Seurat-compatible PCA reduction", {
  obj <- make_scaled_seurat()
  out <- suppressWarnings(RunFastPCA(obj, npcs = 5L, p = 2L, q_iter = 1L,
                                     backend = "r", cores = 1L, verbose = FALSE))
  dr <- out[["pca"]]
  emb <- SeuratObject::Embeddings(dr)
  loadings <- SeuratObject::Loadings(dr)
  expect_s4_class(dr, "DimReduc")
  expect_equal(dim(emb), c(ncol(obj), 5L))
  expect_true(all(grepl("^PC_", colnames(emb))))
  expect_equal(nrow(loadings), length(SeuratObject::VariableFeatures(obj)))
  expect_length(SeuratObject::Stdev(dr), 5L)
})

test_that("RunFastPCA stores under a custom reduction name", {
  obj <- make_scaled_seurat()
  out <- RunFastPCA(obj, npcs = 4L, backend = "r", cores = 1L,
                    reduction.name = "fastpca", verbose = FALSE)
  expect_true("fastpca" %in% SeuratObject::Reductions(out))
})

test_that("RunFastPCA requires a Seurat object", {
  expect_error(RunFastPCA(matrix(1:4, nrow = 2), backend = "r", verbose = FALSE),
               "must be a Seurat object")
})

test_that("RunFastPCA requires scaled data", {
  counts <- matrix(rpois(10 * 20, lambda = 5), nrow = 10)
  rownames(counts) <- paste0("g", seq_len(nrow(counts)))
  colnames(counts) <- paste0("c", seq_len(ncol(counts)))
  obj <- Seurat::CreateSeuratObject(counts = counts)
  expect_error(RunFastPCA(obj, npcs = 3L, backend = "r", verbose = FALSE),
               "scale.data")
})

test_that("RunFastPCA enforces the memory ceiling", {
  obj <- make_scaled_seurat()
  expect_error(RunFastPCA(obj, npcs = 5L, backend = "r",
                          memory_limit = 1e-6, verbose = FALSE),
               "memory limit")
})

test_that(".fastpca_dense_gb reports the raw dense size", {
  expect_equal(.fastpca_dense_gb(1000, 1e6), 1000 * 1e6 * 8 / 1024^3)
})
