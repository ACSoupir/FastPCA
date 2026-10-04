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

test_that("RunNIPALS stores a Seurat-compatible reduction", {
  obj <- make_scaled_seurat()
  out <- suppressWarnings(RunNIPALS(obj, npcs = 5L, backend = "r", cores = 1L,
                                    verbose = FALSE))
  dr <- out[["nipals"]]
  emb <- SeuratObject::Embeddings(dr)
  loadings <- SeuratObject::Loadings(dr)

  expect_s4_class(dr, "DimReduc")
  # NIPALS is run with cells as observations, so the native scores are embeddings
  expect_equal(dim(emb), c(ncol(obj), 5L))
  expect_true(all(grepl("^NP_", colnames(emb))))
  expect_equal(nrow(loadings), length(SeuratObject::VariableFeatures(obj)))
  expect_length(SeuratObject::Stdev(dr), 5L)
})

test_that("RunNIPALS accepts the tinygrad backend", {
  skip_if_not(reticulate::py_available(initialize = FALSE), "Python is not initialized")
  skip_if_not(reticulate::py_module_available("tinygrad"), "tinygrad is not available")

  obj <- make_scaled_seurat()
  out <- suppressWarnings(RunNIPALS(obj, npcs = 3L, backend = "tinygrad",
                                    cores = 1L, verbose = FALSE))
  dr <- out[["nipals"]]
  expect_equal(dim(SeuratObject::Embeddings(dr)), c(ncol(obj), 3L))
  expect_length(SeuratObject::Stdev(dr), 3L)
})

test_that("RunNIPALS coexists with a RunFastPCA reduction", {
  obj <- make_scaled_seurat()
  out <- suppressWarnings(RunFastPCA(obj, npcs = 5L, p = 2L, q_iter = 1L,
                                     backend = "r", cores = 1L, verbose = FALSE))
  out <- suppressWarnings(RunNIPALS(out, npcs = 4L, backend = "r", cores = 1L,
                                    verbose = FALSE))
  expect_true(all(c("pca", "nipals") %in% SeuratObject::Reductions(out)))
})

test_that("RunNIPALS requires a Seurat object and scaled data", {
  expect_error(RunNIPALS(matrix(1:4, nrow = 2), backend = "r", verbose = FALSE),
               "must be a Seurat object")

  counts <- matrix(rpois(10 * 20, lambda = 5), nrow = 10)
  rownames(counts) <- paste0("g", seq_len(nrow(counts)))
  colnames(counts) <- paste0("c", seq_len(ncol(counts)))
  obj <- Seurat::CreateSeuratObject(counts = counts)
  expect_error(RunNIPALS(obj, npcs = 3L, backend = "r", verbose = FALSE),
               "scale.data")
})

test_that("RunNIPALS enforces the memory ceiling", {
  obj <- make_scaled_seurat()
  expect_error(RunNIPALS(obj, npcs = 5L, backend = "r",
                         memory_limit = 1e-6, verbose = FALSE),
               "memory limit")
})
