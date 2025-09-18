# Tests for umap()

test_that("umap with method = 'uwot' runs without error", {
  set.seed(123)
  pc_scores <- matrix(rnorm(100*10), nrow = 100, ncol = 10)

  expect_no_error(umap(pc_scores, method = "uwot", seed = 123, cores = 1))
})

test_that("umap with method = 'uwot' returns correct dimensions", {
  set.seed(123)
  pc_scores <- matrix(rnorm(100*10), nrow = 100, ncol = 10)
  k <- 3

  umap_res <- umap(pc_scores, n_components = k, method = "uwot", seed = 123, cores = 1)

  expect_equal(nrow(umap_res), 100)
  expect_equal(ncol(umap_res), k)
})
