# Tests for prep_matrix()

test_that("prep_matrix works with backend = 'r'", {
  # Create a sample matrix
  mat <- matrix(1:9, nrow = 3, ncol = 3)

  # Test with default arguments
  prepped_mat <- prep_matrix(mat, backend = "r", log2 = FALSE, scale = FALSE)
  expect_equal(prepped_mat, mat)
})

test_that("prep_matrix with log2 = TRUE works", {
  mat <- matrix(2^c(1:9), nrow = 3, ncol = 3)
  prepped_mat <- prep_matrix(mat, backend = "r", log2 = TRUE, scale = FALSE, transpose = FALSE)
  expect_equal(prepped_mat, matrix(1:9, nrow = 3, ncol = 3))
})

test_that("prep_matrix with transpose = TRUE works", {
  mat <- matrix(1:9, nrow = 3, ncol = 3)
  prepped_mat <- prep_matrix(mat, backend = "r", log2 = FALSE, scale = FALSE, transpose = TRUE)
  expect_equal(prepped_mat, t(mat))
})

test_that("prep_matrix with scale = TRUE works", {
  mat <- matrix(rnorm(9), nrow = 3, ncol = 3)
  prepped_mat <- prep_matrix(mat, backend = "r", log2 = FALSE, scale = TRUE, transpose = FALSE)
  # 'scale' centers and scales the columns of a matrix
  expected_mat <- scale(mat, center = TRUE, scale = TRUE)
  expect_equal(prepped_mat, expected_mat)
})

test_that("prep_matrix with all options works", {
  mat <- matrix(2^c(1:9), nrow = 3, ncol = 3)
  prepped_mat <- prep_matrix(mat, backend = "r", log2 = TRUE, scale = TRUE, transpose = TRUE)

  log_mat <- matrix(1:9, nrow = 3, ncol = 3)
  transposed_mat <- t(log_mat)
  scaled_mat <- scale(transposed_mat)

  expect_equal(prepped_mat, scaled_mat)
})
