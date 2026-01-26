test_that("prep matrix", {
  #should work normal
  expect_no_error(prep_matrix(X+1, log2 = TRUE, transpose = TRUE, scale = TRUE, cores = 1))
  expect_no_error(prep_matrix(X+1, log2 = FALSE, transpose = FALSE, scale = FALSE, cores = 1))
  expect_no_error(prep_matrix(X, log2 = FALSE, transpose = FALSE, scale = FALSE, cores = 1))
  #prep matrix doesn't accept tinygrad yet
  expect_error(prep_matrix(X, log2 = FALSE, transpose = FALSE, scale = FALSE, cores = 1, backend = 'tinygrad'))
  expect_warning(suppressMessages(prep_matrix(X, log2 = FALSE, transpose = FALSE, scale = FALSE, cores = 1, device = 'GPU')))
  expect_error(prep_matrix(X, log2 = FALSE, transpose = FALSE, scale = FALSE, cores = 1, device = 'GL'))

  #check against the parameters for simulating
  res = prep_matrix(X+1, log2 = TRUE, transpose = TRUE, scale = TRUE, cores = 1)
  expect_equal(dim(res), c(n_samples, n_features))
  #don't transpose should be opposite
  res2 = prep_matrix(X+1, log2 = TRUE, transpose = FALSE, scale = TRUE, cores = 1)
  expect_equal(dim(res2), c(n_features, n_samples))


  #now python
  expect_error(prep_matrix(X+1, log2 = TRUE, transpose = TRUE, scale = TRUE, cores = 1, backend = "pytorch"))
  suppressMessages(start_FastPCA_env())
  expect_identical(class(prep_matrix(X+1, log2 = TRUE, transpose = TRUE, scale = TRUE, cores = 1, backend = "pytorch")), c("matrix", "array"))
})
