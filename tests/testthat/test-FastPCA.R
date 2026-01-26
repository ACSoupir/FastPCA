test_that("FastPCA", {
  k = 5
  p = 10
  q_iter = 2

  X2 = prep_matrix(X+1, log2 = TRUE, transpose = TRUE, scale = TRUE, cores = 1)

  expect_no_error(FastPCA(X2, k = k, p = p, q_iter = q_iter, cores = 1, backend = "r"))
  #need a matrix
  expect_error(FastPCA(as.data.frame(X2), k = k, p = p, q_iter = q_iter, cores = 1, backend = "r"))
  expect_error(FastPCA(X2, k = k, p = p, q_iter = q_iter, cores = 1, backend = "r", device = "something")) #needs to be CPU/GPU

  #no environment set
  expect_error(FastPCA(X2, k = k, p = p, q_iter = q_iter, cores = 1, backend = "pytorch"))
  #

  #start the environment
  suppressMessages(start_FastPCA_env())
  expect_error(FastPCA(as.data.frame(X2), k = k, p = p, q_iter = q_iter, cores = 1, backend = "pytorch"))
  expect_no_error(suppressMessages(FastPCA(X2, k = k, p = p, q_iter = q_iter, cores = 1, backend = "pytorch")))

  res = suppressMessages(FastPCA(X2, k = k, p = p, q_iter = q_iter, cores = 1, backend = "pytorch"))
  expect_equal(dim(res$U), c(10000, 5))
  expect_equal(dim(res$S), c(5))
  expect_equal(dim(res$Vh), c(5, 200))

  #running rtorch after pytorch - avoid collisions
  expect_error(FastPCA(X2, k = k, p = p, q_iter = q_iter, cores = 1, backend = "rtorch"))
})
