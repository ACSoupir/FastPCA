test_that("calculating PC scores", {
  suppressMessages(start_FastPCA_env())
  k = 5
  p = 10
  q_iter = 2
  X2 = suppressMessages(prep_matrix(X+1, log2 = TRUE, transpose = TRUE, scale = TRUE, cores = 1))
  res = suppressMessages(FastPCA(X2, k = k, p = p, q_iter = q_iter, cores = 1, backend = "pytorch"))
  #has to be the output of FastPCA and not prepmatrix
  expect_error(get_pc_scores(X2))
  expect_no_error(get_pc_scores(res))
  scores = get_pc_scores(res)
  expect_equal(dim(scores), c(10000, 5))
  expect_equal(unname(apply(scores, 2, mean)), rep(0, ncol(scores)))#should really be centered at 0
})
