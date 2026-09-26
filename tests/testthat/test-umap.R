test_that("UMAP tests", {
  suppressMessages(start_FastPCA_env())
  k = 5
  p = 10
  q_iter = 2
  X2 = suppressMessages(prep_matrix(X+1, log2 = TRUE, transpose = TRUE, scale = TRUE, cores = 1))
  res = suppressMessages(FastPCA(X2, k = k, p = p, q_iter = q_iter, cores = 1, backend = "pytorch"))
  #get the scores from the 5 dimensions
  scores = get_pc_scores(res)
  #needs to be scores output
  expect_error(umap(res))
  expect_no_error(umap(scores))
  #not an included method
  expect_error(umap(scores, method = "UMAP"))
})
