# Tests for get_pc_scores()

test_that("get_pc_scores works correctly", {
  # Create a sample FastPCA output
  U <- matrix(rnorm(10*3), nrow = 10, ncol = 3)
  colnames(U) <- paste0("PC", 1:3)
  S <- 1:3
  fastpca_out <- list(U = U, S = S)

  # Calculate scores
  scores <- get_pc_scores(fastpca_out)

  # Expected result
  expected_scores <- U %*% diag(S)
  colnames(expected_scores) <- colnames(U)

  # Compare results
  expect_equal(scores, expected_scores)

  # Check dimensions and names
  expect_equal(nrow(scores), 10)
  expect_equal(ncol(scores), 3)
  expect_equal(colnames(scores), paste0("PC", 1:3))
})
