# Tests for FastPCA()

test_that("FastPCA works with backend = 'irlba'", {
  # Create a sample matrix
  set.seed(123)
  mat <- matrix(rnorm(100*20), nrow = 100, ncol = 20)
  rownames(mat) <- paste0("Sample", 1:100)
  colnames(mat) <- paste0("Feature", 1:20)

  # Run FastPCA with irlba backend
  k <- 5
  fast_pca_res <- FastPCA(mat, k = k, backend = "irlba")

  # Run irlba directly
  irlba_res <- irlba::irlba(mat, nv = k, work = k + 10)

  # Compare singular values (these should be very close)
  expect_equal(fast_pca_res$S, irlba_res$d)

  # Compare singular vectors by checking for high correlation
  # This is robust to sign flips
  for (i in 1:k) {
    u_cor <- cor(fast_pca_res$U[, i], irlba_res$u[, i])
    expect_true(abs(u_cor) > 0.9999)

    v_cor <- cor(fast_pca_res$Vh[i, ], irlba_res$v[, i])
    expect_true(abs(v_cor) > 0.9999)
  }
})

test_that("FastPCA output dimensions are correct", {
  set.seed(123)
  mat <- matrix(rnorm(100*20), nrow = 100, ncol = 20)
  k <- 5
  fast_pca_res <- FastPCA(mat, k = k, backend = "irlba")

  expect_equal(nrow(fast_pca_res$U), 100)
  expect_equal(ncol(fast_pca_res$U), k)

  expect_equal(length(fast_pca_res$S), k)

  expect_equal(nrow(fast_pca_res$Vh), k)
  expect_equal(ncol(fast_pca_res$Vh), 20)
})

test_that("FastPCA output names are correct", {
  set.seed(123)
  mat <- matrix(rnorm(100*20), nrow = 100, ncol = 20)
  rownames(mat) <- paste0("Sample", 1:100)
  colnames(mat) <- paste0("Feature", 1:20)
  k <- 5
  fast_pca_res <- FastPCA(mat, k = k, backend = "irlba")

  expect_equal(rownames(fast_pca_res$U), rownames(mat))
  expect_equal(colnames(fast_pca_res$U), paste0("PC", 1:k))

  expect_equal(rownames(fast_pca_res$Vh), paste0("PC", 1:k))
  expect_equal(colnames(fast_pca_res$Vh), colnames(mat))
})

test_that("FastPCA passes extra arguments to irlba", {
  set.seed(123)
  mat <- matrix(rnorm(100*20), nrow = 100, ncol = 20)
  k <- 5

  # Use 'tol' as an extra argument
  fast_pca_res_tol <- FastPCA(mat, k = k, backend = "irlba", tol = 1e-4)
  irlba_res_tol <- irlba::irlba(mat, nv = k, tol = 1e-4)

  expect_equal(fast_pca_res_tol$S, irlba_res_tol$d)
})
