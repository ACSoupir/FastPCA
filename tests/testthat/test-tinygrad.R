test_that("tinygrad is an accepted backend for FastPCA and prep_matrix", {
  expect_true("tinygrad" %in% eval(formals(FastPCA)$backend))
  expect_true("tinygrad" %in% eval(formals(prep_matrix)$backend))
})

test_that("setup_py_env resolves the backend choice", {
  expect_identical(.resolve_setup_backend(), "torch")          # default
  expect_identical(.resolve_setup_backend("pytorch"), "torch")
  expect_identical(.resolve_setup_backend("tinygrad"), "tinygrad")
  expect_identical(.resolve_setup_backend("all"), c("torch", "tinygrad"))
  expect_error(.resolve_setup_backend("not-a-backend"))
})

test_that("prep_matrix tinygrad transform agrees with the r backend", {
  skip_if_not(reticulate::py_available(initialize = FALSE), "Python is not initialized")
  skip_if_not(reticulate::py_module_available("tinygrad"), "tinygrad is not installed")

  set.seed(11)
  M <- matrix(abs(rnorm(400 * 120)) + 1, nrow = 200)

  for (tp in c(FALSE, TRUE)) {
    ref <- prep_matrix(M, log2 = TRUE, transpose = tp, scale = TRUE, backend = "r")
    got <- suppressMessages(
      prep_matrix(M, log2 = TRUE, transpose = tp, scale = TRUE,
                  backend = "tinygrad", device = "GPU")
    )
    expect_equal(dim(got), dim(ref))
    # float32 transform, so compare on a relative scale
    expect_lt(max(abs(got - ref)) / max(abs(ref)), 1e-4)
  }
})

test_that("FastPCA tinygrad backend agrees with the r backend", {
  skip_if_not(reticulate::py_available(initialize = FALSE), "Python is not initialized")
  skip_if_not(reticulate::py_module_available("tinygrad"), "tinygrad is not installed")

  # a low-rank matrix with a decaying spectrum, as in real PCA data. A flat /
  # near-degenerate spectrum is unstable for *any* randomized SVD (the torch
  # baseline lands ~65 degrees off there too), so it is not a fair comparison.
  set.seed(7)
  m <- 80; n <- 300
  U0 <- qr.Q(qr(matrix(rnorm(m * 10), m)))
  V0 <- qr.Q(qr(matrix(rnorm(n * 10), n)))
  A <- U0 %*% diag(100 * 0.7^(0:9)) %*% t(V0) + matrix(rnorm(m * n), m) * 0.05

  res_r <- FastPCA(A, k = 5, p = 10, q_iter = 2, cores = 1, backend = "r")
  res_tg <- suppressMessages(
    FastPCA(A, k = 5, p = 10, q_iter = 2, cores = 1,
            backend = "tinygrad", device = "GPU")
  )

  expect_equal(dim(res_tg$U), dim(res_r$U))
  expect_equal(length(res_tg$S), length(res_r$S))
  expect_true(all(is.finite(res_tg$U)) && all(is.finite(res_tg$S)))
  expect_true(all(res_tg$S >= 0))

  # the leading subspace must agree to well under a degree (float32 range finder)
  cosang <- svd(crossprod(res_r$U, res_tg$U))$d
  expect_gt(min(cosang), cos(0.25 * pi / 180))
})
