test_that("tinygrad is an accepted FastPCA backend", {
  expect_true("tinygrad" %in% eval(formals(FastPCA)$backend))
})

test_that("setup_py_env resolves the backend choice", {
  expect_identical(.resolve_setup_backend(), "torch")          # default
  expect_identical(.resolve_setup_backend("pytorch"), "torch")
  expect_identical(.resolve_setup_backend("tinygrad"), "tinygrad")
  expect_identical(.resolve_setup_backend("all"), c("torch", "tinygrad"))
  expect_error(.resolve_setup_backend("not-a-backend"))
})

test_that("prep_matrix falls back from tinygrad to pytorch", {
  # the transform side is not implemented for tinygrad yet, so it warns and
  # falls back rather than erroring in match.arg()
  expect_warning(.resolve_transform_backend("tinygrad"), "does not implement")
  expect_identical(suppressWarnings(.resolve_transform_backend("tinygrad")), "pytorch")
  expect_identical(suppressWarnings(.resolve_transform_backend("r")), "r")
  expect_identical(suppressWarnings(.resolve_transform_backend("pytorch")), "pytorch")
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
