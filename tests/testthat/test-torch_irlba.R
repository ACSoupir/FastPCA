test_that("torch_irlba validates its arguments before Python is needed", {
  m <- matrix(rnorm(40), nrow = 10, ncol = 4)
  dimnames(m) <- list(paste0("s", 1:10), paste0("g", 1:4))

  expect_error(torch_irlba(as.data.frame(m), nv = 2L),
               "must be a numeric matrix")
  expect_error(torch_irlba(m, nv = 0L), "greater than or equal to 1")
  expect_error(torch_irlba(m, nv = 4L), "strictly less than")
  expect_error(torch_irlba(m, nv = 2L, work = 1L), "greater than or equal to 2")
  expect_error(torch_irlba(m, nv = 2L, tol = 0), "greater than 0")
  expect_error(torch_irlba(m, nv = 2L, svtol = -1), "between")
  expect_error(torch_irlba(m, nv = 2L, precision = "half"),
               "'arg' should be one of")
  expect_error(torch_irlba(m, nv = 2L, device = "tpu"), "'arg' should be one of")
  expect_error(torch_irlba(m, nv = 2L, v = c(1, 2)), "starting vector")

  m_na <- m
  m_na[1, 1] <- NA_real_
  expect_error(torch_irlba(m_na, nv = 2L), "missing values")
})

test_that("torch_irlba agrees with irlba::irlba on a shared start vector", {
  skip_if_not(reticulate::py_available(initialize = FALSE),
              "Python is not initialized")
  skip_if_not(reticulate::py_module_available("torch"), "torch is not available")

  set.seed(11)
  m <- matrix(rnorm(200 * 12), nrow = 200, ncol = 12)
  v0 <- rnorm(ncol(m))

  ref <- irlba::irlba(m, nv = 3L, v = v0)
  got <- torch_irlba(m, nv = 3L, v = v0)

  expect_s3_class(got, "fastpca")
  expect_identical(attr(got, "backend"), "torch_irlba")
  expect_equal(as.numeric(got$S), as.numeric(ref$d), tolerance = 1e-8)

  # principal angles between the two subspaces
  cosines <- apply(crossprod(ref$u, got$U), 1L,
                   function(x) min(1, max(-1, abs(x))))
  expect_lt(max(abs(sin(acos(cosines)))), 1e-4)

  # a fixed start vector makes the solver reproducible
  again <- torch_irlba(m, nv = 3L, v = v0)
  expect_equal(got$S, again$S, tolerance = 1e-12)

  # result object carries the IRLBA bookkeeping and works with the accessors
  expect_true(is.finite(attr(got, "n_iter")))
  expect_gt(attr(got, "mprod"), attr(got, "n_iter"))
  expect_equal(dim(get_pc_scores(got)), c(200L, 3L))
  expect_output(print(got), "Lanczos SVD \\(IRLBA\\)")
  expect_output(print(got), "precision: double")
})

test_that("torch_irlba runs in single precision and reports it", {
  skip_if_not(reticulate::py_available(initialize = FALSE),
              "Python is not initialized")
  skip_if_not(reticulate::py_module_available("torch"), "torch is not available")

  set.seed(3)
  m <- matrix(rnorm(60 * 8), nrow = 60, ncol = 8)
  v0 <- rnorm(ncol(m))

  double_fit <- torch_irlba(m, nv = 2L, v = v0)
  single_fit <- torch_irlba(m, nv = 2L, v = v0, precision = "single")

  expect_identical(attr(double_fit, "precision"), "double")
  expect_identical(attr(single_fit, "precision"), "single")
  # float32 loses digits but tracks the leading subspace
  expect_equal(single_fit$S, double_fit$S, tolerance = 1e-4)
})
