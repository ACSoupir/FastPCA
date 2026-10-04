# NIPALS: missing-value PCA ------------------------------------------------

small_missing <- function(n = 40, p = 10, na_prop = 0.2, seed = 7) {
  set.seed(seed)
  m <- matrix(rnorm(n * p), nrow = n, ncol = p,
              dimnames = list(paste0("s", seq_len(n)), paste0("g", seq_len(p))))
  m[sample(length(m), floor(na_prop * length(m)))] <- NA
  m
}

test_that("NIPALS with complete data matches the SVD", {
  set.seed(42)
  m <- matrix(rnorm(300), nrow = 30, ncol = 10,
              dimnames = list(paste0("s", seq_len(30)), paste0("g", seq_len(10))))
  res <- NIPALS(m, ncomp = 3L, backend = "r")

  expect_s3_class(res, "fastpca")
  expect_identical(attr(res, "method"), "nipals")
  centered <- sweep(m, 2L, colMeans(m), "-")
  expect_equal(as.numeric(res$S), svd(centered)$d[1:3], tolerance = 1e-4)

  # scores are always populated and equal U diag(S) for both solvers
  expect_equal(unname(res$scores), unname(res$U %*% diag(res$S)), tolerance = 1e-10)
  expect_equal(dim(get_pc_scores(res)), c(30L, 3L))
  expect_identical(colnames(get_pc_scores(res)), paste0("PC", 1:3))

  # complete data gives an orthogonal left basis
  expect_lt(max(abs(crossprod(res$U) - diag(3))), 1e-8)
})

test_that("NIPALS reproduces the reference nipals package on missing data", {
  skip_if_not_installed("nipals")
  m <- small_missing()
  ours <- NIPALS(m, ncomp = 3L, backend = "r")
  ref <- nipals::nipals(m, ncomp = 3L, center = TRUE, scale = FALSE)

  # singular values agree; the reference counts one extra iteration (documented
  # upstream off-by-one), so compare only the eigenvalues and vectors here.
  expect_equal(as.numeric(ours$S), as.numeric(ref$eig), tolerance = 1e-6)
  # signs are not identifiable: compare magnitude, after our sign normalisation
  expect_equal(abs(as.numeric(ours$Vh[1, ])), abs(as.numeric(ref$loadings[, 1])),
               tolerance = 1e-6)
  expect_equal(abs(as.numeric(ours$scores[, 2])), abs(as.numeric(ref$scores[, 2])) * ref$eig[2],
               tolerance = 1e-6)

  # the "nipals" backend is that package behind our object contract
  pkg <- NIPALS(m, ncomp = 3L, backend = "nipals")
  expect_identical(attr(pkg, "backend"), "nipals")
  expect_equal(as.numeric(pkg$S), as.numeric(ours$S), tolerance = 1e-6)
})

test_that("NIPALS normalises the sign of every component", {
  m <- small_missing(seed = 11)
  res <- NIPALS(m, ncomp = 3L, backend = "r")
  biggest <- apply(res$Vh, 1L, function(v) v[which.max(abs(v))])
  expect_true(all(biggest > 0))
})

test_that("entirely missing columns and rows are handled explicitly", {
  set.seed(3)
  m <- matrix(rnorm(200), nrow = 20, ncol = 10,
              dimnames = list(paste0("s", seq_len(20)), paste0("g", seq_len(10))))
  m[, "g4"] <- NA
  m["s2", ] <- NA

  expect_warning(res <- NIPALS(m, ncomp = 2L, backend = "r"), "no observed values")
  expect_true("g4" %in% attr(res, "dropped_features"))
  expect_true(all(is.na(res$Vh[, "g4"])))
  expect_false(anyNA(res$Vh[, -4]))
  # an all-NA observation keeps NA scores, others are finite
  expect_true(all(is.na(res$scores["s2", ])))
  expect_false(anyNA(res$scores[-2, , drop = FALSE]))
})

test_that("fitted() reconstructs and imputes a NIPALS result", {
  set.seed(9)
  U0 <- matrix(rnorm(30 * 3), nrow = 30)
  V0 <- matrix(rnorm(10 * 3), nrow = 10)
  full <- U0 %*% t(V0)

  # with complete data the reconstruction reproduces the input
  res_full <- NIPALS(full, ncomp = 3L, backend = "r")
  expect_lt(max(abs(fitted(res_full) - full)), 0.05)

  # with missing data the masked entries are filled from the components and the
  # observed ones are approximated better than predicting zero
  m <- full
  m[sample(length(m), 60)] <- NA
  res <- NIPALS(m, ncomp = 3L, backend = "r")
  hat <- fitted(res)
  expect_equal(dim(hat), dim(m))
  observed <- !is.na(m)
  missing <- is.na(m)
  expect_false(anyNA(hat))
  expect_lt(sqrt(mean((hat[observed] - m[observed])^2)), 0.3)
  expect_lt(sqrt(mean((hat[missing] - full[missing])^2)),
            sqrt(mean(full[observed]^2)))

  svd_res <- FastPCA(matrix(rnorm(60), nrow = 10, ncol = 6),
                     k = 2L, p = 1L, backend = "r", cores = 1L)
  expect_error(fitted(svd_res), "only available for NIPALS")
})

test_that("print() and summary() report the solver, missingness and iterations", {
  m <- small_missing(seed = 13)
  res <- NIPALS(m, ncomp = 2L, backend = "r")

  out <- utils::capture.output(print(res))
  expect_true(any(grepl("NIPALS", out)))
  expect_true(any(grepl("missing:", out)))

  smry <- summary(res)
  expect_equal(nrow(smry), 2L)
  expect_true(all(c("iterations", "converged") %in% names(smry)))
  expect_true(all(is.finite(smry$iterations)) && all(smry$converged))
  expect_identical(attr(smry, "method"), "nipals")

  # pve uses observed entries only and is finite
  expect_true(all(is.finite(pve(res))))
})

test_that("NIPALS argument validation gives actionable errors", {
  m <- matrix(rnorm(60), nrow = 10, ncol = 6)
  expect_error(NIPALS(m, ncomp = 7L, backend = "r"),
               "cannot exceed the smallest matrix dimension")
  expect_error(NIPALS(m, ncomp = 2L, tol = -1, backend = "r"), "tol")
  expect_error(NIPALS(m, ncomp = 2L, maxit = 0L, backend = "r"), "maxit")
  expect_error(NIPALS(m, ncomp = 2L, max_na_prop = 1.5, backend = "r"), "max_na_prop")
  expect_error(NIPALS(as.data.frame(m), ncomp = 2L, backend = "r"), "data frame")
  expect_error(NIPALS(m, ncomp = 2L, backend = "tinygrad"), "not implemented")
  expect_true("nipals" %in% eval(formals(NIPALS)$backend))
})

test_that("max_na_prop rejects columns that are too sparse", {
  set.seed(2)
  m <- matrix(rnorm(60), nrow = 10, ncol = 6)
  m[, 1] <- NA
  expect_error(NIPALS(m, ncomp = 2L, backend = "r", max_na_prop = 0.5),
               "max_na_prop")
})

test_that("zero-variance columns under scale = TRUE warn and use sd 1", {
  set.seed(4)
  m <- matrix(rnorm(60), nrow = 10, ncol = 6)
  m[, 3] <- 5
  expect_warning(res <- NIPALS(m, ncomp = 2L, scale = TRUE, backend = "r"),
                 "zero variance")
  expect_true(all(is.finite(res$S)))
})

test_that("the nipals package backend refuses unsupported shapes", {
  skip_if_not_installed("nipals")
  m <- small_missing(seed = 5)
  # the CRAN package cannot handle an all-NA row
  m[1, ] <- NA
  expect_error(suppressWarnings(NIPALS(m, ncomp = 2L, backend = "nipals")),
               "entirely missing")
})

test_that("the pytorch backend agrees with the base R implementation", {
  skip_if_not(reticulate::py_available(initialize = FALSE), "Python is not initialized")
  skip_if_not(reticulate::py_module_available("torch"), "torch is not available")

  m <- small_missing(seed = 17)
  ref <- NIPALS(m, ncomp = 3L, backend = "r")
  got <- suppressMessages(NIPALS(m, ncomp = 3L, backend = "pytorch", cores = 1L))

  expect_identical(attr(got, "backend"), "pytorch")
  expect_equal(as.numeric(got$S), as.numeric(ref$S), tolerance = 1e-5)
  expect_equal(unname(abs(got$Vh)), unname(abs(ref$Vh)), tolerance = 1e-5)
})
