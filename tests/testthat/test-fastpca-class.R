test_that("FastPCA returns a fastpca object with usable methods", {
  set.seed(42)
  m <- matrix(rnorm(20 * 15), nrow = 20, ncol = 15,
              dimnames = list(paste0("s", seq_len(20)), paste0("g", seq_len(15))))
  res <- FastPCA(m, k = 4, p = 2, backend = "r", cores = 1)

  expect_s3_class(res, "fastpca")
  expect_equal(dim(res$U), c(20L, 4L))
  expect_length(res$S, 4)
  expect_equal(dim(res$Vh), c(4L, 15L))
  expect_identical(rownames(res$U), rownames(m))
  expect_identical(colnames(res$Vh), colnames(m))

  # the Lanczos backend is near-exact
  expect_equal(res$S, svd(m)$d[1:4], tolerance = 1e-6)

  # percent variation explained is self-consistent and matches the exact SVD
  expected <- svd(m)$d[1:4]^2 / sum(m^2)
  expect_equal(unname(pve(res)), expected, tolerance = 1e-6)

  smry <- summary(res)
  expect_equal(nrow(smry), 4L)
  expect_true(all(is.finite(smry$cumulative_percent_variance)))
  expect_equal(tail(smry$cumulative_percent_variance, 1), sum(expected) * 100)

  scores <- get_pc_scores(res)
  expect_equal(dim(scores), c(20L, 4L))
  expect_identical(colnames(scores), paste0("PC", seq_len(4)))
  expect_identical(scores, scores(res))

  grDevices::pdf(tempfile())
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_silent(plot(res))
})

test_that("pve requires a stored or supplied total variance", {
  set.seed(1)
  m <- matrix(rnorm(300), nrow = 20, ncol = 15)
  res <- FastPCA(m, k = 2, p = 1, backend = "r", cores = 1,
                 store_variance = FALSE)
  expect_error(pve(res), "Total variance is not available")
  expect_equal(unname(pve(res, total = sum(m^2))),
               res$S^2 / sum(m^2), tolerance = 1e-12)
})

test_that("argument validation gives actionable errors", {
  m <- matrix(rnorm(300), nrow = 20, ncol = 15)
  expect_error(FastPCA(m, k = 2, p = 1, q_iter = 0, backend = "r"),
               "q_iter")
  expect_error(FastPCA(m, k = 100, p = 1, backend = "r"),
               "smallest matrix dimension")
  expect_error(FastPCA(m, k = 12, p = 4, backend = "r"), "k \\+ p")
  expect_error(FastPCA(as.data.frame(m), k = 2, p = 1, backend = "r"),
               "data frame")
  m_na <- m
  m_na[1, 1] <- NA_real_
  expect_error(FastPCA(m_na, k = 2, p = 1, backend = "r"), "missing values")
})

test_that("FastPCA dispatches on the resolved backend", {
  m <- matrix(rnorm(60), nrow = 10, ncol = 6)
  res <- with_mocked_bindings(
    FastPCA(m, k = 2, p = 1, backend = "rtorch", cores = 1),
    validate_backend = function(backend, device) {
      list(backend = "r", device = "CPU")
    },
    .package = "FastPCA"
  )
  expect_identical(attr(res, "backend"), "r")
})

test_that("prep_matrix restores dimnames correctly", {
  m <- matrix(abs(rnorm(40)) + 1, nrow = 8, ncol = 5,
              dimnames = list(paste0("f", seq_len(8)), paste0("c", seq_len(5))))

  same <- prep_matrix(m, log2 = TRUE, transpose = FALSE, scale = TRUE,
                      backend = "r", cores = 1)
  expect_identical(rownames(same), rownames(m))
  expect_identical(colnames(same), colnames(m))

  flipped <- prep_matrix(m, log2 = TRUE, transpose = TRUE, scale = TRUE,
                         backend = "r", cores = 1)
  expect_identical(rownames(flipped), colnames(m))
  expect_identical(colnames(flipped), rownames(m))

  # the helper used by the Python/torch branches must match that behaviour
  names_list <- list(paste0("f", seq_len(8)), paste0("c", seq_len(5)))
  not_flipped <- FastPCA:::.prep_dimnames(matrix(0, nrow = 8, ncol = 5),
                                          names_list, transpose = FALSE)
  expect_identical(rownames(not_flipped), paste0("f", seq_len(8)))
  expect_identical(colnames(not_flipped), paste0("c", seq_len(5)))
  is_flipped <- FastPCA:::.prep_dimnames(matrix(0, nrow = 5, ncol = 8),
                                         names_list, transpose = TRUE)
  expect_identical(rownames(is_flipped), paste0("c", seq_len(5)))
  expect_identical(colnames(is_flipped), paste0("f", seq_len(8)))
})
