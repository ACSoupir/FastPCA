test_that("IncrementalPCA matches batch PCA when the statistics are exact", {
  fixture <- incremental_fixture()
  reference <- batch_reference(fixture$X, k = 4L)

  fit <- IncrementalPCA(fixture$files, ncomp = 4L)
  expect_s3_class(fit, "fastpca")
  expect_identical(attr(fit, "method"), "incremental")

  # One streaming pass centres each block with the mean of the preceding blocks,
  # which is what makes it approximate: no buffer size removes that, only a
  # statistics pass does. The running mean itself is exact (checked below).
  expect_equal(as.numeric(fit$S), reference$S, tolerance = 1e-2)
  expect_gt(subspace_min_cosine(fit$Vh, reference$Vh), 0.9999)
  expect_equal(as.numeric(attr(fit, "center")), unname(colMeans(fixture$X)),
               tolerance = 1e-12)
})

test_that("exact statistics (scale) and the oversampling buffer sharpen agreement", {
  fixture <- incremental_fixture(n_features = 12L, spectrum = c(25, 11, 5, 2.5))
  reference <- batch_reference(fixture$X, k = 3L, scale_columns = TRUE)

  # a statistics pass fixes the mean and SD before any block is decomposed, so
  # every block sees identical pre-processing
  exact <- IncrementalPCA(fixture$files, ncomp = 3L, scale = TRUE)
  expect_equal(as.numeric(exact$S), reference$S, tolerance = 1e-6)
  expect_gt(subspace_min_cosine(exact$Vh, reference$Vh), 0.99999)
  expect_equal(as.numeric(attr(exact, "scaling")), unname(apply(fixture$X, 2L, sd)),
               tolerance = 1e-10)

  # with a buffer at least as wide as the rank, no block tail is discarded and
  # the streaming fit reproduces batch PCA exactly
  buffered <- IncrementalPCA(fixture$files, ncomp = 3L, oversample = 20L,
                             scale = TRUE)
  expect_equal(as.numeric(buffered$S), reference$S, tolerance = 1e-10)
  expect_gt(subspace_min_cosine(buffered$Vh, reference$Vh), 1 - 1e-10)
})

test_that("chunk_rows streams a file in blocks without changing the answer", {
  fixture <- incremental_fixture()
  whole <- IncrementalPCA(fixture$files, ncomp = 4L, scale = TRUE)
  blocked <- IncrementalPCA(fixture$files, ncomp = 4L, scale = TRUE,
                            chunk_rows = 17L)

  expect_equal(blocked$S, whole$S)
  expect_equal(unname(blocked$Vh), unname(whole$Vh))
  # the same blocks are reported, so the reader really did split the files
  expect_equal(attr(blocked, "n_blocks") > attr(whole, "n_blocks"), TRUE)

  # a single source split across blocks keeps its feature names
  expect_equal(colnames(blocked$Vh), colnames(fixture$X))
})

test_that("scores are stored on request and match the components", {
  fixture <- incremental_fixture(row.names = TRUE)
  fit <- IncrementalPCA(fixture$files, ncomp = 4L, row.names = 1,
                        scale = TRUE, store_scores = TRUE)

  scores <- get_pc_scores(fit)
  expect_equal(dim(scores), c(nrow(fixture$X), 4L))
  expect_identical(rownames(scores)[1], "s01_0001")
  expect_identical(colnames(scores), paste0("PC", 1:4))

  scaled <- batch_reference(fixture$X, k = 4L, scale_columns = TRUE)$centered
  expect_equal(unname(scores), unname(scaled %*% t(fit$Vh)), tolerance = 1e-8)
  # the documented identity scores == U %*% diag(S) holds for this method too
  expect_equal(unname(scores), unname(fit$U %*% diag(fit$S, 4L)),
               tolerance = 1e-10)

  # without the scoring pass neither scores nor U exist, and asking is actionable
  lean <- IncrementalPCA(fixture$files, ncomp = 4L, row.names = 1)
  expect_null(lean$scores)
  expect_null(lean$U)
  expect_error(get_pc_scores(lean), "store_scores = TRUE")
})

test_that("incremental results print, summarise and report variation explained", {
  fixture <- incremental_fixture()
  fit <- IncrementalPCA(fixture$files, ncomp = 3L)

  expect_output(print(fit), "incremental PCA")
  expect_output(print(fit), "out-of-core")

  summary_table <- summary(fit)
  expect_equal(nrow(summary_table), 3L)
  expect_identical(attr(summary_table, "method"), "incremental")

  pct <- pve(fit)
  expect_equal(length(pct), 3L)
  expect_true(all(is.finite(pct)))
  # pve() is relative to the streamed total sum of squares
  expected_total <- sum((fixture$X - rep(colMeans(fixture$X), each = nrow(fixture$X)))^2)
  expect_equal(attr(fit, "total_variance"), expected_total, tolerance = 1e-8)
  expect_equal(sum(fit$S^2), sum(pct) * expected_total, tolerance = 1e-6)

  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_silent(graphics::plot(fit, type = "cumulative"))
})

test_that("reading is flexible: header row, ID column and compression", {
  fixture <- incremental_fixture(row.names = TRUE)

  by_index <- IncrementalPCA(fixture$files, ncomp = 3L, row.names = 1)
  by_name <- IncrementalPCA(fixture$files, ncomp = 3L, row.names = "id")
  expect_equal(by_name$S, by_index$S)

  # gzip is read through a base R connection
  gz <- file.path(fixture$dir, sprintf("part%02d.csv.gz", seq_along(fixture$parts)))
  for (i in seq_along(gz)) {
    connection <- gzfile(gz[i], "w")
    utils::write.csv(fixture$parts[[i]], connection, row.names = FALSE)
    close(connection)
  }
  # the plain fixture files carry an id column, so read them by identifier
  expect_equal(IncrementalPCA(gz, ncomp = 3L)$S,
               IncrementalPCA(fixture$files, ncomp = 3L, row.names = "id")$S)

  # headerless files are positional
  bare <- file.path(fixture$dir, sprintf("bare%02d.csv", seq_along(fixture$parts)))
  for (i in seq_along(bare)) {
    utils::write.table(fixture$parts[[i]], bare[i], sep = ",", row.names = FALSE,
                       col.names = FALSE)
  }
  bare_fit <- IncrementalPCA(bare, ncomp = 3L, header = FALSE)
  expect_equal(unname(bare_fit$S), unname(by_index$S[1:3]), tolerance = 1e-6)

  # a user-supplied reader is the extension point for other containers; it
  # returns the file as a numeric matrix and puts identifiers in row names
  custom <- IncrementalPCA(fixture$files, ncomp = 3L,
                           reader = function(path) {
                             frame <- utils::read.csv(path, check.names = FALSE)
                             rownames(frame) <- frame$id
                             as.matrix(frame[, setdiff(names(frame), "id")])
                           })
  expect_equal(custom$S, by_index$S[1:3], tolerance = 1e-8)
})

test_that("chunk files must share their features and stay numeric", {
  fixture <- incremental_fixture()
  mismatched <- file.path(fixture$dir, "renamed.csv")
  renamed <- fixture$parts[[1]]
  colnames(renamed) <- paste0("g", seq_len(ncol(renamed)))
  utils::write.csv(renamed, mismatched, row.names = FALSE)

  expect_error(IncrementalPCA(c(fixture$files[1], mismatched), ncomp = 2L),
               "different feature set")
  expect_error(IncrementalPCA(c(fixture$files[1],
                                file.path(fixture$dir, "missing.csv")), ncomp = 2L),
               "not found")

  # a non-numeric column is reported with the row.names hint
  labelled <- file.path(fixture$dir, "labelled.csv")
  frame <- data.frame(id = paste0("r", seq_len(nrow(fixture$parts[[1]]))),
                      fixture$parts[[1]], check.names = FALSE)
  utils::write.csv(frame, labelled, row.names = FALSE)
  expect_error(IncrementalPCA(labelled, ncomp = 2L), "row.names")

  # unsupported containers point at the reader argument
  parquet <- file.path(fixture$dir, "data.parquet")
  writeLines("not really parquet", parquet)
  expect_error(IncrementalPCA(parquet, ncomp = 2L), "Unsupported chunk file type")
})

test_that("missing and non-finite values are rejected with a NIPALS pointer", {
  fixture <- incremental_fixture()
  with_na <- file.path(fixture$dir, "with-na.csv")
  broken <- fixture$parts[[1]]
  broken[2, 3] <- NA_real_
  utils::write.csv(broken, with_na, row.names = FALSE)
  expect_error(IncrementalPCA(c(fixture$files[1], with_na), ncomp = 2L),
               "NIPALS")

  nonpositive <- file.path(fixture$dir, "zeros.csv")
  zeros <- fixture$parts[[1]]
  zeros[1, 1] <- 0
  utils::write.csv(zeros, nonpositive, row.names = FALSE)
  # a zero under log2 becomes -Inf, so the warning is followed by a hard stop
  expect_warning(
    tryCatch(IncrementalPCA(nonpositive, ncomp = 2L, log2 = TRUE),
             error = function(e) NULL),
    "non-positive")
  expect_error(suppressWarnings(IncrementalPCA(nonpositive, ncomp = 2L,
                                               log2 = TRUE)),
               "log2 transformation")
})

test_that("scaling handles zero-variance features and requires two rows", {
  fixture <- incremental_fixture(counts = c(20, 25))
  constant <- lapply(fixture$parts, function(part) {
    part[, 2] <- 5
    part
  })
  files <- file.path(fixture$dir, c("const1.csv", "const2.csv"))
  for (i in seq_along(files)) utils::write.csv(constant[[i]], files[i],
                                              row.names = FALSE)

  expect_warning(fit <- IncrementalPCA(files, ncomp = 2L, scale = TRUE),
                 "zero variance")
  expect_true(all(is.finite(fit$S)))
  expect_equal(as.numeric(attr(fit, "scaling"))[2], 1)

  single <- file.path(fixture$dir, "one-row.csv")
  utils::write.csv(constant[[1]][1, , drop = FALSE], single, row.names = FALSE)
  expect_error(IncrementalPCA(single, ncomp = 1L, scale = TRUE),
               "at least two observations")
})

test_that("argument validation names the offending argument", {
  fixture <- incremental_fixture()
  expect_error(IncrementalPCA(character()), "files")
  expect_error(IncrementalPCA(fixture$files, ncomp = 0L), "ncomp")
  expect_error(IncrementalPCA(fixture$files, oversample = -1L), "oversample")
  expect_error(IncrementalPCA(fixture$files, chunk_rows = 0L), "chunk_rows")
  expect_error(IncrementalPCA(fixture$files, center = NA), "center")
  expect_error(IncrementalPCA(fixture$files, reader = "not a function"),
               "reader")
  expect_error(IncrementalPCA(fixture$files, backend = "rtorch"), "ROADMAP")
  expect_error(IncrementalPCA(fixture$files, backend = "irlba"),
               "Unknown IncrementalPCA")

  # more components than the data can support
  expect_error(IncrementalPCA(fixture$files, ncomp = 50L), "component")
})

test_that("the PyTorch backend agrees with the base R kernel", {
  skip_if_not(reticulate::py_available(initialize = FALSE),
              "Python is not initialized")
  skip_if_not(reticulate::py_module_available("torch"), "torch is not available")

  fixture <- incremental_fixture()
  reference <- IncrementalPCA(fixture$files, ncomp = 4L, scale = TRUE,
                              store_scores = TRUE)
  fit <- IncrementalPCA(fixture$files, ncomp = 4L, scale = TRUE,
                        store_scores = TRUE, backend = "pytorch")

  expect_equal(as.numeric(fit$S), as.numeric(reference$S), tolerance = 1e-10)
  expect_gt(subspace_min_cosine(fit$Vh, reference$Vh), 1 - 1e-10)
  expect_equal(unname(get_pc_scores(fit)), unname(get_pc_scores(reference)),
               tolerance = 1e-8)
})

test_that("the tinygrad backend agrees with the base R kernel", {
  skip_if_not(reticulate::py_available(initialize = FALSE),
              "Python is not initialized")
  skip_if_not(reticulate::py_module_available("tinygrad"),
              "tinygrad is not available")

  fixture <- incremental_fixture()
  reference <- IncrementalPCA(fixture$files, ncomp = 4L, scale = TRUE)
  fit <- IncrementalPCA(fixture$files, ncomp = 4L, scale = TRUE,
                        backend = "tinygrad")

  # float32 plus a randomized sketch per block: agreement is close but not exact
  expect_equal(as.numeric(fit$S), as.numeric(reference$S), tolerance = 1e-4)
  expect_gt(subspace_min_cosine(fit$Vh, reference$Vh), 0.999)
})
