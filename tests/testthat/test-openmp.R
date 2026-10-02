local_omp_env <- function() {
  old <- Sys.getenv("OMP_NUM_THREADS", unset = NA_character_)
  withr::defer({
    if (is.na(old)) Sys.unsetenv("OMP_NUM_THREADS")
    else Sys.setenv(OMP_NUM_THREADS = old)
  })
}

test_that(".backend_uses_openmp identifies OpenMP-backed backends", {
  expect_true(.backend_uses_openmp("pytorch"))
  expect_true(.backend_uses_openmp("rtorch"))
  expect_false(.backend_uses_openmp("r"))
  expect_false(.backend_uses_openmp("irlba"))
})

test_that(".openmp_conflicts only reports loaded offender namespaces", {
  old <- options(FastPCA.openmp_offenders = c("FastPCA"))
  on.exit(options(old), add = TRUE)
  expect_true("FastPCA" %in% .openmp_conflicts())

  options(FastPCA.openmp_offenders = c("definitely_not_a_real_package"))
  expect_length(.openmp_conflicts(), 0L)
})

test_that(".resolve_cores clamps to one core when a conflict is present", {
  local_omp_env()
  old <- options(FastPCA.openmp_warning = "always")
  on.exit(options(old), add = TRUE)
  expect_warning(
    got <- .resolve_cores(4, openmp = TRUE, offenders = "data.table"),
    "OpenMP runtime conflict"
  )
  expect_identical(got, 1L)
})

test_that(".resolve_cores preserves user cores when there is no conflict", {
  expect_silent(got <- .resolve_cores(4, openmp = TRUE, offenders = character()))
  expect_identical(got, 4L)
})

test_that(".resolve_cores ignores non-OpenMP code paths", {
  expect_silent(got <- .resolve_cores(4, openmp = FALSE, offenders = "data.table"))
  expect_identical(got, 4L)
})

test_that(".resolve_cores stays quiet when only one core is requested", {
  local_omp_env()
  expect_silent(got <- .resolve_cores(1, openmp = TRUE, offenders = "data.table"))
  expect_identical(got, 1L)
})

test_that(".resolve_cores caps OMP_NUM_THREADS before torch is imported", {
  local_omp_env()
  Sys.unsetenv("OMP_NUM_THREADS")
  expect_silent(.resolve_cores(1, openmp = TRUE, offenders = "data.table"))
  expect_identical(Sys.getenv("OMP_NUM_THREADS"), "1")
})

test_that(".resolve_cores leaves OMP_NUM_THREADS alone without a conflict", {
  local_omp_env()
  Sys.setenv(OMP_NUM_THREADS = "7")
  expect_silent(.resolve_cores(4, openmp = TRUE, offenders = character()))
  expect_identical(Sys.getenv("OMP_NUM_THREADS"), "7")
})

test_that(".cap_openmp_threads sets a single-threaded pool", {
  local_omp_env()
  Sys.unsetenv("OMP_NUM_THREADS")
  .cap_openmp_threads()
  expect_identical(Sys.getenv("OMP_NUM_THREADS"), "1")
})

test_that(".resolve_cores coerces doubles and rejects invalid values", {
  expect_silent(got <- .resolve_cores(8, openmp = TRUE, offenders = character()))
  expect_identical(got, 8L)
  expect_error(.resolve_cores(0), "greater than or equal to 1")
  expect_error(.resolve_cores(c(2, 4)), "greater than or equal to 1")
})

test_that("warning cadence honours FastPCA.openmp_warning", {
  local_omp_env()
  old <- options(FastPCA.openmp_warning = "silent")
  on.exit(options(old), add = TRUE)
  expect_silent(.resolve_cores(4, openmp = TRUE, offenders = "data.table"))
})

test_that("'once' cadence warns only on the first conflicting call", {
  local_omp_env()
  old <- options(FastPCA.openmp_warning = "once")
  on.exit(options(old), add = TRUE)
  .fastpca_state$warned <- FALSE
  on.exit(.fastpca_state$warned <- FALSE, add = TRUE)
  expect_warning(
    .resolve_cores(4, openmp = TRUE, offenders = "data.table"),
    "OpenMP runtime conflict"
  )
  expect_silent(.resolve_cores(4, openmp = TRUE, offenders = "data.table"))
})

test_that("invalid warning options fall back to 'always'", {
  old <- options(FastPCA.openmp_warning = "nonsense")
  on.exit(options(old), add = TRUE)
  expect_identical(.openmp_warning_mode(), "always")
})
