test_that(".conda_env_names parses conda env list output", {
  out <- c(
    "",
    "# conda environments:",
    "#",
    "                       /Users/someone/env",          # unnamed, other root
    "base                 * /opt/anaconda3",              # active -> extra "*"
    "FastPCA                /opt/anaconda3/envs/FastPCA", # 2 fields
    "GNN_test               /opt/anaconda3/envs/GNN_test"
  )
  expect_identical(.conda_env_names(out, "/opt/anaconda3"),
                   c("base", "FastPCA", "GNN_test"))
})

test_that(".conda_env_names returns nothing when no line matches the root", {
  out <- c("# conda environments:", "base * /opt/anaconda3")
  expect_length(.conda_env_names(out, "/some/other/root"), 0L)
})

test_that(".ensure_numba_cache_dir sets a writable cache directory", {
  old <- Sys.getenv("NUMBA_CACHE_DIR", unset = NA_character_)
  withr::defer({
    if (is.na(old)) Sys.unsetenv("NUMBA_CACHE_DIR")
    else Sys.setenv(NUMBA_CACHE_DIR = old)
  })
  Sys.unsetenv("NUMBA_CACHE_DIR")
  got <- .ensure_numba_cache_dir()
  expect_true(nzchar(got))
  expect_identical(Sys.getenv("NUMBA_CACHE_DIR"), got)
})
