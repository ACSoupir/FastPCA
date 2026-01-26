test_that("environment setup works", {
  #need to check erros and things first
  expect_error(setup_py_env(method = "something else", envname = "FastPCA", python_version = 3.10))
  expect_error(setup_py_env(method = "virtualenv", envname = "FastPCA", python_version = 3.10, cuda = TRUE))
  #can take a bit
  expect_warning(suppressMessages(setup_py_env(method = "conda", envname = "FastPCA", python_version = 3.10, cuda = TRUE)))

  #asking for a tinygrad backend shouldn't fail but rather
  expect_warning(suppressMessages(setup_py_env(method = "conda", envname = "FastPCA", python_version = 3.10, backend = "tinygrad")))
})
