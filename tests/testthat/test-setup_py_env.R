test_that("environment setup works", {
  #need to check erros and things first
  expect_error(setup_py_env(method = "something else", envname = "FastPCA", python_version = 3.10))
  expect_error(setup_py_env(method = "virtualenv", envname = "FastPCA", python_version = 3.10, cuda = TRUE))
  #can take a bit
  expect_warning(suppressMessages(setup_py_env(method = "conda", envname = "FastPCA", python_version = 3.10, cuda = TRUE)))

  #'tinygrad' is now a supported backend choice (it installs the Metal SVD path)
  expect_setequal(eval(formals(setup_py_env)$backend), c("pytorch", "tinygrad", "all"))
})
