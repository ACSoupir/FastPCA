# Tests for utility functions

test_that("get_python_files works correctly", {
  file_list <- get_python_files()

  expect_type(file_list, "list")
  expect_named(file_list, c("torch_random_svd", "torch_exact_svd", "torch_tranformation", "umap_calculation"))

  for (file in file_list) {
    expect_type(file, "character")
  }
})

test_that("validate_backend works for backend = 'r'", {
  # Should return a list with backend = "r" and device = "CPU"
  res <- validate_backend(backend = "r", device = "CPU")
  expect_equal(res, list(backend = "r", device = "CPU"))

  # Should warn and change device to "CPU" if GPU is requested
  expect_warning(
    res_gpu <- validate_backend(backend = "r", device = "GPU"),
    "Currently no GPU device implemented for base R PCA"
  )
  expect_equal(res_gpu, list(backend = "r", device = "CPU"))
})
