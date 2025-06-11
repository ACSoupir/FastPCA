#utilities
.globals <- new.env(parent = emptyenv())
.globals$svd_python_module <- NULL
.globals$transform_python_module = NULL

#' @noRd
# Internal helper to get the path to our Python script within the installed package.
get_python_script_path <- function() {
  system.file("python", "svd_func.py", package = "FastPCA")
}

#' @noRd
# Internal helper to get the path to our Python script within the installed package.
get_transform_script_path <- function() {
  system.file("python", "transform.py", package = "FastPCA")
}
