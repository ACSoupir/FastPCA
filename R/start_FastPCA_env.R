#' Activate python environment for FastPCA
#'
#' @param method Character string. The method to use for environment creation.
#'   Can be "conda" (default) or "virtualenv".
#' @param envname Character string. The name of the Python environment to create/use.
#'   Defaults to "FastPCA".
#' @param required boolean passed to `reticulate`
#'
#' @returns Invisibly returns `TRUE` if setup is successful, `FALSE` otherwise.
#' @export
#'
#' @examples
#' \dontrun{
#'   #for conda
#'   setup_py_env(method = "conda", envname = "FastPCA",
#'     python_version = "3.9", backend = "all", cuda = FALSE)
#'   start_fpca_env
#' }
start_FastPCA_env = function(method = c("conda", "virtualenv"),
                          envname = "FastPCA",
                          required = FALSE){
  #match method
  method <- match.arg(method)
  #for conda
  if(method == "conda"){
    if (!reticulate::condaenv_exists(envname)) {
      stop("Environment is not available through this method.
           Please run `FastPCA::setup_py_env()`.")
    }
    reticulate::use_condaenv(envname, required = required)
  }
  #for virtual env with system python installed
  if(method == "virtualenv"){
    if(!reticulate::virtualenv_exists(envname)){
      stop("Environment is not available through this method.
           Please run `FastPCA::setup_py_env()`.")
    }
    reticulate::use_virtualenv(envname, required = required)
  }
}
