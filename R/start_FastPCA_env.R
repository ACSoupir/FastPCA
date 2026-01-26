#' Activate python environment for FastPCA
#'
#' @param method Character string. The method to use for environment creation.
#'   Can be "conda" (default) or "virtualenv".
#' @param envname Character string. The name of the Python environment to create/use.
#'   Defaults to "FastPCA".
#' @param required boolean passed to `reticulate`
#'
#' @returns Invisibly returns `TRUE` if setup is successful, `FALSE` otherwise.
#'
#' @details
#' Using a conda environment can allow for quick testing of new versions of pytorch and tinygrad.
#' However, it should be used with some caution. For example, if you have already used the 'rtorch' backend
#' in some other functions, not even necessarily part of this package, using `start_FastPCA_env` will attach
#' the conda environment but the modules in the environment won't be available. This is a system level
#' conflict between something in reticulate and torch (maybe libtorch variables?). Because of this,
#' it is really only recommended to use the python envronment if wanting to use the absolute latest
#' version of torch (and in the future tinygrad) to compare performance (accuracy and speed). Otherwise,
#' the rtorch implementation is likely sufficient.
#'
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
    conda_loc = Sys.which("conda")
    conda_res = tryCatch(system2(conda_loc, c("env", "list"), stdout = TRUE),
                         warning = function(w){
                           print(w)
                           stop("Conda produced a warning")
                         })
    conda_envs = conda_res %>%
      grep(gsub("\\/bin.*", "", conda_loc), ., value = TRUE) %>%
      read.table(text = ., header = FALSE)

    if (!envname %in% conda_envs[[1]]) {
      stop("Environment is not available through this method.
           Please run `FastPCA::setup_py_env()`.")
    }
    reticulate::use_condaenv(envname, required = required)
  }
  #for virtual env with system python installed
  if(method == "virtualenv"){
    virtualenv_root_loc = reticulate::virtualenv_root()

    if(!(dir.exists(file.path(virtualenv_root_loc, envname)) &&
         (file.exists(file.path(virtualenv_root_loc, envname, "bin", "python")) ||
          file.exists(file.path(virtualenv_root_loc, envname, "Scripts", "python.exe"))))){
      stop("Environment is not available through this method.
           Please run `FastPCA::setup_py_env()`.")
    }
    reticulate::use_virtualenv(envname, required = required)
  }
}
