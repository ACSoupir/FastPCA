#' Activate python environment for FastPCA
#'
#' @param method Character string. The method to use for environment creation.
#'   Can be "conda" (default) or "virtualenv".
#' @param envname Character string. The name of the Python environment to create/use.
#'   Defaults to "FastPCA".
#' @param required boolean passed to `reticulate`
#'
#' @returns Invisibly returns `TRUE` when the environment is activated and its
#'   modules can be imported.
#'
#' @details
#' A FastPCA Python environment always installs `umap-learn` alongside the SVD
#' backends, and this function verifies that `umap` is importable. That check
#' runs even when you only intend to use the Python SVD backends, so an
#' environment without `umap` is treated as incomplete (re-run
#' [setup_py_env()] to repair it).
#'
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
#'     python_version = "3.10", backend = "all", cuda = FALSE)
#'   start_FastPCA_env()
#' }
start_FastPCA_env = function(method = c("conda", "virtualenv"),
                          envname = "FastPCA",
                          required = TRUE){
  #Sys.setenv(KMP_DUPLICATE_LIB_OK = TRUE)
  #make sure numba/umap can import before the module check below
  .ensure_numba_cache_dir()
  #match method
  method <- match.arg(method)
  #for conda
  if(method == "conda"){
    conda_loc = Sys.which("conda")
    if (!nzchar(conda_loc)) {
      stop("Conda was not found on the PATH.\n",
           "Install conda or use method = 'virtualenv'.")
    }
    conda_res = tryCatch(system2(conda_loc, c("env", "list"), stdout = TRUE),
                         warning = function(w){
                           print(w)
                           stop("Conda produced a warning")
                         })
    conda_root = gsub("\\/(bin|condabin).*", "", conda_loc)
    env_names = .conda_env_names(conda_res, conda_root)

    if (!envname %in% env_names) {
      stop("Environment is not available through conda",
           "\nPlease run `FastPCA::setup_py_env()`.")
    }
    message("Environment found.. Attempting to use..")
    reticulate::use_condaenv(envname, required = required)

  }
  #for virtual env with system python installed
  if(method == "virtualenv"){
    virtualenv_root_loc = reticulate::virtualenv_root()

    if(!(dir.exists(file.path(virtualenv_root_loc, envname)) &&
         (file.exists(file.path(virtualenv_root_loc, envname, "bin", "python")) ||
          file.exists(file.path(virtualenv_root_loc, envname, "Scripts", "python.exe"))))){
      stop("Environment is not available through virtualenv",
           "\nPlease run `FastPCA::setup_py_env()`.")
    }
    message("Environment found.. Attempting to use..")
    reticulate::use_virtualenv(envname, required = required)
  }

  # umap is installed with every FastPCA environment and verified even for
  # SVD-only workflows, so an environment without it is incomplete.
  message("Checking for modules available...")
  if(!reticulate::py_module_available("umap")){
    stop("The Python environment `", envname, "` is missing the 'umap' module.\n",
         "Re-run `FastPCA::setup_py_env()` and then `FastPCA::start_FastPCA_env()`.")
  }
  message("Successfully loaded and identified umap module..")
  invisible(TRUE)
}

# Environment names from `conda env list` output.
#
# conda prints "<name> [*] <path>" for named environments and a bare path for
# unnamed ones, so the number of whitespace-separated fields varies from line to
# line (the active environment also gains a "*"). Comparing only the first token
# avoids read.table()'s "did not have N elements" failure on that format.
.conda_env_names = function(conda_res, conda_root){
  env_lines = grep(conda_root, conda_res, value = TRUE, fixed = TRUE)
  trimws(sub("^\\s*([^[:space:]]+).*$", "\\1", env_lines))
}
