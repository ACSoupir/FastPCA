#' Set up python environment for FastPCA
#'
#' This function ensures that a suitable Python environment (virtualenv or conda)
#' is available and that 'torch' and 'numpy' or 'tinygrad' are installed within it.
#' Users should run this once before using functions that rely on PyTorch.
#'
#' @param method Character string. The method to use for environment creation.
#'   Can be "conda" (default) or "virtualenv".
#' @param envname Character string. The name of the Python environment to create/use.
#'   Defaults to "FastPCA".
#' @param python_version Character string. The Python version to use (e.g., "3.9").
#'   Defaults to "3.9". It's recommended to stick to well-supported versions.
#' @param backend character string of the backend to use. Either 'torch', 'tinygrad', or 'all'.
#'   Defaults to 'all' to install all packages into the environment. **Currently only pytorch. Flag is ignored**
#' @param cuda boolean for whether to install cuda toolkit to leverage the cuda backend
#' @param ... Additional arguments passed to `reticulate::virtualenv_create()`
#'   or `reticulate::conda_create()`.
#'
#' @return Invisibly returns `TRUE` if setup is successful, `FALSE` otherwise.
#' @export
#' @examples
#' \dontrun{
#'   #for conda
#'   FastPCA::setup_py_env(method = "conda")
#'
#'   #reticulate virtualenv:
#'   FastPCA::setup_py_env()
#' }
setup_py_env <- function(method = c("conda", "virtualenv"),
                         envname = "FastPCA",
                         python_version = "3.10",
                         backend = c("pytorch"), #, "tinygrad", "all" #future when tinygrad has better impolementations
                         cuda = FALSE,
                         ...) {
  #so much prefer conda
  #https://github.com/ACSoupir/cuda_conda
  if(backend == "tinygrad") message("Falling back to pytorch - current implementations of SVD in tinygrad are slow and memory hungry.")
  method <- match.arg(method)
  backend <- "pytorch"#match.arg(backend)
  if(backend == "pytorch") backend = 'torch' #think it works with either for conda but not pip?
  #expand backend
  if(backend == "all") backend = c("torch", "tinygrad")
  backend = c("numpy", backend)
  #if cuda, need conda
  if(cuda){
    if(method != "conda") stop("In order to use cuda, must have conda")
  }

  message(paste("Attempting to set up Python environment '", envname, "' using ", method, "...", sep=""))

    #allow for the error to be handled at the end
  tryCatch({
    #create environment
    create_environment(method, envname, python_version)
    installed_packages = reticulate::py_list_packages(envname = envname, type = method)
    #if user wants cuda
    if(cuda && !("cudatoolkit" %in% installed_packages$package)){
      install_python_package(method = method, envname = envname, package = 'cudatoolkit')
    }
    #installing the others
    to_install = !sapply(backend, function(p){
      p %in% installed_packages$package
    })
    to_install = names(to_install[to_install])
    if(length(to_install) == 0){
      message("Already Setup")
      return(invisible(TRUE))
    }
    message("Installing ", paste0(to_install, collapse = ", "))
    install_python_package(method = method, envname = envname, package = to_install)
  #if al else fails:
  }, error = function(e){
    warning(paste("Error setting up Python environment:", e$message))
    message("You may need to manually configure reticulate, e.g., by running:")
    message("  reticulate::conda_create(envname = 'FastPCA', packages = c('pip'))")
    message("  reticulate::use_condaenv('FastPCA', required = TRUE)")
    message("  reticulate::py_install(packages = c('torch', 'numpy'), pip = TRUE)")
    return(invisible(FALSE))
  })
}

install_python_package = function(method, envname, package){
  tryCatch({
    if("cudatoolkit" %in% package){
      reticulate::conda_install(envname = envname, packages = "cudatoolkit", channel = "anaconda")
    } else {
      reticulate::py_install(packages = package,
                             envname = envname,
                             method = method,
                             pip = TRUE)
    }

  }, error = function(e){
    warning(paste("Error installing python packages:", e$message,
                  "\nPossible you provided cuda = TRUE when cuda isn't available?"))
    return(invisible(TRUE))
  })
}

create_environment = function(method, envname, python_version, ...){
  tryCatch({
    #conda
    if(method == "conda"){
      message("Checking if ", envname, " is already an existing Conda environment")
      if(reticulate::condaenv_exists(envname)){
        return(invisible(TRUE))
      } else {
        message("\tCreating ", envname)
        reticulate::conda_create(envname = envname, python_version = python_version,
                                 packages = c("umap-learn"),
                                 ...)
      }
    }
    #virtual environment
    if(method == "virtualenv"){
      message("Checking if ", envname, " is already an existing virtual environment")
      if(reticulate::virtualenv_exists(envname)){
        return(invisible(TRUE))
      } else {
        message("\tCreating ", envname)
        reticulate::virtualenv_create(envname = envname, python = python_version, ...)#strange that virtual envirnments and conda don't use same flags but whatever
      }
    }
  }, error = function(e){
    warning(paste0("Error creating '", envname, "' conda environment:\n", e$message))
    return(invisible(TRUE))
  })
}
