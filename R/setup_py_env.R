#' Set up python environment for FastPCA
#'
#' This function ensures that a suitable Python environment (virtualenv or conda)
#' is available and that 'torch' and 'numpy' are installed within it.
#' Users should run this once before using functions that rely on PyTorch.
#'
#' @param method Character string. The method to use for environment creation.
#'   Can be "virtualenv" (default) or "conda".
#' @param envname Character string. The name of the Python environment to create/use.
#'   Defaults to "FastPCA".
#' @param python_version Character string. The Python version to use (e.g., "3.9").
#'   Defaults to "3.9". It's recommended to stick to well-supported versions.
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
                         cuda = FALSE,
                         ...) {
  #so much prefer conda
  #https://github.com/ACSoupir/cuda_conda
  method <- match.arg(method)

  message(paste("Attempting to set up Python environment '", envname, "' using ", method, "...", sep=""))

  if (reticulate::py_available(initialize = FALSE)) {
    #chck if packages are installed
    if (reticulate::py_module_available("torch") && reticulate::py_module_available("numpy")) {
      message("PyTorch 'torch' and 'numpy' are already available in the current Python environment.")
      return(invisible(TRUE))
    } else {
      message("Current Python environment lacks 'torch' or 'numpy'. Attempting to install...")
      tryCatch({
        if(cuda){
          #trying to install in environment
          if(method == "conda"){
            trycatch({
              #try installing the toolkit
              reticulate::conda_install("FastPCA", packages = "cudatoolkit", channel = "anaconda")
            }, error = function(e){
              warning(paste("Error installing pythong packages:", e$message,
                            "\nPossible you provided cuda = TRUE when cuda isn't available?"))
              return(invisible(TRUE))
            })
          }
        }
        #install packages
        reticulate::py_install(packages = c("torch", "numpy"), pip = TRUE)

        #check if the packages are available now
        if (reticulate::py_module_available("torch") && reticulate::py_module_available("numpy")) {
          message("Successfully installed 'torch' and 'numpy' in the current environment.")
          return(invisible(TRUE))
        } else {
          stop("Failed to install 'torch' and 'numpy' in the current environment.")
        }
      }, error = function(e) {
        warning(paste("Could not install 'torch' in current environment:", e$message,
                      "\nAttempting to create a new dedicated environment."))
      })
    }
  }
  #enironment not activated?
  tryCatch({
    if (method == "conda") { # conda
      if (!envname %in% reticulate::conda_list()$name) {
        message(paste("Creating conda environment '", envname, "' (this may take a moment)...", sep=""))
        reticulate::conda_create(envname = envname, python = python_version, ...)
      }
      reticulate::use_condaenv(envname, required = TRUE)
    } else { #virtualenv
      if (!envname %in% reticulate::virtualenv_list()) {
        message(paste("Creating virtual environment '", envname, "' (this may take a moment)...", sep=""))
        reticulate::virtualenv_create(envname = envname, python = python_version, ...)
      }
      reticulate::use_virtualenv(envname, required = TRUE)
    }

    message("Installing 'torch' and 'numpy' into the environment...")
    if(cuda){
      #trying to install in environment
      if(method == "conda"){
        trycatch({
          reticulate::conda_install("FastPCA", packages = "cudatoolkit", channel = "anaconda")
        }, error = function(e){
          warning(paste("Error installing pythong packages:", e$message,
                        "\nPossible you provided cuda = TRUE when cuda isn't available?"))
          return(invisible(TRUE))
        })
      }

    }

    reticulate::py_install(packages = c("torch", "numpy"), pip = TRUE)

    if (reticulate::py_module_available("torch") && reticulate::py_module_available("numpy")) {
      message(paste("Successfully set up Python environment '", envname, "' with 'torch' and 'numpy'.", sep=""))
      return(invisible(TRUE))
    } else {
      stop("Failed to install 'torch' and 'numpy'. Please check your internet connection or Python setup.")
    }
  }, error = function(e) {
    warning(paste("Error setting up Python environment:", e$message))
    message("You may need to manually configure reticulate, e.g., by running:")
    message("  reticulate::conda_create(envname = 'FastPCA', packages = c('pip'))")
    message("  reticulate::use_condaenv('FastPCA', required = TRUE)")
    message("  reticulate::py_install(packages = c('torch', 'numpy'), pip = TRUE)")
    return(invisible(FALSE))
  })
}
