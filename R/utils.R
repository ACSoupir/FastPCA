#make sure backend is present
check_backend = function(backend = c("tinygrad", "pytorch")){
  backend = match.arg(backend)
  if(!reticulate::py_available(initialize = FALSE)){
    stop("Python is not available in the current session.",
         "\nSave your work if needed, and restart R (this is a reticulate requirement).",
         "\nPlease run `FastPCA::setup_py_env()` and `FastPCA::start_FastPCA_env()`.")
  }
  #check the modules the selected backend actually needs
  required = if(backend == "tinygrad") c("numpy", "tinygrad") else c("torch", "numpy")
  missing = required[!vapply(required, reticulate::py_module_available, logical(1))]
  if(length(missing) > 0){
    stop("The '", backend, "' backend requires the Python module(s): ",
         paste(missing, collapse = ", "), ".",
         "\nPlease run `FastPCA::setup_py_env(backend = '",
         if(backend == "tinygrad") "tinygrad" else "pytorch", "')` and `FastPCA::start_FastPCA_env()`.")
  }
}

#get python file locations from package folder
get_python_files = function(backend = NULL){
  # the tinygrad backend only needs its own module, so selecting it does not
  # require torch to be installed in the same environment.
  if(identical(backend, "tinygrad")){
    file_list = list(
      tinygrad_random_svd = system.file("python", "tinygrad_svd_func.py", package = "FastPCA"),
      tinygrad_tranformation = system.file("python", "tinygrad_transform.py", package = "FastPCA")
    )
  } else {
    file_list = list(
      torch_random_svd = system.file("python", "torch_svd_func.py", package = "FastPCA"),
      torch_exact_svd = system.file("python", "torch_svd_func_exact.py", package = "FastPCA"),
      torch_tranformation = system.file("python", "torch_transform.py", package = "FastPCA"),
      umap_calculation = system.file("python", "umap_func.py", package = "FastPCA")
    )
  }
  missing_files = names(file_list)[!nzchar(unlist(file_list))]
  if(length(missing_files) > 0){
    stop("Python script(s) not found in the installed FastPCA package: ",
         paste(missing_files, collapse = ", "),
         "\nTry reinstalling the package (e.g. `devtools::install()`).")
  }
  return(file_list)
}

#load python files
#
# Only the requested scripts are loaded. This avoids importing heavyweight or
# unrelated modules (for example umap/numba) when only the SVD backend is used.
python_functions = function(backend = NULL,
                            scripts = names(get_python_files(backend))){
  #utilities
  .globals <- new.env(parent = emptyenv())

  #get files (tinygrad selects only its own module)
  script_paths = get_python_files(backend)

  #limit to the scripts actually needed by this call
  scripts <- match.arg(scripts, names(script_paths), several.ok = TRUE)

  # numba (used by umap-learn) needs a writable cache directory when Python is
  # embedded via reticulate, otherwise importing umap aborts with a locator error.
  if (!nzchar(Sys.getenv("NUMBA_CACHE_DIR"))) {
    numba_cache <- file.path(tempdir(), "fastpca-numba")
    dir.create(numba_cache, showWarnings = FALSE, recursive = TRUE)
    Sys.setenv(NUMBA_CACHE_DIR = numba_cache)
  }

  .globals = lapply(script_paths[scripts], function(f){
    reticulate::py_run_file(f)
  })

  return(.globals)
#
#   script_path <- get_python_script_path()
#   if (!file.exists(script_path)) {
#     stop(paste("Python SVD script not found at:", script_path,
#                "\nIs the package installed correctly?"))
#   }
#   .globals$svd_python_module <- reticulate::py_run_file(script_path)
#
#   script_path <- get_exact_svd_script_path()
#   if (!file.exists(script_path)) {
#     stop(paste("Python SVD script not found at:", script_path,
#                "\nIs the package installed correctly?"))
#   }
#   .globals$exact_svd_python_module <- reticulate::py_run_file(script_path)
#   message(paste("Python SVD script '", basename(script_path), "' loaded.", sep=""))
}

validate_backend = function(backend = c("r", "rtorch", "pytorch", "irlba", "tinygrad"),
                            device = c("CPU", "GPU")){
  backend = match.arg(backend)
  device = match.arg(device)
  #quick exit if all good
  if(backend == "r"){
    if(device == "GPU"){
      warning("Currently no GPU device implemented for base R PCA")
      message("Setting device to 'CPU'")
      device = "CPU"
    }
  }
  #more complicated things
  if(backend == "rtorch"){
    if(reticulate::py_available(initialize = FALSE)){
      stop("Looks like python has already been initialized.\n",
           "To use rtorch, you must restart your R session.\n",
           "Only do this after saving everything and if absolutely required.")
    }
    rtorch_avail <- "torch" %in% row.names(installed.packages())
    if(!rtorch_avail){
      warning("Torch is not fully installed.\nPlease run `install.packages('torch')` and `torch::install_torch()`")
      message("Setting backend to 'r' and device to 'CPU'")
      backend = "r"
      device = "CPU"
    } else { #torch is available
      if(device == "GPU" & !torch::cuda_is_available()){
        warning("GPU is not avialable for `torch`")
        message("Setting device to 'CPU'")
        device = "CPU"
      }
      #torch is available and user selected CPU
    }
  }

  #pytorch
  #doesn't matter if gpu or cpu since the function themselves does the checks in python
  if(backend == 'pytorch'){
    check_backend('pytorch')
  }

  #tinygrad
  if(backend == 'tinygrad'){
    check_backend('tinygrad')
    # tinygrad only has a working Metal backend on Apple Silicon, so be explicit
    if(device == "CPU" && .on_arm64_mac()){
      message("FastPCA: on Apple Silicon (arm64) the tinygrad backend runs on METAL; ",
              "'device = \"CPU\"' is mapped to the GPU because tinygrad's CPU codegen ",
              "is unavailable.")
    }
  }
  return(list(backend = backend,
              device = device))
}

# TRUE on Apple Silicon Macs, where tinygrad only has a working Metal backend
.on_arm64_mac = function(){
  info = Sys.info()
  identical(unname(info[["sysname"]]), "Darwin") &&
    grepl("arm64|aarch64", unname(info[["machine"]]))
}


# map setup_py_env()'s backend choice onto the packages to install
.resolve_setup_backend = function(backend = c("pytorch", "tinygrad", "all")){
  backend = match.arg(backend, c("pytorch", "tinygrad", "all"))
  if(identical(backend, "pytorch")) return("torch")
  if(identical(backend, "all")) return(c("torch", "tinygrad"))
  return("tinygrad")
}
