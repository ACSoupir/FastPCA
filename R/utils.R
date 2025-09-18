#make sure backend is present
check_backend = function(backend = c("tinygrad", "pytorch")){
  if(backend == "tinygrad") message("Falling back to pytorch - current implementations of SVD in tinygrad are slow and memory hungry.")
  backend = "pytorch" #match.arg(backend)

  #for torch backend
  if(backend == "pytorch"){
    if (!reticulate::py_module_available("torch") || !reticulate::py_module_available("numpy")) {
      stop("PyTorch 'torch' and/or 'numpy' are not available in the current Python environment.\nPlease run `FastPCA::setup_py_env()` and `FastPCA::start_FastPCA_env()`.")
    }
  } else { #for tinygrad
    if(!reticulate::py_module_available("tinygrad")){
      stop("Tinygrad is not availble in the current Python environment.\nPlease run `FastPCA::setup_py_env()` and `FastPCA::start_FastPCA_env()`.")
    }
  }
}

#get python file locations from package folder
get_python_files = function(){
  file_list = list(
    torch_random_svd = system.file("python", "torch_svd_func.py", package = "FastPCA"),
    torch_exact_svd = system.file("python", "torch_svd_func_exact.py", package = "FastPCA"),
    torch_tranformation = system.file("python", "torch_transform.py", package = "FastPCA"),
    umap_calculation = system.file("python", "umap_func.py", package = "FastPCA")#,
    #tinygrad_devices = system.file("python", "tinygrad_backends.py", package = "FastPCA"),
    #tinygrad_tranformation = system.file("python", "tinygrad_transform.py", package = "FastPCA"),
    #tinygrad_random_svd = system.file("python", "tinygrad_svd_func.py", package = "FastPCA")
  )
  return(file_list)
}

#load python files
python_functions = function(){
  #utilities
  .globals <- new.env(parent = emptyenv())

  #get files
  script_paths = get_python_files()

  .globals = lapply(script_paths, function(f){
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

  #tinygrad for future implementations
  if(backend == 'tinygrad'){
    check_backend('tinygrad')
  }
  return(list(backend = backend,
              device = device))
}
