#make sure backend is present
check_backend = function(backend = c("tinygrad", "pytorch")){
  if(backend == "tinygrad") message("Falling back to pytorch - current implementations of SVD in tinygrad are slow and memory hungry.")
  backend = "pytorch" #match.arg(backend)

  #for torch backend
  if(backend == "pytorch"){
    if (!reticulate::py_module_available("torch") || !reticulate::py_module_available("numpy")) {
      stop("PyTorch 'torch' and/or 'numpy' are not available in the current Python environment.
           Please run `FastPCA::setup_py_env()` and `FastPCA::start_FastPCA_env()`.")
    }
  } else { #for tinygrad
    if(!reticulate::py_module_available("tinygrad")){
      stop("Tinygrad is not availble in the current Python environment.
           Please run `FastPCA::setup_py_env()` and `FastPCA::start_FastPCA_env()`.")
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

validate_backend = function(backend = c("pytorch", "tinygrad")){
  backend = match.arg(backend)
  if(length(backend) == 1 & "tinygrad" %in% backend){
    message("Falling back to pytorch - current implementations of SVD in tinygrad are slow and memory hungry.")
    backend = "pytorch"
  }
  if(length(backend) == 2){
    message("Using pytorch backend.")
    backend = "pytorch"
  }
  return(backend)
}
