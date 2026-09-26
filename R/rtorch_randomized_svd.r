#' Randomized Singular Vector Decomposition using torch
#'
#' @param A_mat matrix passed from R
#' @param k Integer. Number of dimensions to calculate the singular vectors on
#' @param p Intger. Oversampling dimensions
#' @param q_iter Integer. Number of power iterations to perform
#' @param device Character. Either "CPU" or "GPU depending on what user selects and availability
#' @param cores Integer. Number of CPU cores to use for matrix operations
#'
#' @returns a list with U, S, and Vh
#'
rtorch_randomized_svd = function(A_mat, k, p, q_iter, device = "CPU", cores = 1){
  #accuracy
  dtype = "float64"
  #set device and cores
  if(device == "GPU"){
    if(torch::cuda_is_available()){
      device = torch::torch_device("cuda")
    } else {
      warning("CUDA supplied as device but not available to torch")
      message("Falling back to CPU for computation")
      device = torch::torch_device("cpu")
      torch::torch_set_num_threads(cores)
    }
  } else {
    device = torch::torch_device("cpu")
    torch::torch_set_num_threads(cores)
  }

  A = torch::torch_tensor(A_mat, dtype = dtype)
  A = A$to(device = device)

  #random projection
  Omega = torch::torch_randn(dim(A)[2], k + p, device = device, dtype = dtype)

  #subspace
  Y = A$matmul(Omega)
  for(i in seq(q_iter)){
    Q = torch::linalg_qr(torch::torch_transpose(A, 1, 2)$matmul(Y))[[1]]
    Y = torch::linalg_qr(A$matmul(Q))[[1]]
  }
  #orthonormalize
  Q = torch::linalg_qr(Y)[[1]]
  #project A
  B = torch::torch_transpose(Q, 1, 2)$matmul(A)
  #calculate the svd of smaller dimension
  out = torch::linalg_svd(B, full_matrices = FALSE)
  #recover left
  U = Q$matmul(out[[1]])

  #return results
  U = as.matrix(U$to(device = "cpu"))
  S = as.matrix(out[[2]]$to(device = "cpu"))
  Vh = as.matrix(out[[3]]$to(device = "cpu"))
  return(list(U = U[,1:k],
              S = S[1:k],
              Vh = Vh[1:k,]))
}
