#' umap
#'
#' @param pc_scores Matrix: exported from `get_pc_scores()`
#' @param n_neighbors Integer: number of nearest neighbors to use
#' @param n_components Integer: number of UMAP dimensions to calculate
#' @param metric Character: How to calculate similarity. See details below for more informaation.
#' @param min_dist Numeric: minimum distance to consider
#' @param cores Integer: Number of cores to use for UMAP calculation
#' @param seed Integer: reproducibility seed. If cores > 1 or seed equals `-1`, seed will be ignored.
#' @param densemap Boolean: whether to use DensMAP for local densities
#' @param dense_lambda Numeric: value to apply for local density. Default: 2. higher values prioritize local density while low values are closer to typical UMAP
#' @param verbose Boolean: whether to be verbose in function calls
#'
#' @returns matrix with UMAP reductions
#' @export
#'
#' @details
#'
#' `metric`
#' There are many metrics that are supported in the python implementation. Here
#' are the list in the [documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html#metric)
#' for the umap function;
#' - Minkowski syle metrics: "euclidean", "manhattan", "chebyshev", "minkowski";
#' - Miscellaneous spatial metrics: "canberra", "braycurtis", "haversine";
#' - Normalized spatial metrics: "mahalanobis", "wminkowski", "seuclidean";
#' - Angular and correlation metrics: "cosine", "correlation";
#' - Metrics for binary data: "hamming", "jaccard", "dice", "russellrao", "kulsinski", "rogerstanimoto", "sokalmichener", "sokalsneath", "yule"
#'
umap = function(pc_scores,
                n_neighbors = 15,
                n_components = 2,
                metric = "euclidean",
                min_dist = 0.1,
                cores = 4,
                seed = -1,
                densemap = TRUE,
                dense_lambda = 2,
                verbose = FALSE){
  n_neighbors = as.integer(n_neighbors)
  n_components = as.integer(n_components)
  cores = as.integer(cores)
  seed = as.integer(seed)
  densemap = as.integer(ifelse(densemap, 1, 0))
  verbose = as.integer(ifelse(verbose, 1, 0))

  #make sure environment is initialized and the script is loaded.
  if (!reticulate::py_available(initialize = FALSE)) {
    stop("Python environment not initialized. Please run `FastPCA::start_FastPCA_env()` first.")
  }

  .globals = python_functions()
  out = .globals$umap_calculation$umap_learn_py(pc_scores,
                                                n_neighbors = n_neighbors,
                                                n_components = n_components,
                                                metric = metric,
                                                min_dist = min_dist,
                                                cores = cores,
                                                seed = seed,
                                                densemap = densemap,
                                                dens_lambda = dense_lambda)
  return(out)
}

