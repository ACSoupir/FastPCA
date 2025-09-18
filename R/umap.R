#' umap
#'
#' @param pc_scores Matrix: exported from `get_pc_scores()`
#' @param n_neighbors Integer: number of nearest neighbors to use
#' @param n_components Integer: number of UMAP dimensions to calculate
#' @param method Character: The backend method for calculating UMAP (either 'uwot' or 'umap-learn'). Default is 'uwot'.
#' @param metric Character: How to calculate similarity. See details below for more informaation.
#' @param min_dist Numeric: minimum distance to consider
#' @param cores Integer: Number of cores to use for UMAP calculation
#' @param seed Integer: reproducibility seed. If cores > 1 or seed equals `-1`, seed will be ignored.
#' @param densemap Boolean: whether to use DensMAP for local densities
#' @param dense_lambda Numeric: value to apply for local density. Default: 2. higher values prioritize local density while low values are closer to typical UMAP
#' @param verbose Boolean: whether to be verbose in function calls
#' @param ... other parameters to pass to `uwot::umap` directly
#'
#' @returns matrix with UMAP reductions
#' @export
#'
#' @details
#'
#' `method`:
#'
#' Currently, there are two ways that UMAP can be calculated: 'uwot' or 'umap-learn'.
#' To use 'uwot', it's simply the installation of a dependency (likely already installed if FastPCA is installed).
#' This is the same method that is default for Seurat, and is very efficient and fast.
#' Some of the parameters that are explicit in the function call (like `n_neighbors` and `n_components`)
#' are passed while others shown in `uwot::umap`'s documentation can be passed by the `...`.
#'
#' Alternatively, there is a python method called `umap-learn`. To use this, a conda environment is preferred.
#' `FastPCA` provides interfaces to create and then activate the environment with `FastPCA::setup_py_env()` and
#' `FastPCA::start_FastPCA_env()`. Something to keep in mind, that if using this method it's recommended (required?)
#' to restart your R session, load `FastPCA`, then `FastPCA::start_FastPCA_env()`. There are system level conflicts
#' somwhere between `reticulate` and R's `torch` package.
#'
#' `metric`:
#'
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
                method = c("uwot", "umap-learn"),
                metric = "euclidean",
                min_dist = 0.1,
                cores = 4,
                seed = -1,
                densemap = TRUE,
                dense_lambda = 2,
                verbose = FALSE,
                ...){
  dots = list(...)
  method = match.arg(method)
  n_neighbors = as.integer(n_neighbors)
  n_components = as.integer(n_components)
  cores = as.integer(cores)
  seed = as.integer(seed)

  if(method == "umap-learn"){
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
  } else if(method == "uwot"){
    umap_vars = names(formals(uwot::umap))
    explicit_vars = list(X = pc_scores,
                         n_neighbors = n_neighbors,
                         n_components = n_components,
                         metric = metric,
                         min_dist = min_dist,
                         n_threads = cores,
                         seed = seed,
                         verbose = verbose)
    umap_params = dots[names(dots) %in% umap_vars]
    umap_params = umap_params[!(names(umap_params) %in% names(explicit_vars))]
    umap_params = c(explicit_vars, umap_params)
    out = do.call(uwot::umap,
                  umap_params, quote = TRUE)
  }

  return(out)
}

