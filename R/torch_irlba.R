# torch_irlba(): irlba's Lanczos SVD on a PyTorch backend --------------------
#
# The algorithm is a faithful transcription of irlba's own iterative solver
# (Lanczos bidiagonalisation with implicit restart, classical Gram-Schmidt and
# the convtests() stopping rule), so it reaches irlba's answer at PyTorch speed
# instead of estimating a random sketch. Argument handling is shared through
# R/validate.R; only the inner iteration lives in inst/python/torch_irlba_func.py.

#' Lanczos SVD (IRLBA) with a PyTorch backend
#'
#' \code{torch_irlba()} computes the leading singular triplets of a matrix with
#' irlba's own algorithm -- Lanczos bidiagonalisation with implicit restart --
#' run on PyTorch through \pkg{reticulate}. It is the iterative sibling of
#' [FastPCA()]: where FastPCA estimates the subspace from a random sketch,
#' \code{torch_irlba()} refines a Krylov subspace until irlba's convergence test
#' is satisfied, so it stays near-exact on the flat spectra where randomized
#' methods drift (see the section in [FastPCA()]).
#'
#' The arguments mirror [irlba::irlba()]. As with every FastPCA entry point, the
#' input is oriented \strong{observations x features}: rows are samples or cells,
#' columns are genes or other features. For the package example that is
#' \code{199112 x 978} (cells by genes); passing the transpose would decompose
#' features as if they were observations.
#'
#' @param input_r_matrix A dense numeric matrix, with rows as observations
#'   (samples or cells) and columns as features. Missing values are not allowed;
#'   use [NIPALS()] for matrices with \code{NA}.
#' @param nv Integer. Number of singular triplets (components) to compute
#'   (default 5, as in [irlba::irlba()]). Must be strictly less than the
#'   smallest matrix dimension.
#' @param maxit Integer. Maximum number of Lanczos restarts (default 1000).
#' @param work Integer. Number of Lanczos vectors carried between restarts.
#'   Defaults to \code{nv + 7} (as in [irlba::irlba()]) and is capped at the
#'   smallest matrix dimension.
#' @param reorth Logical. Re-orthogonalise each new Lanczos vector against the
#'   previous ones (default \code{TRUE}). Disabling it is faster but numerically
#'   unsafe on clustered singular values.
#' @param tol Numeric > 0. Subspace convergence tolerance, compared against the
#'   residual \code{|A v - sigma u|} relative to the largest singular value
#'   (default 1e-5).
#' @param svtol Numeric > 0. Additional convergence test on the change in each
#'   singular value between restarts. Defaults to \code{tol}.
#' @param v Optional numeric starting vector of length \code{ncol(input_r_matrix)}.
#'   When supplied, the Lanczos iteration starts from it -- this is how a run can
#'   be made reproducible or matched exactly against [irlba::irlba()]. When
#'   \code{NULL}, a random start is drawn by PyTorch (note that R's
#'   \code{set.seed()} does not control it; pass \code{v} for reproducibility).
#' @param center Logical. Subtract the column means before decomposing (default
#'   \code{FALSE}, as in [irlba::irlba()]). Columns are features, so centring is
#'   the usual gene-wise centring.
#' @param scale Logical. Divide by the column standard deviations after centring
#'   (default \code{FALSE}).
#' @param precision Character, either \code{"double"} (default) or
#'   \code{"single"}. PyTorch normally runs in float64; \code{"single"} casts
#'   the working matrix to float32, which roughly halves memory and is often
#'   2x faster on the CPU. It costs about six digits of agreement (measured:
#'   relative singular-value error around \code{1e-5}).
#' @param device Character, either \code{"CPU"} (default) or \code{"GPU"}. A GPU
#'   request selects CUDA when it is available and otherwise Apple Metal (MPS)
#'   on arm64 Macs. \strong{This is the one place precision and device
#'   interact}: MPS has no float64 support, so \code{device = "GPU"} with
#'   \code{precision = "double"} is served on the CPU and reports why; pass
#'   \code{precision = "single"} to reach Metal. A GPU request with no
#'   accelerator available falls back to the CPU, never an error.
#' @param cores Integer. Number of CPU threads (default 1). Clamped to 1 when an
#'   OpenMP runtime conflict is detected (see [FastPCA()]). On Apple silicon the
#'   matmul runs on a single-core matrix unit regardless, so this rarely helps.
#' @param store_variance Logical. When \code{TRUE} (default) the total variance
#'   of the input is stored so [pve()] can report percent variation explained.
#'
#' @return An object of class \code{"fastpca"} with the same elements as a
#'   [FastPCA()] result: \code{U} (observations x nv), \code{S}, and \code{Vh}
#'   (nv x features). The number of restarts is stored as the
#'   \code{"n_iter"} attribute, the count of matrix products as
#'   \code{"mprod"}, and the precision used as \code{"precision"}.
#'
#' @details
#' The Python kernel transcribes the pure-R fallback of \pkg{irlba}
#' (\code{fastpath = FALSE}): the same start-vector handling,
#' \code{irlba::orthog()} reorthogonalisation, the same incremental bidiagonal
#' matrix and its small \code{svd()}, and irlba's own \code{convtests()}
#' stopping rule. Given the same starting vector it reproduces
#' \code{irlba::irlba()} to about \code{1e-11} in the singular values.
#'
#' Some [irlba::irlba()] arguments are intentionally not exposed:
#' \code{nu} (left and right vectors always come in matched pairs here),
#' \code{right_only}, \code{smallest} (leading triplets only), \code{shift},
#' \code{mult}, \code{deflate} and \code{fastpath}.
#'
#' @seealso [FastPCA()] for the randomized/exact SVD backends and
#'   \code{backend = "irlba"} for the CRAN implementation on sparse input;
#'   [NIPALS()] when the matrix contains missing values.
#'
#' @examples
#' \dontrun{
#' # requires setup_py_env()/start_FastPCA_env()
#' set.seed(1)
#' x <- matrix(rnorm(500 * 40), nrow = 500, ncol = 40)
#'
#' # near-exact leading SVD at PyTorch speed
#' fit <- torch_irlba(x, nv = 5)
#'
#' # float32 on the GPU where available
#' fast <- torch_irlba(x, nv = 5, precision = "single", device = "GPU")
#'
#' # reproducible against the CRAN solver: share one starting vector
#' v0 <- rnorm(ncol(x))
#' ref  <- irlba::irlba(scale(x, scale = FALSE), nv = 5, v = v0)
#' got  <- torch_irlba(scale(x, scale = FALSE), nv = 5, v = v0)
#' max(abs(got$S - ref$d))
#' }
#'
#' @export
torch_irlba <- function(input_r_matrix,
                        nv = 5L,
                        maxit = 1000L,
                        work = NULL,
                        reorth = TRUE,
                        tol = 1e-5,
                        svtol = NULL,
                        v = NULL,
                        center = FALSE,
                        scale = FALSE,
                        precision = c("double", "single"),
                        device = c("CPU", "GPU"),
                        cores = 1L,
                        store_variance = TRUE) {
  precision <- .resolve_precision(precision)
  device <- toupper(as.character(device))
  device <- match.arg(device, .fastpca_devices)

  # validate everything before touching Python so argument errors are reported
  # even in a session without an interpreter
  .check_matrix_input(input_r_matrix, allow_sparse = FALSE)

  args <- .check_irlba_args(input_r_matrix, nv = nv, work = work,
                            tol = tol, svtol = svtol, maxit = maxit)
  nv <- args$nv
  work <- args$work
  tol <- args$tol
  svtol <- args$svtol
  maxit <- args$maxit

  if (!is.null(v)) {
    v <- as.numeric(v)
    if (length(v) != ncol(input_r_matrix)) {
      stop("`v` must be a starting vector with one entry per column of ",
           "`input_r_matrix`.", call. = FALSE)
    }
  }

  # clamp to a single thread when an OpenMP runtime conflict is present (R/openmp.R)
  cores <- .resolve_cores(cores, openmp = TRUE)

  check_backend("pytorch")

  total_variance <- if (isTRUE(store_variance)) {
    .fastpca_total_variance(input_r_matrix)
  } else NA_real_

  # torch_irlba has its own module, so loading it never pulls in the SVD or umap
  # scripts (R/utils.R)
  .globals <- python_functions(backend = "torch_irlba")
  on.exit(rm(.globals), add = TRUE)
  out <- .globals$torch_irlba_func$torch_irlba_py(
    input_r_matrix,
    nv = as.integer(nv),
    work = as.integer(work),
    maxit = as.integer(maxit),
    tol = as.numeric(tol),
    svtol = as.numeric(svtol),
    reorth = isTRUE(reorth),
    v0 = if (is.null(v)) NULL else as.numeric(v),
    center = isTRUE(center),
    scale_ = isTRUE(scale),
    precision = precision,
    device = toupper(as.character(device)),
    cores = as.integer(cores)
  )

  U_r <- as.matrix(out$u)       # (rows of input x nv)
  S_r <- as.numeric(out$d)      # singular values
  Vh_r <- t(as.matrix(out$v))   # (nv x columns of input)

  colnames(U_r) <- paste0("dim", seq_len(ncol(U_r)))
  if (!is.null(rownames(input_r_matrix))) {
    rownames(U_r) <- rownames(input_r_matrix)
  }
  rownames(Vh_r) <- paste0("dim", seq_len(nrow(Vh_r)))
  if (!is.null(colnames(input_r_matrix))) {
    colnames(Vh_r) <- colnames(input_r_matrix)
  }

  invisible(gc(full = TRUE))

  new_fastpca(
    U = U_r, S = S_r, Vh = Vh_r,
    k = length(S_r), p = NULL, q_iter = NULL, exact = FALSE,
    backend = "torch_irlba", device = toupper(out$device_resolved),
    n_obs = nrow(input_r_matrix), n_features = ncol(input_r_matrix),
    total_variance = total_variance,
    tol = as.numeric(tol), work = as.integer(work),
    n_iter = as.integer(out$iterations), mprod = as.integer(out$mprod),
    precision = precision, converged = isTRUE(out$converged)
  )
}
