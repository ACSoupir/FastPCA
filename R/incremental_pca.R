# Incremental (out-of-core) PCA ------------------------------------------------
#
# FastPCA(), NIPALS() and torch_irlba() all need the whole matrix in memory.
# IncrementalPCA() does not: it reads one block at a time from a list of chunk
# files and keeps only the O(ncomp * n_features) factorization state plus one
# block, so a dataset far larger than RAM can still yield its top components.
#
# The update is the standard incremental SVD: for a block X (rows = new
# observations, already centred/scaled) stack the previous factors with it and
# take a thin SVD,
#
#     [ diag(S) %*% Vh ]            ^      ^  ^
#     [       X         ]  = svd -> U, S_new, Vh_new ; keep the top ncomp
#
# and discard U (the observation-space factor is never needed: scores are formed
# on demand in the scoring pass). Chunked, that is exactly what a batch SVD of
# the concatenated matrix computes -- up to the centring estimate discussed in
# the help page.
#
# Reading is separated from computing (R/incremental_readers.R) and the SVD
# kernel is pluggable through four small methods below, so every backend runs
# the same driver loop and therefore sees identical pre-processing.

#' Incremental PCA from a list of chunk files
#'
#' Fits the top principal components of data that is split across several CSV
#' (or compressed) files without ever holding the whole matrix in memory. Each
#' file is read one block at a time, so peak R-side memory is roughly one block
#' plus the \code{ncomp x n_features} factorization state.
#'
#' @param files Character vector of chunk file paths, in the order their rows
#'   should be concatenated. Plain text/CSV, \code{.gz}, \code{.bz2} and
#'   \code{.xz} are read directly; \code{.zst}/\code{.zstd} need the external
#'   \code{zstd} binary. Other containers (parquet, HDF5) can be supported by
#'   passing a \code{reader}.
#' @param ncomp Integer. Number of components to extract (the analogue of
#'   \code{k} in [FastPCA()]).
#' @param oversample Integer >= 0. Extra components kept inside the streaming
#'   factorization and discarded at the end (default 10). Each block's tail is
#'   truncated, which is what makes the method approximate: retaining a few more
#'   components than requested preserves that information, and with enough rows
#'   per block (or a large enough buffer) the result becomes exact. Set to 0 for
#'   scikit-learn's \code{IncrementalPCA} behaviour.
#' @param chunk_rows Integer or NULL. Maximum number of rows read per block.
#'   \code{NULL} (default) reads each file in one block; set it when a single
#'   CSV is itself larger than memory. The per-block SVD cost grows with the
#'   square of the block size, so a moderate value (a few thousand rows) is also
#'   faster.
#' @param center Logical. Subtract the column means before decomposing (default
#'   \code{TRUE}). The stored mean is always the exact global mean.
#' @param scale Logical. Divide by the column standard deviations (default
#'   \code{FALSE}). This needs an extra statistics pass over the files so the
#'   scaling is exact rather than incremental.
#' @param log2 Logical. Apply \code{log2()} to every block before centring
#'   (default \code{FALSE}); non-positive entries produce -Inf/NaN, which is
#'   warned about and then rejected.
#' @param store_scores Logical. Also return the \code{n_obs x ncomp} score
#'   matrix (default \code{FALSE}). Scoring needs a pass over the files after
#'   the components are final, and the returned matrix is one row per
#'   observation; leave it \code{FALSE} for a genuinely constant-memory fit.
#' @param store_variance Logical. Store the total sum of squares so [pve()] and
#'   \code{plot(type = "cumulative")} work (default \code{TRUE}). Computed from
#'   streaming sums, so it costs no extra pass.
#' @param header Logical. Whether the files have a header row (passed to
#'   \code{read.table()}).
#' @param row.names Column index or name holding observation identifiers, passed
#'   to \code{read.table()}. Use this when the first column is an ID column;
#'   otherwise every column must be numeric. Default \code{NULL}.
#' @param sep Field separator (default \code{","}).
#' @param reader Optional function taking one file path and returning that whole
#'   file as a numeric matrix or data frame, with observation identifiers (if
#'   any) in the row names. This is the extension point for formats R cannot read
#'   here (parquet, HDF5); block boundaries are drawn after it returns, so a
#'   custom reader is not itself streaming.
#' @param backend Character. \code{"r"} (default, no Python), \code{"pytorch"}
#'   or \code{"tinygrad"}. The tinygrad backend runs float32 with a randomized
#'   sketch per block and is the least accurate of the three; see Details.
#' @param device Character, \code{"CPU"} (default) or \code{"GPU"}. Passed
#'   through the same resolution as [FastPCA()]: an unavailable accelerator falls
#'   back to the CPU with a message.
#' @param cores Integer. Threads for the backend; clamped to 1 when an OpenMP
#'   runtime conflict is detected in the session.
#' @param precision Character, \code{"double"} (default) or \code{"single"},
#'   used by the PyTorch backend. Apple Metal has no float64 support, so a
#'   double-precision GPU request runs on the CPU.
#'
#' @return A \code{"fastpca"} object with \code{method = "incremental"}, so
#'   [print()], \code{summary()}, [pve()] and \code{plot()} work as they do for
#'   an SVD result. It contains:
#'   \describe{
#'     \item{\code{Vh}}{the components, one row per PC (\code{ncomp} x
#'       \code{n_features}).}
#'     \item{\code{S}}{the singular values.}
#'     \item{\code{scores}, \code{U}}{only when \code{store_scores = TRUE};
#'       otherwise both are \code{NULL} and [get_pc_scores()] explains how to
#'       obtain them.}
#'   }
#'
#' @details
#' ## Passes over the data
#'
#' \code{scale} and \code{store_scores} decide how many times the files are
#' read, so the cost is predictable:
#'
#' \tabular{lll}{
#'   \code{scale} \tab \code{store_scores} \tab passes \cr
#'   FALSE \tab FALSE \tab 1: fit \cr
#'   TRUE \tab FALSE \tab 2: statistics, fit \cr
#'   FALSE \tab TRUE \tab 2: fit, score \cr
#'   TRUE \tab TRUE \tab 3: statistics, fit, score \cr
#' }
#'
#' ## Centring
#'
#' With a single pass the running mean is used: each block is centred with the
#' mean of everything read so far, and the stored \code{center} vector is still
#' the exact global mean. This matches scikit-learn's \code{IncrementalPCA} and
#' converges to batch PCA as blocks accumulate, but the components are not
#' bit-identical to \code{FastPCA()} on pre-centred data. Requesting exact
#' statistics (\code{scale = TRUE}, or any configuration with a statistics pass)
#' removes that difference: every block is then centred (and scaled) with the
#' same exact vectors, and the result matches batch PCA of the pre-processed data
#' to numerical precision.
#'
#' ## Backends and accuracy
#'
#' The \code{"r"} and \code{"pytorch"} backends compute an exact thin SVD of
#' each stacked block. The \code{"tinygrad"} backend has no usable exact SVD, so
#' it uses a randomized range finder with Newton-Schulz orthonormalisation and a
#' host-side small SVD, in float32; it is the fastest on Apple Metal but its
#' components are approximate (exact for well-separated spectra).
#'
#' Missing values are not supported here: use [NIPALS()] for matrices with
#' \code{NA}.
#'
#' @seealso [FastPCA()], [NIPALS()], [prep_matrix()]
#'
#' @examples
#' set.seed(1)
#' dir <- tempfile("chunks"); dir.create(dir)
#' files <- file.path(dir, sprintf("part%02d.csv", 1:4))
#' for (i in seq_along(files)) {
#'   part <- matrix(rnorm(25 * 8), nrow = 25)
#'   colnames(part) <- paste0("feature_", seq_len(ncol(part)))
#'   utils::write.csv(part, files[[i]], row.names = FALSE)
#' }
#'
#' fit <- IncrementalPCA(files, ncomp = 3, scale = TRUE)
#' dim(get_pc_scores(IncrementalPCA(files, ncomp = 3, scale = TRUE,
#'                                  store_scores = TRUE)))
#' fit
#'
#' @export
IncrementalPCA <- function(files,
                           ncomp = 10L,
                           oversample = 10L,
                           chunk_rows = NULL,
                           center = TRUE,
                           scale = FALSE,
                           log2 = FALSE,
                           store_scores = FALSE,
                           store_variance = TRUE,
                           header = TRUE,
                           row.names = NULL,
                           sep = ",",
                           reader = NULL,
                           backend = c("r", "pytorch", "tinygrad"),
                           device = c("CPU", "GPU"),
                           cores = 1L,
                           precision = c("double", "single")) {

  args <- .check_incremental_args(files, ncomp = ncomp, oversample = oversample,
                                  chunk_rows = chunk_rows,
                                  center = center, scale = scale, log2 = log2,
                                  store_scores = store_scores,
                                  store_variance = store_variance,
                                  header = header, row.names = row.names,
                                  sep = sep, reader = reader)
  files <- args$files
  ncomp <- args$ncomp

  # An R-torch incremental kernel is deliberately not shipped yet, and match.arg
  # would report it (and any other name) as a bare "invalid 'arg'" error, so the
  # backend vector is checked here where the message can be useful.
  if (is.character(backend)) {
    unknown <- setdiff(tolower(unique(backend)), .incremental_pca_backends)
    if (any(tolower(unique(backend)) == "rtorch")) {
      stop("The 'rtorch' backend is not implemented for IncrementalPCA(); use ",
           "\"r\", \"pytorch\" or \"tinygrad\". An R-torch kernel is tracked in ",
           "ROADMAP.md.", call. = FALSE)
    }
    if (length(unknown) > 0L) {
      stop("Unknown IncrementalPCA() backend(s): ",
           paste(unknown, collapse = ", "), ". Available backends: ",
           paste(.incremental_pca_backends, collapse = ", "), ".", call. = FALSE)
    }
  }

  resolved <- resolve_backend_device(backend, device,
                                     allowed = .incremental_pca_backends)
  backend <- resolved$backend
  device <- resolved$device
  precision <- .resolve_precision(precision)
  # OpenMP conflicts are process-wide, exactly as for the other backends
  cores <- .resolve_cores(cores, openmp = .backend_uses_openmp(backend))

  opts <- list(header = header, row.names = row.names, sep = sep,
               chunk_rows = args$chunk_rows, reader = reader,
               feature_names = NULL)

  # ---- pass 1 (optional): exact column statistics --------------------------
  fixed <- NULL
  if (isTRUE(scale)) {
    stats_pass <- .incremental_column_stats(files, opts, log2 = log2)
    if (stats_pass$n < 2L) {
      stop("\`scale = TRUE\` needs at least two observations to estimate a ",
           "standard deviation; the chunk files contain ", stats_pass$n, ".",
           call. = FALSE)
    }
    fixed <- .incremental_stats_to_vectors(stats_pass)
  }

  # ---- pass 2: fit ---------------------------------------------------------
  # the state keeps ncomp + oversample components; only the top ncomp are
  # returned, which is what limits how much of each block's tail is discarded
  state <- .new_incremental_state(backend = backend,
                                  ncomp = args$ncomp + args$oversample,
                                  device = device, cores = cores,
                                  precision = precision)
  var_sums <- NULL
  var_sumsq <- NULL
  n_seen <- 0L
  running_mean <- NULL
  feature_names <- NULL
  n_blocks <- 0L
  warn_log2 <- new.env(parent = emptyenv())

  stream <- .new_incremental_stream(files, opts)
  # finally = so an error inside a block still closes the open connection
  tryCatch(repeat {
    block <- stream$next_block()
    if (is.null(block)) break
    path <- files[[block$file_index]]
    x <- .incremental_prep_block(block$x, log2 = log2, warn_env = warn_log2)
    .incremental_check_block(x, path, log2 = log2)

    if (is.null(feature_names)) feature_names <- colnames(x)
    n_c <- nrow(x)

    if (isTRUE(store_variance)) {
      chunk_sums <- colSums(x)
      # one block-sized temporary; the alternative (crossprod) is p x p
      chunk_sumsq <- colSums(x * x)
      if (is.null(var_sums)) {
        var_sums <- chunk_sums
        var_sumsq <- chunk_sumsq
      } else {
        var_sums <- var_sums + chunk_sums
        var_sumsq <- var_sumsq + chunk_sumsq
      }
    }

    # first block: no prior estimate, so centre by its own means (nothing is
    # lost); later blocks: the running mean of everything seen so far, or the
    # exact vector when a statistics pass ran
    center_vec <- NULL
    if (isTRUE(center)) {
      center_vec <- if (!is.null(fixed)) fixed$mean else
        (if (n_seen == 0L) colMeans(x) else running_mean)
    }
    xc <- .incremental_center_scale(x, center_vec,
                                    if (is.null(fixed)) NULL else fixed$sd)
    state <- .partial_fit_incremental(state, xc)

    chunk_mean <- colMeans(x)
    running_mean <- if (is.null(running_mean)) chunk_mean else
      (running_mean * n_seen + chunk_mean * n_c) / (n_seen + n_c)
    n_seen <- n_seen + n_c
    n_blocks <- n_blocks + 1L
  }, finally = stream$close())

  if (n_seen == 0L) {
    stop("No data rows were found in the chunk file(s).", call. = FALSE)
  }

  fit <- .finalize_incremental(state, ncomp = ncomp)
  S <- as.numeric(fit$S)
  Vh <- as.matrix(fit$Vh)
  if (length(S) < ncomp) {
    stop(sprintf(paste0("Only %d component(s) are available from %d observation(s) ",
                        "and %d feature(s); reduce \`ncomp\` (requested %d)."),
                 length(S), n_seen, state$n_features, ncomp), call. = FALSE)
  }

  # sign convention shared with FastPCA()/NIPALS(): the largest-magnitude
  # loading of every component is positive, so repeated runs are comparable
  signs <- vapply(seq_along(S), function(j) {
    idx <- which.max(abs(Vh[j, ]))
    if (length(idx) == 0L || Vh[j, idx] >= 0) 1 else -1
  }, numeric(1))
  Vh <- sweep(Vh, 1L, signs, "*")
  dimnames(Vh) <- list(.pc_names(length(S)), feature_names)

  final_mean <- if (!is.null(fixed)) fixed$mean else running_mean

  # ---- pass 3 (optional): scores -------------------------------------------
  scores <- NULL
  U <- NULL
  if (isTRUE(store_scores)) {
    scores <- matrix(NA_real_, nrow = n_seen, ncol = length(S))
    ids_out <- if (is.null(row.names)) NULL else character(n_seen)
    position <- 0L
    warn_score <- new.env(parent = emptyenv())

    score_stream <- .new_incremental_stream(files, opts)
    tryCatch(repeat {
      block <- score_stream$next_block()
      if (is.null(block)) break
      x <- .incremental_prep_block(block$x, log2 = log2,
                                   warn_env = warn_score)
      xc <- .incremental_center_scale(x,
                                      if (isTRUE(center)) final_mean else NULL,
                                      if (is.null(fixed)) NULL else fixed$sd)
      rows <- position + seq_len(nrow(xc))
      scores[rows, ] <- .transform_incremental(state, xc)
      if (!is.null(ids_out)) {
        ids_out[rows] <- if (is.null(block$ids)) as.character(rows) else block$ids
      }
      position <- max(rows)
    }, finally = score_stream$close())

    scores <- sweep(scores, 2L, signs, "*")
    colnames(scores) <- .pc_names(length(S))
    if (!is.null(ids_out)) rownames(scores) <- make.unique(ids_out)
    # U is defined so that scores == U %*% diag(S) holds for every method
    U <- sweep(scores, 2L, S, "/")
    if (any(!is.finite(U))) U[!is.finite(U)] <- 0
    dimnames(U) <- list(rownames(scores), colnames(scores))
  }

  # ---- total variance for pve() -------------------------------------------
  total_variance <- NA_real_
  if (isTRUE(store_variance) && !is.null(var_sumsq)) {
    # sum over columns of the squared entries about the (exact) mean, undoing
    # any scaling; algebraically sum(x^2) - n * mean^2, so no extra pass
    centered_ss <- if (isTRUE(center)) var_sumsq - n_seen * final_mean^2 else
      var_sumsq
    if (!is.null(fixed)) centered_ss <- centered_ss / fixed$sd^2
    total_variance <- sum(pmax(centered_ss, 0))
  }

  out <- new_fastpca(
    U = U,
    S = S,
    Vh = Vh,
    scores = scores,
    method = "incremental",
    k = length(S),
    ncomp = length(S),
    exact = FALSE,
    backend = backend,
    device = if (identical(backend, "r")) "CPU" else device,
    n_obs = n_seen,
    n_features = state$n_features,
    total_variance = total_variance,
    center = if (isTRUE(center)) final_mean else NULL,
    scaling = if (!is.null(fixed) && isTRUE(scale)) fixed$sd else NULL,
    precision = if (identical(backend, "r")) NA_character_ else precision
  )
  attr(out, "files") <- files
  attr(out, "n_blocks") <- n_blocks
  attr(out, "passes") <- 1L + as.integer(!is.null(fixed)) +
    as.integer(isTRUE(store_scores))
  attr(out, "scores_stored") <- isTRUE(store_scores)
  out
}

# ---- shared pre-processing ---------------------------------------------------

.incremental_prep_block <- function(x, log2 = FALSE,
                                    warn_env = new.env(parent = emptyenv())) {
  if (!isTRUE(log2)) return(x)
  if (any(x <= 0) && !isTRUE(warn_env$warned)) {
    warning("'log2 = TRUE' with non-positive values produces -Inf/NaN; ",
            "shift the input before transforming.", call. = FALSE)
    warn_env$warned <- TRUE
  }
  # log2() would also emit one "NaNs produced" per negative entry; the warning
  # above already says what happened, and the block check stops right after
  suppressWarnings(log2(x))
}

.incremental_check_block <- function(x, path, log2 = FALSE) {
  # under log2 a non-positive entry is the likely cause, so say that instead of
  # sending the user to NIPALS() for a NaN this function created itself
  if (isTRUE(log2)) {
    if (!all(is.finite(x))) {
      stop("Chunk file '", basename(path), "' contains non-finite values after ",
           "the log2 transformation; every entry must be positive. Shift the ",
           "input or set 'log2 = FALSE'.", call. = FALSE)
    }
    return(invisible(TRUE))
  }
  if (anyNA(x)) {
    stop("Chunk file '", basename(path), "' contains missing values; ",
         "IncrementalPCA() requires complete data. Use NIPALS() for matrices ",
         "with NA.", call. = FALSE)
  }
  if (!all(is.finite(x))) {
    stop("Chunk file '", basename(path), "' contains non-finite values ",
         "(only -Inf/NaN are rejected here).", call. = FALSE)
  }
  invisible(TRUE)
}

.incremental_center_scale <- function(x, center_vec = NULL, scale_vec = NULL) {
  if (!is.null(center_vec)) x <- sweep(x, 2L, center_vec)
  if (!is.null(scale_vec)) x <- sweep(x, 2L, scale_vec, "/")
  x
}

# Streaming column statistics: sums and sums of squares are enough for an exact
# mean and (unbiased) SD without holding the data.
.incremental_column_stats <- function(files, opts, log2 = FALSE) {
  stream <- .new_incremental_stream(files, opts)
  warn_env <- new.env(parent = emptyenv())
  n_total <- 0L
  sums <- NULL
  sumsq <- NULL

  tryCatch(repeat {
    block <- stream$next_block()
    if (is.null(block)) break
    x <- .incremental_prep_block(block$x, log2 = log2, warn_env = warn_env)
    .incremental_check_block(x, files[[block$file_index]], log2 = log2)
    if (is.null(sums)) {
      sums <- numeric(ncol(x))
      sumsq <- numeric(ncol(x))
    }
    sums <- sums + colSums(x)
    sumsq <- sumsq + colSums(x * x)
    n_total <- n_total + nrow(x)
  }, finally = stream$close())
  if (n_total == 0L) {
    stop("No data rows were found in the chunk file(s).", call. = FALSE)
  }
  list(n = n_total, sum = sums, sumsq = sumsq)
}

.incremental_stats_to_vectors <- function(stats) {
  mean_vec <- stats$sum / stats$n
  var_vec <- pmax((stats$sumsq - stats$n * mean_vec^2) / (stats$n - 1), 0)
  sd_vec <- sqrt(var_vec)
  # a constant column cannot be scaled; sd 1 leaves it centred at zero, which is
  # the same choice NIPALS() makes under scale = TRUE
  constant <- !is.finite(sd_vec) | sd_vec <= 0
  if (any(constant)) {
    warning(sum(constant), " feature(s) have zero variance under 'scale = TRUE'; ",
            "using sd 1 for them.", call. = FALSE)
    sd_vec[constant] <- 1
  }
  list(mean = mean_vec, sd = sd_vec)
}

# ---- backend state ----------------------------------------------------------
#
# One interface for every kernel: new_incremental_state() -> partial_fit() per
# block (returns the state) -> finalize() for S and Vh, plus transform() when
# scores are requested. Python backends keep the factorization inside their own
# object so only one block crosses the R/Python boundary per call.

.new_incremental_state <- function(backend, ncomp, device = "CPU", cores = 1L,
                                   precision = "double") {
  state <- new.env(parent = emptyenv())
  state$backend <- backend
  state$ncomp <- as.integer(ncomp)
  state$device <- device
  state$cores <- as.integer(cores)[1]
  state$precision <- precision
  state$n_features <- NULL
  state$S <- NULL
  state$Vh <- NULL
  state$SV <- NULL      # diag(S) %*% Vh, the base-R factorization state
  state$py <- NULL      # reticulate reference for the Python backends
  state
}

.partial_fit_incremental <- function(state, X) {
  if (is.null(state$n_features)) state$n_features <- ncol(X)
  switch(
    state$backend,
    r = .partial_fit_incremental_r(state, X),
    pytorch = {
      if (is.null(state$py)) state <- .init_incremental_python(state, "incremental")
      invisible(state$py$partial_fit(X))
      state
    },
    tinygrad = {
      if (is.null(state$py)) {
        state <- .init_incremental_python(state, "incremental_tinygrad")
      }
      invisible(state$py$partial_fit(X))
      state
    },
    stop("Unsupported IncrementalPCA() backend: '", state$backend, "'.",
         call. = FALSE)
  )
}

.partial_fit_incremental_r <- function(state, X) {
  # stack the previous factors with the new block and take a thin SVD; U is
  # discarded, only Vh (and its scaled copy) carries over
  B <- if (is.null(state$SV)) X else rbind(state$SV, X)
  fit <- svd(B)
  k <- min(state$ncomp, length(fit$d))
  state$S <- fit$d[seq_len(k)]
  Vh <- t(fit$v)[seq_len(k), , drop = FALSE]
  state$Vh <- Vh
  state$SV <- sweep(Vh, 1L, state$S, "*")
  state
}

.init_incremental_python <- function(state, script) {
  check_backend(if (identical(state$backend, "tinygrad")) "tinygrad" else "pytorch")
  globals <- python_functions(script)
  module <- globals[[1L]]
  state$module <- module
  if (identical(state$backend, "pytorch")) {
    state$py <- module$IncrementalPCAState(
      n_features = as.integer(state$n_features),
      ncomp = as.integer(state$ncomp),
      device = state$device,
      cores = as.integer(state$cores),
      precision = state$precision
    )
  } else {
    state$py <- module$IncrementalPCAState(
      n_features = as.integer(state$n_features),
      ncomp = as.integer(state$ncomp)
    )
  }
  state
}

# Close the factorization and drop the oversampling buffer, so the returned
# components (and every later transform()) are exactly ncomp wide.
.finalize_incremental <- function(state, ncomp = state$ncomp) {
  if (is.null(state$py)) {
    keep <- min(ncomp, length(state$S))
    state$S <- state$S[seq_len(keep)]
    state$Vh <- state$Vh[seq_len(keep), , drop = FALSE]
    return(list(S = as.numeric(state$S), Vh = state$Vh))
  }
  fit <- state$py$finalize(as.integer(ncomp))
  S <- as.numeric(if (!is.null(fit$S)) fit$S else fit[[1L]])
  Vh <- if (!is.null(fit$Vh)) fit$Vh else fit[[2L]]
  if (!is.matrix(Vh)) Vh <- matrix(as.numeric(Vh), nrow = length(S))
  list(S = S, Vh = Vh)
}

.transform_incremental <- function(state, X) {
  if (is.null(state$py)) {
    return(X %*% t(state$Vh))
  }
  as.matrix(state$py$transform(X))
}
