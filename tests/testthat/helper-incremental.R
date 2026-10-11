# Fixtures for IncrementalPCA() tests ------------------------------------------
#
# Chunked data with a well-separated spectrum: the streaming update truncates
# every block to its leading components, so a flat spectrum makes any incremental
# method ill-conditioned and the agreement tests meaningless. Keeping four well
# separated directions makes "matches batch PCA" a real assertion while the files
# stay small enough for the suite to run in seconds.

incremental_fixture <- function(dir = tempfile("fastpca-chunks"),
                                counts = c(60, 75, 40, 55),
                                n_features = 10,
                                spectrum = c(30, 14, 7, 3),
                                seed = 20240501,
                                row.names = FALSE) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  set.seed(seed)
  loadings <- matrix(stats::rnorm(n_features * length(spectrum)),
                     nrow = n_features)
  parts <- vector("list", length(counts))
  for (i in seq_along(counts)) {
    scores_i <- matrix(stats::rnorm(counts[i] * length(spectrum)),
                       nrow = counts[i]) %*% diag(spectrum)
    parts[[i]] <- scores_i %*% t(loadings) +
      matrix(stats::rnorm(counts[i] * n_features), nrow = counts[i])
    colnames(parts[[i]]) <- paste0("f", seq_len(n_features))
    rownames(parts[[i]]) <- sprintf("s%02d_%04d", i, seq_len(counts[i]))
  }
  files <- file.path(dir, sprintf("part%02d.csv", seq_along(counts)))
  for (i in seq_along(parts)) {
    if (isTRUE(row.names)) {
      # a named identifier column, so tests can use row.names = 1 or "id"
      frame <- data.frame(id = rownames(parts[[i]]), parts[[i]],
                          check.names = FALSE)
      utils::write.csv(frame, files[i], row.names = FALSE)
    } else {
      utils::write.csv(parts[[i]], files[i], row.names = FALSE)
    }
  }
  list(files = files, dir = dir, parts = parts, X = do.call(rbind, parts))
}

# smallest cosine between two k-dimensional subspaces spanned by the rows of a
# and b (1 means identical, 0 orthogonal)
subspace_min_cosine <- function(a, b) {
  values <- svd(as.matrix(a) %*% t(as.matrix(b)))$d
  min(pmin(abs(values), 1))
}

# batch reference: singular values and right vectors of the centred/scaled matrix
batch_reference <- function(X, k = 4L, scale_columns = FALSE) {
  M <- if (isTRUE(scale_columns)) scale(X) else sweep(X, 2L, colMeans(X))
  decomposition <- svd(M)
  list(S = decomposition$d[seq_len(k)],
       Vh = t(decomposition$v[, seq_len(k), drop = FALSE]),
       centered = M)
}
