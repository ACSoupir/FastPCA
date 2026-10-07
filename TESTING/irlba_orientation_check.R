#!/usr/bin/env Rscript
# Correct-orientation check: torch_irlba() (the package function) against
# irlba::irlba on the real example data.
#
# ORIENTATION. The Seurat assay behind this matrix has dim
# (features x cells) = 978 x 199112 with gene symbols in the rownames, so
# data/prepped_gex.rds is 199112 cells x 978 genes -- already samples x features.
# The earlier TESTING benchmarks ran on its transpose (nipals_export.R assumed
# rows were features), which reverses the centring axis. This script uses the
# data as-is and centres genes (columns), which is standard PCA.
#
#   Rscript TESTING/irlba_orientation_check.R [--k=10] [--repeats=2]
#
# Requires a Python environment with torch (RETICULATE_PYTHON) and the package
# loadable via pkgload.

args <- commandArgs(trailingOnly = TRUE)
getopt <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0L) return(default)
  sub(paste0("^--", name, "="), "", hit[[1]])
}

k <- as.integer(getopt("k", "10"))
repeats <- as.integer(getopt("repeats", "2"))

file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg)))
} else {
  normalizePath("TESTING")
}
repo <- dirname(here)
setwd(repo)

py <- Sys.getenv("FASTPCA_PYTHON", "/opt/anaconda3/envs/FastPCA/bin/python")
if (!nzchar(Sys.getenv("RETICULATE_PYTHON"))) Sys.setenv(RETICULATE_PYTHON = py)
Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE")

suppressMessages(pkgload::load_all(repo, quiet = TRUE))
invisible(suppressMessages(reticulate::py_config()))

x <- readRDS(file.path("data", "prepped_gex.rds"))
stopifnot(is.matrix(x), is.numeric(x))
cat(sprintf("prepped_gex.rds: %d x %d (samples x features)  %.2f GiB\n",
            nrow(x), ncol(x), as.numeric(object.size(x)) / 1024^3))

# gene-wise (column) centring, computed once and shared by every runner
xc <- sweep(x, 2L, colMeans(x), "-")
rm(x); invisible(gc(full = FALSE))

set.seed(42)
v0 <- rnorm(ncol(xc))
work <- min(k + 7L, min(dim(xc)))

time_it <- function(fun) {
  times <- numeric(repeats)
  fit <- NULL
  for (i in seq_len(repeats)) {
    t0 <- proc.time()[["elapsed"]]
    fit <- fun()
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  list(fit = fit, best = min(times))
}

cat(sprintf("k=%d work=%d, best of %d\n", k, work, repeats))

ref <- time_it(function() irlba::irlba(xc, nv = k, work = work, v = v0))
cat(sprintf("irlba::irlba (C)      %7.3f s\n", ref$best))

fp64 <- time_it(function() torch_irlba(xc, nv = k, work = work, v = v0))
cat(sprintf("torch_irlba fp64 CPU  %7.3f s\n", fp64$best))

fp32 <- time_it(function() torch_irlba(xc, nv = k, work = work, v = v0,
                                       precision = "single"))
cat(sprintf("torch_irlba fp32 CPU  %7.3f s\n", fp32$best))

mps <- try(time_it(function() torch_irlba(xc, nv = k, work = work, v = v0,
                                          precision = "single",
                                          device = "GPU")), silent = TRUE)

ref_sv <- as.numeric(ref$fit$d)
ref_u <- ref$fit$u
report <- function(label, fit) {
  d <- as.numeric(fit$S)
  rel <- max(abs(d^2 - ref_sv^2) / ref_sv^2)
  cosines <- apply(crossprod(ref_u, fit$U), 1L,
                   function(z) min(1, max(-1, abs(z))))
  ang <- acos(pmin(cosines, 1)) * 180 / pi
  cat(sprintf("%-22s rel. eig err %.3e   max angle %.5f deg\n",
              label, rel, max(ang)))
}

report("torch_irlba fp64", fp64$fit)
report("torch_irlba fp32", fp32$fit)

if (!inherits(mps, "try-error")) {
  cat(sprintf("torch_irlba fp32 MPS  %7.3f s\n", mps$best))
  report("torch_irlba fp32 MPS", mps$fit)
} else {
  cat("torch_irlba fp32 MPS: unavailable in this environment\n")
}

# Transpose invariance of the singular values, for context on the old benchmark.
cat("\ntranspose check (singular values are invariant):\n")
xt <- t(xc)
ref_t <- irlba::irlba(xt, nv = k, work = work)
cat(sprintf("  max |d(A) - d(t(A))| / d = %.3e\n",
            max(abs(ref$fit$d - ref_t$d) / ref$fit$d)))
cat("  (the centred subspaces differ because the centring axis changes)\n")
