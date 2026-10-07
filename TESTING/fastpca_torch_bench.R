#!/usr/bin/env Rscript
# FastPCA(backend = "pytorch") as a user calls it: the installed package's R
# API, which converts an R matrix to numpy via reticulate and runs the shipped
# randomized_svd_py kernel. One process per run.
#
#   Rscript TESTING/fastpca_torch_bench.R --x=... --m=978 --n=2000 \
#           --k=10 --p=10 --q_iter=2 --cores=4
#
# The matrix is column-centred here so it matches the iterative IRLBA runs and
# the direct-kernel row. A small warm-up call loads torch + umap first, because
# the package's python_functions() imports every script it ships (including
# umap_func.py) on the first call -- ~3 s that would otherwise be charged to
# the decomposition. Both cold and steady-state times are reported.
#
# Requires NUMBA_CACHE_DIR to point at a writable directory (the package's
# tempdir() fallback is not enough here), otherwise importing umap aborts with
# "cannot cache function ... no locator available".

args <- commandArgs(trailingOnly = TRUE)
getopt <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0L) return(default)
  sub(paste0("^--", name, "="), "", hit[[1]])
}

xpath   <- getopt("x", "")
m       <- as.integer(getopt("m", "0"))
n       <- as.integer(getopt("n", "0"))
tag     <- getopt("tag", "")
k       <- as.integer(getopt("k", "10"))
p       <- as.integer(getopt("p", "10"))
q_iter  <- as.integer(getopt("q_iter", "2"))
cores   <- as.integer(getopt("cores", "4"))
repeats <- as.integer(getopt("repeats", "3"))
center  <- as.integer(getopt("center", "1")) == 1L
max_gb  <- as.numeric(getopt("max-gb", "20"))
save_bin<- getopt("save-bin", "")
json_out<- getopt("json", "")

stopifnot(nzchar(xpath), m > 0, n > 0)
py <- Sys.getenv("FASTPCA_PYTHON", "/opt/anaconda3/envs/FastPCA/bin/python")

jsonlite_like <- function(x) {
  esc <- function(s) gsub('"', '\\\\"', s)
  fmt1 <- function(v) {
    if (is.list(v)) return(jsonlite_like(v))
    if (length(v) == 1L && is.character(v)) return(paste0('"', esc(v), '"'))
    if (length(v) == 1L && is.logical(v))   return(ifelse(is.na(v), "null", ifelse(v, "true", "false")))
    if (length(v) == 1L && is.na(v))        return("null")
    v <- v[!is.na(v)]
    if (length(v) == 0L)                    return("[]")
    if (length(v) == 1L) return(format(v, digits = 10))
    paste0("[", paste(vapply(v, fmt1, character(1)), collapse = ","), "]")
  }
  paste0("{", paste(sprintf('"%s":%s', names(x), vapply(x, fmt1, character(1))), collapse = ","), "}")
}

read_bin <- function(path) {
  con <- file(path, "rb"); on.exit(close(con))
  readBin(con, what = "double", n = m * n, size = 8L, endian = "little")
}

in_gb <- (m * n * 8) / 1e9
if (in_gb > max_gb) {
  cat("RESULT_JSON ", jsonlite_like(list(impl = "fastpca_torch", tag = tag, m = m,
      n = n, status = "skipped",
      reason = sprintf("input alone is %.3g GB > cap %g GB", in_gb, max_gb))),
      "\n", sep = "")
  quit(save = "no")
}

suppressPackageStartupMessages(library(FastPCA))
reticulate::use_python(py, required = TRUE)
invisible(reticulate::py_config())

t_load0 <- Sys.time()
X <- matrix(read_bin(xpath), nrow = m, ncol = n)
load_s <- as.numeric(Sys.time() - t_load0, units = "secs")
if (center) X <- sweep(X, 2L, colMeans(X), "-")

# warm-up: load the package's python scripts (torch + umap) off the clock
t_w0 <- Sys.time()
invisible(FastPCA(matrix(rnorm(40 * 12), 40, 12), k = 3L, p = 2L,
                  q_iter = 1L, backend = "pytorch", cores = cores))
warmup_s <- as.numeric(Sys.time() - t_w0, units = "secs")

times <- numeric(repeats); res <- NULL
for (r in seq_len(repeats)) {
  t0 <- Sys.time()
  res <- FastPCA(X, k = k, p = p, q_iter = q_iter,
                 backend = "pytorch", cores = cores)
  times[r] <- as.numeric(Sys.time() - t0, units = "secs")
  cat(sprintf("[fastpca_torch] repeat %d: %.3fs\n", r, times[r]))
}
best <- min(times)
S <- as.numeric(res$S); Vh <- res$Vh

if (nzchar(save_bin)) {
  v <- t(Vh)                       # nvar x k, column-major like the other runners
  con <- file(paste0(save_bin, ".eig.f64"), "wb")
  writeBin(as.numeric(S^2), con, size = 8L, endian = "little"); close(con)
  con <- file(paste0(save_bin, ".loadings.f64"), "wb")
  writeBin(as.numeric(v), con, size = 8L, endian = "little"); close(con)
  writeLines(sprintf('{"nvar":%d,"ncomp":%d}', nrow(v), ncol(v)),
             paste0(save_bin, ".json"))
}

gcmax <- sum(gc()[, "max used"] * c(8, 56) / 1e9)
out <- list(
  impl = "fastpca_torch", tag = tag, m = m, n = n, status = "ok",
  pkg_version = as.character(utils::packageVersion("FastPCA")),
  k = k, p_oversample = p, q_iter = q_iter, cores = cores,
  center = center, warmup_s = round(warmup_s, 3),
  load_s = round(load_s, 3), times_s = round(times, 4),
  best_s = round(best, 4), sv = round(S, 8), eig = round(S^2, 8),
  r_gc_maxused_gb = round(gcmax, 3)
)
cat(sprintf("[fastpca_torch] best=%.3fs (warmup %.2fs) sv[:5]=%s\n", best, warmup_s,
            paste(round(head(S, 5), 4), collapse = ", ")))
if (nzchar(json_out)) writeLines(jsonlite_like(out), json_out)
cat("RESULT_JSON ", jsonlite_like(out), "\n", sep = "")
