#!/usr/bin/env Rscript
# R side of the IRLBA benchmark. One implementation per process.
#
#   Rscript TESTING/irlba_r.R --impl=fast     --x=... --m=978 --n=2000 --nv=10
#   Rscript TESTING/irlba_r.R --impl=fallback ...
#
# Impls:
#   fast      irlba::irlba() -- the shipped C Lanczos implementation
#             (fastpath = TRUE, the default and what FastPCA(backend="irlba")
#             calls). This is "the irlba package version".
#   fallback  irlba::irlba(fastpath = FALSE) -- the pure-R Lanczos + implicit
#             restart that TESTING/irlba_torch.py ports. Used for parity, not
#             as a speed reference.
#
# A fixed starting vector is generated with set.seed() and handed to irlba via
# its v= argument, so runs are reproducible and the torch port can be given the
# exact same start (--v0). Column centring matches every other runner here.
#
# JSON goes to stdout on a RESULT_JSON line so the orchestrator can collect it.

args <- commandArgs(trailingOnly = TRUE)
getopt <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0L) return(default)
  sub(paste0("^--", name, "="), "", hit[[1]])
}

impl    <- getopt("impl", "fast")
xpath   <- getopt("x", "")
m       <- as.integer(getopt("m", "0"))
n       <- as.integer(getopt("n", "0"))
tag     <- getopt("tag", "")
nv      <- as.integer(getopt("nv", "10"))
work    <- as.integer(getopt("work", as.character(nv + 7L)))
maxit   <- as.integer(getopt("maxit", "1000"))
tol     <- as.numeric(getopt("tol", "1e-5"))
svtol   <- as.numeric(getopt("svtol", as.character(tol)))
center  <- as.integer(getopt("center", "1")) == 1L
seed    <- as.integer(getopt("seed", "42"))
max_gb  <- as.numeric(getopt("max-gb", "20"))
v0_out  <- getopt("v0-out", "")
save_bin<- getopt("save-bin", "")

stopifnot(nzchar(xpath), m > 0, n > 0)
suppressPackageStartupMessages(library(irlba))

read_bin <- function(path, size, what) {
  con <- file(path, "rb"); on.exit(close(con))
  readBin(con, what = what, n = m * n, size = size, endian = "little")
}

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

# pre-flight: the input itself must fit under the cap
in_gb <- (m * n * 8) / 1e9
if (in_gb > max_gb) {
  cat("RESULT_JSON ", jsonlite_like(list(impl = impl, tag = tag, m = m, n = n,
      status = "skipped", reason = sprintf("input alone is %.3g GB > cap %g GB",
                                           in_gb, max_gb))), "\n", sep = "")
  quit(save = "no")
}

t_load0 <- Sys.time()
X <- matrix(read_bin(xpath, 8L, "double"), nrow = m, ncol = n)
load_s <- as.numeric(Sys.time() - t_load0, units = "secs")
cat(sprintf("[R] loaded %dx%d impl=%s in %.2fs\n", m, n, impl, load_s))

if (center) X <- sweep(X, 2L, colMeans(X), "-")

# deterministic start vector shared with the torch port when --v0-out is given
set.seed(seed)
v0 <- rnorm(n)
if (nzchar(v0_out)) {
  con <- file(v0_out, "wb"); writeBin(as.double(v0), con, size = 8L, endian = "little"); close(con)
}

t0 <- Sys.time()
fit <- if (impl == "fast") {
  irlba::irlba(X, nv = nv, work = work, maxit = maxit, tol = tol,
               svtol = svtol, v = v0)
} else if (impl == "fallback") {
  irlba::irlba(X, nv = nv, work = work, maxit = maxit, tol = tol,
               svtol = svtol, v = v0, fastpath = FALSE)
} else {
  stop("unknown --impl=", impl)
}
elapsed <- as.numeric(Sys.time() - t0, units = "secs")

d <- fit$d
v <- fit$v                       # nvar x ncomp (right singular vectors)

if (nzchar(save_bin)) {
  con <- file(paste0(save_bin, ".eig.f64"), "wb")
  writeBin(as.numeric(d^2), con, size = 8L, endian = "little"); close(con)
  con <- file(paste0(save_bin, ".loadings.f64"), "wb")
  writeBin(as.numeric(v), con, size = 8L, endian = "little"); close(con)
  writeLines(sprintf('{"nvar":%d,"ncomp":%d}', nrow(v), ncol(v)),
             paste0(save_bin, ".json"))
}

gcmax <- sum(gc()[, "max used"] * c(8, 56) / 1e9)
out <- list(
  impl = paste0("irlba_", impl), tag = tag, m = m, n = n, status = "ok",
  backend = impl, center = center, nv = nv, work = work,
  tol = tol, svtol = svtol, maxit = maxit,
  load_s = round(load_s, 3), best_s = round(elapsed, 4),
  iterations = as.numeric(fit$iter), mprod = as.numeric(fit$mprod),
  sv = round(as.numeric(d), 8),
  eig = round(as.numeric(d^2), 8),
  r_gc_maxused_gb = round(gcmax, 3)
)
cat(sprintf("[R] %s best=%.3fs iter=%s mprod=%s sv[:5]=%s\n", impl, elapsed,
            format(fit$iter), format(fit$mprod),
            paste(round(d[seq_len(min(5, length(d)))], 4), collapse = ", ")))
cat("RESULT_JSON ", jsonlite_like(out), "\n", sep = "")
