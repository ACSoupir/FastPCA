#!/usr/bin/env Rscript
# R side of the NIPALS benchmark. One implementation per process.
#
#   Rscript TESTING/nipals_r.R --impl=base       --m=978 --n=2000 --x=... [--mask=...] --ncomp=10
#   Rscript TESTING/nipals_r.R --impl=nipals_pkg ...
#   Rscript TESTING/nipals_r.R --impl=irlba      ...            # complete data only
#
# Implementations:
#   base        hand-written NIPALS. Same maths as nipals::nipals (observed-only
#               centre/scale, observed sum-of-squares denominators when NA) but
#               it never builds the O(nobs^2)/O(nvar^2) Gram-Schmidt matrices and
#               it uses colSums() instead of apply(). This is the "what a careful
#               R user would write" reference, and it runs at full data size.
#   nipals_pkg  the vendored CRAN package source (TESTING/vendor/nipals), faithful
#               to upstream including its O(n^2) allocations. Subsample tiers only.
#   irlba       irlba::irlba on the centred matrix -- a non-NIPALS R speed/accuracy
#               yardstick for complete data.
#
# JSON goes to stdout on a RESULT_JSON line so the orchestrator can collect it.

args <- commandArgs(trailingOnly = TRUE)
getopt <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0L) return(default)
  sub(paste0("^--", name, "="), "", hit[[1]])
}
# accept both --x=path and a bare value for convenience
impl    <- getopt("impl", "base")
xpath   <- getopt("x", "")
mpath   <- getopt("mask", "")
m       <- as.integer(getopt("m", "0"))
n       <- as.integer(getopt("n", "0"))
tag     <- getopt("tag", "")
ncomp   <- as.integer(getopt("ncomp", "10"))
tol     <- as.numeric(getopt("tol", "1e-6"))
maxiter <- as.integer(getopt("maxiter", "500"))
scale_it<- as.integer(getopt("scale", "0")) == 1L
gs      <- as.integer(getopt("gramschmidt", "1")) == 1L
max_gb  <- as.numeric(getopt("max-gb", "12"))
save_bin<- getopt("save-bin", "")

stopifnot(nzchar(xpath), m > 0, n > 0)

read_bin <- function(path, size, what) {
  con <- file(path, "rb"); on.exit(close(con))
  readBin(con, what = what, n = m * n, size = size, endian = "little")
}

t_load0 <- Sys.time()
X <- matrix(read_bin(xpath, 8L, "double"), nrow = m, ncol = n)
mask <- if (nzchar(mpath)) {
  matrix(as.logical(read_bin(mpath, 1L, "raw")), nrow = m, ncol = n)
} else NULL
load_s <- as.numeric(Sys.time() - t_load0, units = "secs")
has_na <- !is.null(mask) && any(mask)
cat(sprintf("[R] loaded %dx%d impl=%s na=%s in %.2fs\n", m, n, impl, has_na, load_s))

# --- tiny JSON writer (no jsonlite dependency) -----------------------------
jsonlite_like <- function(x) {
  esc <- function(s) gsub('"', '\\"', s)
  fmt1 <- function(v) {
    if (is.list(v)) return(jsonlite_like(v))
    if (length(v) == 1L && is.character(v)) return(paste0('"', esc(v), '"'))
    if (length(v) == 1L && is.logical(v))   return(ifelse(is.na(v), "null", ifelse(v, "true", "false")))
    if (length(v) == 1L && is.na(v))        return("null")
    v <- v[!is.na(v)]                       # NA (e.g. irlba iteration counts) -> drop
    if (length(v) == 0L)                    return("[]")
    if (length(v) == 1L) return(format(v, digits = 10))
    paste0("[", paste(vapply(v, fmt1, character(1)), collapse = ","), "]")
  }
  paste0("{", paste(sprintf('"%s":%s', names(x), vapply(x, fmt1, character(1))), collapse = ","), "}")
}

# --- peak-memory guard -----------------------------------------------------
alloc_gb <- function() {
  worst <- max(m, n)^2 * 8 / 1024^3
  # the package also copies x (m*n*8) a few times per iteration in the NA path
  worst + if (has_na) 2 * m * n * 8 / 1024^3 else 0
}
if (impl == "nipals_pkg") {
  need <- alloc_gb()
  if (need > max_gb) {
    msg <- sprintf("N/A: %s would allocate ~%.1f GB (nobs^2/nvar^2 Gram-Schmidt matrix); cap is %.0f GB",
                   impl, need, max_gb)
    cat("RESULT_JSON ", jsonlite_like(list(impl = impl, tag = tag, m = m, n = n,
        status = "skipped", reason = msg)), "\n", sep = "")
    quit(save = "no")
  }
}

# --- base R NIPALS ---------------------------------------------------------
nipals_base <- function(X, mask = NULL, ncomp = 10L, center = TRUE,
                        scale = FALSE, maxiter = 500L, tol = 1e-6,
                        gramschmidt = TRUE) {
  nobs <- nrow(X); nvar <- ncol(X)
  has_na <- !is.null(mask) && any(mask)

  obsd <- NULL
  if (has_na) {
    obs <- !mask
    cnt <- colSums(obs)
    cmeans <- as.vector(colSums(X * obs) / cnt)
  } else {
    cmeans <- colMeans(X)
  }
  if (center) X <- sweep(X, 2L, cmeans, "-")
  csds <- NULL
  if (scale) {
    mu2 <- if (has_na) colSums((X * X) * obs) / (cnt - 1) else apply(X, 2L, stats::var)
    csds <- sqrt(mu2)
    if (any(csds == 0)) stop("At least one column has zero standard deviation")
    X <- sweep(X, 2L, csds, "/")
  }
  x0 <- if (has_na) { z <- X; z[mask] <- 0; z } else X
  # a numeric copy of the observation mask: lets every NA denominator be a single
  # BLAS matvec instead of materialising an nobs x nvar temporary per iteration
  if (has_na) obsd <- matrix(as.numeric(obs), nobs, nvar)

  eig      <- numeric(ncomp)
  loadings <- matrix(0, nvar, ncomp)
  scores   <- matrix(0, nobs, ncomp)
  iters    <- integer(ncomp)

  for (h in seq_len(ncomp)) {
    scol <- which.max(if (has_na) colSums(obsd * abs(x0)) else colSums(abs(X)))
    th <- x0[, scol]
    pciter <- 1L
    repeat {
      if (has_na) {
        denom_p <- as.vector(crossprod(obsd, th * th))
        ph <- as.vector(crossprod(x0, th)) / denom_p
      } else {
        ph <- as.vector(crossprod(X, th)) / sum(th * th)
      }
      if (gramschmidt && h > 1L) {
        P <- loadings[, seq_len(h - 1L), drop = FALSE]
        ph <- ph - P %*% (crossprod(P, ph))
      }
      nrm <- sqrt(sum(ph * ph)); if (nrm == 0) stop("zero-norm loading")
      ph <- ph / nrm

      th_old <- th
      if (has_na) {
        denom_t <- as.vector(obsd %*% (ph * ph))       # per-sample observed sums
        th <- as.vector(x0 %*% ph) / denom_t
      } else {
        th <- as.vector(X %*% ph) / sum(ph * ph)
      }
      if (gramschmidt && h > 1L) {
        T <- scores[, seq_len(h - 1L), drop = FALSE]
        eigs_prev <- sqrt(eig[seq_len(h - 1L)])
        Tn <- sweep(T, 2L, eigs_prev, "/")
        th <- th - Tn %*% (crossprod(Tn, th))
      }
      nn <- sqrt(sum(th * th)); no <- sqrt(sum(th_old * th_old))
      if (nn > 0 && no > 0) {
        d <- th / nn - th_old / no
        if (sum(d * d) < tol) break
      }
      if (pciter == maxiter) { warning("maxiter reached for PC ", h); break }
      pciter <- pciter + 1L
    }
    X  <- X - tcrossprod(th, ph)
    if (has_na) { x0 <- X; x0[mask] <- 0 }   # masked entries stay out of every matvec
    loadings[, h] <- ph; scores[, h] <- th
    eig[h] <- sum(th * th); iters[h] <- pciter
  }
  sv <- sqrt(eig)
  list(scores = sweep(scores, 2L, sv, "/"), loadings = loadings,
       eig = eig, iter = iters)
}

# --- run -------------------------------------------------------------------
res <- NULL
t0 <- Sys.time()
if (impl == "base") {
  res <- nipals_base(X, mask = mask, ncomp = ncomp, center = TRUE,
                     scale = scale_it, maxiter = maxiter, tol = tol,
                     gramschmidt = gs)
  res$eig <- as.numeric(res$eig)   # base returns eigenvalues (not sv)
} else if (impl == "nipals_pkg") {
  here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
  source(file.path(here, "vendor", "nipals", "R", "nipals.R"))
  Xuse <- X
  if (has_na) Xuse[mask] <- NA_real_
  fit <- nipals(Xuse, ncomp = ncomp, center = TRUE, scale = scale_it,
                maxiter = maxiter, tol = tol, gramschmidt = gs)
  # nipals() returns $eig already as singular-value scale after sqrt(); recover SS
  res <- list(scores = fit$scores, loadings = fit$loadings,
              eig = (fit$eig)^2, iter = as.integer(fit$iter))
} else if (impl == "irlba") {
  if (has_na) stop("irlba requires complete data")
  Xc <- sweep(X, 2L, colMeans(X), "-")
  fit <- irlba::irlba(Xc, nv = ncomp)
  res <- list(scores = fit$u %*% diag(fit$d), loadings = fit$v,
              eig = fit$d^2, iter = rep(NA_integer_, ncomp))
} else {
  stop("unknown --impl=", impl)
}
elapsed <- as.numeric(Sys.time() - t0, units = "secs")

if (nzchar(save_bin)) {
  con <- file(paste0(save_bin, ".eig.f64"), "wb")
  writeBin(as.numeric(res$eig), con, size = 8L, endian = "little"); close(con)
  con <- file(paste0(save_bin, ".loadings.f64"), "wb")
  writeBin(as.numeric(res$loadings), con, size = 8L, endian = "little"); close(con)
  writeLines(sprintf('{"nvar":%d,"ncomp":%d}', nrow(res$loadings), ncol(res$loadings)),
             paste0(save_bin, ".json"))
}

gcmax <- sum(gc()[, "max used"] * c(8, 56) / 1e9)
out <- list(
  impl = impl, tag = tag, m = m, n = n, status = "ok",
  has_mask = has_na, ncomp = ncomp, scale = scale_it,
  tol = tol, maxiter = maxiter,
  load_s = round(load_s, 3), best_s = round(elapsed, 4),
  iters = as.numeric(res$iter), eig = round(as.numeric(res$eig), 8),
  r_gc_maxused_gb = round(gcmax, 3)
)
cat(sprintf("[R] %s best=%.3fs eig[:5]=%s\n", impl, elapsed,
            paste(round(res$eig[seq_len(min(5, length(res$eig)))], 4), collapse = ", ")))
cat("RESULT_JSON ", jsonlite_like(out), "\n", sep = "")
