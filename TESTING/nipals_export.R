#!/usr/bin/env Rscript
# Export the FastPCA example matrix to a raw binary both python and R can read.
#
# Python cannot read .rds, so every implementation loads the SAME bytes from
# TESTING/_tmp/. The MCAR missing-value mask is generated here too, once, so R
# and pytorch see byte-identical missingness.
#
#   Rscript TESTING/nipals_export.R --F=20000 --na-pct=5
#   Rscript TESTING/nipals_export.R --F=0            # all features (full tier)
#
# Writes TESTING/_tmp/:
#   X_<tag>.f64    float64, column-major, (samples x features)
#   M_<tag>.u8     0/1 bytes, same layout, 1 = missing
#   F_<tag>.i32    selected feature indices into data/prepped_gex.rds rows
#   X_<tag>.json   dims + provenance (hand-written; no jsonlite dependency)
#
# Orientation: the .rds is features x samples (199112 x 978). PCA wants
# samples as rows, so we export the transpose: 978 x F. All cells are kept;
# only features are subsampled, because that is the axis with room to spare
# (the vendored nipals package allocates O(ncol^2)).

args <- commandArgs(trailingOnly = TRUE)
getopt <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0L) return(default)
  sub(paste0("^--", name, "="), "", hit[[1]])
}

F_req  <- as.integer(getopt("F", "20000"))
na_pct <- as.numeric(getopt("na-pct", "0"))
seed   <- as.integer(getopt("seed", "42"))

here  <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
repo  <- dirname(here)
tmp   <- file.path(here, "_tmp")
dir.create(tmp, showWarnings = FALSE)

stopifnot(file.exists(file.path(repo, "data", "prepped_gex.rds")))
t0 <- Sys.time()
x  <- readRDS(file.path(repo, "data", "prepped_gex.rds"))
stopifnot(is.matrix(x), is.numeric(x))
cat(sprintf("loaded data/prepped_gex.rds  %d x %d  %.2f GiB  (%.1fs)\n",
            nrow(x), ncol(x), as.numeric(object.size(x)) / 1024^3,
            as.numeric(Sys.time() - t0, units = "secs")))

# --- feature subsample (rows of x) -----------------------------------------
nfeat <- nrow(x)
if (F_req <= 0L || F_req >= nfeat) {
  feat_idx <- seq_len(nfeat)
} else {
  set.seed(1L)
  feat_idx <- sort(sample.int(nfeat, F_req))
}
xs <- x[feat_idx, , drop = FALSE]
rm(x); invisible(gc(full = FALSE))

# --- transpose to samples x features ---------------------------------------
xt <- t(xs)
rm(xs); invisible(gc(full = FALSE))
storage.mode(xt) <- "double"
m <- nrow(xt); n <- ncol(xt)

# --- MCAR mask (generated once, shared by every implementation) ------------
if (na_pct > 0) {
  set.seed(seed)
  n_miss <- as.integer(round(na_pct / 100 * m * n))
  miss   <- logical(m * n)
  idx    <- sample.int(m * n, n_miss)          # positions in column-major order
  miss[idx] <- TRUE
  mask   <- matrix(miss, nrow = m, ncol = n)
  rm(miss); invisible(gc(full = FALSE))
  # MCAR at these rates cannot empty a row/column, but fail loudly if it does.
  stopifnot(!any(colSums(mask) == m), !any(rowSums(mask) == n))
} else {
  mask <- matrix(FALSE, nrow = m, ncol = n)
}
stopifnot(identical(dim(mask), dim(xt)))

tag <- sprintf("F%d_na%02d", n, as.integer(round(na_pct)))

# --- write ------------------------------------------------------------------
xpath <- file.path(tmp, sprintf("X_%s.f64", tag))
mpath <- file.path(tmp, sprintf("M_%s.u8",  tag))
fpath <- file.path(tmp, sprintf("F_%s.i32", tag))
jpath <- file.path(tmp, sprintf("X_%s.json", tag))

con <- file(xpath, "wb"); writeBin(as.double(xt), con, size = 8L, endian = "little"); close(con)
con <- file(mpath, "wb"); writeBin(as.raw(as.integer(mask)), con); close(con)
con <- file(fpath, "wb"); writeBin(as.integer(feat_idx), con, size = 4L, endian = "little"); close(con)

writeLines(sprintf(
  '{"m":%d,"n":%d,"na_pct":%g,"seed":%d,"tag":"%s","source":"data/prepped_gex.rds","orientation":"samples x features"}',
  m, n, na_pct, seed, tag), jpath)

cat(sprintf("wrote %s  (%d x %d = %.2f GiB, mask NAs=%d [%.3g%%], features kept %d/%d)\n",
            xpath, m, n, as.numeric(object.size(xt)) / 1024^3,
            sum(mask), 100 * mean(mask), length(feat_idx), nfeat))
cat(sprintf("TAG=%s\n", tag))
