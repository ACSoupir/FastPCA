.onLoad <- function(libname, pkgname) {
  # tinygrad opens its cache database the moment it is imported. Point
  # XDG_CACHE_HOME at a writable directory before anything can import it, since
  # the default location is read-only in some sandboxed/reticulate sessions.
  try(.ensure_tinygrad_cache_dir(), silent = TRUE)
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    "Thank you for using FastPCA!\n",
    "To cite this package, run: citation('FastPCA')\n\n",
    "Either python OR R-torch can be used, but not both due to libomp.dylib conflicts.\n",
    "When an OpenMP conflict is detected (e.g. Seurat/data.table loaded), FastPCA clamps 'cores' to 1.\n",
    "Control the warning with options(FastPCA.openmp_warning = 'always'|'once'|'silent')."
  )
}
