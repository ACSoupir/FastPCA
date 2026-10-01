.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    "Thank you for using FastPCA!\n",
    "To cite this package, run: citation('FastPCA')\n\n",
    "Either python OR R-torch can be used, but not both due to libomp.dylib conflicts.\n",
    "When an OpenMP conflict is detected (e.g. Seurat/data.table loaded), FastPCA clamps 'cores' to 1.\n",
    "Control the warning with options(FastPCA.openmp_warning = 'always'|'once'|'silent')."
  )
}
