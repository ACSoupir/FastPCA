.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    "Thank you for using FastPCA!\n",
    "To cite this package, run: citation('FastPCA')\n\n",
    "Either python OR R-torch can be used, but not both due to libomp.dylib conflicts"
  )
}
