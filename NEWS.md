# FastPCA (development version)

* Detect an already-loaded OpenMP runtime (e.g. via data.table/Seurat) and clamp
  `cores` to 1 for OpenMP-backed backends (pytorch, rtorch) before it can crash
  the session. Reporting is configurable with `options(FastPCA.openmp_warning = )`.
* Coerce and validate the user-supplied `cores` in all entry points, fixing a
  "set_num_threads expects an int" error for double-valued input.

## Version 0.0.3

* Added tests
* Added updates to vignettes for "Suggests" packages
* Removed memory benchmarks
* Review comments 1
  * https://github.com/openjournals/joss-reviews/issues/9706#issuecomment-3735025935
  * https://github.com/openjournals/joss-reviews/issues/9706#issuecomment-3733686184

## Version 0.0.2

* Added spatial transcriptomics tutorial
* Added base-R backend as default in preparation for CRAN

## Version 0.0.1

* Initial builds with only pytorch backend
