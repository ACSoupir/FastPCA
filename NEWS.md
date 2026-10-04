# FastPCA 0.9.1

## CRAN readiness

* `TESTING/`, Python bytecode caches and scratch scripts are excluded from the
  built tarball, so no heavy development data ships with the package.
* `LICENSE` is a valid CRAN MIT stub (full text kept in `LICENSE.md`), and
  `SystemRequirements` records the Python version needed for optional backends.
* Help pages regenerated with roxygen2: previously undocumented arguments
  (`device`, `log2`) are documented and the malformed Seurat Rd page is fixed.
* Added `inst/CITATION`, citing the JOSS paper (doi:10.21105/joss.09706).

## Functionality and fixes

* `FastPCA()` returns a `"fastpca"` S3 object with `print()`, `summary()`,
  `plot()` and `pve()` methods; percent variation explained is available via
  `store_variance` (default `TRUE`).
* Every entry point now uses the backend/device resolved by
  `validate_backend()`. Requesting rtorch without torch now falls back to base
  R instead of erroring part-way through a call.
* `magrittr` is imported and the pipe re-exported, fixing `get_pc_scores()`
  when magrittr is not attached.
* Fixed `prep_matrix()` row/column names with `transpose = FALSE` on the Python
  backends, and fixed its rtorch path when `scale = FALSE`.
* Shared argument validation: `q_iter` must be >= 1, and `k`, `p`, matrix type
  and dimension constraints give actionable errors.
* Sparse `Matrix` input is documented as supported by the base R and irlba
  backends only.

* Added `RunFastPCA()`, a Seurat integration that runs FastPCA on an assay's
  scaled data and stores the result as a `DimReduc`, so downstream steps such
  as `Seurat::RunUMAP()` work unchanged. A memory ceiling (default 20 GB, or
  `options(FastPCA.memory_limit = )`) is checked before the dense conversion and
  the call stops rather than exceed it.
* Cap `OMP_NUM_THREADS` at 1 when an OpenMP conflict is detected, *before* any
  Python module is imported. Clamping `torch.set_num_threads()` after import was
  too late: the process could still segfault, especially when Seurat (and hence
  data.table) was loaded before the FastPCA environment was started.
* Load only the Python script(s) each call needs instead of importing every
  bundled module, and point numba at a writable cache directory. This avoids an
  unrelated `umap`/`numba` import failure during SVD-only calls.
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
