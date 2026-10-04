# FastPCA 0.9.2

## NIPALS: PCA with missing values

* Added `NIPALS()`, which extracts principal components one at a time by
  iterative least squares and accepts a matrix containing `NA` (missing
  quantifications are routine in proteomics and metabolomics). It returns the
  same `"fastpca"` object as `FastPCA()`, so `print()`, `summary()`, `pve()`,
  `get_pc_scores()`/`scores()` and `plot()` work unchanged. Backends are base R
  (`"r"`), the CRAN `nipals` package, native torch (`"rtorch"`) and PyTorch
  through reticulate (`"pytorch"`); `backend = "tinygrad"` is accepted but stops
  with an explicit not-implemented message. Note that `backend = "nipals"`
  names the *package* and runs it as the reference implementation; because that
  package allocates an O(max(nrow, ncol)^2) Gram-Schmidt matrix it refuses
  shapes that would exceed `options(FastPCA.memory_limit)` and cannot handle
  all-NA rows or zero-variance columns.
* Missing-value semantics follow the reference implementation: observed-only
  means and unbiased standard deviations, zero-filled products with
  observed-only denominators, and masked rank-one deflation. Columns that are
  entirely missing are dropped with a warning (their loadings stay `NA` so the
  feature axis stays aligned) and rows with no observed entries keep `NA`
  scores. NIPALS signs are not identifiable, so each component is flipped to
  make its largest-magnitude loading positive; the SVD path keeps its own signs.
* `fitted()` reconstructs -- and therefore imputes missing entries of -- a
  NIPALS result, undoing the stored centring and scaling. It is computed on
  demand because the reconstruction of the package example (978 x 199112) would
  be about 1.5 GB.
* Added `RunNIPALS()`, the Seurat counterpart of `RunFastPCA()`: it stores a
  reduction named `"nipals"` (key `NP_`) so it can coexist with a FastPCA
  reduction, and runs NIPALS on the scaled data with cells as observations so
  that the native score vectors are cell embeddings.
* `nipals` is now a hard dependency (Imports), so the reference backend needs no
  separate install step.
* `cores` now defaults to **1** in `FastPCA()`, `RunFastPCA()`, `NIPALS()` and
  `RunNIPALS()`. Thread scaling measured in `TESTING/` buys only ~5-10% for
  NIPALS while peak memory is flat, and the base R/irlba paths were already
  effectively single-threaded. `prep_matrix()` keeps its default of 2 and
  `umap()` keeps 4, where the threads are genuinely used.
* The result object carries a `scores` element for every solver, and
  `.validate_fastpca()` checks its invariants in one place; a three-element list
  written before this release is still accepted (scores are derived from `U` and
  `S`). `get_pc_scores()` now also accepts a Seurat object or `DimReduc`.

## tinygrad backend and preprocessing (earlier unreleased work)

* `FastPCA(backend = "tinygrad")` runs the randomized SVD on Apple Metal, and
  `prep_matrix(backend = "tinygrad")` uses an optimized transform that is about
  3-4x faster than the previous implementation.
* Device resolution warns when a `device = "CPU"` request is mapped to Metal on
  arm64 Macs, where tinygrad has no CPU codegen.
* Documented that centre-scaling data with little structure leaves nearly equal
  top singular values, which is ill-conditioned for *any* randomized SVD; use
  `backend = "r"` or `"irlba"` when the leading components must be exact.

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
