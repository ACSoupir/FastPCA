# FastPCA 0.10.0

## Out-of-core PCA: `IncrementalPCA()`

* Added `IncrementalPCA()`, which fits the leading components of data split
  across many chunk files (one CSV per sample is the intended layout) without
  ever holding more than one block in memory. Peak R-side memory is roughly the
  largest block plus an `ncomp x n_features` factorization state, so datasets
  far larger than RAM can still yield their top ~10-20 PCs. The result is the
  usual `"fastpca"` object (`method = "incremental"`), so `print()`,
  `summary()`, `pve()` and `plot()` work unchanged.
* Backends are base R (`"r"`, no Python), PyTorch through reticulate
  (`"pytorch"`) and tinygrad (`"tinygrad"`, float32 on Metal or CUDA), with
  the same `device`/`cores` arguments as the rest of the package (plus
  `precision` for PyTorch). Base R and PyTorch take an exact thin SVD of each
  stacked block; tinygrad uses a randomized sketch per block and is the fastest
  on Apple Metal but approximate. An R-torch kernel is not shipped yet (see
  `ROADMAP.md`).
* Reading: plain text/CSV, `.gz`, `.bz2` and `.xz` are streamed through base
  R connections, `.zst`/`.zstd` through the external `zstd` binary, and any
  other container (parquet, HDF5) through `reader =`. `header`, `row.names`
  and `sep` are passed to the CSV reader, so a leading identifier column works
  as it does elsewhere. `chunk_rows` splits an individual file that is itself
  too large to hold.
* Passes are explicit: `scale = TRUE` adds a statistics pass so the mean and
  standard deviation are exact rather than incremental, `store_scores = TRUE`
  adds a scoring pass and returns the observation scores (otherwise nothing
  `n_obs x ncomp` is ever allocated). With exact statistics the components match
  batch PCA of the pre-processed data to numerical precision; with a single pass
  each block is centred by the mean of everything read before it, which matches
  scikit-learn's `IncrementalPCA` and is the documented approximation.
* Every block truncates its tail, so `oversample` (default 10) keeps extra
  components internally and discards them at the end; a buffer at least as wide
  as the rank makes an exact-statistics fit reproduce batch PCA exactly.
* Missing values are rejected with a pointer to `NIPALS()`, as for
  `FastPCA()`.

# FastPCA 0.9.3

## Iterative Lanczos SVD: `torch_irlba()`

* Added `torch_irlba()`, an exported iterative SVD that runs irlba's own
  algorithm (Lanczos bidiagonalisation with implicit restart, classical
  Gram-Schmidt and `convtests()`) on PyTorch. It mirrors `irlba::irlba()`'s
  arguments (`nv`, `work`, `maxit`, `reorth`, `tol`, `svtol`, and a
  starting vector `v`) and returns the same `"fastpca"` object as
  `FastPCA()`, so `print()`, `summary()`, `pve()` and the accessors work
  unchanged. Given a shared start vector it agrees with `irlba::irlba()` to
  about 1e-9 in the singular values while running ~2.9x faster on the package
  example.
* Added `precision = "double" | "single"` and `device = "CPU" | "GPU"`
  controls. Single precision roughly halves memory and is faster; on arm64 Macs
  a GPU request reaches Metal only in single precision, because MPS has no
  float64 support (a double-precision GPU request runs on the CPU and reports
  why). Results record their `precision`, restart count (`n_iter`) and
  matrix-product count, which `print()` reports.
* The input is oriented observations x features, as everywhere in the package:
  rows are samples or cells and columns are features.

# FastPCA 0.9.2

## NIPALS: PCA with missing values

* Added `NIPALS()`, which extracts principal components one at a time by
  iterative least squares and accepts a matrix containing `NA` (missing
  quantifications are routine in proteomics and metabolomics). It returns the
  same `"fastpca"` object as `FastPCA()`, so `print()`, `summary()`, `pve()`,
  `get_pc_scores()`/`scores()` and `plot()` work unchanged. Backends are base R
  (`"r"`), the CRAN `nipals` package, native torch (`"rtorch"`), PyTorch
  through reticulate (`"pytorch"`) and TinyJit-compiled tinygrad
  (`"tinygrad"`, float32 on Metal or CUDA). Note that `backend = "nipals"`
  names the *package* and runs it as the reference implementation; because that
  package allocates an O(max(nrow, ncol)^2) Gram-Schmidt matrix it refuses
  shapes that would exceed `options(FastPCA.memory_limit)` and cannot handle
  all-NA rows or zero-variance columns.
* `backend = "tinygrad"` compiles the inner NIPALS loop with TinyJit and runs
  it in float32 on Metal or CUDA. It is the fastest backend for large matrices
  and uses about half the memory of PyTorch once a mask is present, at roughly
  1e-5 relative agreement on singular values. It needs the Python `tinygrad`
  module (`setup_py_env(backend = "tinygrad")`) and honours the
  `FASTPCA_TG_MEM_GB` working-set limit (20 GB by default).
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
