# FastPCA roadmap

Ideas that are intentionally out of scope for the current release, kept here so
they can be picked up later.

## NIPALS / missing-value SVD

Metabolomics and proteomics matrices routinely contain `NA` (missing
quantification). Base R, irlba and the randomized backends all require complete
data; `FastPCA()` therefore rejects missing values with a clear error.

NIPALS (non-linear iterative partial least squares) is planned as its **own
function**, not another `backend =` choice. It solves a different problem -- it
accommodates missing entries and converges one component at a time rather than
computing a fixed-rank randomized sketch -- so its arguments (`tol`, `maxit`)
do not map onto the sketch parameters. The function should still return a
`"fastpca"`-compatible object so the existing accessors (`get_pc_scores()`,
`pve()`, `summary()`) keep working.

**Status (measured in `TESTING/`, see "NIPALS on missing-value data" there).**
A single-core pytorch NIPALS is ~5-6x faster than a careful base-R implementation
and 35-37x faster than the CRAN `nipals` package on the example data, at 3-5x
lower peak RSS. With missing values the gap widens: on 978 x 20000 at 5% MCAR,
pytorch needs ~1.4 GB where the package needs ~12 GB. The CRAN package cannot run
on the full 199112-feature matrix at all -- it allocates an O(n^2) Gram-Schmidt
matrix (~295 GB), which the harness reports as N/A rather than crashing.

**Implemented in 0.9.2.** `NIPALS()` and `RunNIPALS()` ship the base R, native
torch, PyTorch and CRAN-package backends; `TESTING/nipals_torch.py` remains the
standalone benchmark prototype. See `NEWS.md` for the user-facing summary and
`TESTING/NIPALS_PLAN.md` for the design.

**Thread scaling on the full matrix (978 x 199112, ncomp = 10).**
`torch.set_num_threads(N)` with `OMP_NUM_THREADS=N`, one run each:

| torch threads | complete s | peak RSS GB | 5% MCAR s | peak RSS GB |
|---|---|---|---|---|
| 1 | 21.7 | 3.38 | 42.5 | 8.65 |
| 2 | 20.7 | 3.39 | 41.7 | 8.65 |
| 4 | 20.0 | 3.38 | 41.7 | 8.65 |
| 8 | 19.9 | 3.38 | 40.6 | 8.65 |

A best-of-two repeat agreed within ~5%. Extra threads buy only **~5-10%**: the
inner loop is BLAS `gemv`/`ger`, and on this macOS build process CPU time stays
near wall (~1.3-1.4x at 8 threads), so almost no parallelism is extracted and the
run stays core-bound. Peak RSS does not move with thread count, so the single-core
figures elsewhere are an honest comparison rather than a memory trade. Raw logs:
`TESTING/results/nipals_threads_f.txt`, `..._single.txt`.

**Naming and aliases.** The exported function is `NIPALS()`, as specified in the
implementation plan (`TESTING/NIPALS_PLAN.md`) and shipped in 0.9.2. The longer
spelling `FastNIPALS()` / `fastnipals()` is *not* exported. If it turns out to be
a common guess, a roxygen `@aliases FastNIPALS` on the help page is enough to make
it findable. The "Fast" prefix is justified rather than aspirational: the pytorch
backend measured 5-6x faster than base R and ~35x faster than the CRAN package.

**tinygrad placeholder.** `NIPALS(backend = "tinygrad")` is accepted by the
backend vector and then stops with an explicit not-implemented error. A Metal
kernel for missing-value NIPALS is future work.

**Deferred from the first implementation.** The full specification and its settled
decisions are in `TESTING/NIPALS_PLAN.md`. Three items were deliberately left out:

- *Wold / reconstruction deflation.* v1 subtracts the masked rank-1 component,
  matching `nipals::nipals`. The classic variant fills the missing entries with the
  current component fit before subtracting; it imputes better but diverges from the
  CRAN reference and needs its own validation. Planned as a
  `deflation = c("masked", "wold")` argument.
- *Cross-solver sign convention.* NIPALS signs are normalised (largest absolute
  loading positive) so results are stable across backends and runs. The SVD path
  keeps its current signs, because changing them would alter existing user output;
  unifying the two is an opt-in change for later.
- *float32 mode.* Every backend computes in double except `torch_irlba()`,
  which added a `precision = "double" | "single"` argument (measured ~1.5x
  faster and half the memory at ~3e-5 relative singular-value error). Extending
  the same argument to `FastPCA()`/`NIPALS()` is still open.

## Incremental / out-of-core PCA

**Implemented in 0.10.0.** `IncrementalPCA()` streams a list of chunk files and
keeps only one block plus an `ncomp x n_features` factorization state, so the
top components of a dataset larger than RAM can still be extracted. Backends are
base R, PyTorch and tinygrad; `scale = TRUE` adds a statistics pass so centring
and scaling are exact, and `oversample` widens the internal rank that is carried
between blocks (a buffer at least as wide as the rank makes an exact-statistics
fit reproduce batch PCA exactly). See `NEWS.md`.

Deliberate follow-ups:

* **R-torch kernel.** Only the base R and Python kernels exist. An
  `.partial_fit_incremental_rtorch()` would let users on a machine without
  reticulate use the native torch backend, but R `torch` is not installed in
  this environment so it could not be tested; the entry point currently stops
  with an explicit message and `ROADMAP.md` pointer instead.
* **Projecting new data.** Scores are only available for the files that were
  fitted (`store_scores = TRUE`, which needs a scoring pass). A `transform()`
  method (or an exported helper) could project *new* chunk files onto a stored
  result with the same `center`/`scaling` vectors.
* **Other containers.** parquet/HDF5 currently need a user-supplied `reader`
  function, which is loaded whole. A reader interface that streams by row group
  would extend the out-of-core guarantee to those formats.
* **Warm restarts.** A `fit` object cannot currently be extended with further
  chunk files without refitting; the Python kernels already keep their state in a
  reference, so exposing it is mostly API design.

## Iterative / converging SVD

The randomized backends use a fixed number of power iterations. A future
iterative solver would instead refine until the singular values/vectors meet a
`tol`, up to `maxit` (the same contract as `irlba::irlba()`) -- useful when the
top components must be converged rather than "good enough", and as an internal
building block for NIPALS.

Whether this is its own function or a `method = "iterative"` argument on the
existing solver should wait until NIPALS has settled, so both share a single
convergence interface.

**Status (0.9.3).** Shipped as its own exported function, `torch_irlba()`,
which mirrors irlba's iterative arguments and adds the package's
`precision`/`device` controls. On `data/prepped_gex.rds` used as-is
(199112 samples x 978 genes) it reproduces `irlba::irlba()` to ~1e-9 and runs
~2.9x faster in float64; see `TESTING/README.md`, "Orientation correction and
the packaged torch_irlba()".

## Other candidates

* Promote the `"fastpca"` object from S3 to a richer class (or an R6 reference
  object) once the accessors have settled.
* Sparse-aware randomized backends (currently only base R/irlba accept
  `Matrix` input; Python/torch backends require dense data).
* Memoize loaded Python modules per session to avoid repeated import overhead.
