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
matrix (~295 GB), which the harness reports as N/A rather than crashing. No
package code has changed yet; the pytorch implementation is `TESTING/nipals_torch.py`.

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

## Iterative / converging SVD

The randomized backends use a fixed number of power iterations. A future
iterative solver would instead refine until the singular values/vectors meet a
`tol`, up to `maxit` (the same contract as `irlba::irlba()`) -- useful when the
top components must be converged rather than "good enough", and as an internal
building block for NIPALS.

Whether this is its own function or a `method = "iterative"` argument on the
existing solver should wait until NIPALS has settled, so both share a single
convergence interface.

## Other candidates

* Promote the `"fastpca"` object from S3 to a richer class (or an R6 reference
  object) once the accessors have settled.
* Sparse-aware randomized backends (currently only base R/irlba accept
  `Matrix` input; Python/torch backends require dense data).
* Memoize loaded Python modules per session to avoid repeated import overhead.
