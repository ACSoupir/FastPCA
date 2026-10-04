# FastPCA NIPALS: implementation plan

Companion to the `NIPALS / missing-value SVD` item in [ROADMAP.md](../ROADMAP.md).
Benchmark evidence is in `TESTING/README.md` ("NIPALS on missing-value data").
This supersedes the first draft of this plan; only the deltas are new.

## Goal

Add NIPALS (missing-value PCA) as a first-class solver alongside the randomized
and exact SVD path, sharing one result object and every accessory function.

Success criteria:

- `NIPALS()` on the example matrix (978 x 199112, ncomp = 10) with 5% MCAR on
  `backend = "pytorch"`, `cores = 1`: about <= 45 s and peak RSS <= ~9 GB
  (measured 42.5 s / 8.65 GB in TESTING).
- `backend = "r"` reproduces the CRAN package on small data: eigenvalues agree to
  < 1e-8 and iteration counts match apart from the documented upstream off-by-one.
- Every backend returns an interchangeable `"fastpca"` object, so `print()`,
  `summary()`, `pve()`, `get_pc_scores()`/`scores()` and `plot()` work on both
  methods.
- testthat passes with and without Python; R CMD check reports 0 errors,
  0 warnings.

## Public API

```r
NIPALS(input_r_matrix, ncomp = 10L,
       tol = 1e-6, maxit = 500L,
       center = TRUE, scale = FALSE, gramschmidt = TRUE,
       backend = c("r", "rtorch", "pytorch", "nipals", "tinygrad"),
       device = c("CPU", "GPU"), cores = 1L,
       max_na_prop = 1, verbose = FALSE)
```

- `ncomp` plays the role of `k`. NIPALS has no sketch, so there is **no** `p` and
  no `q_iter`; the mapping to the SVD arguments is documented in Details.
- No `dtype`: every backend computes in double precision (decision D8). The
  torch backends must build tensors with `dtype = torch.float64` explicitly,
  because R's own default would otherwise be float32.
- No `k` alias: the argument is `ncomp` only (decision D5).
- Returns a `"fastpca"` object with `method = "nipals"`.

```r
RunNIPALS(object, assay = NULL, features = NULL, npcs = 50L,
          reduction.name = "nipals", reduction.key = "NP_",
          backend = c("r", "rtorch", "pytorch"), device = c("CPU", "GPU"),
          cores = 1L, tol = 1e-6, maxit = 500L,
          memory_limit = getOption("FastPCA.memory_limit", 20),
          verbose = TRUE)
```

`RunNIPALS()` ships in the same release as `NIPALS()`. It is **not** deferred.

## Backends

| backend | engine | notes |
|---|---|---|
| `"r"` | port of the base implementation in TESTING/nipals_r.R | no Python; reference fallback |
| `"rtorch"` | R torch (native) | needs Suggests: torch; OpenMP-clamped |
| `"pytorch"` | new inst/python/nipals_func.py via reticulate | fastest; 5-6x base R |
| `"nipals"` | the CRAN **nipals package** (Suggests) or its vendored source | the string names that *package*; help page must say so explicitly. Errors on large data: O(n^2) allocation |
| `"tinygrad"` | placeholder only | accepted by the vector, then stops with a clear not-implemented error |

- Resolution goes through `resolve_backend_device()` with a new `.nipals_backends`
  vector in R/validate.R, so missing torch or a missing nipals package downgrades
  to `"r"` with the message style FastPCA() already uses.
- `device` applies only to torch and tinygrad; `"r"` and `"nipals"` ignore it with a
  message.
- `cores` defaults to **1** in FastPCA(), RunFastPCA(), NIPALS() and RunNIPALS()
  (decision D2). `prep_matrix()` keeps its default of 2 and `umap()` keeps 4: uwot
  genuinely uses its threads, so a uniform default would cost more there than it buys.

## Missing-value semantics

| situation | behaviour |
|---|---|
| `NA` present | allowed in NIPALS only; FastPCA() keeps its existing error |
| masked entries | set to 0 for every matvec; the mask is retained and reapplied after deflation |
| centre / scale | observed-only mean, unbiased sd (n-1), matching `nipals::nipals` |
| column above `max_na_prop` | error naming the offending columns; default 1 means never |
| all-NA or zero-variance column | dropped with a warning, recorded in `attr(x, "dropped_features")` |
| all-NA row | kept; its scores are NA, with a warning |
| zero-variance column when `scale = TRUE` | sd treated as 1 plus a warning (the CRAN package stops instead) |
| start vector | column with the largest observed absolute sum, as in `nipals::nipals` (deterministic, no RNG) |
| convergence | on the normalised score vector; per-component iteration counts stored, and a non-convergence warning names `maxit` |
| total variance | sum of squares over observed entries of the working matrix, stored so `pve()` works |
| sparse input | rejected with an actionable message (dense only); noted in the roadmap |

## Deflation with missing values (decision D3 — resolved)

**Implement masked rank-1 subtraction now**: `X <- X - t p`, then re-mask. This
matches `nipals::nipals` and the TESTING benchmark, so measured numbers stay
directly comparable to both references.

The classic Wold variant (**reconstruction deflation**: fill NA with the current
component fit before subtracting) is deferred. It imputes better but diverges from
the CRAN reference and needs its own validation, so it goes to ROADMAP as a future
`deflation = c("masked", "wold")` argument rather than shipping in v1. The
deflation step stays behind one internal function so adding that option later does
not touch the body of `NIPALS()`.

## Sign convention (decision D7 — resolved)

Component signs are arbitrary in NIPALS: a score/loading pair can be negated and
fit equally well, so two correct runs on different backends can disagree in sign.
This is a real interoperability papercut when NIPALS scores are compared against an
SVD run.

**Normalise the NIPALS path only.** After each component converges, flip so that
the loading with the largest absolute value is positive, and apply the same flip to
its score vector so `scores %*% t(loadings)` is unchanged. Output then agrees
across backends and across runs with no seed dependence.

The SVD path is deliberately untouched: changing its signs would alter existing
user output for no correctness gain. ROADMAP carries "unify the sign convention
across both solvers" as a future, opt-in change.

Tests assert the rule directly (largest-|loading| > 0 per component) and that
`"r"` vs `"pytorch"` parity needs no post-hoc sign correction.

## Object model and interoperability

**Does the object need a deeper reorganisation? No move to S4/R6, but yes:
formalise the contract.** It stays one S3 class `"fastpca"` -- a list plus
attributes. Two solver families now write it, so the attribute set grows from 9 to
about 15 and several fields mean something for only one family. Rather than change
representation, the plan documents three explicit groups and adds one internal
invariant check:

| group | contents | always present? |
|---|---|---|
| payload elements | `U`, `S`, `Vh`, and now `scores` | yes |
| identity attributes | `method`, `backend`, `device`, `n_obs`, `n_features` | yes |
| SVD-only attributes | `k`, `p`, `q_iter`, `exact` | NA for NIPALS |
| NIPALS-only attributes | `ncomp`, `tol`, `maxit`, `iterations`, `converged`, `missing_fraction` | NA for SVD |
| shared fitting attributes | `total_variance`, and (NIPALS) `center` / `scale` vectors | as applicable |

Why not subclasses (`c("fastpca_nipals", "fastpca")`) or S4?

- the list contract is documented and already exercised by tests, so subclassing
  would not remove a branch -- it would only move the same `switch(method)` into
  method dispatch, while every accessor gains a second definition;
- with exactly two solver families the branching is trivial. A third family (the
  iterative SVD in ROADMAP) is the trigger to reconsider, not NIPALS;
- promoting to S4/R6 stays in ROADMAP under "Other candidates".

New internal `.validate_fastpca(x)`, called by every accessor. It checks that
`length(S) == ncol(Vh)`, `nrow(U) == n_obs`, that the scalars are finite, and that
`method` is known. A hand-built or older object then fails loudly instead of
producing silent nonsense, and the table above has one place where it is enforced.

### `scores` becomes a fourth element (decision D1 -- resolved)

`new_fastpca()` always returns `list(U, S, Vh, scores)`, so the payload shape is
uniform across methods:

- **SVD**: `scores <- U %*% diag(S)`, computed at construction. This duplicates an
  `n_obs x k` matrix that is already derivable from `U`, but at roughly 78 KB for
  978 x 10 that is not a memory concern, and it removes the "does this object have
  scores?" question entirely.
- **NIPALS**: `scores` are the native score vectors and `U` is defined as their
  unit-norm version, so the same identity (`scores == U %*% diag(S)`) holds. The
  real difference is that NIPALS' left vectors are *not* mutually orthogonal, so
  `crossprod(U)` is not the identity even though each column has unit norm.

No existing test asserts `length(res) == 3` or the names of the list, so this is
additive: code that indexes `$U`, `$S` or `$Vh` keeps working. NEWS records it as an
addition rather than a break.

### Method-aware accessors

| function | change |
|---|---|
| `get_pc_scores()` / `scores()` | return the stored element (always present after D1); still accept a bare list with `U`/`S`, and now also a Seurat object or DimReduc |
| `print.fastpca()` | add method, dims, missing-data share and PVE lines |
| `summary.fastpca()` | add `iterations` and `converged` columns (NA for SVD); carry the method on the data.frame |
| `pve()` | unchanged; document that the NIPALS denominator uses observed entries only |
| `plot.fastpca()` | unchanged (scree / cumulative) |
| `fitted.fastpca()` | **new**, NIPALS only: reconstruct the input from `scores`, `Vh` and the stored centring/scaling |

Target `print()` shape:

```
<fastpca> 978 x 199112 input; k = 10
  method: NIPALS (missing-value) | backend: pytorch | device: CPU
  missing: 5.0% of entries (masked)
  converged in median 14 iterations (tol = 1e-06, maxit = 500)
  PVE: PC1 18.3%, PC2 9.7%, ... | cumulative (10 PCs) 61.2%
```

For SVD results the missing line is omitted and method reads "randomized SVD" or
"exact SVD", taken from the existing `exact` attribute.

### Imputation (decision D6 -- resolved)

Recovering the missing entries is often the real reason to run NIPALS on
proteomics or metabolomics data, so v1 supports it -- but **never as a stored
element**:

- `new_fastpca()` stores the centring and scaling vectors that were applied. These
  are length-`n_features` numerics (about 1.6 MB for the 199112-feature example),
  so they are cheap to keep and also make a result reproducible.
- `fitted(x)` returns the reconstruction on demand. For the example matrix that is
  a dense 978 x 199112 double, about 1.5 GB -- exactly why it is a call and not an
  element; the help page states that size up front.
- It errors for SVD results, where there is no stored centring to undo.

This is the one step that can be cut to a follow-up without affecting anything else,
should it turn out to need more design (masking policy, per-column offsets).

## Seurat integration

`RunNIPALS()` mirrors `RunFastPCA()` and reuses its helpers:

- takes the `scale.data` layer, which is where NA usually appears;
- calls `NIPALS()` rather than `FastPCA()`;
- builds the DimReduc from `res$scores` (embeddings) and loadings, with
  `stdev = S / sqrt(n_cells - 1)`, matching RunFastPCA();
- reuses the memory ceiling, with the dense overhead constant raised to 4 for
  NIPALS (input + mask + working copy + torch tensor);
- defaults to `reduction.name = "nipals"` and key `"NP_"`, so it can coexist with
  a RunFastPCA reduction named "pca".

## Files

New:

- `R/nipals.R`: exported `NIPALS()`, argument validation and dispatch.
- `R/seurat_nipals.R`: `RunNIPALS()`, kept next to seurat.R for symmetry.
- `R/rtorch_nipals.r`: torch backend, mirroring rtorch_randomized_svd.r.
- `inst/python/nipals_func.py`: promoted from TESTING/nipals_torch.py with the CLI
  removed, exposing `nipals_py(X, ncomp, tol, maxit, center, scale, gramschmidt,
  device, cores)` and returning a dict (`eig`, `sv`, `scores`, `loadings`,
  `iters`, `converged`). NA is detected in Python with `~isfinite(X)`, so no mask
  matrix crosses the reticulate boundary.
- `tests/testthat/test-nipals.R`, `tests/testthat/test-seurat-nipals.R`.
- `man/NIPALS.Rd`, `man/RunNIPALS.Rd` (roxygen-generated).

Changed:

- `R/validate.R`: `.nipals_backends`, `.check_nipals_args()`, and an `allow_na`
  argument for `.check_matrix_input()`.
- `R/class.R`: the four-element payload, method attribute, centre/scale storage,
  `.validate_fastpca()`, the `fitted()` method, and print/summary updates.
- `R/utils.R`: a "nipals" branch in `get_python_files()` so only the NIPALS module
  is imported for this backend.
- `R/FastPCA.R`, `R/seurat.R`: cores default 4 -> 1 and updated @param text.
- `DESCRIPTION` (Suggests: nipals; description mentions missing-value PCA),
  `NAMESPACE`, `NEWS.md`.
- `ROADMAP.md`: the two deferred items (Wold deflation, cross-solver sign
  unification), plus the alias and tinygrad notes already recorded there.
- `tests/testthat/test-prep_matrix.R`: the stale expectation on line 7 still asserts
  that `prep_matrix()` rejects tinygrad. It must become `expect_no_error()`. It
  currently passes only because the test environment has no tinygrad installed.

## Tests and acceptance

- Parity: `NIPALS(backend = "r")` against `nipals::nipals()` on small NA matrices,
  eigenvalues within 1e-8 (`skip_if_not_installed("nipals")`).
- Parity across backends: `"r"` vs `"pytorch"` on 500 x 200 with 5% MCAR, subspace
  angle < 1e-6 (`skip_if_no_python`).
- Edge cases: all-NA column dropped with a warning, `max_na_prop` error,
  zero-variance scaling warning, all-NA row kept.
- Object contract: method attribute set; `print()` output contains the method,
  dims and PVE lines (`expect_output`); `summary()` has iterations/converged;
  `pve()` works for NIPALS; `.validate_fastpca()` rejects a malformed object and
  accepts an older three-element list by deriving scores.
- `scores` element: present for both methods; equals `U %*% diag(S)` for SVD;
  finite and different from it for NIPALS; `get_pc_scores()` returns the stored
  element in both cases.
- Sign convention: every NIPALS component has its largest absolute loading
  positive, and `"r"` vs `"pytorch"` agree with no post-hoc flip.
- Imputation: `fitted()` errors for SVD results; for NIPALS it reproduces observed
  entries to tolerance and fills the masked ones.
- Backend handling: tinygrad placeholder error; a missing nipals package falls
  back to `"r"`.
- Seurat: `RunNIPALS()` stores a DimReduc with the expected dimensions and key.
- Performance stays tracked in TESTING rather than CRAN tests, to keep CI fast.

## NEWS

Step 0, independent of NIPALS: catch the current section up with work that has
already shipped but is not recorded.

- added the tinygrad backend (randomized SVD on Apple Metal), including device
  resolution and the arm64 CPU -> METAL behaviour;
- `prep_matrix()` transform on tinygrad, measured 3-4x faster than the torch path;
- note that a near-flat spectrum makes any randomized method ill-conditioned.

Then add the NIPALS entries: new `NIPALS()` and `RunNIPALS()`, missing-value
handling, the r/rtorch/pytorch/nipals backends with a tinygrad placeholder,
method-aware print/summary/accessors, and the deterministic sign convention.

Two changes to existing behaviour get their own bullets rather than being buried:
`FastPCA()` and `RunFastPCA()` now default to `cores = 1`, and the `"fastpca"`
object gains a fourth `scores` element (additive; no call site breaks).

## Decisions

Settled during planning:

| id | question | decision |
|---|---|---|
| D1 | where PC scores live | fourth element `x$scores`, always populated for both methods |
| D2 | scope of the `cores = 1` default | FastPCA(), RunFastPCA(), NIPALS(), RunNIPALS(); `prep_matrix()` stays 2, `umap()` stays 4 |
| D3 | deflation with NA | masked subtraction now; Wold variant deferred to ROADMAP |
| D4 | backend string for the CRAN package | `"nipals"`, with help text stating it means that *package* |
| D5 | `k` alias for `ncomp` | no; `ncomp` only, documented against the SVD arguments |
| D6 | imputation output | on-demand `fitted()` from stored centre/scale, never a stored matrix |
| D7 | sign convention | normalise the NIPALS path only; ROADMAP the cross-solver unification |
| D8 | float32 option | no `dtype` argument; every backend computes in double |

Still open, to settle while implementing:

- whether `.validate_fastpca()` errors or warns on a three-element list missing
  `scores` (recommend: accept it and derive, so previously saved objects load);
- the wording of the `"nipals"` backend message when that package is absent, so it
  is unambiguous that the string names a package and not this function.

## Non-goals

- tinygrad/Metal NIPALS (placeholder only in this release).
- sparse input.
- Wold/reconstruction deflation, and unifying the sign convention across both
  solvers: ROADMAP carries both as future, opt-in changes.
- float32 storage for any backend (decision D8).
- an iterative/converging SVD solver: ROADMAP keeps that item until NIPALS has
  settled, so both can share one convergence interface.
