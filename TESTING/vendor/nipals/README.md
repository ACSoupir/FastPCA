# Vendored: nipals 1.2 (Kevin Wright)

`nipals` is the CRAN reference implementation this benchmark compares against.
It cannot be installed here: **CRAN and r-universe are unreachable from the
sandbox** (only `raw.githubusercontent.com` and `api.github.com` respond), so
the two source files are vendored instead. The package is pure R
(`NeedsCompilation: no`, depends only on `R >= 3.4.0`) so `source()`-ing
`R/nipals.R` is enough to run it.

Upstream: <https://github.com/kwstat/nipals> (DESCRIPTION reports version 1.2,
matching CRAN). Licence: MIT + file LICENSE — see `LICENSE`.

| file | url fetched from |
|---|---|
| `R/nipals.R`   | https://raw.githubusercontent.com/kwstat/nipals/master/R/nipals.R |
| `R/empca.R`    | https://raw.githubusercontent.com/kwstat/nipals/master/R/empca.R |
| `DESCRIPTION`  | https://raw.githubusercontent.com/kwstat/nipals/master/DESCRIPTION |
| `LICENSE`      | https://raw.githubusercontent.com/kwstat/nipals/master/LICENSE |

sha256 at the time of vendoring (2026-10):

```
ea8c332733796694309355d05c43ed2b1e9501ed17c9c2ab87749ca9ea121ae5  R/empca.R
0a378041945d64e0fc112302a41f067ce353a813ce2c96fee7356523ff22857c  R/nipals.R
2989fd733ade3d9fb14b8580d87fdb333fd4c641a590d64d04383f4037de627a  DESCRIPTION
```

Re-fetch with:

```bash
B=https://raw.githubusercontent.com/kwstat/nipals/master
curl -sf $B/R/nipals.R  -o TESTING/vendor/nipals/R/nipals.R
curl -sf $B/R/empca.R   -o TESTING/vendor/nipals/R/empca.R
curl -sf $B/DESCRIPTION -o TESTING/vendor/nipals/DESCRIPTION
curl -sf $B/LICENSE     -o TESTING/vendor/nipals/LICENSE
```

## Why this matters for the benchmark

Two properties of `nipals::nipals()` shape the comparison (see TESTING/README.md):

1. It allocates `PPp <- matrix(0, nvar, nvar)` and `TTp <- matrix(0, nobs, nobs)`
   for its Gram-Schmidt step. At the example data's 199112 features that is a
   **~295 GB** allocation in either orientation, so the package cannot run at full
   size; `TESTING/nipals_r.R` detects this and reports "N/A" instead of dying.
2. With missing values its inner loop builds a full `nobs x nvar` matrix
   (`T2`, then `P2`) **every iteration** to form the observed sum-of-squares
   denominators.
