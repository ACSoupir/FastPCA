# TESTING — optimised PyTorch and native-Metal tinygrad SVD kernels

Research sandbox for faster randomized-SVD kernels than the one shipped in
`inst/python/torch_svd_func.py`. Everything here is committed to `alex_optim`
for version control but **never pushed** (the repo owner pushes manually).

## TL;DR

The default kernel to use is **`stable`** (`randomized_svd_stable`). It keeps the
baseline's accuracy at every dtype while removing its two real bottlenecks.

| matrix (k=50, p=10, q_iter=2) | baseline fp64 | `stable` fp64 | speed-up |
|---|---|---|---|
| 4000 × 1000   | 46.3 ms | **21.5 ms** | **2.15×** |
| 6000 × 1000   | 59.5 ms | **30.0 ms** | **1.98×** |
| 15000 × 2000  | 167.4 ms | **114.5 ms** | **1.46×** |

If float32 is acceptable (typical for PCA/embedding), the win is much larger,
because `stable` does not collapse in single precision:

| matrix | baseline fp64 (package default) | `stable` fp32 CPU | speed-up |
|---|---|---|---|
| 4000 × 1000   | 46.3 ms | **8.9 ms** | **5.2×** |
| 6000 × 1000   | 59.5 ms | **12.8 ms** | **4.6×** |
| 15000 × 2000  | 167.4 ms | **46.3 ms** | **3.6×** |

Apple GPU (MPS), float32 — the baseline cannot run on Metal at all:

| matrix | best baseline (CPU fp64) | `stable` on MPS fp32 | speed-up |
|---|---|---|---|
| 4000 × 1000   | ~46 ms | **10.2 ms** | **4.5×** |
| 15000 × 2000  | ~167 ms | **29.2 ms** | **5.7×** |

Reconstruction error is identical to the baseline in every row
(`recon_relFro` matches), i.e. no accuracy is traded for the speed-up.

## Files

| file | purpose |
|---|---|
| `pytorch_rsvd.py`  | the kernels + registry (`KERNELS`) |
| `benchmark_pytorch.py` | correctness check and timing harness |
| `torch_svd_func_optim.py` | drop-in replacement exposing the baseline signature |
| `run_benchmarks.sh` | reproduces correctness + the benchmark sweep |
| `results/*.json`   | raw measurements (PyTorch) |
| `tinygrad_native.py` | native-Metal tinygrad kernel (Newton–Schulz + TinyJit) |
| `tinygrad_validate.py` | native vs in-package torch accuracy check |
| `tg_mem_probe.py` | per-backend timing + peak-RSS probe |
| `_gen_matrices.py` | writes deterministic test matrices to `.cache/` |
| `run_tinygrad_bench.sh` | reproduces the tinygrad vs baseline comparison |
| `results/tinygrad_bench.txt` | raw tinygrad measurements |
| `tinygrad_transform_optim.py` | optimized tinygrad prep_matrix transform (log2/transpose/scale) |
| `tg_transform_bench.py` | transform correctness + speed/memory benchmark |
| `irlba_torch.py` | PyTorch port of irlba's Lanczos + implicit-restart iteration |
| `irlba_r.R` | irlba::irlba fast C path and pure-R fallback, for comparison |
| `fastpca_torch_bench.R` | FastPCA(backend = "pytorch") through the installed R API |
| `fastpca_kernel_bench.py` | the shipped randomized_svd_py called directly (no reticulate) |
| `irlba_accuracy.py` | randomized-kernel error vs q_iter against the exact irlba reference |
| `memguard.py` | hard RSS cap: SIGKILL the process group above --cap-gb |
| `memguard_selftest.sh` | proves the cap (survive / kill at 137 / pre-flight refusal) |
| `run_irlba_bench.sh` | reproduces the IRLBA sweep, one implementation per process |
| `irlba_collect.py` | merges RESULT_JSON + peak RSS into results/irlba_bench_*.json |
| `results/irlba_bench_{s,m,f}.{txt,json}` | raw IRLBA measurements |
| `results/irlba_accuracy_{s,f}.json` | randomized error vs q_iter |
| `thread_probe.py`, `thread_probe2.py` | measures cores actually used (cpu/wall) vs the requested count |
| `run_thread_verify.sh` | reproduces the thread-binding evidence into results/thread_verify.txt |
| `results/thread_verify.txt` | raw thread-binding measurements |

## Why it is faster

The in-package baseline (`inst/python/torch_svd_func.py`) does, per call:

1. **A redundant tall QR on every power iteration.** It computes
   `Q = qr(A.T @ Y)` (small, needed for stability) *and then* `Y = qr(A @ Q)`
   (tall m×b, **not** needed). Only the small QR is required to stop the subspace
   collapsing in low precision. `stable` keeps the small QR and drops the tall one.
2. **An unconditional full `gc.collect()`** at the end (~12–15 ms on this
   machine, measured; 2000 iterations show no leak without it). `stable` frees
   via reference counting and only collects when `FASTPCA_SVD_GC=1`.
3. **An autograd graph that is discarded** — all kernels run under
   `torch.no_grad()`.
4. **No MPS support / float64 only.** `_prepare` maps MPS to a supported dtype
   and `_qr` falls back to a CPU Householder QR where Metal lacks one.

## Kernels

| name | scheme | notes |
|---|---|---|
| `baseline`  | package kernel, wrapped for timing | fp64 CPU/CUDA only |
| `orth`      | baseline math, no_grad + reduced QR | isolates autograd/QR cost |
| `stable`    | small-QR power iterations, one final QR | **recommended**; safe in fp32 |
| `direct`    | Halko Alg. 4.1, no per-iteration QR | fastest in fp64; **unstable in fp32** |
| `triangular`| Q-free / triangular solve | stable, ≈ `stable` here |
| `lowrank`   | `torch.svd_lowrank` reference point | slower than our kernels |

### Accuracy warning: `direct` + float32

Without per-iteration reorthogonalisation the power method raises the singular
value ratio to `(σ1/σk)^4`. In float32 that pushes subdominant directions below
machine epsilon and the top-k singular values are wrong by ~80%
(`sv_relerr ≈ 0.8`, visible in every fp32 `direct` row). Use `stable` for
float32 / MPS. In float64 `direct` is fine and ~10% faster than `stable`.

## Running

```bash
export KMP_DUPLICATE_LIB_OK=TRUE           # libomp double-init on macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1       # torch 2.9 lacks native MPS qr

# correctness on a small matrix
/opt/anaconda3/envs/FastPCA/bin/python TESTING/benchmark_pytorch.py --check

# full sweep
DEVICES=cpu,mps DTYPES=64,32 SIZES="4000x1000,15000x2000" ./TESTING/run_benchmarks.sh
```

## Using it from R

Copy `pytorch_rsvd.py` + `torch_svd_func_optim.py` into `inst/python/`, point
the loader in `R/utils.R` at `torch_svd_func_optim.py`, and the R call site in
`R/FastPCA.R:81` keeps working unchanged (`randomized_svd_py(...)` signature is
identical). Nothing outside `TESTING/` has been modified yet.

## Native-Metal tinygrad kernel (`tinygrad_native.py`)

A pure-tinygrad implementation of the same randomized SVD whose range finder runs
entirely on Apple's Metal GPU. It matches the **in-package baseline**
(`inst/python/torch_svd_func.py::randomized_svd_py`) in results and beats it on
both speed and memory, once the JIT is warm.

tinygrad is **not** in `/opt/anaconda3/envs/FastPCA`. A workspace venv is used
instead (gitignored):

```bash
/opt/anaconda3/envs/FastPCA/bin/python -m venv --system-site-packages TESTING/.venv-tg
TESTING/.venv-tg/bin/python -m pip install tinygrad        # 0.11.0
# tinygrad caches kernels under ~/.cache (read-only in this sandbox):
export XDG_CACHE_HOME="$PWD/TESTING/.cache"
```

### Approach

tinygrad's bundled `Tensor.qr` / `Tensor.svd` are unusable at these sizes: the
QR materialises a full m×m Q (`eye(m).expand(...)`, ~0.9 GB at m=15000 in
float32 alone) and the SVD runs hundreds of Jacobi rounds. Instead:

* **Newton–Schulz polar iteration** replaces every QR — matmul-only, so nothing
  larger than m×b / b×b is ever allocated:
  `X <- a X + b (X G) + c ((X G) G)`, `G = XᵀX`, `(a,b,c)=(15,-10,3)/8`.
  Newton–Schulz *grows* small singular directions (~1.875× per step), which is
  what keeps float32 stable; the plain power scheme collapses.
* The whole range finder is wrapped in **`TinyJit`**, which removes the per-op
  Python dispatch that otherwise dominates (≈220 ms → ≈10 ms on 2000×500).
* The only host-side step is the small SVD of `B = QᵀA` (b×n, b=k+p), computed
  via the Gram route — see below.

### The final small SVD: why it stays on the host

After the range finder, `B = QᵀA` (b×n) still needs an SVD. Measured here with
b=60:

| final step (b = 60) | m=2000, n=500 | m=15000, n=2000 |
|---|---|---|
| `np.linalg.svd(B)` (old default) | 5.11 ms | 10.42 ms |
| **Gram + `np.linalg.eigh` (new default)** | **1.29 ms** | **3.19 ms** |
| Gram matmul on Metal + host `eigh` (device hybrid) | 6.63 ms | 7.48 ms |
| tinygrad `Tensor.svd` | unusable (below) | |

* The default now uses the **Gram route**: `C = B Bᵀ` (b×b), symmetric
  eigendecomposition, then `Vh = diag(1/S) Uᵀ B`. It is exact to ~1e-15 and
  **~4× faster** than LAPACK's direct SVD at these shapes; end-to-end that is
  `12.7 ms → 9.7 ms` on 2000×500.
* **tinygrad cannot do this step.** `Tensor.svd` unrolls
  `num·log2(num)·2 ≈ 712` Jacobi rounds into one lazy graph: it took **14 s for
  an 8×8** matrix and raises `RecursionError` from ~20×20 upward (even with
  `sys.setrecursionlimit(10**6)`). tinygrad also exposes no `eigh` / `eig` /
  `cholesky` / `solve`, so a native replacement means hand-writing Jacobi —
  ~10 sweeps × b/2 rotations ≈ hundreds of sequential kernel launches, versus one
  LAPACK call whose hard part is only a b×b eigenproblem.
* **Pushing the Gram matmul onto Metal does not help** (6.63/7.48 ms). Device→host
  transfers are cheap on unified memory (`B.numpy()` ≈ 0.26 ms), but the eager,
  non-JIT'd small ops cost more than numpy saves.
* Net: shuffling the final b×n SVD to numpy is the right choice, and it has been
  optimised rather than left as LAPACK's direct SVD. `final="svd"` restores the
  old path for comparison.

### Results vs the in-package torch baseline

Measured by `TESTING/run_tinygrad_bench.sh`, both backends in the **same**
interpreter (`TESTING/.venv-tg/bin/python` sees torch *and* tinygrad), k=50,
p=10, q_iter=2:

| matrix | baseline steady | native tg steady | speed-up | baseline peak RSS | native tg peak RSS |
|---|---|---|---|---|---|
| 2000 × 500   | 32.2 ms | **9.8 ms** | **3.27×** | 0.270 GB | **0.161 GB** |
| 4000 × 1000  | 48.8 ms | **20.4 ms** | **2.39×** | 0.304 GB | **0.221 GB** |
| 8000 × 3000  | 142.0 ms | **50.1 ms** | **2.83×** | 0.664 GB | **0.586 GB** |
| 15000 × 2000 | 164.9 ms | **70.7 ms** | **2.33×** | 0.810 GB | **0.715 GB** |

(Run-to-run spread on this shared machine is ±10–20% — e.g. the baseline at
2000×500 measured 28–32 ms across runs; ratios are more stable than absolutes.)

* Accuracy matches: reconstruction error is identical to 5 decimals, top-10
  singular values agree to ~1e-7 (the float32 limit) and the leading subspaces
  align to ≤0.0001° (`TESTING/tinygrad_validate.py`). The one large angle in the
  `shifted` spectrum is between *near-degenerate tail* directions, not a real
  discrepancy — reconstruction and singular values still match.
* Peak RSS is **12–40% lower** than the baseline at every size (fresh process,
  matrix pre-loaded).
* **Caveat — cold start.** The first call for a new shape pays the `TinyJit`
  trace, ≈0.64–0.69 s vs the baseline's 31–165 ms first call; every later call is
  13–73 ms. Most of that is fixed backend cost: `import tinygrad` alone is 0.25 s
  and only ~0.1 s of the trace scales with `ns_steps`. Measured as **total** cold
  time in a fresh process (interpreter + numpy + library import + load A +
  6 SVDs) the two backends are 0.80 s vs 1.16 s at 2000×500, but **1.64 s vs
  1.56 s** at 15000×2000 — cold-start parity from medium sizes up, plus the
  2.1–2.5× steady-state win everywhere. Call `warmup(m, n)` once at startup to
  move the trace into initialisation.
* **20 GB memory safeguard:** `check_shape()` refuses a shape whose projected
  working set would exceed the cap *before* allocating, and `check_mem()`
  re-checks peak RSS at each stage. The cap is `FASTPCA_TG_MEM_GB` (default 20).

### Backend readiness and device handling

The package backend `inst/python/tinygrad_svd_func.py` exposes the exact baseline
signature (`randomized_svd_py_tg(A, k, p=10, q_iter=2, device="cpu", cores=2)`)
and the same float64 `(U, S, Vh)` contract; `cores` is accepted but ignored
(tinygrad schedules its own GPU work). It mirrors the kernel developed in
`TESTING/tinygrad_native.py`.

R passes `"CPU"`/`"GPU"`. On this machine **only METAL works** — tinygrad's CPU
backend shells out to a cross-compiler the toolchain rejects and there is no CUDA
device — so `resolve_device()` maps `"CPU"` straight to METAL (quietly; probing
CPU leaks a compiler error) and `"GPU"` prefers METAL then CUDA. Override with
`FASTPCA_TG_DEVICE`. On non-Darwin, `"CPU"` honours tinygrad's CPU backend.

Wired into the package: `FastPCA(backend = "tinygrad")` dispatches to this
kernel, the loader is backend-aware (`R/utils.R`), `setup_py_env(backend =
"tinygrad")` installs tinygrad, and the device mapping above applies there too
(on arm64 Macs a `device = "CPU"` request runs on METAL and says so). Tests live
in `tests/testthat/test-tinygrad.R`.

### Matrix transform (prep_matrix)

`inst/python/tinygrad_transform.py` implements log2 / transpose / centre-scale.
The torch baseline materialises every step in float64 (log2, a copy-transpose,
mean+subtract, std+divide). The tinygrad version:

* never builds a transposed copy -- it takes the statistics along whichever axis
  becomes dim=0 after the transpose and only transposes at the end;
* lets tinygrad's scheduler fuse log2 + subtract + divide into one kernel
  (~3 reads + 1 write of the data);
* uses float32 buffers (Metal has no double), casting back to the input dtype on
  return, and matches torch to ~1e-6 relative.

Measured with `tg_transform_bench.py` (median of 3, same interpreter; the
optimization was developed as `TESTING/tinygrad_transform_optim.py`):

| shape | config | torch | tinygrad | speedup |
|---|---|---|---|---|
| 6000x1500 | log2, scale | 74.8 ms | **25.2 ms** | 3.0x |
| 6000x1500 | log2, transpose, scale | 74.1 ms | **22.0 ms** | 3.4x |
| 15000x2500 | log2, scale | 314.3 ms | **82.7 ms** | 3.8x |
| 15000x2500 | log2, transpose, scale | 309.3 ms | **77.8 ms** | 4.0x |

Correctness vs a float64 NumPy reference is ~3e-7..9e-7 (float32). Whole-process
peak RSS is comparable to torch -- lower at medium sizes, marginally higher at the
largest because of the float32 staging copy plus device and host output buffers.

On a flat / near-degenerate spectrum *every* randomized method is ill-conditioned:
the torch baseline and this kernel were both ~7% off the exact top singular values
(tinygrad 6.66%, torch 6.90%). That is a property of the random sketch, not a
backend regression -- use an exact solver (`r`/`irlba`) when it matters.

### Environment findings

* The *previous* `inst/python/tinygrad_svd_func.py` did **not run as shipped**:
  its CPU path shells out to a cross-compiler (`--target=arm64-none-unknown-elf`)
  that the conda `cc` rejects, and its GPU path selects OpenCL (`OpenCL Error
  -48: CL_INVALID_KERNEL`). Only tinygrad's native **METAL** device works, which
  is why the rewritten file maps `"CPU"` to METAL on Apple Silicon.
* tinygrad resolves its kernel-cache path at import, and R's
  `py_module_available()` imports it before the backend file runs. The module now
  re-points `tinygrad.helpers.cache_dir` when the default is read-only, which
  otherwise fails on first use with "attempt to write a readonly database".
* `Tensor.qr` / `Tensor.svd` are unusable for the shapes here (see above), so
  this kernel avoids both. An earlier Metal-matmul + numpy-QR *hybrid* was
  79.8 ms vs 21.5 ms (4000×1000) — slower, and has been removed.

## NIPALS on missing-value data: pytorch vs R

`FastPCA()` rejects `NA`: base R, irlba and every randomized backend need
complete data. NIPALS is the usual way to handle missing quantification
(metabolomics / proteomics) and, per `ROADMAP.md`, is planned as its own
function rather than another `backend =`. This section measures what it costs.

### Harness

| file | purpose |
|---|---|
| `nipals_export.R` | transposes the example matrix to samples x features, subsamples features, writes raw float64 + a shared MCAR mask so R and pytorch see identical missingness |
| `nipals_torch.py` | single-core pytorch NIPALS, a faithful port of the R package's algorithm |
| `nipals_r.R` | base-R NIPALS, the vendored CRAN package, and an irlba reference |
| `nipals_tinygrad.py` | the same NIPALS loop as one TinyJit graph on tinygrad Metal (or CPU); float32, two m x n device buffers |
| `nipals_validate.py` | relative eigenvalue error + principal angles between loading subspaces |
| `nipals_collect.py` | merges per-run JSON and peak RSS into one results file |
| `run_nipals_bench.sh` | tiers s / m / f, one implementation per process under `/usr/bin/time -l` |
| `vendor/nipals/` | vendored CRAN source (CRAN is unreachable from this sandbox) |

Data: `data/prepped_gex.rds`, **199112 samples (cells) x 978 features
(genes)**, **zero NAs** -- already the samples x features orientation PCA wants.
The Seurat assay behind it has dim `features x cells = 978 x 199112` with gene
symbols in the feature rownames, so no transpose is needed. `nipals_export.R`
nevertheless treats rows as features and writes the transpose (`978 x F`, genes
as observations), which is what every table below measured; see the orientation
correction near the end for exactly what that does and does not change. Peak RSS
is measured identically for R and python by `/usr/bin/time -l`.

### Tier m -- 978 x 20,000 features (10% of genes), ncomp = 10

| implementation | complete s | peak RSS GB | 5% MCAR s | peak RSS GB |
|---|---|---|---|---|
| **pytorch fp64, 1 thread** | **2.12** | **0.53** | **4.36** | **1.06** |
| base R NIPALS (deflation, O(k) Gram-Schmidt) | 13.75 | 1.25 | 26.41 | 2.20 |
| `nipals` CRAN package (vendored) | 94.72 | 7.97 | 159.62 | **13.43** |
| `irlba` (not NIPALS; complete-data yardstick) | 1.40 | 0.65 | -- | -- |

pytorch is **6.5x** faster than careful base R and **45x** faster than the CRAN
package; at 5% missing it is 6.1x / 37x and uses **12.7x** less peak RSS than the
package.

### Tier f -- the full 978 x 199,112 matrix, ncomp = 10

| implementation | complete s | peak RSS GB | 5% MCAR s | peak RSS GB |
|---|---|---|---|---|
| **pytorch fp64, 1 thread** | **21.24** | **3.39** | **44.37** | 8.65 |
| base R NIPALS | 130.87 | 11.30 | 262.72 | 12.92 |
| `nipals` CRAN package | **N/A** (~295 GB allocation) | -- | N/A (~298 GB) | -- |
| `irlba` (complete data only) | 10.52 | 4.78 | -- | -- |

The CRAN package allocates `TTp <- matrix(0, nobs, nobs)` and
`PPp <- matrix(0, nvar, nvar)` for its Gram-Schmidt step, so at 199112 features
it cannot run in either orientation. `nipals_r.R` pre-checks the allocation and
reports N/A rather than being OOM-killed.

### Tier m and f -- tinygrad Metal (fp32)

A third engine runs the same NIPALS loop as a single `TinyJit` graph on Metal.
float32 is forced (Metal has no fp64), the data and mask are the only two m x n
device buffers, and `--block` unrolls B iterations per graph replay so the host
syncs once per block. `--device CPU` also works, but only with Apple clang on
PATH and CC: the anaconda wrapper rejects tinygrad's `--target` flag. CPU is
~10x slower than Metal at tier m.

| implementation | complete s | peak RSS GB | 5% MCAR s | peak RSS GB |
|---|---|---|---|---|
| **tinygrad Metal fp32, block=8** | **0.91** | 0.53 | **1.56** | **0.75** |
| pytorch fp64, 1 thread | 2.02 | **0.53** | 4.13 | 1.06 |
| base R NIPALS | 13.48 | 1.19 | 26.17 | 2.13 |
| `nipals` CRAN package | 96.06 | 7.54 | 160.82 | 11.62 |
| tinygrad CPU fp32, block=8 | 13.81 | 0.56 | 24.63 | 0.77 |

At the full 978 x 199,112 shape (fresh process per run):

| implementation | complete s | peak RSS GB | 5% MCAR s | peak RSS GB |
|---|---|---|---|---|
| **tinygrad Metal fp32** | **16.22** | **3.17** | **19.30** | 4.53 |
| pytorch fp64, 1 thread | 20.94 | 3.39 | 46.17 | **8.64** |

tinygrad is ~2.2x / 2.6x faster than pytorch at tier m (complete / 5% MCAR) and
~1.3x / 2.4x at full size, using about half the RSS once a mask is involved
(0.75 vs 1.06 GB at tier m, 4.53 vs 8.64 GB full); on complete data it only
edges pytorch out (0.53 vs 0.55 GB at tier m). At the small F2000 shape it is
*slower* (0.30 s vs 0.08 s) because kernel-launch latency dominates, though it
still uses less memory (0.16 vs 0.23 GB).

Accuracy is float32-limited. Against the pytorch fp64 reference, at F2000
(k=5) the max relative eigenvalue error is 3.4e-6 complete and 1.6e-5 at 5%
MCAR, with loading-subspace angles <=0.41 and <=0.42 degrees; over all ten
components at tier m it is 2.1e-5 / 4.1e-5 (angles <=0.45 / <=0.54 degrees) and
at full size 1.8e-5 / 2.2e-5 (angles <=0.64 / <=0.66 degrees). Because
`tol = 1e-6` sits at the fp32 rounding floor, the stopping test is noisier than
in fp64: tinygrad spends slightly more iterations at F2000 (448 vs 386 complete,
416 vs 379 with MCAR) without moving the eigenvalues further. Run-to-run spread
on this machine remains +/-20%.

These numbers were re-measured after the Gram-Schmidt buffers in
`nipals_tinygrad.py` were corrected: on this tinygrad version `.assign()` is a
no-op when applied to an indexed view, so Gram-Schmidt had silently been skipped.
The timings barely move (the projection is O(nvar x k) per iteration), but the
later components are now orthogonal to the earlier ones and agree with fp64.

Results: `results/nipals_tinygrad_bench.json`.

### Correctness

* pytorch vs base R: identical eigenvalues **and** iteration counts at every tier
  (max relative eigenvalue error 1.7e-14, subspace angle <=0.0004 deg).
* pytorch vs the CRAN package: max relative eigenvalue error 6.8e-08 -- they agree
  to the `tol = 1e-6` convergence tolerance.
* Both are within ~2e-6 of `irlba`, the more converged solver.
* The package reports **one extra iteration per component** (17 vs our 16):
  upstream increments its counter after convergence is detected. Eigenvalues are
  unaffected; `nipals_torch.py` reports the true count.

### Missing-value behaviour

The real dataset has zero NAs, so missingness is injected as MCAR at a fixed
seed in `nipals_export.R` and the *same* mask is read by both languages. On
tier m, 5% MCAR costs pytorch 2.1x the complete-data time and base R 1.9x, while
the package pays in both time (1.7x) and memory (**+69% RSS**, 8.0 -> 13.4 GB),
because its inner loop materialises a full nobs x nvar matrix every iteration.

### Caveats

* **R runs on reference BLAS here.** `extSoftVersion()["BLAS"]` is
  `libRblas.0.dylib`, linked by absolute path; the framework also ships
  `libRblas.vecLib.dylib`, but it cannot be selected without modifying the R
  framework (outside this workspace). Part of pytorch's margin is therefore BLAS
  quality, not algorithm. Raw single-thread power-iteration cost on the full
  matrix: **R 341 ms vs torch fp64 53 ms** (fp32 26 ms).
* Single thread throughout (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`);
  run-to-run spread on this machine is +/-10-20%.
* `scale = FALSE` (centre only) everywhere, so every implementation does the same
  work; `--scale=1` is available on both sides.
* Missingness here is synthetic MCAR; the shipped example data has no NAs.
## Iterative (IRLBA) PCA: PyTorch port vs irlba::irlba and FastPCA-pytorch

`FastPCA(backend = "pytorch")` is a **randomized** SVD: it estimates the leading
subspace from one random sketch plus `q_iter` power iterations. An **iterative**
Lanczos solver (what `irlba::irlba` implements) instead grows an explicit Krylov
basis and restarts it until the singular values stop moving -- many more
matrix-vector products, but near-exact. This section asks how a faithful PyTorch
port of irlba's own algorithm compares, on the shipped example matrix, with (a)
the `irlba` package and (b) FastPCA's randomized backend.

### Implementations

| file | what it runs |
|---|---|
| `TESTING/irlba_torch.py` | PyTorch transcription of irlba's pure-R fallback: Lanczos bidiagonalization + implicit restart, `svd(B)` per sweep, irlba's own `convtests()` stopping rule |
| `TESTING/irlba_r.R --impl=fast` | `irlba::irlba()` C fast path (the package version) |
| `TESTING/irlba_r.R --impl=fallback` | `irlba::irlba(fastpath = FALSE)`, the pure-R reference |
| `TESTING/fastpca_torch_bench.R` | installed FastPCA 0.9.0.0 `FastPCA(backend = "pytorch")` through reticulate |
| `TESTING/fastpca_kernel_bench.py` | the shipped `torch_svd_func.py::randomized_svd_py` called directly, no reticulate |

Data: `data/prepped_gex.rds` (199112 features x 978 samples) transposed to
samples x features and column-centred by every runner, so all five see identical
bytes. `k = 10`; the iterative solvers use `work = k + 7 = 17`, the randomized
ones their package defaults (`p = 10`, `q_iter = 2` -> a 20-column sketch).

### Harness

One implementation per process. Each run is wrapped by
`TESTING/memguard.py --cap-gb 20`, which polls the resident size of the whole
process tree (via `proc_pid_rusage`; `/bin/ps` is denied in this sandbox) and
SIGKILLs the process group above 20 GB, then by `/usr/bin/time -l` for an
independent peak-RSS reading. Thread count is set with `--threads/--cores` and
swept over 1 / 2 / 4. `TESTING/run_irlba_bench.sh` reproduces everything;
`memguard_selftest.sh` proves the cap (a 1.5 GB child under a 0.25 GB ceiling is
SIGKILLed, exit 137).

`DTYPE=fp64|fp32` and `DEVICE=cpu|gpu` are passed through to
[irlba_torch.py](TESTING/irlba_torch.py), so the precision x device sweep is
just repeated harness invocations:

    TIER=f DTYPE=fp64 DEVICE=cpu ./TESTING/run_irlba_bench.sh
    TIER=f DTYPE=fp32 DEVICE=cpu ./TESTING/run_irlba_bench.sh
    TIER=f DTYPE=fp32 DEVICE=gpu ./TESTING/run_irlba_bench.sh

Measured results for that sweep are recorded in
[results/precision_device.txt](TESTING/results/precision_device.txt).

### Results -- full `prepped_gex` (978 x 199112), k = 10

best of 3; RSS is `/usr/bin/time -l` peak:

| implementation | cores=1 | cores=2 | cores=4 | peak RSS GB |
|---|---|---|---|---|
| **`torch_irlba`** (this port) | 1.780 s | **1.764 s** | 1.777 s | **3.31** |
| `fastpca_kernel` (direct) | 0.701 s | 0.645 s | **0.624 s** | 3.49 |
| `fastpca_torch` (R API) | 0.779 s | **0.719 s** | 0.734 s | 5.28 |
| `irlba::irlba` (C fast path) | -- | 8.640 s | -- | 4.97 |
| `irlba(fastpath = FALSE)` (R) | -- | 11.170 s | -- | 5.82 |

### Accuracy

Against the C fast path (itself equal to a full `numpy.linalg.svd` on F2000):

| implementation | max rel. eigenvalue error | max loading-subspace angle |
|---|---|---|
| `torch_irlba` | **1.5e-11** | 0.0006 deg |
| `fastpca_kernel`, q_iter=2 (default) | 1.6e-01 | 54 deg |
| `fastpca_torch`, q_iter=2 (default) | 1.4e-01 | 38 deg |

The port matches irlba's *own* pure-R fallback even more tightly: given the same
start vector, F2000 agrees to 3.5e-15 on the eigenvalues and ~1e-6 degrees
between loading subspaces, with identical iteration counts (`iter = 7`,
`mprod = 82`) and the same convergence test.

The randomized error is a property of this near-flat spectrum, not a bug --
extra power iterations converge it:

| q_iter | max rel. singular-value error, F2000 | full |
|---|---|---|
| 2 (default) | 8.8 - 10.9% | 3.9 - 10.6% |
| 4 | 1.7 - 3.4% | 0.13 - 1.35% |
| 6 | 0.29 - 0.34% | 0.05 - 0.24% |

Even at `q_iter = 6` the randomized kernel is orders of magnitude less accurate
than the iterative port, which reaches machine precision.

### Size scaling (k = 10)

| matrix | `torch_irlba` best | `fastpca_kernel` best | FastPCA-R best | `irlba_fast` |
|---|---|---|---|---|
| 978 x 2000   | **12 ms** (c2) | 21 ms | 90 ms | 105 ms |
| 978 x 20000  | 190 ms (c2) | **77 ms** (c4) | 141 ms | 865 ms |
| 978 x 199112 | 1.76 s (c2) | **0.62 s** (c4) | 0.72 s | 8.64 s |

### Does the thread knob actually bind? (yes -- but not to matmul)

`--threads/--cores` reaches `torch.set_num_threads()`, and torch honours it:
inside every run `get_num_threads()` returns exactly the requested value, and
the knob is demonstrably live because ATen's elementwise/reduction kernels scale:

| requested 1 -> 4 | op (978 x 199112 fp64) | wall @1 thr | wall @4 thr | parallelism |
|---|---|---|---|---|
| `x.mul(2)`   | elementwise (ATen pool) | 152 ms | 72 ms | **1.00x -> 2.77x** |
| `x.sum()`    | reduction   (ATen pool) | 24 ms  | 14 ms | **1.00x -> 3.95x** |

Every matrix product in these solvers, however, is dispatched to **Apple
Accelerate** (`torch.__config__`: `BLAS_INFO=accelerate`, `USE_MKL=OFF`), and
Accelerate executes these GEMMs single-threaded:

| shape | requested=1 | requested=4 | parallelism |
|---|---|---|---|
| skinny `(978 x 199112) @ (199112 x 17)` | 85.6 ms | 85.1 ms | **1.00x / 1.00x** |
| square `2048 x 2048` fp64 | 49.2 ms | 49.8 ms | **1.00x / 1.00x** |

`OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=8` and requested=8 make no
difference either. End to end on the real tier-f workload,
`cpu/wall = (user+sys)/real` is ~1.0x at both settings:

| run | threads=1 | threads=4 |
|---|---|---|
| `torch_irlba` best / cpu-wall | 2.04 s / **0.98x** | 2.00 s / **1.07x** |
| `fastpca_kernel` best / cpu-wall | 0.75 s / **1.07x** | 0.66 s / **1.15x** |

So the flat core sweep is real, and it means *one core does all of the work* --
not that a fixed larger pool was always engaged. The knob is respected by
PyTorch; it simply has nothing to act on, because the cost lives in BLAS. A
single core sustains ~350 GFLOPS fp64 on a 4096-square GEMM (Apple's matrix
unit, single-threaded), which is why spreading identical work over more cores
does not pay. The real levers on this hardware are precision and execution unit
(fp32, then MPS/GPU), not more CPU threads -- see the next section.

`TESTING/run_thread_verify.sh` reproduces every number above into
`results/thread_verify.txt`.

### Can the single core be avoided? (yes -- but it is slower)

Single-core GEMM is a property of **Accelerate**, not of the hardware. Accelerate
dispatches these GEMMs to Apple's AMX matrix unit automatically and exposes no
supported runtime switch to disable it, so there is nothing to toggle inside the
torch wheel. But you are not stuck on one core: swapping in a BLAS that threads
gets you real parallelism. In this same conda env `numpy` links such a library:

| fp64 3072-square GEMM | wall | cpu/wall |
|---|---|---|
| torch, `threads=1` (Accelerate/AMX) | 162.0 ms | **1.00x** |
| torch, `threads=4` (Accelerate/AMX) | 162.2 ms | **1.00x** |
| numpy (its own BLAS) | 272.7 ms | **3.99x** |

So multi-core *is* reachable -- and it still loses: one AMX core (162 ms) beats
~4 OpenBLAS cores (273 ms) by 1.7x, because one AMX core sustains ~350 GFLOPS
fp64 while the threaded library manages only ~50 GFLOPS/core. That is why every
torch row here is single-core and why the core sweep is flat: routing these GEMMs
through a threaded BLAS would add parallelism and *reduce* performance.

The other execution unit is Metal/MPS, and it *is* fast in fp32 (3072-square
GEMM: 11.9 ms on MPS vs 44.5 ms for CPU fp32). But it refuses float64 outright:

    Cannot convert a MPS Tensor to float64 dtype as the MPS framework
    doesn't support float64. Please use float32 instead.

`TESTING/blas_backend_probe.py` reproduces those GEMM numbers
(`KMP_DUPLICATE_LIB_OK=TRUE python TESTING/blas_backend_probe.py`).

### Precision x device: fp32 is the real lever, not cores

[irlba_torch.py](TESTING/irlba_torch.py) takes `--precision double|single`
(`fp64`/`fp32` are accepted aliases) and `--device CPU|GPU`, treated as
**independent** choices. The vocabulary deliberately matches FastPCA's own:
`R/validate.R` defines `.fastpca_devices <- c("CPU", "GPU")`, a GPU request
means CUDA when available and otherwise falls back to the CPU **with a message**,
and that is exactly what this port does now (same wording as
[inst/python/nipals_func.py](inst/python/nipals_func.py)). One platform rule is
added on top: because MPS cannot hold float64, an fp64 + Apple-GPU request is
served on the CPU and *reported* (`device_status = "mps-fp64-downgrade"`)
instead of silently changing dtype.

Because the package has no Apple value in `.fastpca_devices`, MPS is reachable
here only through an explicit benchmark-only `--device MPS`, and it is fp32
only. Verified on this machine (tier s, k = 5):

| request | resolved | `device_status` |
|---|---|---|
| `--precision double --device GPU` | cpu (no CUDA here) | `no-gpu-fallback` |
| `--precision single --device GPU` | cpu (no CUDA here) | `no-gpu-fallback` |
| `--precision single --device MPS` | **mps** | `requested` |
| `--precision double --device MPS` | cpu | `mps-fp64-downgrade` |

**This whole constraint is Apple-only.** CUDA supports float64 natively, so on a
Linux/NVIDIA host `device = "GPU"` with the fp64 default runs on the GPU
unchanged -- there is nothing to fix in the package for that case. Two things are
worth documenting rather than changing: on consumer GeForce parts fp64 runs at a
fraction of the fp32 rate (datacentre cards are fine), so `GPU` + float64 can
still lose to the CPU; and on arm64 Macs `torch_svd_func_exact.py` would report
success while silently running the whole solve on the host, because MPS has no
native `linalg.svd`.

Tier f (978 x 199112, k = 10), best of 2, identical start vector and `tol`:

| config | best | vs fp64 CPU | iter / mprod | max rel. eigenvalue err vs `irlba` |
|---|---|---|---|---|
| fp64 CPU (AMX) | 2.128 s | 1.00x | 5 / 66 | 1.5e-11 |
| **fp32 CPU (AMX)** | **1.042 s** | **2.04x** | 5 / 66 | 6.1e-06 |
| **fp32 MPS (GPU)** | **0.661 s** | **3.22x** | 5 / 66 | 1.6e-06 |

So the way around "one core" is not more cores, and not necessarily even the
GPU: fp32 on the *same* single AMX core is 2x faster than fp64, and MPS adds
another ~1.5x at full width -- 3.2x over the fp64 baseline, with principal
angles under 0.003 degrees and a relative eigenvalue error two orders below
`tol = 1e-5`.

**MPS only wins once the matrices are wide.** At tier s (978 x 2000) the same
solve is **120 ms on MPS vs 6 ms fp32 CPU** -- 20x *slower*, because each of the
~74 products is tiny and every call pays dispatch plus a host round-trip for the
small SVD. Related op gaps in torch 2.9: there is no native MPS
`linalg.qr`, `svdvals` or `eigh`, and plain `linalg.svd` silently falls back
to the CPU with a warning per call, so [irlba_torch.py](TESTING/irlba_torch.py)
moves the `work x work` matrix to the host explicitly. The first MPS call in a
process also pays ~3.5 s of kernel compilation (repeat 1 = 4.31 s, repeat 2 =
0.66 s), so GPU timings are best-of-N after a warm-up.

**The RSS cap does not see Metal memory.** `ru_maxrss` stayed at 3.31 GB while
the fp32 matrix sat on the GPU (memguard saw a 2.78 GB tree peak), so
[irlba_torch.py](TESTING/irlba_torch.py) also checks
`torch.mps.driver_allocated_memory()` after every repeat and aborts above
`--max-gb`. An RSS-only guard would under-count an MPS run. (The empty-subspace
guard added to [orthog()](TESTING/irlba_torch.py#L66-L75) is likewise defensive:
the `(m, 0) @ (0,)` projection trips an MPS-internal *"Placeholder tensor is
empty!"* assert that the CPU tolerates.)

[device_dtype_probe.py](TESTING/device_dtype_probe.py) records the platform
facts behind this section:

    KMP_DUPLICATE_LIB_OK=TRUE python TESTING/device_dtype_probe.py

### What this says

* The PyTorch port is **4.9x faster than `irlba::irlba`'s C path** and 6.3x
  faster than the R fallback on full data, with results identical to ~1e-11.
  That is the "better performance without losing accuracy" result.
* FastPCA's randomized backend is another ~2.5x faster again, but on this
  near-flat spectrum its default `q_iter = 2` is ~16% off on the later singular
  values. Use `torch_irlba` (or raise `q_iter` and pay for it) when the tail
  matters.
* **Core count barely moves anything, and that is expected.** The flag is
  honoured by PyTorch but matmul runs single-threaded through Apple Accelerate,
  so one core performs all the work: `torch_irlba` is flat across 1/2/4 threads
  and the randomized kernel's ~11% (0.70 -> 0.62 s) comes from its non-BLAS
  elementwise work, not from a faster GEMM. See "Does the thread knob actually
  bind?" above for the `cpu/wall` measurements.
* **Precision beats core count.** fp32 on the same AMX core is 2.04x faster
  than fp64 at tier f (1.042 s vs 2.128 s) and MPS adds another ~1.5x
  (0.661 s), all within `tol`: see "Precision x device" above.
* Memory on full data: `torch_irlba` 3.31 GB < direct kernel 3.49 GB <
  `irlba` C 4.97 GB < FastPCA-R API 5.28 GB < R fallback 5.82 GB. Every run
  stayed well under the 20 GB ceiling; nothing was killed.

### Caveats

* R runs on **reference BLAS** (`libRblas.0.dylib`, single-threaded) while the
  torch rows use Apple Accelerate (also effectively single-threaded here), so
  part of the port's margin is BLAS quality rather than algorithm. The
  algorithmic match was verified separately with a shared start vector.
* FastPCA-pytorch pays for the R -> numpy conversion: 5.28 GB versus 3.49 GB for
  the same kernel called directly from Python.
* The FastPCA rows benchmark the **installed** package (0.9.0.0); this checkout's
  `R/FastPCA.R` differs only in defaults (`cores`, variance bookkeeping) and
  adds the tinygrad backend.
* The full tier is 1.45 GiB of doubles; its `X_*.f64` export is regenerated by
  the orchestrator and deleted again on this machine's small disk.
* Run-to-run spread is a few percent; the core-sweep differences sit inside it
  except for the randomized kernel's consistent ~11% at four threads.

## Orientation correction and the packaged `torch_irlba()`

The IRLBA/NIPALS benchmarks above ran on the transpose of the example data, and
that is worth stating plainly because it flips which axis gets centred.

**The evidence.** `data/seurat_object_v5.rds` carries the same matrix: its
default assay (`Nanostring`) is `978 x 199112` with gene symbols in the
**feature** rownames (`ABL1`, `ABL2`, ...) and cell barcodes in the columns
(`Run5448.RCC3_1_1`). So `data/prepped_gex.rds`, at `199112 x 978`, is
**samples (cells) x features (genes)** already, and standard PCA centres its 978
columns. `nipals_export.R` assumed rows were features, so it wrote the
transpose (`978 x F`) and every runner column-centred *cells* instead of genes.

**What it changes.** Singular values are transpose-invariant, so the
accuracy-vs-`irlba` comparisons stand: on the full matrix,
`max |d(A) - d(t(A))| / d = 2.8e-11`. What changes is the centring axis, hence
the actual principal subspace and PVE -- these are gene-wise PCs only in the
correct orientation.

**The packaged function.** The port was promoted into the package as an exported
`torch_irlba()` ([R/torch_irlba.R](R/torch_irlba.R), kernel
[inst/python/torch_irlba_func.py](inst/python/torch_irlba_func.py)). It mirrors
`irlba::irlba()`'s iterative arguments (`nv`, `work`, `maxit`, `reorth`,
`tol`, `svtol`, `v`) and adds the package's backend controls:
`precision = "double" | "single"` and `device = "CPU" | "GPU"`, with the same
fastpca result contract as `FastPCA()`/`NIPALS()`. It validates arguments
before touching Python, and with a shared start vector it reproduces
`irlba::irlba()`.

**Correct-orientation measurement.** `data/prepped_gex.rds` used as-is
(199112 samples x 978 genes), gene-wise centring, `k = 10`, shared start
vector, best of 2 ([TESTING/irlba_orientation_check.R](TESTING/irlba_orientation_check.R),
raw output [results/irlba_orientation.txt](TESTING/results/irlba_orientation.txt)):

| implementation | time s | rel. eigenvalue error vs `irlba` | max principal angle |
|---|---|---|---|
| `irlba::irlba()` (C fast path) | 8.438 | reference | -- |
| `torch_irlba()` fp64, CPU | 2.961 | 2.19e-09 | 0.0027 deg |
| `torch_irlba()` fp32, CPU | **1.897** | 3.46e-05 | 0.238 deg |
| `torch_irlba()` fp32, GPU (Metal) | **1.718** | 1.66e-06 | 0.037 deg |

On the correctly-oriented data the PyTorch Lanczos solver is **~2.9x faster than
irlba's C path in float64**, and ~4.5-4.9x faster once precision is lowered,
while agreeing with `irlba` to 1e-9 in float64.

## Next steps

- ~~NIPALS is measured and validated here but not yet in the package~~ —
  done: `NIPALS(backend = "tinygrad")` ships with its own module
  (`inst/python/nipals_tinygrad_func.py`) and tests in
  `tests/testthat/test-nipals.R`.
- ~~Enable `backend="tinygrad"`~~ — done for both `FastPCA()` and
  `prep_matrix()`, including the optimized matrix transform.
- ~~Add a low-precision (float32) option to the R API~~ — partially done:
  `torch_irlba()` takes `precision = "double" | "single"` and
  `device = "CPU" | "GPU"`. Still to do: thread the same two arguments through
  `FastPCA()`/`NIPALS()` and their Python kernels.
- Optional/future: a custom native SVD kernel for `B`. Investigated and rejected
  for now (see "The final small SVD"): tinygrad's `Tensor.svd` is unusable and it
  exposes no eigh/eig/cholesky/solve, so numpy's Gram+eigh is the fastest route.
