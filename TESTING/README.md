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

## Next steps

- ~~Enable `backend="tinygrad"`~~ — done: the backend choice, a backend-aware
  loader and `setup_py_env(backend = "tinygrad")` are wired; `prep_matrix()`
  still warns and falls back to pytorch because the transform is not implemented.
- Add a low-precision (float32) option to the R API so users can opt into the
  ~3–5× PyTorch path.
- Optional/future: a custom native SVD kernel for `B`. Investigated and rejected
  for now (see "The final small SVD"): tinygrad's `Tensor.svd` is unusable and it
  exposes no eigh/eig/cholesky/solve, so numpy's Gram+eigh is the fastest route.
