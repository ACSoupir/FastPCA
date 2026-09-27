# TESTING — optimised PyTorch SVD kernels

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
| `results/*.json`   | raw measurements |

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

## tinygrad (status: experimental, not yet a win)

tinygrad is **not** in `/opt/anaconda3/envs/FastPCA`. A workspace venv is used
instead (gitignored):

```bash
/opt/anaconda3/envs/FastPCA/bin/python -m venv --system-site-packages TESTING/.venv-tg
TESTING/.venv-tg/bin/python -m pip install tinygrad        # 0.11.0
# tinygrad caches kernels under ~/.cache (read-only in this sandbox):
export XDG_CACHE_HOME="$PWD/TESTING/.cache"
```

Findings on this machine (macOS arm64, tinygrad 0.11):

* `inst/python/tinygrad_svd_func.py` does **not run as shipped**:
  * its CPU path shells out to a cross-compiler
    (`--target=arm64-none-unknown-elf`) that the conda `cc` rejects;
  * its GPU path selects OpenCL (`ops_gpu.py`) and dies with
    `OpenCL Error -48: CL_INVALID_KERNEL`. Only tinygrad's native **METAL**
    device works.
* A pure-tinygrad kernel is not obviously worthwhile: its bundled Householder
  QR / Jacobi SVD are Python-graph heavy (hundreds of tiny kernels) and lose to
  LAPACK, and Metal matmuls are ~4–5 ms each for the shapes here.
* `tinygrad_rsvd.py` is a **hybrid** (Metal matmuls + numpy for the small QR /
  SVD). Measured against torch-CPU `stable`: 79.8 ms vs 21.5 ms (4000×1000) and
  255 ms vs 114.5 ms (15000×2000) — i.e. currently *slower*. Getting a real win
  would mean native Metal kernels for the orthonormalisation/SVD and eliminating
  host round-trips, which is a much larger piece of work.

So the tinygrad backend needs design/effort before it can be called an
optimisation; the PyTorch kernels are where the concrete speed-up is today.

## Next steps

- Decide whether to invest in native-Metal tinygrad kernels (large) or leave the
  backend disabled.
- Wire `stable` behind `R/utils.R` / `FastPCA()`.
- Add a low-precision (float32) option to the R API so users can opt into the
  ~3–5× path.
