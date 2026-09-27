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
  Python dispatch that otherwise dominates (≈220 ms → ≈13 ms on 2000×500).
* The only host-side step is the SVD of `B = QᵀA` (b×n, b=k+p).

### Results vs the in-package torch baseline

Measured by `TESTING/run_tinygrad_bench.sh`, both backends in the **same**
interpreter (`TESTING/.venv-tg/bin/python` sees torch *and* tinygrad), k=50,
p=10, q_iter=2:

| matrix | baseline steady | native tg steady | speed-up | baseline peak RSS | native tg peak RSS |
|---|---|---|---|---|---|
| 2000 × 500   | 28.9 ms | **12.9 ms** | **2.24×** | 0.296 GB | **0.165 GB** |
| 4000 × 1000  | 45.1 ms | **19.9 ms** | **2.27×** | 0.317 GB | **0.225 GB** |
| 8000 × 3000  | 135.6 ms | **53.4 ms** | **2.54×** | 0.657 GB | **0.598 GB** |
| 15000 × 2000 | 157.9 ms | **73.3 ms** | **2.15×** | 0.808 GB | **0.715 GB** |

* Accuracy matches: reconstruction error is identical to 5 decimals, top-10
  singular values agree to ~1e-7 (the float32 limit) and the leading subspaces
  align to ≤0.0001° (`TESTING/tinygrad_validate.py`). The one large angle in the
  `shifted` spectrum is between *near-degenerate tail* directions, not a real
  discrepancy — reconstruction and singular values still match.
* Peak RSS is **9–44% lower** than the baseline at every size (fresh process,
  matrix pre-loaded).
* **Caveat — cold start.** The first call for a new shape pays the `TinyJit`
  trace, ≈0.64–0.69 s vs the baseline's 31–165 ms first call; every later call is
  13–73 ms. Call `warmup(m, n)` once at startup (≈0.62 s) to move that cost into
  initialisation, exactly as a torch import is paid today.
* **20 GB memory safeguard:** `check_shape()` refuses a shape whose projected
  working set would exceed the cap *before* allocating, and `check_mem()`
  re-checks peak RSS at each stage. The cap is `FASTPCA_TG_MEM_GB` (default 20).

`randomized_svd_py_tg(...)` is a signature-compatible drop-in for the disabled
`inst/python/tinygrad_svd_func.py`, and `randomized_svd_tg(...)` returns the
same `(U, S, Vh)` contract as `torch_svd_func.randomized_svd_py`.

### Environment findings

* `inst/python/tinygrad_svd_func.py` does **not run as shipped**: its CPU path
  shells out to a cross-compiler (`--target=arm64-none-unknown-elf`) that the
  conda `cc` rejects, and its GPU path selects OpenCL (`OpenCL Error -48:
  CL_INVALID_KERNEL`). Only tinygrad's native **METAL** device works.
* `Tensor.qr` / `Tensor.svd` are unusable for the shapes here (see above), so
  this kernel avoids both. An earlier Metal-matmul + numpy-QR *hybrid* was
  79.8 ms vs 21.5 ms (4000×1000) — slower, and has been removed.

## Next steps

- Wire `stable` (and/or the tinygrad kernel) behind `R/utils.R` / `FastPCA()`;
  the tinygrad loader already calls `randomized_svd_py_tg(...)`, so only the
  module path needs pointing at `TESTING/tinygrad_native.py`.
- Add a low-precision (float32) option to the R API so users can opt into the
  ~3–5× PyTorch path.
- Optional: remove the last host-side step by replacing numpy's SVD of `B`
  (b×n) with a TinyJit'd native Metal Jacobi SVD.
