#!/usr/bin/env bash
# IRLBA (iterative Lanczos) benchmark: PyTorch port vs irlba::irlba vs
# FastPCA-pytorch.
#
#   TIER=f CORES=1,2,4 ./TESTING/run_irlba_bench.sh
#   TIER=s CORES=1 ./TESTING/run_irlba_bench.sh
#
# Every run is one implementation in its own process, wrapped first by the hard
# RSS cap (TESTING/memguard.py) and then by /usr/bin/time -l for an independent
# peak-RSS reading. Results land in TESTING/results/irlba_bench_<tier>.{txt,json}.
#
# Knobs: TIER=s|m|f  NCOMP=10  WORK=k+7  CORES=1,2,4  MAX_GB=20
#        REPEATS=3   IMPLS=comma,list  SEED=42
#        PRECISION=double|single   DEVICE=CPU|GPU|MPS   (torch_irlba only)
#        DTYPE=fp64|fp32 still works as an alias of PRECISION. DEVICE=GPU means
#        CUDA when available and otherwise the CPU with a message -- exactly
#        FastPCA's contract (R/validate.R .fastpca_devices). DEVICE=MPS is a
#        benchmark-only extra for Apple GPUs (single precision only; an fp64 +
#        MPS request is served on the CPU and reported).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
# optional knobs may be unset: read them, then re-enable nounset
set +u
PY="$FASTPCA_PYTHON"; RSCRIPT="$FASTPCA_R"
TIER="$TIER"; NCOMP="$NCOMP"; WORK="$WORK"; CORES_LIST="$CORES"
CAP_GB="$MAX_GB"; REPEATS="$REPEATS"; IMPLS="$IMPLS"; SEED="$SEED"
DTYPE_KNB="$DTYPE"; DEVICE_KNB="$DEVICE"; PRECISION_KNB="${PRECISION:-}"
set -u
[ -n "$PY" ] || PY="/opt/anaconda3/envs/FastPCA/bin/python"
[ -n "$RSCRIPT" ] || RSCRIPT="/usr/local/bin/Rscript"
[ -n "$TIER" ] || TIER="f"
[ -n "$NCOMP" ] || NCOMP=10
[ -n "$CORES_LIST" ] || CORES_LIST="1,2,4"
[ -n "$CAP_GB" ] || CAP_GB=20
[ -n "$REPEATS" ] || REPEATS=3
[ -n "$IMPLS" ] || IMPLS="torch_irlba,fastpca_kernel,fastpca_torch,irlba_fast,irlba_fallback"
[ -n "$SEED" ] || SEED=42
[ -n "$WORK" ] || WORK=$((NCOMP + 7))
# PRECISION=double|single is preferred; DTYPE=fp64|fp32 still works.
# Normalise here so result tags stay stable across spellings.
DTYPE="${PRECISION_KNB:-${DTYPE_KNB:-double}}"
case "$DTYPE" in double) DTYPE=fp64;; single) DTYPE=fp32;; esac
DEVICE="${DEVICE_KNB:-cpu}"

export KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p "$HERE/results" "$HERE/.cache/numba"
export NUMBA_CACHE_DIR="$HERE/.cache/numba"

case "$TIER" in
  s) TAGF="F2000_na00"; FEATS=2000;;
  m) TAGF="F20000_na00"; FEATS=20000;;
  f) TAGF="F199112_na00"; FEATS=0;;
  *) echo "unknown TIER=$TIER" >&2; exit 1;;
esac

X="$HERE/_tmp/X_$TAGF.f64"
if [ ! -f "$X" ]; then
  echo "generating $TAGF (features=$FEATS) ..."
  "$RSCRIPT" --vanilla "$HERE/nipals_export.R" --F="$FEATS" --na-pct=0 \
    | tee "$HERE/_tmp/export_$TIER.log"
fi
M="$(grep -o '"m":[0-9]*' "$HERE/_tmp/X_$TAGF.json" | cut -d: -f2)"
N="$(grep -o '"n":[0-9]*' "$HERE/_tmp/X_$TAGF.json" | cut -d: -f2)"

OUT="$HERE/results/irlba_bench_$TIER.txt"
JSONL="$(mktemp)"; RSSF="$(mktemp)"; MGF="$(mktemp)"
: > "$OUT"

has() { case ",$IMPLS," in *",$1,"*) return 0;; esac; return 1; }

run_one() { # run_one <label> <cmd...>
  local label="$1"; shift
  echo "--- $label ---" | tee -a "$OUT"
  local rc=0
  /usr/bin/time -l "$PY" "$HERE/memguard.py" --cap-gb "$CAP_GB" \
      --interval-ms 100 --label "$label" -- "$@" \
      > "$HERE/_tmp/run_out.txt" 2> "$HERE/_tmp/run_err.txt" || rc=$?
  if [ "$rc" -eq 0 ]; then
    grep 'RESULT_JSON' "$HERE/_tmp/run_out.txt" >> "$JSONL" || true
  else
    echo "FAILED/KILLED: $label (exit $rc)" | tee -a "$OUT"
    grep 'RESULT_JSON' "$HERE/_tmp/run_out.txt" >> "$JSONL" || true
    tail -3 "$HERE/_tmp/run_err.txt" | tee -a "$OUT" || true
  fi
  grep 'MEMGUARD_JSON' "$HERE/_tmp/run_out.txt" >> "$MGF" || true
  grep -E '^\[|best=|KILLED' "$HERE/_tmp/run_out.txt" | tee -a "$OUT" || true
  local rss; rss="$(awk '/maximum resident set size/{print $1}' "$HERE/_tmp/run_err.txt" | tail -1)"
  [ -n "$rss" ] || rss=0
  printf '%s\t%s\n' "$label" "$rss" >> "$RSSF"
}

echo "======== tier=$TIER  $M x $N  ncomp=$NCOMP work=$WORK cap=$CAP_GB GB ========" | tee -a "$OUT"

TC1="$TAGF"_"$DTYPE"-"$DEVICE"_c1
for C in $(echo "$CORES_LIST" | tr ',' ' '); do
  T="$TAGF"_c"$C"
  if has torch_irlba; then
    TT="$TAGF"_"$DTYPE"-"$DEVICE"_c"$C"
    run_one "torch_irlba-$TT" "$PY" "$HERE/irlba_torch.py" \
      --x="$X" --m="$M" --n="$N" --tag="$TT" --nv="$NCOMP" --work="$WORK" \
      --threads="$C" --repeats="$REPEATS" --seed="$SEED" \
      --dtype="$DTYPE" --device="$DEVICE" \
      --save-bin="$HERE/_tmp/torch_irlba_$TT"
  fi
  if has fastpca_kernel; then
    run_one "fastpca_kernel-$T" "$PY" "$HERE/fastpca_kernel_bench.py" \
      --x="$X" --m="$M" --n="$N" --tag="$T" --k="$NCOMP" --p=10 --q_iter=2 \
      --threads="$C" --repeats="$REPEATS" \
      --kernel="$HERE/../inst/python/torch_svd_func.py" \
      --save-bin="$HERE/_tmp/fastpca_kernel_$T"
  fi
  if has fastpca_torch; then
    run_one "fastpca_torch-$T" "$RSCRIPT" --vanilla "$HERE/fastpca_torch_bench.R" \
      --x="$X" --m="$M" --n="$N" --tag="$T" --k="$NCOMP" --p=10 --q_iter=2 \
      --cores="$C" --repeats="$REPEATS" \
      --save-bin="$HERE/_tmp/fastpca_torch_$T"
  fi
done

if has irlba_fast; then
  run_one "irlba_fast-$TAGF" "$RSCRIPT" --vanilla "$HERE/irlba_r.R" \
    --impl=fast --x="$X" --m="$M" --n="$N" --tag="$TAGF" \
    --nv="$NCOMP" --work="$WORK" --seed="$SEED" \
    --save-bin="$HERE/_tmp/irlba_fast_$TAGF"
fi
if has irlba_fallback; then
  run_one "irlba_fallback-$TAGF" "$RSCRIPT" --vanilla "$HERE/irlba_r.R" \
    --impl=fallback --x="$X" --m="$M" --n="$N" --tag="$TAGF" \
    --nv="$NCOMP" --work="$WORK" --seed="$SEED" \
    --v0-out="$HERE/_tmp/v0_$TAGF.f64" \
    --save-bin="$HERE/_tmp/irlba_fallback_$TAGF"
fi

echo "== validation (singular values + principal angles) ==" | tee -a "$OUT"
VAL() { # VAL <ref-prefix> <test-prefix>
  "$PY" "$HERE/nipals_validate.py" --ref-bin="$1" --test-bin="$2" \
    --m="$M" --n="$N" 2>&1 | tee -a "$OUT" || true
}
if has torch_irlba; then VAL "$HERE/_tmp/irlba_fallback_$TAGF" "$HERE/_tmp/torch_irlba_${TC1}"; fi
if has torch_irlba; then VAL "$HERE/_tmp/irlba_fast_$TAGF" "$HERE/_tmp/torch_irlba_${TC1}"; fi
if has fastpca_kernel; then VAL "$HERE/_tmp/irlba_fast_$TAGF" "$HERE/_tmp/fastpca_kernel_"$TAGF"_c1"; fi
if has fastpca_torch; then VAL "$HERE/_tmp/irlba_fast_$TAGF" "$HERE/_tmp/fastpca_torch_"$TAGF"_c1"; fi

echo "== collected results ==" | tee -a "$OUT"
"$PY" "$HERE/irlba_collect.py" --jsonl="$JSONL" --rss="$RSSF" \
  --memguard="$MGF" --out="$HERE/results/irlba_bench_$TIER.json" | tee -a "$OUT"
rm -f "$JSONL" "$RSSF" "$MGF"
