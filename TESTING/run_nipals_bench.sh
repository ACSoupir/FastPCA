#!/usr/bin/env bash
# NIPALS benchmark: pytorch vs R implementations on the FastPCA example data.
#
#   ./TESTING/run_nipals_bench.sh              # tier m (default): head-to-head
#   TIER=s ./TESTING/run_nipals_bench.sh       # tier s: correctness
#   TIER=f NCOMP=10 ./TESTING/run_nipals_bench.sh
#
# One implementation per process, each wrapped in /usr/bin/time -l so peak RSS is
# measured identically for python and R. Results land in TESTING/results/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${FASTPCA_PYTHON:-/opt/anaconda3/envs/FastPCA/bin/python}"
RSCRIPT="${FASTPCA_R:-/usr/local/bin/Rscript}"

TIER="${TIER:-m}"
NCOMP="${NCOMP:-10}"
NA_RATES="${NA_RATES:-0,5}"
MAX_GB="${MAX_GB:-12}"

export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
cd "$REPO"
mkdir -p TESTING/results TESTING/_tmp

case "$TIER" in
  s) FEATS="2000";;   m) FEATS="20000";;
  f) FEATS="0";;      *) echo "unknown TIER=$TIER" >&2; exit 1;;
esac

OUT="TESTING/results/nipals_bench_${TIER}.txt"
JSONL="$(mktemp)"; RSSF="$(mktemp)"
: > "$OUT"

run_one() {  # run_one <label> <command...>
  local label="$1"; shift
  echo "--- $label ---" | tee -a "$OUT"
  if /usr/bin/time -l "$@" > TESTING/_tmp/run_out.txt 2> TESTING/_tmp/run_err.txt; then
    grep 'RESULT_JSON' TESTING/_tmp/run_out.txt | sed 's/RESULT_JSON //' >> "$JSONL" || true
    grep -E '^\[|VALIDATE_JSON' TESTING/_tmp/run_out.txt | tee -a "$OUT" || true
  else
    echo "FAILED: $label" | tee -a "$OUT"
    grep 'RESULT_JSON' TESTING/_tmp/run_out.txt | sed 's/RESULT_JSON //' >> "$JSONL" || true
    tail -3 TESTING/_tmp/run_err.txt | tee -a "$OUT" || true
  fi
  local rss; rss="$(awk '/maximum resident set size/{print $1}' TESTING/_tmp/run_err.txt || echo 0)"
  printf '%s\t%s\n' "$label" "${rss:-0}" >> "$RSSF"
}

for NA in $(echo "$NA_RATES" | tr ',' ' '); do
  echo "======== tier=$TIER features=$FEATS na=${NA}% ncomp=$NCOMP ========" | tee -a "$OUT"
  TAG="$("$RSCRIPT" --vanilla TESTING/nipals_export.R --F="$FEATS" --na-pct="$NA" 2>&1 | tee -a "$OUT" | grep '^TAG=' | cut -d= -f2)"
  M="$(grep -o '"m":[0-9]*' "TESTING/_tmp/X_${TAG}.json" | cut -d: -f2)"
  N="$(grep -o '"n":[0-9]*' "TESTING/_tmp/X_${TAG}.json" | cut -d: -f2)"
  X="TESTING/_tmp/X_${TAG}.f64"; MK=()
  if [ "$NA" -gt 0 ]; then MK=(--mask="TESTING/_tmp/M_${TAG}.u8"); fi

  run_one "torch-${TAG}" "$PY" TESTING/nipals_torch.py --x="$X" ${MK[@]+"${MK[@]}"} \
    --m="$M" --n="$N" --tag="${TAG}" --ncomp="$NCOMP" \
    --save-npz="TESTING/_tmp/torch_${TAG}.npz"
  run_one "base-${TAG}" "$RSCRIPT" --vanilla TESTING/nipals_r.R --impl=base \
    --x="$X" ${MK[@]+"${MK[@]}"} --m="$M" --n="$N" --tag="${TAG}" --ncomp="$NCOMP" \
    --save-bin="TESTING/_tmp/base_${TAG}"
  run_one "nipals_pkg-${TAG}" "$RSCRIPT" --vanilla TESTING/nipals_r.R --impl=nipals_pkg \
    --x="$X" ${MK[@]+"${MK[@]}"} --m="$M" --n="$N" --tag="${TAG}" --ncomp="$NCOMP" \
    --max-gb="$MAX_GB" --save-bin="TESTING/_tmp/pkg_${TAG}"

  if [ "$NA" -eq 0 ]; then
    run_one "irlba-${TAG}" "$RSCRIPT" --vanilla TESTING/nipals_r.R --impl=irlba \
      --x="$X" ${MK[@]+"${MK[@]}"} --m="$M" --n="$N" --tag="${TAG}" --ncomp="$NCOMP"
    "$PY" TESTING/nipals_validate.py \
      --ref="TESTING/_tmp/torch_${TAG}.npz" --test-bin="TESTING/_tmp/base_${TAG}" \
      --m="$M" --n="$N" 2>&1 | tee -a "$OUT" || true
    if [ -f "TESTING/_tmp/pkg_${TAG}.json" ]; then
      "$PY" TESTING/nipals_validate.py \
        --ref="TESTING/_tmp/torch_${TAG}.npz" --test-bin="TESTING/_tmp/pkg_${TAG}" \
        --m="$M" --n="$N" 2>&1 | tee -a "$OUT" || true
    fi
  fi

  # reclaim disk between tiers (15 GB free on this machine)
  if [ "$TIER" = "f" ]; then rm -f TESTING/_tmp/X_${TAG}.f64; fi
done

echo "== collected results ==" | tee -a "$OUT"
"$PY" TESTING/nipals_collect.py --jsonl "$JSONL" --rss "$RSSF" \
  --out "TESTING/results/nipals_bench_${TIER}.json" | tee -a "$OUT"
rm -f "$JSONL" "$RSSF"
