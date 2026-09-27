#!/usr/bin/env bash
# Reproduce the native-Metal tinygrad vs in-package torch baseline comparison.
#
#   ./TESTING/run_tinygrad_bench.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/TESTING/.venv-tg/bin/python"
SIZES="${SIZES:-2000x500,4000x1000,8000x3000,15000x2000}"
OUT="$REPO/TESTING/results/tinygrad_bench.txt"

export KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"
mkdir -p TESTING/results

{
  echo "# native-Metal tinygrad vs package torch baseline"
  echo "# interpreter: $PY (has both torch and tinygrad)"
  "$REPO/TESTING/.venv-tg/bin/python" TESTING/_gen_matrices.py
  echo
  echo "== numerical validation =="
  "$PY" -u TESTING/tinygrad_validate.py

  echo
  echo "== speed and peak RSS (fresh process per backend/size) =="
  IFS=',' read -ra ARR <<< "$SIZES"
  for size in "${ARR[@]}"; do
    echo "-- $size --"
    "$PY" -u TESTING/tg_mem_probe.py baseline "$size"
    "$PY" -u TESTING/tg_mem_probe.py tinygrad "$size"
  done
} | tee "$OUT"

echo "wrote $OUT"
