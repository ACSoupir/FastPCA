#!/usr/bin/env bash
# Reproduce the TESTING benchmark suite.
#
#   ./TESTING/run_benchmarks.sh            # quick sweep
#   DEVICES=cpu,mps ./TESTING/run_benchmarks.sh
#
set -euo pipefail

PY="${FASTPCA_PYTHON:-/opt/anaconda3/envs/FastPCA/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
DEVICES="${DEVICES:-cpu}"
DTYPES="${DTYPES:-64,32}"
SIZES="${SIZES:-4000x1000,6000x1000}"
K="${K:-50}"

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

cd "$REPO"
mkdir -p TESTING/results
echo "== correctness =="
"$PY" TESTING/benchmark_pytorch.py --check

echo "== benchmark: devices=$DEVICES dtypes=$DTYPES sizes=$SIZES =="
"$PY" TESTING/benchmark_pytorch.py \
  --model decay --devices "$DEVICES" --dtypes "$DTYPES" \
  --sizes "$SIZES" --k "$K" --p 10 --q-iter 2 \
  --variants baseline,direct,stable,triangular,lowrank \
  --cores 4 --warmup 1 --repeats 3 \
  --json TESTING/results/bench.json
