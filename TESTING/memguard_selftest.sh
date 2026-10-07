#!/usr/bin/env bash
# Self-test for the hard RSS cap that wraps every benchmark run.
#
#   ./TESTING/memguard_selftest.sh
#
# Case 1 allocates ~500 MB under a 2 GB cap -> must survive (exit 0).
# Case 2 allocates ~1.5 GB under a 0.25 GB cap -> must be SIGKILLed (exit 137).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
set +u; PY="$FASTPCA_PYTHON"; set -u
[ -n "$PY" ] || PY="/opt/anaconda3/envs/FastPCA/bin/python"

echo "== case 1: ~500 MB under a 2 GB cap (expect killed=false, exit 0) =="
"$PY" "$HERE/memguard.py" --cap-gb 2 --interval-ms 30 --label selftest-under -- \
  "$PY" -c "import time
buf = bytearray(500*1024*1024)
for i in range(0, len(buf), 4096): buf[i] = 1
time.sleep(0.3)
print('child survived')"
echo "case 1 exit=$?"

echo
echo "== case 2: ~1.5 GB under a 0.25 GB cap (expect killed=true, exit 137) =="
"$PY" "$HERE/memguard.py" --cap-gb 0.25 --interval-ms 20 --label selftest-over -- \
  "$PY" -c "import time
buf = bytearray(1500*1024*1024)
for i in range(0, len(buf), 4096): buf[i] = 1
print('child should not reach here')
time.sleep(2)"
rc=$?
echo "case 2 exit=$rc"
if [ "$rc" -eq 137 ]; then
  echo "OK: hard cap enforced (SIGKILL at the ceiling)"
else
  echo "FAIL: expected exit 137"; exit 1
fi

echo
echo "== case 3: pre-flight refusal, no allocation (expect status=skipped) =="
X="$HERE/_tmp/X_F2000_na00.f64"
if [ -f "$X" ]; then
  "$PY" "$HERE/irlba_torch.py" --x="$X" --m=978 --n=2000 --nv=10 \
    --max-gb=0.001 2>/dev/null | grep RESULT_JSON
else
  echo "(skipped: run TIER=s once to create X_F2000_na00.f64)"
fi
