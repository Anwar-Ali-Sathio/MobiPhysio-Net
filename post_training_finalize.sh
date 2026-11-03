#!/usr/bin/env bash
# Post-training finalization chain. Waits for the run-all process (PID $1) to exit, then:
# evaluate/compare/verify with the current code -> controlled benchmark on the idle machine ->
# determinism rerun -> paper assets (+ consistency check) -> final validation -> release archive.
# Each step logs to logs/finalize/<step>.log; the chain stops at the first failing step.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
L=logs/finalize
mkdir -p "$L"
PID=${1:-}
if [ -n "$PID" ]; then
  while kill -0 "$PID" 2>/dev/null; do sleep 30; done
fi
step() {
  name=$1; shift
  echo "$(date -u +%FT%TZ) START $name" | tee -a "$L/chain.log"
  if "$@" >"$L/$name.log" 2>&1; then
    echo "$(date -u +%FT%TZ) OK    $name" | tee -a "$L/chain.log"
  else
    echo "$(date -u +%FT%TZ) FAIL  $name (exit $?) — see $L/$name.log" | tee -a "$L/chain.log"
    exit 1
  fi
}
step registry_check python - <<'EOF'
import json, sys
r = json.load(open("experiments/full_v2/registry.json"))["seeds"]
keys = ["svm", "cnn2d", "lstm", "cnn_lstm", "resnet50", "mobiphysio_net", "mobiphysio_net__no_dtw",
        "mobiphysio_net__dtw_no_temporal", "mobiphysio_net__finetuned_vit"]
miss = [(k, s) for k in keys for s in ("1", "2", "42") if s not in r.get(k, {})]
print("missing:", miss)
sys.exit(1 if miss else 0)
EOF
step evaluate python -m mobiphysio.cli evaluate
step compare python -m mobiphysio.cli compare
step verify python -m mobiphysio.cli verify-results
# keep the smoke-test outputs as evidence, out of the reported directory
SM=archive/smoke_tests/efficiency_smoke_2026-10-28T0120Z
if [ -f results/efficiency/SMOKE_TEST_NOT_FOR_REPORTING.md ]; then
  mkdir -p "$SM" && mv results/efficiency/* "$SM"/
fi
step benchmark python scripts/benchmark_efficiency.py --videos 50 --warmup 5 --threads 4 1
step determinism python scripts/determinism_check.py svm:42 mobiphysio_net:42
step paper_assets python scripts/make_paper_assets.py
step validation python scripts/final_validation.py
step release python scripts/make_release.py
echo "$(date -u +%FT%TZ) CHAIN COMPLETE" | tee -a "$L/chain.log"
