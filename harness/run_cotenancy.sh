#!/bin/bash
# harness/run_cotenancy.sh
# Co-tenancy experiment orchestration — run from repo root.
#
# Two attack modes depending on whether NVIDIA MPS is active:
#
#   Without MPS (default): GPU context lock is held per-process.
#     The attacker is blocked at cudaEventSynchronize while the victim
#     runs. Wall-clock timing (wall_us column) captures the contention.
#     GPU-event timing (gpu_us) is unaffected — it only starts once the
#     attacker acquires the GPU.
#
#   With MPS: kernels from both processes truly overlap on the GPU.
#     The victim's DRAM traffic slows the attacker's kernel directly.
#     GPU-event timing becomes the meaningful column.
#     Enable:  nvidia-cuda-mps-control -d
#     Disable: echo quit | nvidia-cuda-mps-control
#
# Usage:
#   bash harness/run_cotenancy.sh [--mps]

set -euo pipefail

BINARY="./baseline/atpqc-cuda/target/trace_kyber512.out"
ATTACKER="./harness/cotenancy_attacker"
OUTDIR="experiments/traces"
NTRACES=10000

if [ ! -x "$BINARY" ]; then
    echo "ERROR: victim binary not found: $BINARY" >&2
    echo "  Build with: make -C baseline/atpqc-cuda trace_kyber512" >&2
    exit 1
fi
if [ ! -x "$ATTACKER" ]; then
    echo "ERROR: attacker binary not found: $ATTACKER" >&2
    echo "  Build with: make -C harness cotenancy_attacker" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

echo "=== Phase 5: Co-tenancy experiment ==="
echo "    victim:   $BINARY"
echo "    attacker: $ATTACKER"
echo "    ntraces:  $NTRACES"
echo ""

run_trial() {
    local class=$1
    local label=$2
    local outfile="$OUTDIR/cotenancy_attacker_class${class}.csv"

    echo "--- Class $class ($label) ---"

    # Launch victim in background with effectively infinite trace count.
    # Stdout (CSV) discarded; stderr (progress %) visible for diagnostics.
    echo "1 4 4 4 4 999999999 $class" | "$BINARY" > /dev/null &
    VICTIM_PID=$!

    # Wait for victim to warm up (GPU frequencies settle, caches filled).
    echo "  Waiting 3s for victim warmup..."
    sleep 3

    # Verify victim is still alive before starting attacker.
    if ! kill -0 "$VICTIM_PID" 2>/dev/null; then
        echo "ERROR: victim process died during warmup" >&2
        exit 1
    fi

    echo "  Starting attacker ($NTRACES traces) -> $outfile"
    "$ATTACKER" "$NTRACES" "$outfile"

    # Terminate victim cleanly.
    kill "$VICTIM_PID" 2>/dev/null || true
    wait "$VICTIM_PID" 2>/dev/null || true
    echo "  Victim terminated."
}

run_trial 0 "valid ciphertexts"

echo ""
echo "Cooling down (5s)..."
sleep 5
echo ""

run_trial 1 "invalid ciphertexts"

echo ""
echo "=== Running TVLA analysis (gpu_us column) ==="
python3 harness/tvla_analysis.py \
    --col gpu_us \
    "$OUTDIR/cotenancy_attacker_class0.csv" \
    "$OUTDIR/cotenancy_attacker_class1.csv" || \
python3 harness/tvla_analysis.py \
    "$OUTDIR/cotenancy_attacker_class0.csv" \
    "$OUTDIR/cotenancy_attacker_class1.csv"

echo ""
echo "=== Running TVLA analysis (wall_us column) ==="
python3 harness/tvla_analysis.py \
    --col wall_us \
    "$OUTDIR/cotenancy_attacker_class0.csv" \
    "$OUTDIR/cotenancy_attacker_class1.csv" 2>/dev/null || \
echo "  (tvla_analysis.py does not support --col; analyse wall_us manually)"

echo ""
echo "Done. Results in $OUTDIR/cotenancy_attacker_class{0,1}.csv"
