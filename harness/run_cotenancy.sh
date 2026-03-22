#!/bin/bash
# harness/run_cotenancy.sh
# Co-tenancy experiment orchestration — run from repo root.
#
# Two attack modes depending on whether NVIDIA MPS is active:
#
#   Without MPS (default): GPU context lock is held per-process.
#     The attacker is blocked at cudaEventSynchronize while the victim
#     runs. Wall-clock timing (wall_*.csv) captures the contention.
#     GPU-event timing (gpu_*.csv) is unaffected — it only starts once
#     the attacker acquires the GPU.
#
#   With MPS: kernels from both processes truly overlap on the GPU.
#     The victim's DRAM traffic slows the attacker's kernel directly.
#     GPU-event timing (gpu_*.csv) becomes the meaningful measurement.
#     Enable:  nvidia-cuda-mps-control -d
#     Disable: echo quit | nvidia-cuda-mps-control
#
# Usage:
#   bash harness/run_cotenancy.sh

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
    local gpu_out="$OUTDIR/cotenancy_gpu_class${class}.csv"
    local wall_out="$OUTDIR/cotenancy_wall_class${class}.csv"

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

    echo "  Starting attacker ($NTRACES traces)..."
    echo "    gpu  -> $gpu_out"
    echo "    wall -> $wall_out"
    "$ATTACKER" "$NTRACES" "$gpu_out" "$wall_out"

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
echo "=== TVLA: GPU-event timing (meaningful under MPS) ==="
python3 harness/tvla_analysis.py \
    "$OUTDIR/cotenancy_gpu_class0.csv" \
    "$OUTDIR/cotenancy_gpu_class1.csv"

echo ""
echo "=== TVLA: Wall-clock timing (captures contention without MPS) ==="
python3 harness/tvla_analysis.py \
    "$OUTDIR/cotenancy_wall_class0.csv" \
    "$OUTDIR/cotenancy_wall_class1.csv"

echo ""
echo "Done."
echo "  GPU  traces: $OUTDIR/cotenancy_gpu_class{0,1}.csv"
echo "  Wall traces: $OUTDIR/cotenancy_wall_class{0,1}.csv"
