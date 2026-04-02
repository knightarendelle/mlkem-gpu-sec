#!/usr/bin/env bash
# experiments/cotenancy/test_local.sh
#
# Local (no-Docker) smoke test for the co-tenancy experiment.
# Builds victim and attacker directly with nvcc, runs them in parallel,
# then runs the analysis script.
#
# Usage (from any directory):
#   bash experiments/cotenancy/test_local.sh
#
# Requirements:
#   - nvcc in PATH (CUDA toolkit installed)
#   - Python 3 + numpy + scipy
#   - GPU accessible (nvidia-smi works)
#
# The victim and attacker hardcode /logs/ for output.
# This script creates a symlink /logs -> experiments/cotenancy/logs/
# so the CSV files land in the local repo directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"
ATPQC_DIR="$REPO_ROOT/baseline/atpqc-cuda"

mkdir -p "$LOGS_DIR"

echo "======================================================"
echo "  mlkem-gpu-sec Co-tenancy Local Smoke Test"
echo "======================================================"
echo "  Repo:  $REPO_ROOT"
echo "  Logs:  $LOGS_DIR"
echo ""

# ── Detect CUDA and GPU ────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
    echo "[ERROR] nvcc not found — add CUDA to PATH."
    echo "        e.g.  export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
# compute_cap returns e.g. "8.9" → strip dot → "89"
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
              | head -1 | tr -d ' .')
SM="$COMPUTE_CAP"
echo "[detect] GPU:     $GPU_NAME"
echo "[detect] SM arch: sm_$SM"
echo ""

ARCH_FLAG="-arch=compute_${SM} -code=sm_${SM}"

# Derive CUDAPATH from nvcc location (nvcc lives in $CUDAPATH/bin/)
NVCC_ABS="$(command -v nvcc)"
CUDAPATH="$(cd "$(dirname "$NVCC_ABS")/.." && pwd)"
echo "[detect] CUDA:    $CUDAPATH"
echo ""

# ── Set up /logs → local logs dir ─────────────────────────
# Both programs open /logs/victim_log.csv and /logs/attacker_log.csv directly.
# We create a symlink so those writes land in $LOGS_DIR.
echo "[0/4] Setting up /logs symlink..."
if [[ -L /logs ]]; then
    EXISTING_TARGET="$(readlink /logs)"
    if [[ "$EXISTING_TARGET" == "$LOGS_DIR" ]]; then
        echo "  /logs -> $LOGS_DIR  (already set, OK)"
    else
        echo "  /logs -> $EXISTING_TARGET  (points elsewhere — removing and re-creating)"
        sudo rm /logs
        sudo ln -s "$LOGS_DIR" /logs
        echo "  /logs -> $LOGS_DIR  (created)"
    fi
elif [[ -d /logs ]]; then
    echo "  WARNING: /logs is a real directory (not a symlink)."
    echo "           Logs will go to /logs/ — copying to $LOGS_DIR after the run."
    LOGS_DIR=/logs
elif [[ -e /logs ]]; then
    echo "[ERROR] /logs exists but is neither a directory nor a symlink."
    echo "        Remove it and re-run."
    exit 1
else
    ln -s "$LOGS_DIR" /logs 2>/dev/null || sudo ln -s "$LOGS_DIR" /logs
    echo "  /logs -> $LOGS_DIR  (created)"
fi
echo ""

# Clean any stale logs from a previous run
rm -f "$LOGS_DIR/victim_log.csv" "$LOGS_DIR/attacker_log.csv"

# ── Build victim ───────────────────────────────────────────
echo "[1/4] Building victim (kyber512) via Makefile..."
(
  cd "$ATPQC_DIR"
  # Override arch and CUDAPATH in case they differ from Makefile defaults
  make victim_loop_kyber512 \
    CUDA_GENCODE_FLAG="$ARCH_FLAG" \
    CUDAPATH="$CUDAPATH" \
    2>&1
)
VICTIM_BIN="$ATPQC_DIR/target/victim_loop_kyber512.out"
if [[ ! -x "$VICTIM_BIN" ]]; then
    echo "[ERROR] Victim binary not found after build: $VICTIM_BIN"
    exit 1
fi
echo "  Built: $VICTIM_BIN"
echo ""

# ── Build attacker ─────────────────────────────────────────
ATTACKER_SRC="$SCRIPT_DIR/attacker_probe.cu"
ATTACKER_BIN="$SCRIPT_DIR/attacker_probe"

echo "[2/4] Building attacker (standalone)..."
nvcc -O3 $ARCH_FLAG \
    -std=c++17 \
    "$ATTACKER_SRC" \
    -o "$ATTACKER_BIN"
if [[ ! -x "$ATTACKER_BIN" ]]; then
    echo "[ERROR] Attacker binary not found after build: $ATTACKER_BIN"
    exit 1
fi
echo "  Built: $ATTACKER_BIN"
echo ""

# ── Run victim and attacker ────────────────────────────────
# Timing layout (mirrors Docker run_cotenancy.sh):
#   t=0s  victim starts  (runs 30s → finishes at t=30s)
#   t=2s  attacker starts (runs 35s → finishes at t=37s)
#
# After attacker exits (foreground), victim has already finished;
# the 'wait' below returns immediately.

echo "[3/4] Running victim (30s) and attacker (35s)..."
echo "      (victim starts now; attacker starts in 2s)"
echo ""

"$VICTIM_BIN" &
VICTIM_PID=$!
echo "  [victim]   PID $VICTIM_PID  — running..."

sleep 2

echo "  [attacker] starting..."
"$ATTACKER_BIN"
echo "  [attacker] done"

echo "  Waiting for victim..."
wait "$VICTIM_PID" || true
echo "  [victim]   done"
echo ""

# ── Verify log files ───────────────────────────────────────
VICTIM_CSV="$LOGS_DIR/victim_log.csv"
ATTACKER_CSV="$LOGS_DIR/attacker_log.csv"

FAIL=0
if [[ ! -s "$VICTIM_CSV" ]]; then
    echo "[ERROR] victim_log.csv is missing or empty: $VICTIM_CSV"
    FAIL=1
fi
if [[ ! -s "$ATTACKER_CSV" ]]; then
    echo "[ERROR] attacker_log.csv is missing or empty: $ATTACKER_CSV"
    FAIL=1
fi
if [[ $FAIL -eq 1 ]]; then
    echo ""
    echo "  Files in $LOGS_DIR:"
    ls -lh "$LOGS_DIR/" 2>/dev/null || echo "  (empty)"
    exit 1
fi

V_LINES=$(wc -l < "$VICTIM_CSV")
A_LINES=$(wc -l < "$ATTACKER_CSV")
echo "  victim_log.csv:   $V_LINES lines  (expect ~500–2000 depending on GPU)"
echo "  attacker_log.csv: $A_LINES lines"
echo ""

# Quick CSV sanity check
HEAD_V=$(head -2 "$VICTIM_CSV")
HEAD_A=$(head -2 "$ATTACKER_CSV")
echo "  victim header+sample:   $HEAD_V"
echo "  attacker header+sample: $HEAD_A"
echo ""

# ── Run analysis ───────────────────────────────────────────
echo "[4/4] Running co-tenancy analysis..."

# On a local machine both processes share the GPU without Docker-level
# context serialization, so wall_us signal may be weak.  The script still
# exercises the full pipeline and produces a t-statistic.
python3 "$SCRIPT_DIR/analyse_cotenancy.py" \
    "$VICTIM_CSV" \
    "$ATTACKER_CSV" \
    --out "$LOGS_DIR/cotenancy_tvla_report.txt"

echo ""
echo "======================================================"
echo "  Done."
echo "  victim_log.csv:          $VICTIM_CSV"
echo "  attacker_log.csv:        $ATTACKER_CSV"
echo "  TVLA report:             $LOGS_DIR/cotenancy_tvla_report.txt"
echo ""
echo "  NOTE: Without Docker process isolation the attacker's"
echo "  wall_us timing may not show a strong t-statistic here."
echo "  The goal of this run is to confirm both programs compile"
echo "  and produce well-formed CSV output — not to measure leakage."
echo "======================================================"
