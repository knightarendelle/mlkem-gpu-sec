#!/usr/bin/env bash
# experiments/network_timing/run_network_timing.sh
#
# End-to-end runner for the network timing attack experiment.
#
#   1. Detects GPU SM arch.
#   2. Builds the TCP server (net_server_kyber512) via Makefile.
#   3. Starts the server in the background.
#   4. Waits for the server to write /tmp/mlkem_valid_ct.bin (ready signal).
#   5. Runs the Python client (200,000 measurements).
#   6. Kills the server.
#   7. Runs analyse_network.py and prints the TVLA report.
#
# Usage:
#   bash experiments/network_timing/run_network_timing.sh
# Run from anywhere — paths are resolved relative to this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ATPQC_DIR="$REPO_ROOT/baseline/atpqc-cuda"
LOGS_DIR="$SCRIPT_DIR/logs"
VALID_CT="/tmp/mlkem_valid_ct.bin"

mkdir -p "$LOGS_DIR"

echo "======================================================"
echo "  mlkem-gpu-sec Network Timing Attack Experiment"
echo "======================================================"
echo "  Repo:  $REPO_ROOT"
echo "  Logs:  $LOGS_DIR"
echo ""

# ── Detect GPU and CUDA ────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
    echo "[ERROR] nvcc not found — add CUDA to PATH."
    exit 1
fi
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
              | head -1 | tr -d ' .')
SM="$COMPUTE_CAP"
ARCH_FLAG="-arch=compute_${SM} -code=sm_${SM}"

NVCC_ABS="$(command -v nvcc)"
CUDAPATH="$(cd "$(dirname "$NVCC_ABS")/.." && pwd)"

echo "[detect] GPU:  $GPU_NAME"
echo "[detect] Arch: sm_$SM"
echo "[detect] CUDA: $CUDAPATH"
echo ""

# ── Build server ───────────────────────────────────────────
echo "[1/4] Building net_server (kyber512)..."
(
  cd "$ATPQC_DIR"
  make net_server \
    CUDA_GENCODE_FLAG="$ARCH_FLAG" \
    CUDAPATH="$CUDAPATH" \
    2>&1
)
SERVER_BIN="$ATPQC_DIR/target/net_server_kyber512.out"
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "[ERROR] Server binary not found after build: $SERVER_BIN"
    exit 1
fi
echo "  Built: $SERVER_BIN"
echo ""

# ── Start server in background ─────────────────────────────
echo "[2/4] Starting server..."
rm -f "$VALID_CT"

"$SERVER_BIN" >"$LOGS_DIR/server_stdout.log" 2>"$LOGS_DIR/server_stderr.log" &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"
echo "  Logs: $LOGS_DIR/server_stderr.log"
echo ""

# Trap to kill server on script exit (normal or error)
trap 'echo "  [cleanup] Killing server PID $SERVER_PID"; kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT

# ── Wait for server ready ──────────────────────────────────
echo "  Waiting for server to write $VALID_CT ..."
WAIT_LIMIT=30    # seconds
WAITED=0
while [[ ! -f "$VALID_CT" ]]; do
    sleep 0.5
    WAITED=$((WAITED + 1))
    if [[ $WAITED -ge $((WAIT_LIMIT * 2)) ]]; then
        echo "[ERROR] Server did not write $VALID_CT within ${WAIT_LIMIT}s."
        echo "  Last server log:"
        tail -20 "$LOGS_DIR/server_stderr.log" || true
        exit 1
    fi
    # Check server hasn't crashed
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[ERROR] Server process died before becoming ready."
        echo "  Last server log:"
        tail -20 "$LOGS_DIR/server_stderr.log" || true
        exit 1
    fi
done

# Wait an extra moment for the "Listening on port" message
sleep 0.5
echo "  Server ready (valid_ct.bin present, port 9999 open)"
echo ""

# ── Run client ─────────────────────────────────────────────
echo "[3/4] Running client (200,000 measurements)..."
echo "      This takes approximately 20–60s depending on GPU speed."
echo ""

CSV_OUT="$LOGS_DIR/network_timing.csv"
rm -f "$CSV_OUT"

python3 "$SCRIPT_DIR/client.py" \
    --host 127.0.0.1 \
    --port 9999 \
    --out "$CSV_OUT"

echo ""

# ── Kill server (trap will also run, that's fine) ──────────
echo "[cleanup] Stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT   # disarm trap — already cleaned up
echo "  Server stopped"
echo ""

# ── Verify CSV ─────────────────────────────────────────────
if [[ ! -s "$CSV_OUT" ]]; then
    echo "[ERROR] network_timing.csv is missing or empty."
    exit 1
fi
LINES=$(wc -l < "$CSV_OUT")
echo "  network_timing.csv: $LINES lines"
echo ""

# ── Run analysis ───────────────────────────────────────────
echo "[4/4] Running TVLA analysis..."
python3 "$SCRIPT_DIR/analyse_network.py" \
    "$CSV_OUT" \
    --out "$LOGS_DIR/network_tvla_report.txt"

echo ""
echo "======================================================"
echo "  Done."
echo "  CSV:    $CSV_OUT"
echo "  Report: $LOGS_DIR/network_tvla_report.txt"
echo "======================================================"
