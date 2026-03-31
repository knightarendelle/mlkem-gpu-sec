#!/usr/bin/env bash
# experiments/cotenancy/run_cotenancy.sh
#
# Co-tenancy side-channel experiment.
#
# Launches the ML-KEM victim and the DRAM-bandwidth attacker in
# separate Docker containers — each gets its own CUDA context with
# no MPS, no shared memory, and no IPC between them.
# Both mount a shared /logs volume for CSV output.
# After both finish, runs the post-hoc analysis script.
#
# Usage:
#   bash experiments/cotenancy/run_cotenancy.sh
#
# Requirements:
#   - Docker with NVIDIA Container Toolkit (nvidia-docker2)
#   - Python 3 + numpy + scipy (for analysis step)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOGS_DIR"

echo "======================================================"
echo "  mlkem-gpu-sec Co-tenancy Experiment"
echo "======================================================"
echo ""
echo "  Repo:    $REPO_ROOT"
echo "  Logs:    $LOGS_DIR"
echo ""

# ── Build Docker images ───────────────────────────────────
echo "[1/4] Building victim Docker image (mlkem-cotenancy-victim)..."
docker build \
    -f "$SCRIPT_DIR/Dockerfile.victim" \
    -t mlkem-cotenancy-victim \
    "$REPO_ROOT"

echo "[1/4] Building attacker Docker image (mlkem-cotenancy-attacker)..."
docker build \
    -f "$SCRIPT_DIR/Dockerfile.attacker" \
    -t mlkem-cotenancy-attacker \
    "$REPO_ROOT"

# Clean up any leftover containers from a previous run
docker rm -f mlkem_cotenancy_victim  2>/dev/null || true
docker rm -f mlkem_cotenancy_attacker 2>/dev/null || true

# ── Launch victim ─────────────────────────────────────────
echo ""
echo "[2/4] Starting victim container (runs for 30s)..."
docker run \
    --detach \
    --gpus all \
    --rm \
    --name mlkem_cotenancy_victim \
    --volume "$LOGS_DIR:/logs" \
    mlkem-cotenancy-victim

# Give the victim 2 seconds to initialize its CUDA context and
# begin the warmup phase before the attacker starts probing.
echo "       Waiting 2s for victim to initialize..."
sleep 2

# ── Launch attacker ───────────────────────────────────────
echo "[3/4] Starting attacker container (runs for 35s)..."
docker run \
    --detach \
    --gpus all \
    --rm \
    --name mlkem_cotenancy_attacker \
    --volume "$LOGS_DIR:/logs" \
    mlkem-cotenancy-attacker

# ── Wait for completion ───────────────────────────────────
echo ""
echo "       Waiting for victim  to finish (~30s from victim start)..."
docker wait mlkem_cotenancy_victim  2>/dev/null || true

echo "       Waiting for attacker to finish (~35s from attacker start)..."
docker wait mlkem_cotenancy_attacker 2>/dev/null || true

# ── Verify log files ──────────────────────────────────────
if [[ ! -s "$LOGS_DIR/victim_log.csv" ]]; then
    echo "[ERROR] victim_log.csv is missing or empty — victim may have crashed."
    echo "        Check: docker logs mlkem_cotenancy_victim"
    exit 1
fi
if [[ ! -s "$LOGS_DIR/attacker_log.csv" ]]; then
    echo "[ERROR] attacker_log.csv is missing or empty — attacker may have crashed."
    echo "        Check: docker logs mlkem_cotenancy_attacker"
    exit 1
fi

echo ""
echo "  victim_log.csv:   $(wc -l < "$LOGS_DIR/victim_log.csv") lines"
echo "  attacker_log.csv: $(wc -l < "$LOGS_DIR/attacker_log.csv") lines"

# ── Run analysis ──────────────────────────────────────────
echo ""
echo "[4/4] Running co-tenancy analysis..."
python3 "$SCRIPT_DIR/analyse_cotenancy.py" \
    "$LOGS_DIR/victim_log.csv" \
    "$LOGS_DIR/attacker_log.csv" \
    --out "$LOGS_DIR/cotenancy_tvla_report.txt"

echo ""
echo "======================================================"
echo "  Done. Report: $LOGS_DIR/cotenancy_tvla_report.txt"
echo "======================================================"
