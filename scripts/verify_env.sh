#!/bin/bash
# =============================================================
# verify_env.sh
# Run this first after connecting to a new RunPod instance.
# It checks that everything needed for the project is in place.
# =============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

check() {
  local label=$1
  local cmd=$2
  local expected=$3

  if eval "$cmd" &>/dev/null; then
    echo -e "${GREEN}[PASS]${NC} $label"
    PASS=$((PASS+1))
  else
    echo -e "${RED}[FAIL]${NC} $label"
    if [ -n "$expected" ]; then
      echo -e "       ${YELLOW}Fix: $expected${NC}"
    fi
    FAIL=$((FAIL+1))
  fi
}

echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  gpu-mlkem-security: Environment Verification  ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# ── OS Check ──────────────────────────────────────────────
echo -e "${BLUE}[ OS ]${NC}"
OS=$(uname -s)
if [ "$OS" = "Linux" ]; then
  echo -e "${GREEN}[PASS]${NC} Linux detected: $(uname -r)"
  PASS=$((PASS+1))
else
  echo -e "${RED}[FAIL]${NC} Not Linux — detected: $OS"
  echo -e "       ${YELLOW}Fix: Use Ubuntu 22.04 on RunPod. Do not run on Windows/WSL2.${NC}"
  FAIL=$((FAIL+1))
fi

# Check it's not WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
echo -e "${YELLOW}[WARN]${NC} WSL2 detected — TVLA timing traces must run on RunPod, not here"
  PASS=$((PASS+1))
else
  echo -e "${GREEN}[PASS]${NC} Not WSL2 — bare-metal Linux confirmed"
  PASS=$((PASS+1))
fi
echo ""

# ── GPU Check ─────────────────────────────────────────────
echo -e "${BLUE}[ GPU ]${NC}"
check "nvidia-smi available" "command -v nvidia-smi" "Install NVIDIA drivers"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
  echo -e "       GPU: ${GPU_NAME}"
  echo -e "       Driver: ${DRIVER}"
  echo -e "       VRAM: ${VRAM}"

  # Warn if GTX 16xx
  if echo "$GPU_NAME" | grep -qi "GTX 16"; then
    echo -e "${YELLOW}[WARN]${NC} GTX 16xx detected — use RTX 30xx or 40xx for final experiments"
  fi
fi
echo ""

# ── CUDA Check ────────────────────────────────────────────
echo -e "${BLUE}[ CUDA ]${NC}"
check "nvcc available" "command -v nvcc" "Install CUDA Toolkit 12.x"
if command -v nvcc &>/dev/null; then
  CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
  echo -e "       CUDA version: ${CUDA_VER}"
  MAJOR=$(echo $CUDA_VER | cut -d'.' -f1 | tr -d 'V')
  if [ "$MAJOR" -ge 12 ] 2>/dev/null; then
    echo -e "${GREEN}[PASS]${NC} CUDA 12.x confirmed"
    PASS=$((PASS+1))
  else
    echo -e "${YELLOW}[WARN]${NC} CUDA version < 12 — recommend upgrading to CUDA 12.x"
  fi
fi
echo ""

# ── Nsight Compute Check ──────────────────────────────────
echo -e "${BLUE}[ Nsight Compute ]${NC}"
check "ncu (Nsight Compute) available" "command -v ncu" "sudo apt-get install -y nsight-compute OR download from developer.nvidia.com/nsight-compute"
if command -v ncu &>/dev/null; then
  NCU_VER=$(ncu --version 2>/dev/null | head -1)
  echo -e "       Version: ${NCU_VER}"
fi
echo ""

# ── Build Tools ───────────────────────────────────────────
echo -e "${BLUE}[ Build Tools ]${NC}"
check "git" "command -v git" "sudo apt-get install -y git"
check "cmake (3.20+)" "cmake --version | grep -E '3\.[2-9][0-9]'" "sudo apt-get install -y cmake"
check "make" "command -v make" "sudo apt-get install -y build-essential"
check "gcc" "command -v gcc" "sudo apt-get install -y build-essential"
check "g++" "command -v g++" "sudo apt-get install -y build-essential"
check "cuobjdump" "command -v cuobjdump" "Should be bundled with CUDA Toolkit"
echo ""

# ── Python Check ──────────────────────────────────────────
echo -e "${BLUE}[ Python & Analysis Libraries ]${NC}"
check "python3 (3.10+)" "python3 -c 'import sys; assert sys.version_info >= (3,10)'" "sudo apt-get install -y python3.10"
check "numpy" "python3 -c 'import numpy'" "pip install numpy"
check "scipy" "python3 -c 'import scipy'" "pip install scipy"
check "matplotlib" "python3 -c 'import matplotlib'" "pip install matplotlib"
check "pandas" "python3 -c 'import pandas'" "pip install pandas"
check "sklearn (scikit-learn)" "python3 -c 'import sklearn'" "pip install scikit-learn"
echo ""

# ── Summary ───────────────────────────────────────────────
echo -e "${BLUE}=================================================${NC}"
echo -e "  Results: ${GREEN}${PASS} passed${NC}  |  ${RED}${FAIL} failed${NC}"
echo -e "${BLUE}=================================================${NC}"

if [ $FAIL -eq 0 ]; then
  echo -e "${GREEN}  Environment is ready. Proceed to baseline setup.${NC}"
  echo -e "  Run: cd baseline && bash setup.sh"
else
  echo -e "${RED}  Fix the failed checks above before proceeding.${NC}"
  echo -e "  Re-run this script after fixing: bash scripts/verify_env.sh"
fi
echo ""

# ── Log environment info ──────────────────────────────────
LOG_FILE="experiments/results/env_$(date +%Y%m%d_%H%M%S).log"
mkdir -p experiments/results
{
  echo "=== Environment Log ==="
  echo "Date: $(date)"
  echo "Hostname: $(hostname)"
  echo "OS: $(uname -a)"
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
  echo "CUDA: $(nvcc --version 2>/dev/null | tail -1 || echo 'N/A')"
  echo "NCU: $(ncu --version 2>/dev/null | head -1 || echo 'N/A')"
  echo "Python: $(python3 --version 2>/dev/null || echo 'N/A')"
  echo "CMake: $(cmake --version 2>/dev/null | head -1 || echo 'N/A')"
} > "$LOG_FILE"

echo -e "  Environment log saved to: ${LOG_FILE}"
echo ""
