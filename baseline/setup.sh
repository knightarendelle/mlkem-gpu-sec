#!/bin/bash
# =============================================================
# baseline/setup.sh
# Run this once to configure and build the GPU Kyber baseline.
# Automatically detects GPU architecture and patches Makefile.
# Usage: bash setup.sh
# =============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  mlkem-gpu-sec: Baseline Setup                  ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

ATPQC_DIR="$(dirname "$0")/atpqc-cuda"

# ── Check atpqc-cuda exists ───────────────────────────────
if [ ! -d "$ATPQC_DIR" ]; then
  echo -e "${RED}[ERROR]${NC} atpqc-cuda directory not found at: $ATPQC_DIR"
  echo -e "        Make sure you're running this from the baseline/ folder."
  echo -e "        Run: bash baseline/setup.sh from the repo root, or"
  echo -e "             cd baseline && bash setup.sh"
  exit 1
fi

cd "$ATPQC_DIR"
echo -e "${BLUE}[ Step 1/4 ]${NC} Detecting GPU compute capability..."

# ── Detect GPU compute capability ────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
  echo -e "${RED}[ERROR]${NC} nvidia-smi not found. Is the GPU driver installed?"
  exit 1
fi

# Get compute capability from deviceQuery or nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo -e "        GPU detected: ${GPU_NAME}"

# Map known GPUs to compute capability
if echo "$GPU_NAME" | grep -qi "RTX 40\|RTX 4050\|RTX 4060\|RTX 4070\|RTX 4080\|RTX 4090\|RTX 4000 Ada\|RTX 4500\|RTX 6000 Ada"; then
  CC="89"
elif echo "$GPU_NAME" | grep -qi "RTX 30\|RTX 3050\|RTX 3060\|RTX 3070\|RTX 3080\|RTX 3090\|A40\|A10\|A16\|A30"; then
  CC="86"
elif echo "$GPU_NAME" | grep -qi "RTX 20\|RTX 2060\|RTX 2070\|RTX 2080\|T4"; then
  CC="75"
elif echo "$GPU_NAME" | grep -qi "A100"; then
  CC="80"
elif echo "$GPU_NAME" | grep -qi "H100\|H200"; then
  CC="90"
elif echo "$GPU_NAME" | grep -qi "GTX 16\|GTX 1650\|GTX 1660"; then
  CC="75"
else
  # Fallback: try to detect via python
  CC=$(python3 -c "
import subprocess, re
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                          capture_output=True, text=True)
    cap = result.stdout.strip().replace('.', '')
    print(cap)
except:
    print('86')
" 2>/dev/null || echo "86")
  echo -e "${YELLOW}[WARN]${NC}  Could not auto-detect CC for '$GPU_NAME', using compute_${CC}"
  echo -e "        If build fails, manually edit Makefile: CUDA_GENCODE_FLAG := -arch=compute_XX -code=sm_XX"
fi

echo -e "${GREEN}[OK]${NC}   Compute capability: ${CC}"

# ── Patch Makefile ────────────────────────────────────────
echo ""
echo -e "${BLUE}[ Step 2/4 ]${NC} Patching Makefile for this environment..."

# Check if already patched
if grep -q "CUDAPATH := /usr/local/cuda$" Makefile && \
   grep -q "CXX := g++$" Makefile && \
   grep -q "compute_${CC}" Makefile; then
  echo -e "${GREEN}[OK]${NC}   Makefile already correctly configured, skipping patch."
else
  # Backup original
  cp Makefile Makefile.orig

  # Apply patches
  sed -i 's|CXX := g++-[0-9]*|CXX := g++|' Makefile
  sed -i 's|CUDAPATH := /usr/local/cuda-[0-9.]*|CUDAPATH := /usr/local/cuda|' Makefile
  sed -i "s|-arch=compute_[0-9]* -code=sm_[0-9]*|-arch=compute_${CC} -code=sm_${CC}|" Makefile

  echo -e "${GREEN}[OK]${NC}   Makefile patched:"
  echo -e "        CXX      → g++"
  echo -e "        CUDAPATH → /usr/local/cuda"
  echo -e "        Arch     → compute_${CC} / sm_${CC}"
fi

# ── Create build directory ────────────────────────────────
echo ""
echo -e "${BLUE}[ Step 3/4 ]${NC} Creating build directory..."
mkdir -p target
echo -e "${GREEN}[OK]${NC}   target/ directory ready"

# ── Build and test ────────────────────────────────────────
echo ""
echo -e "${BLUE}[ Step 4/4 ]${NC} Building all three parameter sets (this takes 2-3 minutes)..."
echo ""

make test 2>&1

echo ""

# ── Run tests ─────────────────────────────────────────────
echo -e "${BLUE}[ Running correctness tests ]${NC}"
echo ""

PASS=0
FAIL=0

for variant in 512 768 1024; do
  echo -n "  Testing Kyber-${variant}... "
  if ./target/test_kyber${variant}.out &>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    PASS=$((PASS+1))
  else
    echo -e "${RED}FAIL${NC}"
    FAIL=$((FAIL+1))
  fi
done

echo ""
echo -e "${BLUE}=================================================${NC}"
if [ $FAIL -eq 0 ]; then
  echo -e "${GREEN}  All tests passed. Baseline is ready.${NC}"
  echo ""
  echo -e "  Run benchmarks:"
  echo -e "    echo \"1024 4 4 4 4\" | ./target/bench_kyber512.out"
  echo -e "    echo \"1024 4 4 4 4\" | ./target/bench_kyber768.out"
  echo -e "    echo \"1024 4 4 4 4\" | ./target/bench_kyber1024.out"
else
  echo -e "${RED}  ${FAIL} test(s) failed. Check output above.${NC}"
fi
echo -e "${BLUE}=================================================${NC}"
echo ""