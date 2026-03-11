#!/bin/bash
# =============================================================
# runpod_install.sh
# Run this ONCE on a fresh RunPod instance to install all
# dependencies needed for the mlkem-gpu-sec project.
# Usage: bash scripts/runpod_install.sh
# Run from: /workspace/mlkem-gpu-sec/
# =============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Resolve project root relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  mlkem-gpu-sec: RunPod Dependency Install       ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "  Project root: $PROJECT_ROOT"
echo ""

# ── System packages ───────────────────────────────────────
echo -e "${BLUE}[1/5]${NC} Installing system packages..."
apt-get update -qq 2>/dev/null
apt-get install -y \
  git \
  cmake \
  build-essential \
  python3-pip \
  python3-dev \
  libssl-dev \
  libffi-dev \
  wget \
  curl \
  unzip \
  pkg-config \
  2>/dev/null && echo -e "${GREEN}      Done.${NC}" || echo -e "${YELLOW}      Some packages may have failed — continuing.${NC}"

# ── Python packages ───────────────────────────────────────
echo -e "${BLUE}[2/5]${NC} Installing Python analysis libraries..."
pip install -q --break-system-packages \
  numpy \
  scipy \
  matplotlib \
  pandas \
  scikit-learn \
  seaborn 2>/dev/null || \
pip install -q \
  numpy \
  scipy \
  matplotlib \
  pandas \
  scikit-learn \
  seaborn 2>/dev/null
echo -e "${GREEN}      Done.${NC}"

# ── Nsight Compute (CLI) ──────────────────────────────────
echo -e "${BLUE}[3/5]${NC} Checking Nsight Compute..."
if command -v ncu &>/dev/null; then
  echo -e "${GREEN}      Already installed:${NC} $(ncu --version 2>/dev/null | head -1)"
  echo -e "${YELLOW}      NOTE: ncu perf counters are blocked on RunPod (ERR_NVGPUCTRPERM).${NC}"
  echo -e "${YELLOW}      Phase 3 root cause analysis must run on local Docker instead.${NC}"
else
  echo -e "${YELLOW}      ncu not found. Phase 3 must run on local Docker.${NC}"
fi

# ── Clone reference implementations ───────────────────────
echo -e "${BLUE}[4/5]${NC} Cloning reference implementations to /workspace/refs/..."

mkdir -p /workspace/refs
cd /workspace/refs

# CPU reference (correctness validation)
if [ ! -d "mlkem-native" ]; then
  echo "      Cloning mlkem-native (CPU reference)..."
  git clone --depth 1 https://github.com/pq-code-package/mlkem-native.git 2>/dev/null && \
    echo -e "${GREEN}      mlkem-native cloned.${NC}" || \
    echo -e "${YELLOW}      WARNING: mlkem-native clone failed.${NC}"
else
  echo -e "${GREEN}      mlkem-native already exists, skipping.${NC}"
fi

# Python reference (KAT validation)
if [ ! -d "kyber-py" ]; then
  echo "      Cloning kyber-py (Python KAT reference)..."
  git clone --depth 1 https://github.com/GiacomoPope/kyber-py.git 2>/dev/null && \
    echo -e "${GREEN}      kyber-py cloned.${NC}" || \
    echo -e "${YELLOW}      WARNING: kyber-py clone failed.${NC}"
else
  echo -e "${GREEN}      kyber-py already exists, skipping.${NC}"
fi

# ML-KEM CUDA source reference (no standalone build — reference only)
if [ ! -d "liboqs-cupqc-meta" ]; then
  echo "      Cloning liboqs-cupqc-meta (ML-KEM CUDA source reference)..."
  git clone --depth 1 https://github.com/open-quantum-safe/liboqs-cupqc-meta.git 2>/dev/null && \
    echo -e "${GREEN}      liboqs-cupqc-meta cloned.${NC}" || \
    echo -e "${YELLOW}      WARNING: liboqs-cupqc-meta clone failed.${NC}"
else
  echo -e "${GREEN}      liboqs-cupqc-meta already exists, skipping.${NC}"
fi

cd /workspace

# ── Final verification ─────────────────────────────────────
echo -e "${BLUE}[5/5]${NC} Running environment verification..."
echo ""
bash "$PROJECT_ROOT/scripts/verify_env.sh"

echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}  Install complete. RunPod is ready.${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo "  Reference implementations: /workspace/refs/"
echo "  Project directory:         $PROJECT_ROOT"
echo ""
echo "  Next step:"
echo "    cd $PROJECT_ROOT"
echo "    bash baseline/setup.sh"
echo ""