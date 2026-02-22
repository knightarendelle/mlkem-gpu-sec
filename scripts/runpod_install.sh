#!/bin/bash
# =============================================================
# runpod_install.sh
# Run this ONCE on a fresh RunPod instance to install all
# dependencies needed for the gpu-mlkem-security project.
# =============================================================

set -e

echo ""
echo "================================================="
echo "  gpu-mlkem-security: RunPod Dependency Install  "
echo "================================================="
echo ""

# ── System packages ───────────────────────────────────────
echo "[1/5] Installing system packages..."
apt-get update -qq
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
  2>/dev/null
echo "      Done."

# ── Python packages ───────────────────────────────────────
echo "[2/5] Installing Python analysis libraries..."
pip install -q \
  numpy \
  scipy \
  matplotlib \
  pandas \
  scikit-learn \
  seaborn \
  jupyterlab
echo "      Done."

# ── Nsight Compute (CLI) ──────────────────────────────────
echo "[3/5] Checking Nsight Compute..."
if command -v ncu &>/dev/null; then
  echo "      Already installed: $(ncu --version 2>/dev/null | head -1)"
else
  echo "      Not found. Attempting install..."
  apt-get install -y nsight-compute 2>/dev/null || \
    echo "      WARNING: Could not auto-install ncu. Download manually from developer.nvidia.com/nsight-compute"
fi

# ── Clone reference implementations ───────────────────────
echo "[4/5] Cloning reference implementations..."

mkdir -p /workspace/refs
cd /workspace/refs

# CPU reference (correctness validation)
if [ ! -d "mlkem-native" ]; then
  echo "      Cloning mlkem-native (CPU reference)..."
  git clone --depth 1 https://github.com/pq-code-package/mlkem-native.git
else
  echo "      mlkem-native already exists, skipping."
fi

# Python reference (KAT validation)
if [ ! -d "kyber-py" ]; then
  echo "      Cloning kyber-py (Python KAT reference)..."
  git clone --depth 1 https://github.com/GiacomoPope/kyber-py.git
else
  echo "      kyber-py already exists, skipping."
fi

# GPU ML-KEM baseline (primary)
if [ ! -d "liboqs-cupqc-meta" ]; then
  echo "      Cloning liboqs-cupqc-meta (GPU baseline)..."
  git clone --depth 1 https://github.com/open-quantum-safe/liboqs-cupqc-meta.git || \
    echo "      WARNING: liboqs-cupqc-meta clone failed. Will try atpqc-cuda fallback."
else
  echo "      liboqs-cupqc-meta already exists, skipping."
fi

# Fallback GPU baseline
if [ ! -d "atpqc-cuda" ]; then
  echo "      Cloning atpqc-cuda (fallback GPU baseline)..."
  git clone --depth 1 https://github.com/tono-satolab/atpqc-cuda.git || \
    echo "      WARNING: atpqc-cuda clone failed. Check internet connection."
else
  echo "      atpqc-cuda already exists, skipping."
fi

cd /workspace

# ── Final verification ─────────────────────────────────────
echo "[5/5] Running environment verification..."
echo ""
bash /workspace/gpu-mlkem-security/scripts/verify_env.sh

echo ""
echo "================================================="
echo "  Install complete. Your RunPod is ready."
echo "================================================="
echo ""
echo "  Reference implementations cloned to: /workspace/refs/"
echo "  Project directory:                   /workspace/gpu-mlkem-security/"
echo ""
echo "  Next step: cd baseline && bash setup.sh"
echo ""
