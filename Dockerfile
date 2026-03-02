# =============================================================
# gpu-mlkem-security Dockerfile
# Base: NVIDIA CUDA 12.6 + Ubuntu 22.04
# Includes: CUDA toolkit, Nsight Compute, build tools,
#           Python analysis stack, ML-KEM reference codebases
# =============================================================

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ── System packages ───────────────────────────────────────
RUN apt-get update && apt-get install -y \
    # Build essentials
    git \
    cmake \
    make \
    build-essential \
    ninja-build \
    # Crypto / SSL
    libssl-dev \
    libffi-dev \
    # Python
    python3 \
    python3-pip \
    python3-dev \
    # Utilities
    wget \
    curl \
    unzip \
    pkg-config \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# ── Nsight Compute (CLI) ──────────────────────────────────
# Install ncu for microarchitectural profiling
RUN apt-get update && apt-get install -y \
    nsight-compute \
    && rm -rf /var/lib/apt/lists/* \
    || echo "WARNING: nsight-compute package not found via apt. Install manually if needed."

# ── Python analysis stack ─────────────────────────────────
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    seaborn \
    jupyterlab \
    tqdm

# ── Environment variables ─────────────────────────────────
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"

# ── Clone reference implementations ───────────────────────
WORKDIR /workspace/refs

# CPU reference ML-KEM (correctness validation)
RUN git clone --depth 1 \
    https://github.com/pq-code-package/mlkem-native.git \
    || echo "WARNING: mlkem-native clone failed"

# Python reference (KAT cross-check)
RUN git clone --depth 1 \
    https://github.com/GiacomoPope/kyber-py.git \
    || echo "WARNING: kyber-py clone failed"

# GPU ML-KEM baseline (primary)
RUN git clone --depth 1 \
    https://github.com/open-quantum-safe/liboqs-cupqc-meta.git \
    || echo "WARNING: liboqs-cupqc-meta clone failed - use atpqc-cuda fallback"

# GPU ML-KEM fallback
RUN git clone --depth 1 \
    https://github.com/tono-satolab/atpqc-cuda.git \
    || echo "WARNING: atpqc-cuda clone failed"

# ── Set working directory to project ─────────────────────
WORKDIR /workspace/gpu-mlkem-security

# ── Verify environment on build ──────────────────────────
RUN nvcc --version && \
    python3 --version && \
    python3 -c "import numpy, scipy, matplotlib; print('Python stack OK')"

# ── Default command ───────────────────────────────────────
CMD ["/bin/bash"]
