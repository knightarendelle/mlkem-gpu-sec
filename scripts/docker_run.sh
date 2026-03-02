#!/bin/bash
# =============================================================
# scripts/docker_run.sh
# Wrapper script — run this to start the research environment
# Usage: bash scripts/docker_run.sh
# =============================================================

set -e

IMAGE="gpu-mlkem-security:latest"
CONTAINER="gpu-mlkem-security"

echo ""
echo "================================================="
echo "  gpu-mlkem-security: Docker Environment"
echo "================================================="
echo ""

# Check Docker is running
if ! docker info &>/dev/null; then
  echo "ERROR: Docker is not running. Start Docker Desktop first."
  exit 1
fi

# Check GPU is accessible
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
  echo "ERROR: Cannot access GPU inside Docker."
  echo "Fix: Make sure Docker Desktop has GPU support enabled."
  exit 1
fi

# Build image if it doesn't exist
if ! docker image inspect "$IMAGE" &>/dev/null; then
  echo "Image not found. Building now (first time takes 5-10 minutes)..."
  docker build -t "$IMAGE" .
  echo "Build complete."
fi

# Stop existing container if running
if docker ps -q -f name="$CONTAINER" | grep -q .; then
  echo "Stopping existing container..."
  docker stop "$CONTAINER" &>/dev/null
fi

# Remove existing container if stopped
if docker ps -aq -f name="$CONTAINER" | grep -q .; then
  docker rm "$CONTAINER" &>/dev/null
fi

echo "Starting container..."
echo ""

# Run container
docker run -it \
  --gpus all \
  --cap-add=SYS_ADMIN \
  --name "$CONTAINER" \
  -v "$(pwd):/workspace/gpu-mlkem-security" \
  -w "/workspace/gpu-mlkem-security" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  "$IMAGE" \
  /bin/bash

echo ""
echo "Container stopped."
