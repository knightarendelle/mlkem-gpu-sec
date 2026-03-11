# harness/Makefile
# Build the Phase 2 timing harness

CUDAPATH := /usr/local/cuda
NVCC     := $(CUDAPATH)/bin/nvcc
CXX      := g++

# Detect compute capability from baseline Makefile if available
# Default to sm_89 (RTX 4050/4090 Ada) — override with: make CC=86
CC ?= 89
GENCODE := -arch=compute_$(CC) -code=sm_$(CC)

NVCC_FLAGS := $(GENCODE) -O2 -std=c++14 \
              -I$(CUDAPATH)/include \
              -lcudart -lm

TARGET_DIR := ../baseline/atpqc-cuda/target
OUT_DIR    := .

.PHONY: all clean

all: collect_traces

collect_traces: collect_traces.cu
	@echo "Building collect_traces (sm_$(CC))..."
	@mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCC_FLAGS) -o collect_traces collect_traces.cu
	@echo "Done: ./harness/collect_traces"

clean:
	rm -f collect_traces

help:
	@echo "Usage:"
	@echo "  make              — build with default sm_89 (RTX 40xx)"
	@echo "  make CC=86        — build for RTX 30xx (Ampere)"
	@echo "  make CC=75        — build for RTX 20xx / GTX 16xx (Turing)"
	@echo "  make clean        — remove build artifacts"
