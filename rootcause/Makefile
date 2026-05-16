# rootcause/Makefile

CUDAPATH := /usr/local/cuda
NVCC     := $(CUDAPATH)/bin/nvcc

# Default sm_89 (RTX 40xx Ada) — override: make SM=86
SM ?= 89

NVCC_FLAGS := -arch=compute_$(SM) -code=sm_$(SM) \
              -O2 -std=c++14 \
              -I$(CUDAPATH)/include \
              -lcudart

.PHONY: all clean

all: dcc_microbench

dcc_microbench: dcc_microbench.cu
	$(NVCC) $(NVCC_FLAGS) -o dcc_microbench dcc_microbench.cu

clean:
	rm -f dcc_microbench
