## Please edit according to your environment
CXX := g++
CUDAPATH := /usr/local/cuda
CUDA_GENCODE_FLAG := -arch=compute_89 -code=sm_89

## You don't need to edit the following

NVCC := $(CUDAPATH)/bin/nvcc
CUDAINCLUDEDIR := $(CUDAPATH)/include
CUDALIBDIR := $(CUDAPATH)/lib64

NVCCFLAGS := -std=c++17 --compiler-bindir=$(CXX) $(CUDA_GENCODE_FLAG) -rdc=true \
-Werror=all-warnings -lcuda -lnvToolsExt $(NVCC_DEFINES) \
-O3
# -g -O0 -Xcompiler -rdynamic -lineinfo

SRCDIR := ./src
BLDDIR := ./target

FIPS202_WS_DIR := $(SRCDIR)/lib/fips202_ws
FIPS202_WS_SOURCE = $(wildcard $(FIPS202_WS_DIR)/*.cu)
TIMING_FENCE_WS_DIR := $(SRCDIR)/lib/timing_fence_ws
TIMING_FENCE_WS_SOURCE = $(wildcard $(TIMING_FENCE_WS_DIR)/*.cu)
VERIFY_CMOV_WS_DIR := $(SRCDIR)/lib/verify_cmov_ws
VERIFY_CMOV_WS_SOURCE = $(wildcard $(VERIFY_CMOV_WS_DIR)/*.cu)
KYBER_DIR := $(SRCDIR)/lib/kyber
KYBER_SOURCE = $(wildcard $(KYBER_DIR)/*.cu) $(wildcard $(KYBER_DIR)/*/*.cu)

SOURCE = $(FIPS202_WS_SOURCE) $(TIMING_FENCE_WS_SOURCE) $(VERIFY_CMOV_WS_SOURCE) $(KYBER_SOURCE)

.PHONY: all clean test bench bench_l2thrash trace trace_ser trace_ser_interleaved trace_ser_l2flush trace_ser_l2thrash trace_ser_l2thrash_interleaved trace_negctrl trace_keydist trace_genmatrix trace_hashct ncu_hashct ncu_genmatrix victim_loop net_server

test: test_kyber512 test_kyber768 test_kyber1024
bench: bench_kyber512 bench_kyber768 bench_kyber1024
bench_l2thrash: bench_l2thrash_kyber512 bench_l2thrash_kyber768 bench_l2thrash_kyber1024
trace: trace_kyber512 trace_kyber768 trace_kyber1024
trace_ser: trace_ser_kyber512 trace_ser_kyber768 trace_ser_kyber1024
trace_ser_l2flush: trace_ser_l2flush_kyber512 trace_ser_l2flush_kyber768 trace_ser_l2flush_kyber1024
trace_ser_l2thrash: trace_ser_l2thrash_kyber512 trace_ser_l2thrash_kyber768 trace_ser_l2thrash_kyber1024
trace_ser_interleaved: trace_ser_interleaved_kyber512 trace_ser_interleaved_kyber768 trace_ser_interleaved_kyber1024
trace_ser_l2thrash_interleaved: trace_ser_l2thrash_interleaved_kyber512 trace_ser_l2thrash_interleaved_kyber768 trace_ser_l2thrash_interleaved_kyber1024
victim_loop: victim_loop_kyber512 victim_loop_kyber768 victim_loop_kyber1024
net_server: $(BLDDIR)/net_server_kyber512.out

victim_loop_kyber512: $(BLDDIR)/victim_loop_kyber512.out
victim_loop_kyber768: $(BLDDIR)/victim_loop_kyber768.out
victim_loop_kyber1024: $(BLDDIR)/victim_loop_kyber1024.out

test_kyber512: $(BLDDIR)/test_kyber512.out
test_kyber768: $(BLDDIR)/test_kyber768.out
test_kyber1024: $(BLDDIR)/test_kyber1024.out

bench_kyber512: $(BLDDIR)/bench_kyber512.out
bench_kyber768: $(BLDDIR)/bench_kyber768.out
bench_kyber1024: $(BLDDIR)/bench_kyber1024.out

trace_kyber512: $(BLDDIR)/trace_kyber512.out
trace_kyber768: $(BLDDIR)/trace_kyber768.out
trace_kyber1024: $(BLDDIR)/trace_kyber1024.out

$(BLDDIR)/test_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/test/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 -DRNG_KIND=zero -DCUDA_DEBUG $^ -o $@
$(BLDDIR)/test_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/test/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 -DRNG_KIND=zero -DCUDA_DEBUG $^ -o $@
$(BLDDIR)/test_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/test/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 -DRNG_KIND=zero -DCUDA_DEBUG $^ -o $@

$(BLDDIR)/bench_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/bench/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 $^ -o $@
$(BLDDIR)/bench_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/bench/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 $^ -o $@
$(BLDDIR)/bench_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/bench/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

bench_l2thrash_kyber512: $(BLDDIR)/bench_l2thrash_kyber512.out
bench_l2thrash_kyber768: $(BLDDIR)/bench_l2thrash_kyber768.out
bench_l2thrash_kyber1024: $(BLDDIR)/bench_l2thrash_kyber1024.out

$(BLDDIR)/bench_l2thrash_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/bench/bench_l2thrash.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 $^ -o $@
$(BLDDIR)/bench_l2thrash_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/bench/bench_l2thrash.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 $^ -o $@
$(BLDDIR)/bench_l2thrash_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/bench/bench_l2thrash.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

$(BLDDIR)/trace_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 $^ -o $@
$(BLDDIR)/trace_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 $^ -o $@
$(BLDDIR)/trace_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

trace_ser_kyber512: $(BLDDIR)/trace_ser_kyber512.out
trace_ser_kyber768: $(BLDDIR)/trace_ser_kyber768.out
trace_ser_kyber1024: $(BLDDIR)/trace_ser_kyber1024.out

trace_ser_interleaved_kyber512: $(BLDDIR)/trace_ser_interleaved_kyber512.out
trace_ser_interleaved_kyber768: $(BLDDIR)/trace_ser_interleaved_kyber768.out
trace_ser_interleaved_kyber1024: $(BLDDIR)/trace_ser_interleaved_kyber1024.out

trace_ser_l2flush_kyber512: $(BLDDIR)/trace_ser_l2flush_kyber512.out
trace_ser_l2flush_kyber768: $(BLDDIR)/trace_ser_l2flush_kyber768.out
trace_ser_l2flush_kyber1024: $(BLDDIR)/trace_ser_l2flush_kyber1024.out

trace_ser_l2thrash_kyber512: $(BLDDIR)/trace_ser_l2thrash_kyber512.out
trace_ser_l2thrash_kyber768: $(BLDDIR)/trace_ser_l2thrash_kyber768.out
trace_ser_l2thrash_kyber1024: $(BLDDIR)/trace_ser_l2thrash_kyber1024.out

$(BLDDIR)/trace_ser_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_serialized.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 $^ -o $@
$(BLDDIR)/trace_ser_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_serialized.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 $^ -o $@
$(BLDDIR)/trace_ser_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_serialized.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

$(BLDDIR)/trace_ser_interleaved_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_interleaved.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 $^ -o $@
$(BLDDIR)/trace_ser_interleaved_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_interleaved.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 $^ -o $@
$(BLDDIR)/trace_ser_interleaved_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_interleaved.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

$(BLDDIR)/trace_ser_l2flush_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_l2flush.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 $^ -o $@
$(BLDDIR)/trace_ser_l2flush_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_l2flush.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 $^ -o $@
$(BLDDIR)/trace_ser_l2flush_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_l2flush.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

$(BLDDIR)/trace_ser_l2thrash_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_l2thrash.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 $^ -o $@
$(BLDDIR)/trace_ser_l2thrash_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_l2thrash.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 $^ -o $@
$(BLDDIR)/trace_ser_l2thrash_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_l2thrash.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

trace_ser_l2thrash_interleaved_kyber512: $(BLDDIR)/trace_ser_l2thrash_interleaved_kyber512.out
trace_ser_l2thrash_interleaved_kyber768: $(BLDDIR)/trace_ser_l2thrash_interleaved_kyber768.out
trace_ser_l2thrash_interleaved_kyber1024: $(BLDDIR)/trace_ser_l2thrash_interleaved_kyber1024.out

$(BLDDIR)/trace_ser_l2thrash_interleaved_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_l2thrash_interleaved.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 $^ -o $@
$(BLDDIR)/trace_ser_l2thrash_interleaved_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_l2thrash_interleaved.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 $^ -o $@
$(BLDDIR)/trace_ser_l2thrash_interleaved_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_ser_l2thrash_interleaved.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

trace_negctrl: trace_negctrl_kyber512
trace_negctrl_kyber512: $(BLDDIR)/trace_negctrl_kyber512.out

$(BLDDIR)/trace_negctrl_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_negctrl.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 $^ -o $@

trace_keydist: trace_keydist_kyber1024
trace_keydist_kyber1024: $(BLDDIR)/trace_keydist_kyber1024.out

$(BLDDIR)/trace_keydist_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_keydist.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

trace_genmatrix: trace_genmatrix_kyber1024
trace_genmatrix_kyber1024: $(BLDDIR)/trace_genmatrix_kyber1024.out

$(BLDDIR)/trace_genmatrix_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_genmatrix.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

trace_hashct: trace_hashct_kyber1024
trace_hashct_kyber1024: $(BLDDIR)/trace_hashct_kyber1024.out

$(BLDDIR)/trace_hashct_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/trace_hashct.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

ncu_hashct: ncu_hashct_kyber1024
ncu_hashct_kyber1024: $(BLDDIR)/ncu_hashct_kyber1024.out

$(BLDDIR)/ncu_hashct_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/ncu_hashct.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

ncu_genmatrix: ncu_genmatrix_kyber1024
ncu_genmatrix_kyber1024: $(BLDDIR)/ncu_genmatrix_kyber1024.out

$(BLDDIR)/ncu_genmatrix_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/trace/ncu_genmatrix.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

VICTIM_SRC := ../../experiments/cotenancy/victim_loop.cu
NET_SERVER_SRC := ../../experiments/network_timing/server.cu

$(BLDDIR)/net_server_kyber512.out: $(SOURCE) $(NET_SERVER_SRC)
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 -I$(SRCDIR) $^ -o $@

$(BLDDIR)/victim_loop_kyber512.out: $(SOURCE) $(VICTIM_SRC)
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 -I$(SRCDIR) $^ -o $@
$(BLDDIR)/victim_loop_kyber768.out: $(SOURCE) $(VICTIM_SRC)
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 -I$(SRCDIR) $^ -o $@
$(BLDDIR)/victim_loop_kyber1024.out: $(SOURCE) $(VICTIM_SRC)
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 -I$(SRCDIR) $^ -o $@

clean:
	$(RM) -fv $(BLDDIR)/*.out
