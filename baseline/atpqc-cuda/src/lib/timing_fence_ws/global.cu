//
// global.cu
// Kernel implementations for GPU-side timing fence.
//

#include "global.cuh"

namespace atpqc_cuda::timing_fence_ws::global {

// Record the SM clock at graph-start into *ts.
// Runs as a single thread with no dependencies so it fires immediately
// when the graph is launched.
__global__ void record_timestamp(unsigned long long* ts) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    *ts = clock64();
}

// Spin until clock64() >= *start_ts + fixed_cycles.
// Scheduled as the final node of the decaps graph (after kdf_node),
// so it pads the observable execution time to a fixed floor, eliminating
// the timing difference between valid and invalid ciphertext paths.
__global__ void timing_fence(const unsigned long long* start_ts,
                              unsigned long long fixed_cycles) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const unsigned long long target = *start_ts + fixed_cycles;
    while (clock64() < target) { /* spin */ }
  }
}

}  // namespace atpqc_cuda::timing_fence_ws::global
