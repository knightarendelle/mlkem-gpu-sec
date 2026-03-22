//
// global.cuh
// Kernel declarations for GPU-side timing fence.
//
// Two-kernel protocol:
//   1. record_timestamp — call with no dependencies at graph start;
//      writes clock64() into a device buffer.
//   2. timing_fence     — call as last node of the graph (after kdf);
//      spins until clock64() >= *start_ts + fixed_cycles, making total
//      graph execution time approximately constant regardless of which
//      decaps path was faster.
//

#ifndef ATPQC_CUDA_LIB_TIMING_FENCE_WS_GLOBAL_CUH_
#define ATPQC_CUDA_LIB_TIMING_FENCE_WS_GLOBAL_CUH_

namespace atpqc_cuda::timing_fence_ws::global {

__global__ void record_timestamp(unsigned long long* ts);

__global__ void timing_fence(const unsigned long long* start_ts,
                              unsigned long long fixed_cycles);

}  // namespace atpqc_cuda::timing_fence_ws::global

#endif
