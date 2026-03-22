//
// host.cuh
// Host-side wrapper for the GPU timing fence.
//
// Usage in a CUDA graph:
//
//   timing_fence tf(700);   // 700 µs floor — above worst-case observed decaps
//
//   // Node 1: no deps, fires at graph start
//   auto rec_args = tf.generate_record_args(ts_buf_ptr);
//   cudaGraphAddKernelNode(&rec_node, graph, nullptr, 0, &rec_params);
//
//   // ... all decaps nodes ...
//
//   // Node N: depends on {kdf_node, rec_node}
//   auto fence_args = tf.generate_fence_args(ts_buf_ptr);
//   std::array deps{kdf_node, rec_node};
//   cudaGraphAddKernelNode(&fence_node, graph, deps.data(), 2, &fence_params);
//
//   *ss_available_ptr = fence_node;   // ss only "ready" after fence
//

#ifndef ATPQC_CUDA_LIB_TIMING_FENCE_WS_HOST_CUH_
#define ATPQC_CUDA_LIB_TIMING_FENCE_WS_HOST_CUH_

#include <cstdio>
#include <memory>

#include "../cuda_debug.hpp"
#include "../cuda_resource.hpp"
#include "global.cuh"

namespace atpqc_cuda::timing_fence_ws::host {

class timing_fence {
 private:
  unsigned long long fixed_cycles_;

 public:
  // kernel_args types follow the same pattern as every other host.cuh in
  // this codebase: cuda_resource::kernel_args<Arg0, Arg1, ...>.
  using record_args_type =
      cuda_resource::kernel_args<unsigned long long*>;

  using fence_args_type =
      cuda_resource::kernel_args<const unsigned long long*,
                                  unsigned long long>;

  std::unique_ptr<record_args_type> generate_record_args(
      unsigned long long* ts_buf) const noexcept {
    return std::make_unique<record_args_type>(ts_buf);
  }

  std::unique_ptr<fence_args_type> generate_fence_args(
      const unsigned long long* ts_buf) const noexcept {
    return std::make_unique<fence_args_type>(ts_buf, fixed_cycles_);
  }

  void* get_record_func() const noexcept {
    return reinterpret_cast<void*>(global::record_timestamp);
  }
  void* get_fence_func() const noexcept {
    return reinterpret_cast<void*>(global::timing_fence);
  }

  // Both kernels are single-thread; grid/block/smem are identical for each.
  dim3     get_grid_dim()    const noexcept { return {1, 1, 1}; }
  dim3     get_block_dim()   const noexcept { return {1, 1, 1}; }
  unsigned get_shared_bytes() const noexcept { return 0; }

  // target_us: the fixed decaps floor in microseconds.
  // Must be set above the worst-case observed decaps latency for both
  // ciphertext classes. 700 µs is conservative for Kyber-768 on RTX 40xx.
  explicit timing_fence(unsigned target_us) {
    int clock_khz = 0;
    CCC(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    fixed_cycles_ = static_cast<unsigned long long>(target_us) *
                    static_cast<unsigned long long>(clock_khz) / 1000ULL;
    std::printf("timing_fence: clock_khz=%u, target_us=%u, fixed_cycles=%llu\n",
                clock_khz, target_us, fixed_cycles_);
  }
};

}  // namespace atpqc_cuda::timing_fence_ws::host

#endif
