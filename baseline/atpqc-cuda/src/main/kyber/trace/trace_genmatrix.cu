//
// trace_genmatrix.cu
// Isolates the generate_at (genmatrix transposed) kernel for root-cause analysis.
//
// Two classes, strictly interleaved in a single run:
//   Class 0: 100k generate_at calls with seed = rho0 (random 32-byte value)
//   Class 1: 100k generate_at calls with seed = rho1 (different random 32-byte value)
//
// Before the main loop, an instrumented count kernel runs once per class to
// report per-polynomial SHAKE128 extra-block counts (rejection sampling loop
// iterations beyond the initial xof_nblocks squeeze) to stderr.
//
// If class0 and class1 have DIFFERENT max extra-block counts, timing diverges
// because of rejection sampling — that is the leakage mechanism.
// If counts are identical, the source is something else (DCC context bleed,
// downstream memory effects).
//
// Usage:
//   echo "1 4 4 4 4 100000" | ./target/trace_genmatrix_kyber1024.out \
//     experiments/traces/per_kernel/genmatrix_class0_n100000.csv \
//     experiments/traces/per_kernel/genmatrix_class1_n100000.csv
//
// Build:
//   make -C baseline/atpqc-cuda trace_genmatrix_kyber1024
//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "../../../lib/cuda_debug.hpp"
#include "../../../lib/cuda_resource.hpp"
#include "../../../lib/kyber/genpoly_warp/device.cuh"
#include "../../../lib/kyber/genpoly_warp/host.cuh"
#include "../../../lib/kyber/genpoly_warp/kernel_params.cuh"
#include "../../../lib/kyber/params.cuh"
#include "../../../lib/kyber/variants.cuh"
#include "../../../lib/rng/std_random_device.hpp"

#ifndef KYBER_VARIANT
#define KYBER_VARIANT kyber1024
#endif

#undef CCC
static inline cudaError_t gm_check(cudaError_t e, const char* call,
                                    const char* file, int line) {
  if (e != cudaSuccess) {
    std::fprintf(stderr,
                 "[GENMATRIX CUDA ERROR] %s:%d\n  call: %s\n  error: %s\n",
                 file, line, call, cudaGetErrorString(e));
    std::fflush(stderr);
    std::abort();
  }
  return e;
}
static inline CUresult gm_check(CUresult e, const char* call,
                                  const char* file, int line) {
  if (e != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* str  = nullptr;
    cuGetErrorName(e, &name);
    cuGetErrorString(e, &str);
    std::fprintf(stderr,
                 "[GENMATRIX CUDA ERROR] %s:%d\n  call: %s\n  error: %s (%s)\n",
                 file, line, call, str ? str : "?", name ? name : "?");
    std::fflush(stderr);
    std::abort();
  }
  return e;
}
#define CCC(call) gm_check((call), #call, __FILE__, __LINE__)

namespace atpqc_cuda::kyber::trace_genmatrix {

using variant = variants::KYBER_VARIANT;

// ── Instrumented count kernel ─────────────────────────────────────────────
// Mirrors global::genmatrix<K, true> (transposed) but writes extra SHAKE
// block counts per polynomial into 'counts' instead of writing poly output.
// Grid/block dimensions must match genpoly_warp::host::genat<K>.
template <unsigned K>
__global__ void genmatrix_at_count(std::uint32_t* counts,
                                    const std::uint8_t* seed,
                                    std::size_t seed_pitch,
                                    unsigned npolys) {
  using kp = genpoly_warp::kernel_params::genmatrix;

  extern __shared__ std::uint8_t shared_buf[];
  const unsigned polyid = blockIdx.x * blockDim.y + threadIdx.y;

  if (polyid < npolys) {
    std::int16_t* poly_buf = reinterpret_cast<std::int16_t*>(
        shared_buf + kp::smem_byte_per_warp * threadIdx.y);
    std::uint8_t* bytes_buf =
        reinterpret_cast<std::uint8_t*>(poly_buf) + kp::poly_bytes;

    seed += seed_pitch * (polyid / (K * K));
    const unsigned x = polyid / K % K;
    const unsigned y = polyid % K;

    symmetric_ws::device::state_type state;
    symmetric_ws::device::keccak_type keccak;
    symmetric_ws::device::xof xof;
    genpoly_warp::device::rej rej;

    bytes_buf[threadIdx.x] = seed[threadIdx.x];
    if (threadIdx.x == 0) {
      bytes_buf[params::symbytes]     = x;  // transposed: (i,j) absorbed as (x,y)
      bytes_buf[params::symbytes + 1] = y;
    }
    __syncwarp();

    state = xof.absorb(bytes_buf, keccak);
    state = xof.squeezeblocks(bytes_buf, kp::xof_nblocks, state, keccak);
    __syncwarp();

    unsigned ctr    = rej(poly_buf, params::n, bytes_buf, kp::rej_bytes);
    unsigned buflen = kp::rej_bytes;
    unsigned extra  = 0;
    while (ctr < params::n) {
      const unsigned off = buflen % 3;
      if (threadIdx.x < off)
        bytes_buf[threadIdx.x] = bytes_buf[buflen - off + threadIdx.x];
      __syncwarp();

      state = xof.squeezeblocks(bytes_buf + off, 1, state, keccak);
      buflen = off + kp::xof_blockbytes;
      ++extra;
      __syncwarp();

      ctr += rej(poly_buf + ctr, params::n - ctr, bytes_buf, buflen);
    }

    // Only thread 0 per warp writes the count (extra is uniform across the warp
    // since the while condition is evaluated identically by all threads).
    if (threadIdx.x == 0)
      counts[polyid] = extra;
  }
}

// ── Per-kernel timed launch ───────────────────────────────────────────────
#define LAUNCH_TIMED(label, obj, stream, ev_a, ev_b, record, fout, tid, ...)  \
  do {                                                                          \
    auto _a = (obj).generate_args(__VA_ARGS__);                                \
    CCC(cudaEventRecord((ev_a), (stream)));                                     \
    CCC(cudaLaunchKernel((obj).get_func(), (obj).get_grid_dim(),               \
                         (obj).get_block_dim(), _a->get_args_ptr(),            \
                         (obj).get_shared_bytes(), (stream)));                 \
    CCC(cudaEventRecord((ev_b), (stream)));                                     \
    CCC(cudaDeviceSynchronize());                                               \
    if (record) {                                                               \
      float _ms = 0.f;                                                         \
      CCC(cudaEventElapsedTime(&_ms, (ev_a), (ev_b)));                         \
      std::fprintf((fout), "%u,%s,%.3f\n", (tid), (label), _ms * 1000.f);    \
    }                                                                           \
  } while (0)

#define LAUNCH_TIMED_CAPTURE(label, out_us, obj, stream, ev_a, ev_b, record, fout, tid, ...) \
  do {                                                                          \
    auto _a = (obj).generate_args(__VA_ARGS__);                                \
    CCC(cudaEventRecord((ev_a), (stream)));                                     \
    CCC(cudaLaunchKernel((obj).get_func(), (obj).get_grid_dim(),               \
                         (obj).get_block_dim(), _a->get_args_ptr(),            \
                         (obj).get_shared_bytes(), (stream)));                 \
    CCC(cudaEventRecord((ev_b), (stream)));                                     \
    CCC(cudaDeviceSynchronize());                                               \
    { float _ms = 0.f;                                                         \
      CCC(cudaEventElapsedTime(&_ms, (ev_a), (ev_b)));                         \
      (out_us) = _ms * 1000.f;                                                 \
      if (record) std::fprintf((fout), "%u,%s,%.3f\n", (tid), (label), (out_us)); } \
  } while (0)

void trace_genmatrix_fn(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "Usage: %s class0.csv class1.csv\n"
                 "  stdin: ninputs genmat_nw genvec_nw genpoly_nw fips_nw"
                 " ntraces_per_class\n",
                 argv[0]);
    std::abort();
  }
  FILE* out0 = std::fopen(argv[1], "w");
  FILE* out1 = std::fopen(argv[2], "w");
  if (!out0 || !out1) {
    std::fprintf(stderr, "[genmatrix] failed to open output files: %s %s\n",
                 argv[1], argv[2]);
    std::abort();
  }

  unsigned ninputs, genmat_nw, genvec_nw, genpoly_nw, fips_nw;
  unsigned ntraces_per_class;
  std::cin >> ninputs >> genmat_nw >> genvec_nw >> genpoly_nw >> fips_nw
           >> ntraces_per_class;

  constexpr unsigned k = params::k<variant>;

  const char* variant_name =
      params::ciphertextbytes<variant> == 1568 ? "Kyber-1024" :
      params::ciphertextbytes<variant> == 1088 ? "Kyber-768"  : "Kyber-512";

  std::fprintf(stderr,
               "[genmatrix] variant: %s  k=%u  ninputs=%u  ntraces_per_class=%u\n"
               "[genmatrix] out0: %s\n[genmatrix] out1: %s\n",
               variant_name, k, ninputs, ntraces_per_class, argv[1], argv[2]);
  std::fflush(stderr);

  // ── Two random rho seeds ──────────────────────────────────────────────────
  // rho is the 32-byte public seed used as input to the genmatrix SHAKE128 XOF.
  // In Kyber keypair generation, rho is sampled uniformly at random, so two
  // fresh random 32-byte values are cryptographically equivalent to rho from
  // two real keypairs (which is exactly the key-distinguishing scenario).
  rng::std_random_device randombytes;
  std::uint8_t rho0[params::symbytes], rho1[params::symbytes];
  randombytes(rho0, params::symbytes);
  randombytes(rho1, params::symbytes);

  std::fprintf(stderr, "[genmatrix] rho0: %02x%02x%02x%02x%02x%02x%02x%02x...\n",
               rho0[0], rho0[1], rho0[2], rho0[3],
               rho0[4], rho0[5], rho0[6], rho0[7]);
  std::fprintf(stderr, "[genmatrix] rho1: %02x%02x%02x%02x%02x%02x%02x%02x...\n",
               rho1[0], rho1[1], rho1[2], rho1[3],
               rho1[4], rho1[5], rho1[6], rho1[7]);
  std::fflush(stderr);

  // ── Device seed buffers ───────────────────────────────────────────────────
  cuda_resource::device_pitched_memory<std::uint8_t> seed0_d(params::symbytes,
                                                              ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> seed1_d(params::symbytes,
                                                              ninputs);
  CCC(cudaMemcpy2D(seed0_d.get_ptr(), seed0_d.get_pitch(),
                   rho0, params::symbytes,
                   params::symbytes, ninputs, cudaMemcpyHostToDevice));
  CCC(cudaMemcpy2D(seed1_d.get_ptr(), seed1_d.get_pitch(),
                   rho1, params::symbytes,
                   params::symbytes, ninputs, cudaMemcpyHostToDevice));
  CCC(cudaDeviceSynchronize());

  // ── Output buffer for genmatrix (flat) ───────────────────────────────────
  // The kernel writes k*k*ninputs polynomials each of n/2 short2 elements.
  const unsigned at_nelems = k * k * (params::n / 2) * ninputs;
  short2* at_d = nullptr;
  CCC(cudaMalloc(&at_d, at_nelems * sizeof(short2)));

  // ── Kernel object ─────────────────────────────────────────────────────────
  genpoly_warp::host::genat<k> generate_at(ninputs, genmat_nw);

  // ── CUDA Events + stream ──────────────────────────────────────────────────
  cuda_resource::stream stream(cudaStreamNonBlocking);
  cudaEvent_t ev_start, ev_stop;
  CCC(cudaEventCreate(&ev_start));
  CCC(cudaEventCreate(&ev_stop));

  // ── Block count kernel ────────────────────────────────────────────────────
  // Runs once per class; grid/block/smem match generate_at so the GPU sees the
  // same occupancy as during timing.
  const unsigned npolys = k * k * ninputs;
  std::uint32_t* counts_d = nullptr;
  CCC(cudaMalloc(&counts_d, npolys * sizeof(std::uint32_t)));

  auto run_count = [&](const std::uint8_t* seed_ptr, std::size_t seed_pitch,
                       const char* label) -> unsigned /*max_extra*/ {
    CCC(cudaMemset(counts_d, 0, npolys * sizeof(std::uint32_t)));
    genmatrix_at_count<k><<<generate_at.get_grid_dim(),
                            generate_at.get_block_dim(),
                            generate_at.get_shared_bytes(),
                            stream>>>(counts_d, seed_ptr, seed_pitch, npolys);
    CCC(cudaDeviceSynchronize());

    std::vector<std::uint32_t> counts_h(npolys);
    CCC(cudaMemcpy(counts_h.data(), counts_d,
                   npolys * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));

    std::fprintf(stderr,
                 "[count/%s] extra SHAKE blocks per polynomial"
                 " (transposed, k=%u, initial=%u blocks):\n",
                 label, k, genpoly_warp::kernel_params::genmatrix::xof_nblocks);
    unsigned total = 0, max_extra = 0;
    for (unsigned i = 0; i < npolys; ++i) {
      const unsigned ix = i / k % k, iy = i % k;
      std::fprintf(stderr, "  poly[%u][%u] extra=%u  total=%u\n",
                   ix, iy, counts_h[i],
                   genpoly_warp::kernel_params::genmatrix::xof_nblocks + counts_h[i]);
      total += counts_h[i];
      if (counts_h[i] > max_extra) max_extra = counts_h[i];
    }
    std::fprintf(stderr,
                 "[count/%s] sum_extra=%u  max_extra=%u"
                 "  (max_total=%u blocks)\n\n",
                 label, total, max_extra,
                 genpoly_warp::kernel_params::genmatrix::xof_nblocks + max_extra);
    std::fflush(stderr);
    return max_extra;
  };

  const unsigned max0 = run_count(seed0_d.get_ptr(), seed0_d.get_pitch(), "class0");
  const unsigned max1 = run_count(seed1_d.get_ptr(), seed1_d.get_pitch(), "class1");

  if (max0 == max1)
    std::fprintf(stderr,
                 "[count] SAME max extra blocks (%u) for both classes.\n"
                 "  => rejection sampling alone does not explain timing differences.\n\n",
                 max0);
  else
    std::fprintf(stderr,
                 "[count] DIFFERENT max extra blocks: class0=%u class1=%u.\n"
                 "  => rejection sampling IS key-dependent; timing leakage confirmed.\n\n",
                 max0, max1);
  std::fflush(stderr);

  // ── Warmup: 200 alternating calls ─────────────────────────────────────────
  std::fprintf(stderr, "[warmup] 200 iterations (interleaved)...\n");
  float _wu = 0.f;
  for (unsigned w = 0; w < 200; ++w) {
    if (w % 2 == 0)
      LAUNCH_TIMED_CAPTURE("generate_at", _wu, generate_at, stream,
                           ev_start, ev_stop, false, stderr, 0,
                           at_d, seed0_d.get_ptr(), seed0_d.get_pitch());
    else
      LAUNCH_TIMED_CAPTURE("generate_at", _wu, generate_at, stream,
                           ev_start, ev_stop, false, stderr, 0,
                           at_d, seed1_d.get_ptr(), seed1_d.get_pitch());
  }
  std::fprintf(stderr, "[warmup] done\n");

  // ── Pilot: 1000 per class, interleaved ────────────────────────────────────
  {
    std::fprintf(stderr,
                 "[pilot] 1000 traces per class (interleaved)...\n");
    double sum0 = 0.0, sum1 = 0.0;
    unsigned cnt0 = 0, cnt1 = 0;
    float us = 0.f;
    while (cnt0 < 1000 || cnt1 < 1000) {
      if (cnt0 < 1000) {
        LAUNCH_TIMED_CAPTURE("generate_at", us, generate_at, stream,
                             ev_start, ev_stop, false, stderr, 0,
                             at_d, seed0_d.get_ptr(), seed0_d.get_pitch());
        sum0 += us;
        ++cnt0;
      }
      if (cnt1 < 1000) {
        LAUNCH_TIMED_CAPTURE("generate_at", us, generate_at, stream,
                             ev_start, ev_stop, false, stderr, 0,
                             at_d, seed1_d.get_ptr(), seed1_d.get_pitch());
        sum1 += us;
        ++cnt1;
      }
    }
    const double mean0 = sum0 / 1000.0, mean1 = sum1 / 1000.0;
    const double diff  = (mean0 > mean1) ? (mean0 - mean1) : (mean1 - mean0);
    std::fprintf(stderr, "[pilot] class0 mean: %.3f µs\n", mean0);
    std::fprintf(stderr, "[pilot] class1 mean: %.3f µs\n", mean1);
    std::fprintf(stderr, "[pilot] diff: %.3f µs — %s\n\n", diff,
                 diff > 2.0 ? "SIGNIFICANT (proceeding anyway — this IS what we measure)"
                            : "within 2 µs");
    std::fflush(stderr);
  }

  // ── CSV headers ───────────────────────────────────────────────────────────
  auto write_header = [&](FILE* f, int cls) {
    std::fprintf(f, "# mlkem-gpu-sec genmatrix isolation timing traces\n");
    std::fprintf(f, "# variant: %s  k=%u\n", variant_name, k);
    std::fprintf(f, "# class: %d (rho%d, fixed 32-byte seed, no decaps context)\n",
                 cls, cls);
    std::fprintf(f, "# n_traces: %u\n", ntraces_per_class);
    std::fprintf(f, "# unit: microseconds (elapsed_us)\n");
    std::fprintf(f, "trace_id,kernel_name,elapsed_us\n");
  };
  write_header(out0, 0);
  write_header(out1, 1);
  std::fflush(out0);
  std::fflush(out1);

  // ── Interleaved collection ────────────────────────────────────────────────
  unsigned cnt0 = 0, cnt1 = 0;
  unsigned last_pct = 0;
  while (cnt0 < ntraces_per_class || cnt1 < ntraces_per_class) {
    if (cnt0 < ntraces_per_class) {
      LAUNCH_TIMED("generate_at", generate_at, stream, ev_start, ev_stop,
                   true, out0, cnt0,
                   at_d, seed0_d.get_ptr(), seed0_d.get_pitch());
      ++cnt0;
    }
    if (cnt1 < ntraces_per_class) {
      LAUNCH_TIMED("generate_at", generate_at, stream, ev_start, ev_stop,
                   true, out1, cnt1,
                   at_d, seed1_d.get_ptr(), seed1_d.get_pitch());
      ++cnt1;
    }
    const unsigned total = cnt0 + cnt1;
    const unsigned pct   = total * 100 / (ntraces_per_class * 2);
    if (pct / 10 > last_pct / 10) {
      std::fprintf(stderr, "  %u%% (class0: %u  class1: %u)\n", pct, cnt0, cnt1);
      std::fflush(stderr);
      last_pct = pct;
    }
  }

  std::fprintf(stderr,
               "[genmatrix] collection done — class0: %u traces  class1: %u traces\n",
               cnt0, cnt1);
  std::fflush(stderr);

  // ── Cleanup ───────────────────────────────────────────────────────────────
  CCC(cudaFree(at_d));
  CCC(cudaFree(counts_d));
  std::fclose(out0);
  std::fclose(out1);
  CCC(cudaEventDestroy(ev_start));
  CCC(cudaEventDestroy(ev_stop));
}

}  // namespace atpqc_cuda::kyber::trace_genmatrix

int main(int argc, char** argv) {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::trace_genmatrix::trace_genmatrix_fn(argc, argv);
    CCC(cuCtxSynchronize());
  }

  return 0;
}
