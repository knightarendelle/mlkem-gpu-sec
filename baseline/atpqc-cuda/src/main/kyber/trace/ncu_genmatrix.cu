//
// ncu_genmatrix.cu
// Single-class generate_at profiler for NCU hardware cache-metric analysis.
//
// Runs generate_at (SHAKE128-based matrix generation) <ntraces> times for
// one class.  No warmup, no interleaving, no output — clean kernel launches.
//
// Both classes use independent uniform-random 32-byte rho seeds (matching the
// semantics of trace_genmatrix.cu).  In the full Kyber pipeline both classes
// use the SAME rho (fixed public key), so any intrinsic metric difference here
// is due to rejection-sampling variance across seeds, not ciphertext class.
// Expected result: metrics essentially identical between classes, confirming
// that generate_at's pipeline leakage comes from upstream dirty-line context.
//
// stdin: ninputs genmat_nw genvec_nw genpoly_nw fips_nw ntraces ct_class
// (genvec_nw, genpoly_nw, fips_nw are read but unused)
// ct_class 0 → rho0 (one random 32-byte seed)
// ct_class 1 → rho1 (independent random 32-byte seed)
//
// Usage:
//   echo "1 4 4 4 4 1000 0" | ncu --metrics lts__t_sectors_op_read.sum,... \
//     --csv --kernel-name-base demangled ./target/ncu_genmatrix_kyber1024.out
// Build:
//   make -C baseline/atpqc-cuda ncu_genmatrix_kyber1024
//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "../../../lib/cuda_debug.hpp"
#include "../../../lib/cuda_resource.hpp"
#include "../../../lib/kyber/genpoly_warp/host.cuh"
#include "../../../lib/kyber/params.cuh"
#include "../../../lib/kyber/variants.cuh"
#include "../../../lib/rng/std_random_device.hpp"

#ifndef KYBER_VARIANT
#define KYBER_VARIANT kyber1024
#endif

#undef CCC
static inline cudaError_t gm_check(cudaError_t e, const char* c,
                                    const char* f, int l) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[NCU_GENMATRIX CUDA ERROR] %s:%d  %s: %s\n",
                 f, l, c, cudaGetErrorString(e));
    std::fflush(stderr);
    std::abort();
  }
  return e;
}
static inline CUresult gm_check(CUresult e, const char* c,
                                  const char* f, int l) {
  if (e != CUDA_SUCCESS) {
    const char *n = nullptr, *s = nullptr;
    cuGetErrorName(e, &n);
    cuGetErrorString(e, &s);
    std::fprintf(stderr, "[NCU_GENMATRIX CUDA ERROR] %s:%d  %s: %s (%s)\n",
                 f, l, c, s ? s : "?", n ? n : "?");
    std::fflush(stderr);
    std::abort();
  }
  return e;
}
#define CCC(call) gm_check((call), #call, __FILE__, __LINE__)

namespace atpqc_cuda::kyber::ncu_genmatrix {

using variant = variants::KYBER_VARIANT;

void run(int /*argc*/, char** /*argv*/) {
  unsigned ninputs, genmat_nw, genvec_nw, genpoly_nw, fips_nw;
  unsigned ntraces, ct_class;
  std::cin >> ninputs >> genmat_nw >> genvec_nw >> genpoly_nw >> fips_nw
           >> ntraces >> ct_class;
  (void)genvec_nw; (void)genpoly_nw; (void)fips_nw;

  constexpr unsigned k = params::k<variant>;
  const char* variant_name =
      params::ciphertextbytes<variant> == 1568 ? "Kyber-1024" :
      params::ciphertextbytes<variant> == 1088 ? "Kyber-768"  : "Kyber-512";

  std::fprintf(stderr,
               "[ncu_genmatrix] %s  k=%u  ninputs=%u  ntraces=%u  ct_class=%u\n",
               variant_name, k, ninputs, ntraces, ct_class);
  std::fflush(stderr);

  // ── Two independent random rho seeds ─────────────────────────────────────
  rng::std_random_device randombytes;
  std::uint8_t rho0[params::symbytes], rho1[params::symbytes];
  randombytes(rho0, params::symbytes);
  randombytes(rho1, params::symbytes);

  std::fprintf(stderr, "[ncu_genmatrix] rho0: %02x%02x%02x%02x...\n",
               rho0[0], rho0[1], rho0[2], rho0[3]);
  std::fprintf(stderr, "[ncu_genmatrix] rho1: %02x%02x%02x%02x...\n",
               rho1[0], rho1[1], rho1[2], rho1[3]);
  std::fflush(stderr);

  // ── Device seed buffers ───────────────────────────────────────────────────
  cuda_resource::device_pitched_memory<std::uint8_t> seed0_d(params::symbytes, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> seed1_d(params::symbytes, ninputs);

  CCC(cudaMemcpy2D(seed0_d.get_ptr(), seed0_d.get_pitch(),
                   rho0, params::symbytes,
                   params::symbytes, ninputs, cudaMemcpyHostToDevice));
  CCC(cudaMemcpy2D(seed1_d.get_ptr(), seed1_d.get_pitch(),
                   rho1, params::symbytes,
                   params::symbytes, ninputs, cudaMemcpyHostToDevice));
  CCC(cudaDeviceSynchronize());

  // ── Output buffer for the generated matrix ────────────────────────────────
  const unsigned at_nelems = k * k * (params::n / 2) * ninputs;
  short2* at_d = nullptr;
  CCC(cudaMalloc(&at_d, at_nelems * sizeof(short2)));

  // ── Kernel object and stream ──────────────────────────────────────────────
  genpoly_warp::host::genat<k> generate_at(ninputs, genmat_nw);
  cuda_resource::stream stream(cudaStreamNonBlocking);

  // Select seed based on class
  const std::uint8_t* seed_ptr =
      (ct_class == 0) ? seed0_d.get_ptr() : seed1_d.get_ptr();
  const std::size_t seed_pitch =
      (ct_class == 0) ? seed0_d.get_pitch() : seed1_d.get_pitch();

  // ── Measurement loop: ntraces sequential single-class launches ────────────
  std::fprintf(stderr,
               "[ncu_genmatrix] measuring: %u generate_at invocations (class %u)...\n",
               ntraces, ct_class);
  std::fflush(stderr);

  for (unsigned i = 0; i < ntraces; ++i) {
    auto _a = generate_at.generate_args(at_d, seed_ptr, seed_pitch);
    CCC(cudaLaunchKernel(
        generate_at.get_func(),
        generate_at.get_grid_dim(),
        generate_at.get_block_dim(),
        _a->get_args_ptr(),
        generate_at.get_shared_bytes(),
        stream));
    CCC(cudaStreamSynchronize(stream));
  }
  CCC(cudaDeviceSynchronize());
  std::fprintf(stderr, "[ncu_genmatrix] done.\n");
  std::fflush(stderr);

  CCC(cudaFree(at_d));
}

}  // namespace atpqc_cuda::kyber::ncu_genmatrix

int main(int argc, char** argv) {
  CUDA_DEBUG_RESET();
  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));
  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::ncu_genmatrix::run(argc, argv);
    CCC(cuCtxSynchronize());
  }
  return 0;
}
