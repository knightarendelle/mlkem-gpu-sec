//
// trace_hashct.cu
// Isolates hash_ct (SHA3-256 over 1568-byte ciphertext) from Kyber-1024.
//
// Class 0: SHA3-256 over a single valid Kyber-1024 ciphertext (generated once,
//          fixed for all traces — structured compressed polynomial data)
// Class 1: SHA3-256 over a single random 1568-byte string (also fixed)
// Interleaved in one run to eliminate GPU thermal/clock drift.
//
// If |t| >= 4.5: hashing structured vs random data takes different time,
//   consistent with DCC (Data Compression Cache) treating them differently.
// If |t| < 4.5: hash_ct is clean; timing in the full pipeline is from
//   ciphertext being loaded into context before hash_ct executes.
//
// Usage:
//   echo "1 4 4 4 4 100000" | ./target/trace_hashct_kyber1024.out \
//     experiments/traces/per_kernel/hashct_class0_n100000.csv \
//     experiments/traces/per_kernel/hashct_class1_n100000.csv
//
// Build:
//   make -C baseline/atpqc-cuda trace_hashct_kyber1024
//

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "../../../lib/cuda_debug.hpp"
#include "../../../lib/cuda_resource.hpp"
#include "../../../lib/fips202_ws/host.cuh"
#include "../../../lib/kyber/arithmetic_mt/host.cuh"
#include "../../../lib/kyber/endecode_mt/host.cuh"
#include "../../../lib/kyber/genpoly_warp/host.cuh"
#include "../../../lib/kyber/ntt_ctgs_128t/host.cuh"
#include "../../../lib/kyber/ntt_ctgs_64t/host.cuh"
#include "../../../lib/kyber/params.cuh"
#include "../../../lib/kyber/primitive/ccakem_enc.cuh"
#include "../../../lib/kyber/primitive/ccakem_keypair.cuh"
#include "../../../lib/kyber/primitive/cpapke_enc.cuh"
#include "../../../lib/kyber/primitive/cpapke_keypair.cuh"
#include "../../../lib/kyber/symmetric_ws/host.cuh"
#include "../../../lib/kyber/variants.cuh"
#include "../../../lib/rng/std_random_device.hpp"

#ifndef KYBER_VARIANT
#define KYBER_VARIANT kyber1024
#endif

#undef CCC
static inline cudaError_t hc_check(cudaError_t e, const char* call,
                                    const char* file, int line) {
  if (e != cudaSuccess) {
    std::fprintf(stderr,
                 "[HASHCT CUDA ERROR] %s:%d\n  call: %s\n  error: %s\n",
                 file, line, call, cudaGetErrorString(e));
    std::fflush(stderr);
    std::abort();
  }
  return e;
}
static inline CUresult hc_check(CUresult e, const char* call,
                                  const char* file, int line) {
  if (e != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* str  = nullptr;
    cuGetErrorName(e, &name);
    cuGetErrorString(e, &str);
    std::fprintf(stderr,
                 "[HASHCT CUDA ERROR] %s:%d\n  call: %s\n  error: %s (%s)\n",
                 file, line, call, str ? str : "?", name ? name : "?");
    std::fflush(stderr);
    std::abort();
  }
  return e;
}
#define CCC(call) hc_check((call), #call, __FILE__, __LINE__)

namespace atpqc_cuda::kyber::trace_hashct {

using rng_type = rng::std_random_device;
using variant  = variants::KYBER_VARIANT;
constexpr variant variant_v;

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

void trace_hashct_fn(int argc, char** argv) {
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
    std::fprintf(stderr, "[hashct] failed to open output files: %s %s\n",
                 argv[1], argv[2]);
    std::abort();
  }

  unsigned ninputs, genmat_nw, genvec_nw, genpoly_nw, fips_nw;
  unsigned ntraces_per_class;
  std::cin >> ninputs >> genmat_nw >> genvec_nw >> genpoly_nw >> fips_nw
           >> ntraces_per_class;

  constexpr unsigned ctbytes = params::ciphertextbytes<variant>;

  std::fprintf(stderr,
               "[hashct] variant: Kyber-%u  ctbytes=%u  ninputs=%u"
               "  ntraces_per_class=%u\n",
               ctbytes == 1568 ? 1024u : ctbytes == 1088 ? 768u : 512u,
               ctbytes, ninputs, ntraces_per_class);
  std::fprintf(stderr, "[hashct] out0: %s\n[hashct] out1: %s\n",
               argv[1], argv[2]);
  std::fflush(stderr);

  // ── Kernel objects for keypair + enc ──────────────────────────────────────
  // Needed only to produce one valid ciphertext for class 0.
  rng_type randombytes;
  symmetric_ws::host::hash_g hash_seed(ninputs, fips_nw);
  genpoly_warp::host::gena<params::k<variant>>    generate_a(ninputs, genmat_nw);
  genpoly_warp::host::genat<params::k<variant>>   generate_at(ninputs, genmat_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>> generate_s(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>> generate_e(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>> generate_r(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta2>          generate_e1(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<1, params::eta2>                           generate_e2(ninputs, genpoly_nw);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>>        fwdnttvec_s(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>>        fwdnttvec_e(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>>        fwdnttvec_r(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<params::k<variant>> intt_ar(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<1>                  intt_tr(ninputs);
  arithmetic_mt::host::mattimesvec_tomont_plusvec<params::k<variant>> mtvpv(ninputs);
  arithmetic_mt::host::mattimesvec<params::k<variant>>  mtv(ninputs);
  arithmetic_mt::host::vectimesvec<params::k<variant>>  ttimesr(ninputs);
  arithmetic_mt::host::vecadd2<params::k<variant>>       vpv(ninputs);
  arithmetic_mt::host::polyadd3                          padd3(ninputs);
  endecode_mt::host::polyvec_tobytes<params::k<variant>>               encodet(ninputs);
  endecode_mt::host::polyvec_tobytes<params::k<variant>>               encodes(ninputs);
  endecode_mt::host::polyvec_frombytes<params::k<variant>>             decodet(ninputs);
  endecode_mt::host::poly_frommsg                                      frommsg(ninputs);
  endecode_mt::host::polyvec_compress<params::k<variant>, params::du<variant>> compressu(ninputs);
  endecode_mt::host::poly_compress<params::dv<variant>>                compressv(ninputs);
  symmetric_ws::host::hash_h keypair_hash_pk(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_rand(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_pk(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_ct_inner(ninputs, fips_nw);
  symmetric_ws::host::hash_g enc_hash_coin(ninputs, fips_nw);
  symmetric_ws::host::kdf    enc_kdf(ninputs, fips_nw);

  // The kernel under test
  symmetric_ws::host::hash_h hash_ct_kernel(ninputs, fips_nw);

  primitive::ccakem_keypair::keypair keypair_obj(
      ninputs, variant_v,
      primitive::cpapke_keypair::cpapke_keypair(
          ninputs, variant_v, randombytes, hash_seed, generate_a, generate_s,
          generate_e, fwdnttvec_s, fwdnttvec_e, mtvpv, encodet, encodes),
      randombytes, keypair_hash_pk);

  primitive::ccakem_enc::enc enc_obj(
      ninputs, variant_v,
      primitive::cpapke_enc::cpapke_enc(
          ninputs, variant_v, generate_at, generate_r, generate_e1, generate_e2,
          fwdnttvec_r, intt_ar, intt_tr, mtv, ttimesr, vpv, padd3, decodet,
          frommsg, compressu, compressv),
      randombytes, enc_hash_rand, enc_hash_pk, enc_hash_ct_inner,
      enc_hash_coin, enc_kdf);

  // ── Device memory ─────────────────────────────────────────────────────────
  cuda_resource::device_pitched_memory<std::uint8_t> pk_d(params::publickeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> sk_d(params::secretkeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ss_d(params::ssbytes,   ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ct0_d(ctbytes, ninputs);  // valid ct
  cuda_resource::device_pitched_memory<std::uint8_t> ct1_d(ctbytes, ninputs);  // random bytes
  cuda_resource::device_pitched_memory<std::uint8_t> hash_d(32,     ninputs);  // output (reused)

  primitive::ccakem_keypair::mem_resource<variant> keypair_mr(ninputs);
  primitive::ccakem_enc::mem_resource<variant>     enc_mr(ninputs);

  // ── Generate one keypair ───────────────────────────────────────────────────
  std::fprintf(stderr, "[hashct] generating keypair...\n");
  std::fflush(stderr);
  {
    cuda_resource::graph kp_graph;
    cudaGraphNode_t dummy, pk_avail, sk_avail;
    CCC(cudaGraphAddEmptyNode(&dummy, kp_graph, nullptr, 0));
    randombytes(keypair_mr.pke_keypair_mr.rand_host.get_ptr(),
                params::symbytes * ninputs);
    randombytes(keypair_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    auto kr_args = keypair_obj.join_graph(
        kp_graph,
        pk_d.get_ptr(), pk_d.get_pitch(), dummy, &pk_avail,
        sk_d.get_ptr(), sk_d.get_pitch(), dummy, &sk_avail,
        keypair_mr);
    cuda_resource::graph_exec kp_exec(kp_graph);
    cuda_resource::stream     kp_stream(cudaStreamNonBlocking);
    CCC(cudaGraphLaunch(kp_exec, kp_stream));
    CCC(cudaStreamSynchronize(kp_stream));
  }
  std::fprintf(stderr, "[hashct] keypair done\n");

  // ── Generate one valid ciphertext into ct0_d ───────────────────────────────
  std::fprintf(stderr, "[hashct] generating valid ciphertext...\n");
  std::fflush(stderr);
  {
    cuda_resource::graph enc_graph;
    cudaGraphNode_t dummy_e, ct_avail, ss_avail, pk_used;
    CCC(cudaGraphAddEmptyNode(&dummy_e, enc_graph, nullptr, 0));
    randombytes(enc_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    auto er_args = enc_obj.join_graph(
        enc_graph,
        ct0_d.get_ptr(), ct0_d.get_pitch(), dummy_e, &ct_avail,
        ss_d.get_ptr(),  ss_d.get_pitch(),  dummy_e, &ss_avail,
        pk_d.get_ptr(),  pk_d.get_pitch(),  dummy_e, &pk_used,
        enc_mr);
    cuda_resource::graph_exec enc_exec(enc_graph);
    cuda_resource::stream     enc_stream(cudaStreamNonBlocking);
    CCC(cudaGraphLaunch(enc_exec, enc_stream));
    CCC(cudaStreamSynchronize(enc_stream));
  }
  std::fprintf(stderr, "[hashct] valid ciphertext ready in ct0_d\n");

  // ── Fill ct1_d with random bytes ───────────────────────────────────────────
  {
    std::vector<std::uint8_t> rand_host(ctbytes * ninputs);
    randombytes(rand_host.data(), rand_host.size());
    CCC(cudaMemcpy2D(ct1_d.get_ptr(), ct1_d.get_pitch(),
                     rand_host.data(), ctbytes,
                     ctbytes, ninputs, cudaMemcpyHostToDevice));
    CCC(cudaDeviceSynchronize());
  }
  std::fprintf(stderr, "[hashct] random bytes ready in ct1_d\n");
  std::fflush(stderr);

  // ── CUDA stream and events ─────────────────────────────────────────────────
  cuda_resource::stream stream(cudaStreamNonBlocking);
  cudaEvent_t ev_start, ev_stop;
  CCC(cudaEventCreate(&ev_start));
  CCC(cudaEventCreate(&ev_stop));

  // ── Warmup: 200 alternating calls ─────────────────────────────────────────
  std::fprintf(stderr, "[warmup] 200 iterations (interleaved)...\n");
  float _wu = 0.f;
  for (unsigned w = 0; w < 200; ++w) {
    if (w % 2 == 0)
      LAUNCH_TIMED_CAPTURE("hash_ct", _wu, hash_ct_kernel, stream,
                           ev_start, ev_stop, false, stderr, 0,
                           hash_d.get_ptr(), hash_d.get_pitch(),
                           ct0_d.get_ptr(), ct0_d.get_pitch(), ctbytes);
    else
      LAUNCH_TIMED_CAPTURE("hash_ct", _wu, hash_ct_kernel, stream,
                           ev_start, ev_stop, false, stderr, 0,
                           hash_d.get_ptr(), hash_d.get_pitch(),
                           ct1_d.get_ptr(), ct1_d.get_pitch(), ctbytes);
  }
  std::fprintf(stderr, "[warmup] done\n");

  // ── Pilot: 1000 per class, interleaved ────────────────────────────────────
  {
    std::fprintf(stderr, "[pilot] 1000 traces per class (interleaved)...\n");
    double sum0 = 0.0, sum1 = 0.0;
    unsigned cnt0 = 0, cnt1 = 0;
    float us = 0.f;
    while (cnt0 < 1000 || cnt1 < 1000) {
      if (cnt0 < 1000) {
        LAUNCH_TIMED_CAPTURE("hash_ct", us, hash_ct_kernel, stream,
                             ev_start, ev_stop, false, stderr, 0,
                             hash_d.get_ptr(), hash_d.get_pitch(),
                             ct0_d.get_ptr(), ct0_d.get_pitch(), ctbytes);
        sum0 += us;
        ++cnt0;
      }
      if (cnt1 < 1000) {
        LAUNCH_TIMED_CAPTURE("hash_ct", us, hash_ct_kernel, stream,
                             ev_start, ev_stop, false, stderr, 0,
                             hash_d.get_ptr(), hash_d.get_pitch(),
                             ct1_d.get_ptr(), ct1_d.get_pitch(), ctbytes);
        sum1 += us;
        ++cnt1;
      }
    }
    const double mean0 = sum0 / 1000.0, mean1 = sum1 / 1000.0;
    const double diff  = (mean0 > mean1) ? (mean0 - mean1) : (mean1 - mean0);
    std::fprintf(stderr, "[pilot] class0 (valid ct) mean: %.3f µs\n", mean0);
    std::fprintf(stderr, "[pilot] class1 (random)   mean: %.3f µs\n", mean1);
    std::fprintf(stderr, "[pilot] diff: %.3f µs\n\n", diff);
    std::fflush(stderr);
  }

  // ── CSV headers ───────────────────────────────────────────────────────────
  auto write_header = [&](FILE* f, int cls) {
    std::fprintf(f, "# mlkem-gpu-sec hash_ct isolation timing traces\n");
    std::fprintf(f, "# variant: Kyber-1024  ctbytes=%u\n", ctbytes);
    std::fprintf(f, "# class: %d (%s, fixed input for all traces)\n",
                 cls, cls == 0 ? "valid ciphertext" : "random bytes");
    std::fprintf(f, "# n_traces: %u\n", ntraces_per_class);
    std::fprintf(f, "# unit: microseconds (elapsed_us)\n");
    std::fprintf(f, "trace_id,kernel_name,elapsed_us\n");
  };
  write_header(out0, 0);
  write_header(out1, 1);
  std::fflush(out0);
  std::fflush(out1);

  // ── Interleaved collection ─────────────────────────────────────────────────
  unsigned cnt0 = 0, cnt1 = 0;
  float us = 0.f;
  unsigned last_pct = 0;
  while (cnt0 < ntraces_per_class || cnt1 < ntraces_per_class) {
    if (cnt0 < ntraces_per_class) {
      LAUNCH_TIMED_CAPTURE("hash_ct", us, hash_ct_kernel, stream,
                           ev_start, ev_stop, true, out0, cnt0,
                           hash_d.get_ptr(), hash_d.get_pitch(),
                           ct0_d.get_ptr(), ct0_d.get_pitch(), ctbytes);
      ++cnt0;
    }
    if (cnt1 < ntraces_per_class) {
      LAUNCH_TIMED_CAPTURE("hash_ct", us, hash_ct_kernel, stream,
                           ev_start, ev_stop, true, out1, cnt1,
                           hash_d.get_ptr(), hash_d.get_pitch(),
                           ct1_d.get_ptr(), ct1_d.get_pitch(), ctbytes);
      ++cnt1;
    }
    const unsigned total   = cnt0 + cnt1;
    const unsigned pct     = total * 100 / (ntraces_per_class * 2);
    if (pct / 10 > last_pct / 10) {
      std::fprintf(stderr, "  %u%% (class0: %u  class1: %u)\n", pct, cnt0, cnt1);
      std::fflush(stderr);
      last_pct = pct;
    }
  }

  std::fprintf(stderr,
               "[hashct] done — class0: %u traces  class1: %u traces\n",
               cnt0, cnt1);
  std::fflush(stderr);

  std::fclose(out0);
  std::fclose(out1);
  CCC(cudaEventDestroy(ev_start));
  CCC(cudaEventDestroy(ev_stop));
}

}  // namespace atpqc_cuda::kyber::trace_hashct

int main(int argc, char** argv) {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::trace_hashct::trace_hashct_fn(argc, argv);
    CCC(cuCtxSynchronize());
  }

  return 0;
}
