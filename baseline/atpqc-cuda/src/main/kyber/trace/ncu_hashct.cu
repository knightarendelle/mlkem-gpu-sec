//
// ncu_hashct.cu
// Single-class hash_ct profiler for NCU hardware cache-metric analysis.
//
// Runs hash_ct (SHA3-256 over the Kyber-1024 ciphertext) <ntraces> times
// for one class only — clean sequential kernel launches suitable for NCU.
// No warmup, no interleaving, no output files.
//
// ct_class 0 → valid Kyber-1024 ciphertext (structured polynomial data)
// ct_class 1 → uniform-random 1568-byte string
//
// stdin: ninputs genmat_nw genvec_nw genpoly_nw fips_nw ntraces ct_class
// Usage:
//   echo "1 4 4 4 4 1000 0" | ncu --metrics lts__t_sectors_op_read.sum,... \
//     --csv --kernel-name-base demangled ./target/ncu_hashct_kyber1024.out
// Build:
//   make -C baseline/atpqc-cuda ncu_hashct_kyber1024
//

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
static inline cudaError_t hc_check(cudaError_t e, const char* c,
                                    const char* f, int l) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[NCU_HASHCT CUDA ERROR] %s:%d  %s: %s\n",
                 f, l, c, cudaGetErrorString(e));
    std::fflush(stderr);
    std::abort();
  }
  return e;
}
static inline CUresult hc_check(CUresult e, const char* c,
                                  const char* f, int l) {
  if (e != CUDA_SUCCESS) {
    const char *n = nullptr, *s = nullptr;
    cuGetErrorName(e, &n);
    cuGetErrorString(e, &s);
    std::fprintf(stderr, "[NCU_HASHCT CUDA ERROR] %s:%d  %s: %s (%s)\n",
                 f, l, c, s ? s : "?", n ? n : "?");
    std::fflush(stderr);
    std::abort();
  }
  return e;
}
#define CCC(call) hc_check((call), #call, __FILE__, __LINE__)

namespace atpqc_cuda::kyber::ncu_hashct {

using rng_type = rng::std_random_device;
using variant  = variants::KYBER_VARIANT;
constexpr variant variant_v;

void run(int /*argc*/, char** /*argv*/) {
  unsigned ninputs, genmat_nw, genvec_nw, genpoly_nw, fips_nw;
  unsigned ntraces, ct_class;
  std::cin >> ninputs >> genmat_nw >> genvec_nw >> genpoly_nw >> fips_nw
           >> ntraces >> ct_class;

  constexpr unsigned ctbytes = params::ciphertextbytes<variant>;
  const unsigned kyber_bits =
      ctbytes == 1568 ? 1024u : ctbytes == 1088 ? 768u : 512u;

  std::fprintf(stderr,
               "[ncu_hashct] Kyber-%u  ctbytes=%u  ninputs=%u"
               "  ntraces=%u  ct_class=%u\n",
               kyber_bits, ctbytes, ninputs, ntraces, ct_class);
  std::fflush(stderr);

  rng_type randombytes;

  // ── Kernel objects for keypair+enc setup (class 0 only) ───────────────────
  symmetric_ws::host::hash_g hash_seed(ninputs, fips_nw);
  genpoly_warp::host::gena<params::k<variant>>    generate_a(ninputs, genmat_nw);
  genpoly_warp::host::genat<params::k<variant>>   generate_at_obj(ninputs, genmat_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>>
                                                  generate_s(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>>
                                                  generate_e(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>>
                                                  generate_r(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta2>
                                                  generate_e1(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<1, params::eta2>   generate_e2(ninputs, genpoly_nw);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>>        fwdnttvec_s(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>>        fwdnttvec_e(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>>        fwdnttvec_r(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<params::k<variant>> intt_ar(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<1>                  intt_tr(ninputs);
  arithmetic_mt::host::mattimesvec_tomont_plusvec<params::k<variant>> mtvpv(ninputs);
  arithmetic_mt::host::mattimesvec<params::k<variant>>  mtv(ninputs);
  arithmetic_mt::host::vectimesvec<params::k<variant>>  ttimesr(ninputs);
  arithmetic_mt::host::vecadd2<params::k<variant>>      vpv(ninputs);
  arithmetic_mt::host::polyadd3                         padd3(ninputs);
  endecode_mt::host::polyvec_tobytes<params::k<variant>>              encodet(ninputs);
  endecode_mt::host::polyvec_tobytes<params::k<variant>>              encodes(ninputs);
  endecode_mt::host::polyvec_frombytes<params::k<variant>>            decodet(ninputs);
  endecode_mt::host::poly_frommsg                                     frommsg(ninputs);
  endecode_mt::host::polyvec_compress<params::k<variant>, params::du<variant>>
                                                                      compressu(ninputs);
  endecode_mt::host::poly_compress<params::dv<variant>>               compressv(ninputs);
  symmetric_ws::host::hash_h keypair_hash_pk(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_rand(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_pk(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_ct_inner(ninputs, fips_nw);
  symmetric_ws::host::hash_g enc_hash_coin(ninputs, fips_nw);
  symmetric_ws::host::kdf    enc_kdf(ninputs, fips_nw);

  primitive::ccakem_keypair::keypair keypair_obj(
      ninputs, variant_v,
      primitive::cpapke_keypair::cpapke_keypair(
          ninputs, variant_v, randombytes, hash_seed, generate_a, generate_s,
          generate_e, fwdnttvec_s, fwdnttvec_e, mtvpv, encodet, encodes),
      randombytes, keypair_hash_pk);

  primitive::ccakem_enc::enc enc_obj(
      ninputs, variant_v,
      primitive::cpapke_enc::cpapke_enc(
          ninputs, variant_v, generate_at_obj, generate_r, generate_e1, generate_e2,
          fwdnttvec_r, intt_ar, intt_tr, mtv, ttimesr, vpv, padd3, decodet,
          frommsg, compressu, compressv),
      randombytes, enc_hash_rand, enc_hash_pk, enc_hash_ct_inner,
      enc_hash_coin, enc_kdf);

  // ── Device memory ─────────────────────────────────────────────────────────
  cuda_resource::device_pitched_memory<std::uint8_t>
      pk_d(params::publickeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t>
      sk_d(params::secretkeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t>
      ss_d(params::ssbytes, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t>
      ct_d(ctbytes, ninputs);   // the input buffer for hash_ct
  cuda_resource::device_pitched_memory<std::uint8_t>
      hash_d(32, ninputs);      // SHA3-256 output (32 bytes)

  primitive::ccakem_keypair::mem_resource<variant> keypair_mr(ninputs);
  primitive::ccakem_enc::mem_resource<variant>     enc_mr(ninputs);

  if (ct_class == 0) {
    // ── Class 0: produce one real Kyber ciphertext via keypair + enc ─────────
    std::fprintf(stderr, "[ncu_hashct] class 0: generating keypair...\n");
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
      (void)kr_args;
    }
    std::fprintf(stderr, "[ncu_hashct] class 0: generating valid ciphertext...\n");
    std::fflush(stderr);
    {
      cuda_resource::graph enc_graph;
      cudaGraphNode_t dummy_e, ct_avail, ss_avail, pk_used;
      CCC(cudaGraphAddEmptyNode(&dummy_e, enc_graph, nullptr, 0));
      randombytes(enc_mr.rand_host.get_ptr(), params::symbytes * ninputs);
      auto er_args = enc_obj.join_graph(
          enc_graph,
          ct_d.get_ptr(), ct_d.get_pitch(), dummy_e, &ct_avail,
          ss_d.get_ptr(), ss_d.get_pitch(), dummy_e, &ss_avail,
          pk_d.get_ptr(), pk_d.get_pitch(), dummy_e, &pk_used,
          enc_mr);
      cuda_resource::graph_exec enc_exec(enc_graph);
      cuda_resource::stream     enc_stream(cudaStreamNonBlocking);
      CCC(cudaGraphLaunch(enc_exec, enc_stream));
      CCC(cudaStreamSynchronize(enc_stream));
      (void)er_args;
    }
    std::fprintf(stderr, "[ncu_hashct] class 0: valid ciphertext ready in ct_d.\n");
  } else {
    // ── Class 1: fill ct_d with uniform random bytes — no graph launch ────────
    std::vector<std::uint8_t> rand_host(ctbytes * ninputs);
    randombytes(rand_host.data(), rand_host.size());
    CCC(cudaMemcpy2D(ct_d.get_ptr(), ct_d.get_pitch(),
                     rand_host.data(), ctbytes,
                     ctbytes, ninputs, cudaMemcpyHostToDevice));
    CCC(cudaDeviceSynchronize());
    std::fprintf(stderr, "[ncu_hashct] class 1: random bytes ready in ct_d.\n");
  }
  std::fflush(stderr);

  // ── Kernel under test ─────────────────────────────────────────────────────
  symmetric_ws::host::hash_h hash_ct_kernel(ninputs, fips_nw);
  cuda_resource::stream stream(cudaStreamNonBlocking);

  // ── Measurement loop: ntraces sequential single-class launches ────────────
  std::fprintf(stderr,
               "[ncu_hashct] measuring: %u hash_ct invocations (class %u)...\n",
               ntraces, ct_class);
  std::fflush(stderr);

  for (unsigned i = 0; i < ntraces; ++i) {
    auto _a = hash_ct_kernel.generate_args(
        hash_d.get_ptr(), hash_d.get_pitch(),
        ct_d.get_ptr(), ct_d.get_pitch(), ctbytes);
    CCC(cudaLaunchKernel(
        hash_ct_kernel.get_func(),
        hash_ct_kernel.get_grid_dim(),
        hash_ct_kernel.get_block_dim(),
        _a->get_args_ptr(),
        hash_ct_kernel.get_shared_bytes(),
        stream));
    CCC(cudaStreamSynchronize(stream));
  }
  CCC(cudaDeviceSynchronize());
  std::fprintf(stderr, "[ncu_hashct] done.\n");
  std::fflush(stderr);
}

}  // namespace atpqc_cuda::kyber::ncu_hashct

int main(int argc, char** argv) {
  CUDA_DEBUG_RESET();
  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));
  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::ncu_hashct::run(argc, argv);
    CCC(cuCtxSynchronize());
  }
  return 0;
}
