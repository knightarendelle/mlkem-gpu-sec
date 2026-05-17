//
// bench_l2thrash.cu
// Throughput comparison: unmitigated vs. L2-thrash-mitigated decapsulation.
//
// Runs each mode for `duration_sec` wall-clock seconds, counts iterations,
// and prints ops/sec + percentage overhead.
//
// stdin:  ninputs genmat_nw genvec_nw genpoly_nw fips_nw [duration_sec]
//         (duration_sec defaults to 10 if omitted)
//
// stdout: plaintext table (variant, unmit_ops_sec, mit_ops_sec, overhead_pct)
//
// Build:
//   make -C baseline/atpqc-cuda bench_l2thrash
// Run:
//   echo "1 4 4 4 4" | ./target/bench_l2thrash_kyber1024.out
//

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "../../../lib/cuda_debug.hpp"
#include "../../../lib/cuda_resource.hpp"
#include "../../../lib/fips202_ws/host.cuh"
#include "../../../lib/kyber/arithmetic_mt/host.cuh"
#include "../../../lib/kyber/endecode_mt/host.cuh"
#include "../../../lib/kyber/genpoly_warp/host.cuh"
#include "../../../lib/kyber/ntt_ctgs_128t/host.cuh"
#include "../../../lib/kyber/ntt_ctgs_64t/host.cuh"
#include "../../../lib/kyber/params.cuh"
#include "../../../lib/kyber/primitive/ccakem_dec.cuh"
#include "../../../lib/kyber/primitive/ccakem_enc.cuh"
#include "../../../lib/kyber/primitive/ccakem_keypair.cuh"
#include "../../../lib/kyber/primitive/cpapke_dec.cuh"
#include "../../../lib/kyber/primitive/cpapke_enc.cuh"
#include "../../../lib/kyber/primitive/cpapke_keypair.cuh"
#include "../../../lib/kyber/symmetric_ws/host.cuh"
#include "../../../lib/kyber/variants.cuh"
#include "../../../lib/rng/std_random_device.hpp"
#include "../../../lib/verify_cmov_ws/host.cuh"

#ifndef KYBER_VARIANT
#define KYBER_VARIANT kyber512
#endif

#undef CCC
static inline cudaError_t blt_check(cudaError_t e, const char* call,
                                     const char* file, int line) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[BENCH_L2THRASH CUDA ERROR] %s:%d\n  call: %s\n  error: %s\n",
                 file, line, call, cudaGetErrorString(e));
    std::fflush(stderr);
    std::abort();
  }
  return e;
}
static inline CUresult blt_check(CUresult e, const char* call,
                                  const char* file, int line) {
  if (e != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* str  = nullptr;
    cuGetErrorName(e, &name);
    cuGetErrorString(e, &str);
    std::fprintf(stderr,
                 "[BENCH_L2THRASH CUDA ERROR] %s:%d\n  call: %s\n  error: %s (%s)\n",
                 file, line, call, str ? str : "?", name ? name : "?");
    std::fflush(stderr);
    std::abort();
  }
  return e;
}
#define CCC(call) blt_check((call), #call, __FILE__, __LINE__)

// Plain launch (no timing, no thrash)
#define LAUNCH_PLAIN(obj, stream, ...)                                     \
  do {                                                                     \
    auto _a = (obj).generate_args(__VA_ARGS__);                            \
    CCC(cudaLaunchKernel((obj).get_func(), (obj).get_grid_dim(),           \
                         (obj).get_block_dim(), _a->get_args_ptr(),        \
                         (obj).get_shared_bytes(), (stream)));             \
  } while (0)

// Plain launch + full L2 thrash (no timing)
#define LAUNCH_THRASH(obj, stream, thrash_buf, l2_bytes, ...)              \
  do {                                                                     \
    auto _a = (obj).generate_args(__VA_ARGS__);                            \
    CCC(cudaLaunchKernel((obj).get_func(), (obj).get_grid_dim(),           \
                         (obj).get_block_dim(), _a->get_args_ptr(),        \
                         (obj).get_shared_bytes(), (stream)));             \
    CCC(cudaDeviceSynchronize());                                          \
    CCC(cudaMemsetAsync((thrash_buf), 0xA5, (l2_bytes), (stream)));        \
    CCC(cudaDeviceSynchronize());                                          \
    CCC(cudaCtxResetPersistingL2Cache());                                  \
  } while (0)

namespace atpqc_cuda::kyber::bench_l2thrash {

using rng_type = rng::std_random_device;
using variant  = variants::KYBER_VARIANT;
constexpr variant variant_v;

void run() {
  unsigned ninputs, genmat_nw, genvec_nw, genpoly_nw, fips_nw;
  unsigned duration_sec = 10;
  std::cin >> ninputs >> genmat_nw >> genvec_nw >> genpoly_nw >> fips_nw;
  std::cin >> duration_sec;  // optional; leaves duration_sec=10 if not provided
  if (std::cin.fail()) {
    std::cin.clear();
    duration_sec = 10;
  }

  // ── L2 size ──────────────────────────────────────────────────────────────
  int l2_bytes_int = 0;
  CCC(cudaDeviceGetAttribute(&l2_bytes_int, cudaDevAttrL2CacheSize, 0));
  const std::size_t l2_bytes = static_cast<std::size_t>(l2_bytes_int) * 2;
  std::fprintf(stderr, "[bench_l2thrash] L2: %d bytes (%.1f MB); thrash buf: %.1f MB\n",
               l2_bytes_int, l2_bytes_int / (1024.0 * 1024.0),
               l2_bytes / (1024.0 * 1024.0));

  void* thrash_buf = nullptr;
  CCC(cudaMalloc(&thrash_buf, l2_bytes));

  // ── Memory ───────────────────────────────────────────────────────────────
  cuda_resource::device_pitched_memory<std::uint8_t> pk_d(params::publickeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> sk_d(params::secretkeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ct_d(params::ciphertextbytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ss_d(params::ssbytes, ninputs);

  cuda_resource::pinned_memory<std::uint8_t> pk_h(params::publickeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> sk_h(params::secretkeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ct_h(params::ciphertextbytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ss_h(params::ssbytes * ninputs);

  // ── Kernel objects ───────────────────────────────────────────────────────
  rng_type randombytes;
  symmetric_ws::host::hash_g hash_seed(ninputs, fips_nw);
  genpoly_warp::host::gena<params::k<variant>>    generate_a(ninputs, genmat_nw);
  genpoly_warp::host::genat<params::k<variant>>   generate_at(ninputs, genmat_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>> generate_s(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>> generate_e(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>> generate_r(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta2>          generate_e1(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<1, params::eta2>                           generate_e2(ninputs, genpoly_nw);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_s(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_e(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_r(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_u(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<params::k<variant>> intt_ar(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<1>                  intt_tr(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<1>                  intt_su(ninputs);
  arithmetic_mt::host::mattimesvec_tomont_plusvec<params::k<variant>> mtvpv(ninputs);
  arithmetic_mt::host::mattimesvec<params::k<variant>>  mtv(ninputs);
  arithmetic_mt::host::vectimesvec<params::k<variant>>  ttimesr(ninputs);
  arithmetic_mt::host::vectimesvec<params::k<variant>>  stimesu(ninputs);
  arithmetic_mt::host::vecadd2<params::k<variant>>       vpv(ninputs);
  arithmetic_mt::host::polyadd3                          padd3(ninputs);
  arithmetic_mt::host::polysub                           psub(ninputs);
  endecode_mt::host::polyvec_tobytes<params::k<variant>>       encodet(ninputs);
  endecode_mt::host::polyvec_tobytes<params::k<variant>>       encodes(ninputs);
  endecode_mt::host::polyvec_frombytes<params::k<variant>>     decodet(ninputs);
  endecode_mt::host::polyvec_frombytes<params::k<variant>>     decodes(ninputs);
  endecode_mt::host::poly_frommsg                              frommsg(ninputs);
  endecode_mt::host::poly_tomsg                                tomsg(ninputs);
  endecode_mt::host::polyvec_compress<params::k<variant>, params::du<variant>> compressu(ninputs);
  endecode_mt::host::poly_compress<params::dv<variant>>        compressv(ninputs);
  endecode_mt::host::polyvec_decompress<params::k<variant>, params::du<variant>> decompressu(ninputs);
  endecode_mt::host::poly_decompress<params::dv<variant>>      decompressv(ninputs);
  symmetric_ws::host::hash_h keypair_hash_pk(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_rand(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_pk(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_ct(ninputs, fips_nw);
  symmetric_ws::host::hash_g enc_hash_coin(ninputs, fips_nw);
  symmetric_ws::host::kdf    enc_kdf(ninputs, fips_nw);
  symmetric_ws::host::hash_h dec_hash_ct(ninputs, fips_nw);
  symmetric_ws::host::hash_g dec_hash_coin(ninputs, fips_nw);
  symmetric_ws::host::kdf    dec_kdf(ninputs, fips_nw);
  verify_cmov_ws::host::verify_cmov dec_verify_cmov(ninputs);

  primitive::ccakem_keypair::keypair keypair(
      ninputs, variant_v,
      primitive::cpapke_keypair::cpapke_keypair(
          ninputs, variant_v, randombytes, hash_seed, generate_a, generate_s,
          generate_e, fwdnttvec_s, fwdnttvec_e, mtvpv, encodet, encodes),
      randombytes, keypair_hash_pk);
  primitive::ccakem_enc::enc enc(
      ninputs, variant_v,
      primitive::cpapke_enc::cpapke_enc(
          ninputs, variant_v, generate_at, generate_r, generate_e1, generate_e2,
          fwdnttvec_r, intt_ar, intt_tr, mtv, ttimesr, vpv, padd3, decodet,
          frommsg, compressu, compressv),
      randombytes, enc_hash_rand, enc_hash_pk, enc_hash_ct, enc_hash_coin, enc_kdf);

  primitive::ccakem_keypair::mem_resource<variant> keypair_mr(ninputs);
  primitive::ccakem_enc::mem_resource<variant>     enc_mr(ninputs);
  primitive::ccakem_dec::mem_resource<variant>     dec_mr(ninputs);

  // ── Keypair + encaps via CUDA Graph (same setup as trace_ser_l2thrash) ───
  {
    cuda_resource::graph setup_graph;
    cudaGraphNode_t dummy, pk_avail, sk_avail, ct_avail, ssb_avail, pk_used;
    CCC(cudaGraphAddEmptyNode(&dummy, setup_graph, nullptr, 0));

    randombytes(keypair_mr.pke_keypair_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    randombytes(keypair_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    randombytes(enc_mr.rand_host.get_ptr(), params::symbytes * ninputs);

    auto kr_args = keypair.join_graph(
        setup_graph,
        pk_d.get_ptr(), pk_d.get_pitch(), dummy, &pk_avail,
        sk_d.get_ptr(), sk_d.get_pitch(), dummy, &sk_avail,
        keypair_mr);
    auto er_args = enc.join_graph(
        setup_graph,
        ct_d.get_ptr(), ct_d.get_pitch(), dummy, &ct_avail,
        ss_d.get_ptr(), ss_d.get_pitch(), dummy, &ssb_avail,
        pk_d.get_ptr(), pk_d.get_pitch(), pk_avail, &pk_used,
        enc_mr);

    cudaGraphNode_t cpyct, cpysk;
    {
      cudaMemcpy3DParms p = {};
      p.srcPtr = make_cudaPitchedPtr(ct_d.get_ptr(), ct_d.get_pitch(),
          params::ciphertextbytes<variant>, ninputs);
      p.dstPtr = make_cudaPitchedPtr(ct_h.get_ptr(),
          params::ciphertextbytes<variant>, params::ciphertextbytes<variant>, ninputs);
      p.extent = make_cudaExtent(params::ciphertextbytes<variant>, ninputs, 1);
      p.kind = cudaMemcpyDeviceToHost;
      std::array dep{ct_avail};
      CCC(cudaGraphAddMemcpyNode(&cpyct, setup_graph, dep.data(), dep.size(), &p));
    }
    {
      cudaMemcpy3DParms p = {};
      p.srcPtr = make_cudaPitchedPtr(sk_d.get_ptr(), sk_d.get_pitch(),
          params::secretkeybytes<variant>, ninputs);
      p.dstPtr = make_cudaPitchedPtr(sk_h.get_ptr(),
          params::secretkeybytes<variant>, params::secretkeybytes<variant>, ninputs);
      p.extent = make_cudaExtent(params::secretkeybytes<variant>, ninputs, 1);
      p.kind = cudaMemcpyDeviceToHost;
      std::array dep{sk_avail};
      CCC(cudaGraphAddMemcpyNode(&cpysk, setup_graph, dep.data(), dep.size(), &p));
    }

    cuda_resource::graph_exec setup_exec(setup_graph);
    cuda_resource::stream     setup_stream(cudaStreamNonBlocking);
    CCC(cudaGraphLaunch(setup_exec, setup_stream));
    CCC(cudaStreamSynchronize(setup_stream));
  }

  CCC(cudaMemcpy2D(ct_d.get_ptr(), ct_d.get_pitch(),
                   ct_h.get_ptr(), params::ciphertextbytes<variant>,
                   params::ciphertextbytes<variant>, ninputs,
                   cudaMemcpyHostToDevice));
  CCC(cudaMemcpy2D(sk_d.get_ptr(), sk_d.get_pitch(),
                   sk_h.get_ptr(), params::secretkeybytes<variant>,
                   params::secretkeybytes<variant>, ninputs,
                   cudaMemcpyHostToDevice));
  CCC(cudaDeviceSynchronize());

  // ── Convenience pointers ─────────────────────────────────────────────────
  std::uint8_t* buf_ptr   = dec_mr.buf.get_ptr();
  std::size_t   buf_pitch = dec_mr.buf.get_pitch();
  std::uint8_t* kr_ptr    = dec_mr.kr.get_ptr();
  std::size_t   kr_pitch  = dec_mr.kr.get_pitch();
  std::uint8_t* cmp_ptr   = dec_mr.cmp.get_ptr();
  std::size_t   cmp_pitch = dec_mr.cmp.get_pitch();

  const std::uint8_t* pk_in_sk =
      sk_d.get_ptr() + params::indcpa_secretkeybytes<variant>;
  const std::uint8_t* z_in_sk =
      sk_d.get_ptr() + params::secretkeybytes<variant> - params::symbytes;

  short2* bp_dec   = dec_mr.pke_dec_mr.bp.get_ptr();
  short2* skpv_dec = dec_mr.pke_dec_mr.skpv.get_ptr();
  short2* v_dec    = dec_mr.pke_dec_mr.v.get_ptr();
  short2* mp_dec   = dec_mr.pke_dec_mr.mp.get_ptr();

  short2* at_enc   = dec_mr.pke_enc_mr.at.get_ptr();
  short2* sp_enc   = dec_mr.pke_enc_mr.sp.get_ptr();
  short2* pkpv_enc = dec_mr.pke_enc_mr.pkpv.get_ptr();
  short2* ep_enc   = dec_mr.pke_enc_mr.ep.get_ptr();
  short2* bp_enc   = dec_mr.pke_enc_mr.bp.get_ptr();
  short2* v_enc    = dec_mr.pke_enc_mr.v.get_ptr();
  short2* k_enc    = dec_mr.pke_enc_mr.k.get_ptr();
  short2* epp_enc  = dec_mr.pke_enc_mr.epp.get_ptr();

  cuda_resource::stream stream(cudaStreamNonBlocking);

  // ── One decapsulation (unmitigated): 27 kernels, sync once at end ────────
  // Kernels are serialized on the same stream; DeviceSync only at the end.
  auto run_unmit = [&]() {
    LAUNCH_PLAIN(decompressu, stream, bp_dec, ct_d.get_ptr(), ct_d.get_pitch());
    LAUNCH_PLAIN(decompressv, stream, v_dec,
                 ct_d.get_ptr() + params::polyveccompressedbytes<variant>,
                 ct_d.get_pitch());
    LAUNCH_PLAIN(decodes, stream, skpv_dec, sk_d.get_ptr(), sk_d.get_pitch());
    LAUNCH_PLAIN(fwdnttvec_u, stream, bp_dec);
    LAUNCH_PLAIN(stimesu, stream, mp_dec, skpv_dec, bp_dec);
    LAUNCH_PLAIN(intt_su, stream, mp_dec);
    LAUNCH_PLAIN(psub, stream, mp_dec, v_dec, mp_dec);
    LAUNCH_PLAIN(tomsg, stream, buf_ptr, buf_pitch, mp_dec);
    CCC(cudaMemcpy2DAsync(
        buf_ptr + params::symbytes, buf_pitch,
        sk_d.get_ptr() + (params::secretkeybytes<variant> - 2 * params::symbytes),
        sk_d.get_pitch(), params::symbytes, ninputs,
        cudaMemcpyDeviceToDevice, stream));
    LAUNCH_PLAIN(dec_hash_coin, stream, kr_ptr, kr_pitch, buf_ptr, buf_pitch,
                 2 * params::symbytes);
    LAUNCH_PLAIN(decodet, stream, pkpv_enc, pk_in_sk, sk_d.get_pitch());
    LAUNCH_PLAIN(frommsg, stream, k_enc, buf_ptr, buf_pitch);
    LAUNCH_PLAIN(generate_at, stream, at_enc,
                 pk_in_sk + params::polyvecbytes<variant>, sk_d.get_pitch());
    LAUNCH_PLAIN(generate_r, stream, sp_enc, kr_ptr + params::symbytes, kr_pitch,
                 static_cast<std::uint8_t>(0));
    LAUNCH_PLAIN(generate_e1, stream, ep_enc, kr_ptr + params::symbytes, kr_pitch,
                 static_cast<std::uint8_t>(params::k<variant>));
    LAUNCH_PLAIN(generate_e2, stream, epp_enc, kr_ptr + params::symbytes, kr_pitch,
                 static_cast<std::uint8_t>(params::k<variant> * 2));
    LAUNCH_PLAIN(fwdnttvec_r, stream, sp_enc);
    LAUNCH_PLAIN(mtv, stream, bp_enc, at_enc, sp_enc);
    LAUNCH_PLAIN(ttimesr, stream, v_enc, pkpv_enc, sp_enc);
    LAUNCH_PLAIN(intt_ar, stream, bp_enc);
    LAUNCH_PLAIN(intt_tr, stream, v_enc);
    LAUNCH_PLAIN(vpv, stream, bp_enc, bp_enc, ep_enc);
    LAUNCH_PLAIN(padd3, stream, v_enc, v_enc, epp_enc, k_enc);
    LAUNCH_PLAIN(compressu, stream, cmp_ptr, cmp_pitch, bp_enc);
    LAUNCH_PLAIN(compressv, stream,
                 cmp_ptr + params::polyveccompressedbytes<variant>, cmp_pitch, v_enc);
    LAUNCH_PLAIN(dec_hash_ct, stream, kr_ptr + params::symbytes, kr_pitch,
                 ct_d.get_ptr(), ct_d.get_pitch(),
                 params::ciphertextbytes<variant>);
    LAUNCH_PLAIN(dec_verify_cmov, stream, kr_ptr, kr_pitch,
                 z_in_sk, sk_d.get_pitch(), params::symbytes,
                 ct_d.get_ptr(), ct_d.get_pitch(),
                 cmp_ptr, cmp_pitch, params::ciphertextbytes<variant>);
    LAUNCH_PLAIN(dec_kdf, stream, ss_d.get_ptr(), ss_d.get_pitch(),
                 kr_ptr, kr_pitch, 2 * params::symbytes);
    CCC(cudaDeviceSynchronize());
  };

  // ── One decapsulation (mitigated): 27 kernels + L2 thrash after each ────
  auto run_mit = [&]() {
    LAUNCH_THRASH(decompressu, stream, thrash_buf, l2_bytes,
                  bp_dec, ct_d.get_ptr(), ct_d.get_pitch());
    LAUNCH_THRASH(decompressv, stream, thrash_buf, l2_bytes,
                  v_dec,
                  ct_d.get_ptr() + params::polyveccompressedbytes<variant>,
                  ct_d.get_pitch());
    LAUNCH_THRASH(decodes, stream, thrash_buf, l2_bytes,
                  skpv_dec, sk_d.get_ptr(), sk_d.get_pitch());
    LAUNCH_THRASH(fwdnttvec_u, stream, thrash_buf, l2_bytes, bp_dec);
    LAUNCH_THRASH(stimesu, stream, thrash_buf, l2_bytes, mp_dec, skpv_dec, bp_dec);
    LAUNCH_THRASH(intt_su, stream, thrash_buf, l2_bytes, mp_dec);
    LAUNCH_THRASH(psub, stream, thrash_buf, l2_bytes, mp_dec, v_dec, mp_dec);
    LAUNCH_THRASH(tomsg, stream, thrash_buf, l2_bytes, buf_ptr, buf_pitch, mp_dec);
    CCC(cudaMemcpy2DAsync(
        buf_ptr + params::symbytes, buf_pitch,
        sk_d.get_ptr() + (params::secretkeybytes<variant> - 2 * params::symbytes),
        sk_d.get_pitch(), params::symbytes, ninputs,
        cudaMemcpyDeviceToDevice, stream));
    CCC(cudaDeviceSynchronize());
    CCC(cudaMemsetAsync(thrash_buf, 0xA5, l2_bytes, stream));
    CCC(cudaDeviceSynchronize());
    CCC(cudaCtxResetPersistingL2Cache());
    LAUNCH_THRASH(dec_hash_coin, stream, thrash_buf, l2_bytes,
                  kr_ptr, kr_pitch, buf_ptr, buf_pitch, 2 * params::symbytes);
    LAUNCH_THRASH(decodet, stream, thrash_buf, l2_bytes,
                  pkpv_enc, pk_in_sk, sk_d.get_pitch());
    LAUNCH_THRASH(frommsg, stream, thrash_buf, l2_bytes, k_enc, buf_ptr, buf_pitch);
    LAUNCH_THRASH(generate_at, stream, thrash_buf, l2_bytes,
                  at_enc, pk_in_sk + params::polyvecbytes<variant>, sk_d.get_pitch());
    LAUNCH_THRASH(generate_r, stream, thrash_buf, l2_bytes,
                  sp_enc, kr_ptr + params::symbytes, kr_pitch,
                  static_cast<std::uint8_t>(0));
    LAUNCH_THRASH(generate_e1, stream, thrash_buf, l2_bytes,
                  ep_enc, kr_ptr + params::symbytes, kr_pitch,
                  static_cast<std::uint8_t>(params::k<variant>));
    LAUNCH_THRASH(generate_e2, stream, thrash_buf, l2_bytes,
                  epp_enc, kr_ptr + params::symbytes, kr_pitch,
                  static_cast<std::uint8_t>(params::k<variant> * 2));
    LAUNCH_THRASH(fwdnttvec_r, stream, thrash_buf, l2_bytes, sp_enc);
    LAUNCH_THRASH(mtv, stream, thrash_buf, l2_bytes, bp_enc, at_enc, sp_enc);
    LAUNCH_THRASH(ttimesr, stream, thrash_buf, l2_bytes, v_enc, pkpv_enc, sp_enc);
    LAUNCH_THRASH(intt_ar, stream, thrash_buf, l2_bytes, bp_enc);
    LAUNCH_THRASH(intt_tr, stream, thrash_buf, l2_bytes, v_enc);
    LAUNCH_THRASH(vpv, stream, thrash_buf, l2_bytes, bp_enc, bp_enc, ep_enc);
    LAUNCH_THRASH(padd3, stream, thrash_buf, l2_bytes, v_enc, v_enc, epp_enc, k_enc);
    LAUNCH_THRASH(compressu, stream, thrash_buf, l2_bytes,
                  cmp_ptr, cmp_pitch, bp_enc);
    LAUNCH_THRASH(compressv, stream, thrash_buf, l2_bytes,
                  cmp_ptr + params::polyveccompressedbytes<variant>, cmp_pitch, v_enc);
    LAUNCH_THRASH(dec_hash_ct, stream, thrash_buf, l2_bytes,
                  kr_ptr + params::symbytes, kr_pitch,
                  ct_d.get_ptr(), ct_d.get_pitch(),
                  params::ciphertextbytes<variant>);
    LAUNCH_THRASH(dec_verify_cmov, stream, thrash_buf, l2_bytes,
                  kr_ptr, kr_pitch,
                  z_in_sk, sk_d.get_pitch(), params::symbytes,
                  ct_d.get_ptr(), ct_d.get_pitch(),
                  cmp_ptr, cmp_pitch, params::ciphertextbytes<variant>);
    LAUNCH_THRASH(dec_kdf, stream, thrash_buf, l2_bytes,
                  ss_d.get_ptr(), ss_d.get_pitch(),
                  kr_ptr, kr_pitch, 2 * params::symbytes);
  };

  const char* variant_name =
      params::ciphertextbytes<variant> == 768  ? "Kyber-512"  :
      params::ciphertextbytes<variant> == 1088 ? "Kyber-768"  : "Kyber-1024";

  // ── Warmup (50 iters each) ───────────────────────────────────────────────
  std::fprintf(stderr, "[bench_l2thrash] Warmup (50 unmit + 50 mit)...\n");
  for (int i = 0; i < 50; i++) run_unmit();
  for (int i = 0; i < 50; i++) run_mit();
  std::fprintf(stderr, "[bench_l2thrash] Warmup done.\n");

  // ── Unmitigated throughput ───────────────────────────────────────────────
  std::fprintf(stderr, "[bench_l2thrash] Running unmitigated for %u seconds...\n",
               duration_sec);
  long long unmit_count = 0;
  {
    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::seconds(duration_sec);
    while (clock::now() < deadline) {
      run_unmit();
      unmit_count += ninputs;
    }
  }

  // ── Mitigated throughput ─────────────────────────────────────────────────
  std::fprintf(stderr, "[bench_l2thrash] Running mitigated for %u seconds...\n",
               duration_sec);
  long long mit_count = 0;
  {
    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::seconds(duration_sec);
    while (clock::now() < deadline) {
      run_mit();
      mit_count += ninputs;
    }
  }

  double unmit_ops_sec = static_cast<double>(unmit_count) / duration_sec;
  double mit_ops_sec   = static_cast<double>(mit_count)   / duration_sec;
  double overhead_pct  = (unmit_ops_sec - mit_ops_sec) / unmit_ops_sec * 100.0;

  std::printf("\n");
  std::printf("%-12s  %16s  %16s  %12s\n",
              "variant", "unmit_ops_sec", "mit_ops_sec", "overhead_pct");
  std::printf("%-12s  %16.1f  %16.1f  %11.1f%%\n",
              variant_name, unmit_ops_sec, mit_ops_sec, overhead_pct);

  CCC(cudaFree(thrash_buf));
}

}  // namespace atpqc_cuda::kyber::bench_l2thrash

int main() {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::bench_l2thrash::run();
    CCC(cuCtxSynchronize());
  }

  return 0;
}
