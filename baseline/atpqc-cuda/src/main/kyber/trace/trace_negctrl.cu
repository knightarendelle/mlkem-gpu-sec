//
// trace_negctrl.cu
// Negative-control per-kernel serialized timing for TVLA decomposition.
//
// Both class 0 and class 1 use VALID ciphertexts under the SAME keypair.
// The only difference is the encryption randomness (coins).
// There is zero algorithmic reason for any kernel timing to differ.
//
// Expected result:
//   Clean measurement setup  →  all 27 kernels |t| < 4.5
//   Systematic uniform offset →  measurement artifact, not leakage
//
// Output (stdout): CSV with columns trace_id,kernel_name,elapsed_us
//
// Usage:
//   echo "1 4 4 4 4 100000 0" | ./target/trace_negctrl_kyber512.out \
//     > experiments/traces/per_kernel/kyber512_negctrl_class0_n100000_per_kernel.csv
//   echo "1 4 4 4 4 100000 1" | ./target/trace_negctrl_kyber512.out \
//     > experiments/traces/per_kernel/kyber512_negctrl_class1_n100000_per_kernel.csv
//
// Build:
//   make -C baseline/atpqc-cuda trace_negctrl_kyber512
//

#include <algorithm>
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

// Hard error-checking regardless of CUDA_DEBUG (no-op in release builds).
#undef  CCC
static inline cudaError_t ser_check(cudaError_t e,
                                     const char* call, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[NEGCTRL CUDA ERROR] %s:%d\n  call: %s\n  error: %s\n",
                     file, line, call, cudaGetErrorString(e));
        std::fflush(stderr);
        std::abort();
    }
    return e;
}
static inline CUresult ser_check(CUresult e,
                                  const char* call, const char* file, int line) {
    if (e != CUDA_SUCCESS) {
        const char* name = nullptr; const char* str = nullptr;
        cuGetErrorName(e, &name); cuGetErrorString(e, &str);
        std::fprintf(stderr, "[NEGCTRL CUDA ERROR] %s:%d\n  call: %s\n  error: %s (%s)\n",
                     file, line, call, str ? str : "?", name ? name : "?");
        std::fflush(stderr);
        std::abort();
    }
    return e;
}
#define CCC(call) ser_check((call), #call, __FILE__, __LINE__)

namespace atpqc_cuda::kyber::trace_negctrl {

using rng_type = rng::std_random_device;
using variant  = variants::KYBER_VARIANT;
constexpr variant variant_v;

#define LAUNCH_TIMED(label, obj, stream, ev_a, ev_b, record, tid, ...)      \
  do {                                                                        \
    auto _a = (obj).generate_args(__VA_ARGS__);                              \
    CCC(cudaEventRecord((ev_a), (stream)));                                   \
    CCC(cudaLaunchKernel((obj).get_func(), (obj).get_grid_dim(),             \
                         (obj).get_block_dim(), _a->get_args_ptr(),          \
                         (obj).get_shared_bytes(), (stream)));               \
    CCC(cudaEventRecord((ev_b), (stream)));                                   \
    CCC(cudaDeviceSynchronize());                                             \
    if (record) {                                                             \
      float _ms = 0.f;                                                       \
      CCC(cudaEventElapsedTime(&_ms, (ev_a), (ev_b)));                       \
      std::printf("%u,%s,%.3f\n", (tid), (label), _ms * 1000.f);            \
    }                                                                         \
  } while (0)

void trace_negctrl() {
  unsigned ninputs, genmat_nw, genvec_nw, genpoly_nw, fips_nw;
  unsigned ntraces;
  int ct_class;
  std::cin >> ninputs >> genmat_nw >> genvec_nw >> genpoly_nw >> fips_nw
           >> ntraces >> ct_class;

  // ── Device memory ─────────────────────────────────────────────────────────
  cuda_resource::device_pitched_memory<std::uint8_t> pk_d(
      params::publickeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> sk_d(
      params::secretkeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ct_d(
      params::ciphertextbytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ss_d(
      params::ssbytes, ninputs);

  // ── Host memory: sk + TWO ciphertext buffers (ct_a, ct_b) ─────────────────
  cuda_resource::pinned_memory<std::uint8_t> sk_h(
      params::secretkeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ct_h_a(
      params::ciphertextbytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ct_h_b(
      params::ciphertextbytes<variant> * ninputs);

  // ── Kernel objects ────────────────────────────────────────────────────────
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

  // ── Step 1: keypair + encaps A → sk_h, ct_h_a ────────────────────────────
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

    // ct_d → ct_h_a
    cudaGraphNode_t cpyct_a;
    {
      cudaMemcpy3DParms p = {};
      p.srcPtr = make_cudaPitchedPtr(ct_d.get_ptr(), ct_d.get_pitch(),
          params::ciphertextbytes<variant>, ninputs);
      p.dstPtr = make_cudaPitchedPtr(ct_h_a.get_ptr(),
          params::ciphertextbytes<variant>,
          params::ciphertextbytes<variant>, ninputs);
      p.extent = make_cudaExtent(params::ciphertextbytes<variant>, ninputs, 1);
      p.kind = cudaMemcpyDeviceToHost;
      std::array dep{ct_avail};
      CCC(cudaGraphAddMemcpyNode(&cpyct_a, setup_graph, dep.data(), dep.size(), &p));
    }
    // sk_d → sk_h
    cudaGraphNode_t cpysk;
    {
      cudaMemcpy3DParms p = {};
      p.srcPtr = make_cudaPitchedPtr(sk_d.get_ptr(), sk_d.get_pitch(),
          params::secretkeybytes<variant>, ninputs);
      p.dstPtr = make_cudaPitchedPtr(sk_h.get_ptr(),
          params::secretkeybytes<variant>,
          params::secretkeybytes<variant>, ninputs);
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

  // ── Step 2: encaps B (new random coins, same pk) → ct_h_b ─────────────────
  // pk_d is already populated and stable from step 1.
  {
    randombytes(enc_mr.rand_host.get_ptr(), params::symbytes * ninputs);

    cuda_resource::graph enc2_graph;
    cudaGraphNode_t dummy2, ct2_avail, ssb2_avail, pk2_used;
    CCC(cudaGraphAddEmptyNode(&dummy2, enc2_graph, nullptr, 0));

    auto er2_args = enc.join_graph(
        enc2_graph,
        ct_d.get_ptr(), ct_d.get_pitch(), dummy2, &ct2_avail,
        ss_d.get_ptr(), ss_d.get_pitch(), dummy2, &ssb2_avail,
        pk_d.get_ptr(), pk_d.get_pitch(), dummy2, &pk2_used,
        enc_mr);

    // ct_d → ct_h_b
    cudaGraphNode_t cpyct_b;
    {
      cudaMemcpy3DParms p = {};
      p.srcPtr = make_cudaPitchedPtr(ct_d.get_ptr(), ct_d.get_pitch(),
          params::ciphertextbytes<variant>, ninputs);
      p.dstPtr = make_cudaPitchedPtr(ct_h_b.get_ptr(),
          params::ciphertextbytes<variant>,
          params::ciphertextbytes<variant>, ninputs);
      p.extent = make_cudaExtent(params::ciphertextbytes<variant>, ninputs, 1);
      p.kind = cudaMemcpyDeviceToHost;
      std::array dep{ct2_avail};
      CCC(cudaGraphAddMemcpyNode(&cpyct_b, enc2_graph, dep.data(), dep.size(), &p));
    }

    cuda_resource::graph_exec enc2_exec(enc2_graph);
    cuda_resource::stream     enc2_stream(cudaStreamNonBlocking);
    CCC(cudaGraphLaunch(enc2_exec, enc2_stream));
    CCC(cudaStreamSynchronize(enc2_stream));
  }

  // ── Step 3: upload selected valid ct to device (fixed for all traces) ──────
  {
    auto* sel = (ct_class == 0) ? ct_h_a.get_ptr() : ct_h_b.get_ptr();
    CCC(cudaMemcpy2D(ct_d.get_ptr(), ct_d.get_pitch(),
        sel, params::ciphertextbytes<variant>,
        params::ciphertextbytes<variant>, ninputs,
        cudaMemcpyHostToDevice));
  }
  CCC(cudaMemcpy2D(
      sk_d.get_ptr(), sk_d.get_pitch(),
      sk_h.get_ptr(), params::secretkeybytes<variant>,
      params::secretkeybytes<variant>, ninputs,
      cudaMemcpyHostToDevice));
  CCC(cudaDeviceSynchronize());

  // ── Derive convenience pointers (mirror ccakem_dec::join_graph) ────────────
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

  // ── Single stream + reused events ─────────────────────────────────────────
  cuda_resource::stream stream(cudaStreamNonBlocking);
  cudaEvent_t ev_start, ev_stop;
  CCC(cudaEventCreate(&ev_start));
  CCC(cudaEventCreate(&ev_stop));

  // ── Serialized decaps: identical to trace_serialized.cu ───────────────────
  auto run_one = [&](unsigned trace_i, bool record) {

    LAUNCH_TIMED("decompress_u", decompressu, stream, ev_start, ev_stop,
                 record, trace_i,
                 bp_dec, ct_d.get_ptr(), ct_d.get_pitch());

    LAUNCH_TIMED("decompress_v", decompressv, stream, ev_start, ev_stop,
                 record, trace_i,
                 v_dec,
                 ct_d.get_ptr() + params::polyveccompressedbytes<variant>,
                 ct_d.get_pitch());

    LAUNCH_TIMED("decode_s", decodes, stream, ev_start, ev_stop,
                 record, trace_i,
                 skpv_dec, sk_d.get_ptr(), sk_d.get_pitch());

    LAUNCH_TIMED("fwdntt_u", fwdnttvec_u, stream, ev_start, ev_stop,
                 record, trace_i,
                 bp_dec);

    LAUNCH_TIMED("s_times_u", stimesu, stream, ev_start, ev_stop,
                 record, trace_i,
                 mp_dec, skpv_dec, bp_dec);

    LAUNCH_TIMED("intt_su", intt_su, stream, ev_start, ev_stop,
                 record, trace_i,
                 mp_dec);

    LAUNCH_TIMED("v_minus_su", psub, stream, ev_start, ev_stop,
                 record, trace_i,
                 mp_dec, v_dec, mp_dec);

    LAUNCH_TIMED("poly_tomsg", tomsg, stream, ev_start, ev_stop,
                 record, trace_i,
                 buf_ptr, buf_pitch, mp_dec);

    CCC(cudaMemcpy2DAsync(
        buf_ptr + params::symbytes, buf_pitch,
        sk_d.get_ptr() + (params::secretkeybytes<variant> - 2 * params::symbytes),
        sk_d.get_pitch(),
        params::symbytes, ninputs,
        cudaMemcpyDeviceToDevice, stream));
    CCC(cudaDeviceSynchronize());

    LAUNCH_TIMED("hash_coin", dec_hash_coin, stream, ev_start, ev_stop,
                 record, trace_i,
                 kr_ptr, kr_pitch, buf_ptr, buf_pitch, 2 * params::symbytes);

    LAUNCH_TIMED("decode_t", decodet, stream, ev_start, ev_stop,
                 record, trace_i,
                 pkpv_enc, pk_in_sk, sk_d.get_pitch());

    LAUNCH_TIMED("poly_frommsg", frommsg, stream, ev_start, ev_stop,
                 record, trace_i,
                 k_enc, buf_ptr, buf_pitch);

    LAUNCH_TIMED("generate_at", generate_at, stream, ev_start, ev_stop,
                 record, trace_i,
                 at_enc,
                 pk_in_sk + params::polyvecbytes<variant>,
                 sk_d.get_pitch());

    LAUNCH_TIMED("generate_r", generate_r, stream, ev_start, ev_stop,
                 record, trace_i,
                 sp_enc, kr_ptr + params::symbytes, kr_pitch,
                 static_cast<std::uint8_t>(0));

    LAUNCH_TIMED("generate_e1", generate_e1, stream, ev_start, ev_stop,
                 record, trace_i,
                 ep_enc, kr_ptr + params::symbytes, kr_pitch,
                 static_cast<std::uint8_t>(params::k<variant>));

    LAUNCH_TIMED("generate_e2", generate_e2, stream, ev_start, ev_stop,
                 record, trace_i,
                 epp_enc, kr_ptr + params::symbytes, kr_pitch,
                 static_cast<std::uint8_t>(params::k<variant> * 2));

    LAUNCH_TIMED("fwdntt_r", fwdnttvec_r, stream, ev_start, ev_stop,
                 record, trace_i,
                 sp_enc);

    LAUNCH_TIMED("a_times_r", mtv, stream, ev_start, ev_stop,
                 record, trace_i,
                 bp_enc, at_enc, sp_enc);

    LAUNCH_TIMED("t_times_r", ttimesr, stream, ev_start, ev_stop,
                 record, trace_i,
                 v_enc, pkpv_enc, sp_enc);

    LAUNCH_TIMED("intt_ar", intt_ar, stream, ev_start, ev_stop,
                 record, trace_i,
                 bp_enc);

    LAUNCH_TIMED("intt_tr", intt_tr, stream, ev_start, ev_stop,
                 record, trace_i,
                 v_enc);

    LAUNCH_TIMED("ar_plus_e1", vpv, stream, ev_start, ev_stop,
                 record, trace_i,
                 bp_enc, bp_enc, ep_enc);

    LAUNCH_TIMED("tr_plus_e2_plus_m", padd3, stream, ev_start, ev_stop,
                 record, trace_i,
                 v_enc, v_enc, epp_enc, k_enc);

    LAUNCH_TIMED("compress_u", compressu, stream, ev_start, ev_stop,
                 record, trace_i,
                 cmp_ptr, cmp_pitch, bp_enc);

    LAUNCH_TIMED("compress_v", compressv, stream, ev_start, ev_stop,
                 record, trace_i,
                 cmp_ptr + params::polyveccompressedbytes<variant>,
                 cmp_pitch, v_enc);

    LAUNCH_TIMED("hash_ct", dec_hash_ct, stream, ev_start, ev_stop,
                 record, trace_i,
                 kr_ptr + params::symbytes, kr_pitch,
                 ct_d.get_ptr(), ct_d.get_pitch(),
                 params::ciphertextbytes<variant>);

    LAUNCH_TIMED("verify_cmov", dec_verify_cmov, stream, ev_start, ev_stop,
                 record, trace_i,
                 kr_ptr, kr_pitch,
                 z_in_sk, sk_d.get_pitch(), params::symbytes,
                 ct_d.get_ptr(), ct_d.get_pitch(),
                 cmp_ptr, cmp_pitch,
                 params::ciphertextbytes<variant>);

    LAUNCH_TIMED("kdf", dec_kdf, stream, ev_start, ev_stop,
                 record, trace_i,
                 ss_d.get_ptr(), ss_d.get_pitch(),
                 kr_ptr, kr_pitch, 2 * params::symbytes);
  };

  // ── Warmup ─────────────────────────────────────────────────────────────────
  fprintf(stderr, "[warmup] 200 iterations...\n");
  for (unsigned w = 0; w < 200; w++)
    run_one(0, false);
  fprintf(stderr, "[warmup] done\n");

  // ── CSV header ─────────────────────────────────────────────────────────────
  std::printf("# mlkem-gpu-sec per-kernel serialized timing traces (negative control)\n");
  std::printf("# variant: Kyber-512\n");
  std::printf("# class: %d (%s valid ct)\n", ct_class,
              ct_class == 0 ? "ct_A —" : "ct_B —");
  std::printf("# both classes are valid ciphertexts under the same keypair\n");
  std::printf("# n_traces: %u\n", ntraces);
  std::printf("# unit: microseconds (elapsed_us)\n");
  std::printf("trace_id,kernel_name,elapsed_us\n");

  // ── Trace collection ───────────────────────────────────────────────────────
  for (unsigned i = 0; i < ntraces; i++) {
    run_one(i, true);
    if (ntraces >= 10 && (i + 1) % (ntraces / 10) == 0)
      fprintf(stderr, "  %u%%\n", (i + 1) * 100 / ntraces);
  }

  CCC(cudaEventDestroy(ev_start));
  CCC(cudaEventDestroy(ev_stop));
}

}  // namespace atpqc_cuda::kyber::trace_negctrl

int main() {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::trace_negctrl::trace_negctrl();
    CCC(cuCtxSynchronize());
  }

  return 0;
}
