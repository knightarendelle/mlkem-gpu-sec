//
// trace_ser_l2thrash.cu
// Serialized Kyber decapsulation timing with full L2 cache eviction between
// every kernel.
//
// Identical to trace_serialized.cu except that a full-L2-sized cudaMemset is
// issued after every kernel launch + sync, before the next kernel starts.
// This forces all L2 lines (DCC-compressed or not, persisting or not) to be
// written back to DRAM and replaced by the thrash pattern, giving the next
// kernel a genuinely cold L2.
//
// L2 size is queried at runtime via cudaDevAttrL2CacheSize so the binary
// works across Ada variants (4070 = 36 MB, 4080 = 64 MB, 4090/L40 = 72 MB).
// A stderr line reports the detected size.
//
// Mitigation hypothesis: if L2 dirty-line residue (upstream context, carried
// by DCC compression state) is the only mechanism, a cold-L2 pipeline should
// drop leaking kernels from 22/27 to 0/27.
// If some kernels still leak, their DCC effect is intrinsic to their own I/O
// (the ciphertext data itself is more/less compressible), which requires a
// data-level mitigation.
//
// NOTE: thrash overhead = L2_size × 1 memset per kernel × ntraces.
//   On RTX 4090 (72 MB): ~1–2 ms/memset × 28 × 100 k ≈ 3–8 hours.
//   Run at --ntraces 10000 first to confirm the pattern, then scale.
//
// Output: identical CSV format to trace_serialized.cu (trace_id,kernel,us).
//
// Usage:
//   echo "1 4 4 4 4 10000 0" | ./target/trace_ser_l2thrash_kyber1024.out \
//     > experiments/traces/l2thrash/kyber1024_l2thrash_class0_n10000.csv
// Build:
//   make -C baseline/atpqc-cuda trace_ser_l2thrash_kyber1024
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

#undef  CCC
static inline cudaError_t lt_check(cudaError_t e,
                                    const char* call, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[L2THRASH CUDA ERROR] %s:%d\n  call: %s\n  error: %s\n",
                     file, line, call, cudaGetErrorString(e));
        std::fflush(stderr);
        std::abort();
    }
    return e;
}
static inline CUresult lt_check(CUresult e,
                                  const char* call, const char* file, int line) {
    if (e != CUDA_SUCCESS) {
        const char* name = nullptr; const char* str = nullptr;
        cuGetErrorName(e, &name); cuGetErrorString(e, &str);
        std::fprintf(stderr, "[L2THRASH CUDA ERROR] %s:%d\n  call: %s\n  error: %s (%s)\n",
                     file, line, call, str ? str : "?", name ? name : "?");
        std::fflush(stderr);
        std::abort();
    }
    return e;
}
#define CCC(call) lt_check((call), #call, __FILE__, __LINE__)

namespace atpqc_cuda::kyber::trace_ser_l2thrash {

using rng_type = rng::std_random_device;
using variant  = variants::KYBER_VARIANT;
constexpr variant variant_v;

// ── Per-kernel timed launch + full L2 thrash ─────────────────────────────
//
// Timing window:  ev_start → kernel → ev_stop   (does NOT include the thrash)
// L2 flush:       cudaMemset(thrash_buf, l2_bytes) + DeviceSync AFTER timing,
//                 so the NEXT kernel starts with a cold L2.
//
#define LAUNCH_TIMED_THRASH(label, obj, stream, ev_a, ev_b,                    \
                             record, tid, ...)                                  \
  do {                                                                          \
    auto _a = (obj).generate_args(__VA_ARGS__);                                 \
    CCC(cudaEventRecord((ev_a), (stream)));                                     \
    CCC(cudaLaunchKernel((obj).get_func(), (obj).get_grid_dim(),                \
                         (obj).get_block_dim(), _a->get_args_ptr(),             \
                         (obj).get_shared_bytes(), (stream)));                  \
    CCC(cudaEventRecord((ev_b), (stream)));                                     \
    CCC(cudaDeviceSynchronize());                                               \
    if (record) {                                                               \
      float _ms = 0.f;                                                          \
      CCC(cudaEventElapsedTime(&_ms, (ev_a), (ev_b)));                          \
      std::printf("%u,%s,%.3f\n", (tid), (label), _ms * 1000.f);               \
    }                                                                           \
    /* Evict entire L2 before next kernel */                                    \
    CCC(cudaMemsetAsync(thrash_buf, 0xA5, l2_bytes, (stream)));                 \
    CCC(cudaDeviceSynchronize());                                               \
    CCC(cudaCtxResetPersistingL2Cache());                                       \
  } while (0)

void trace_serialized_l2thrash() {
  unsigned ninputs, genmat_nw, genvec_nw, genpoly_nw, fips_nw;
  unsigned ntraces;
  int ct_class;
  std::cin >> ninputs >> genmat_nw >> genvec_nw >> genpoly_nw >> fips_nw
           >> ntraces >> ct_class;

  // ── Query L2 size ─────────────────────────────────────────────────────
  int l2_bytes_int = 0;
  CCC(cudaDeviceGetAttribute(&l2_bytes_int, cudaDevAttrL2CacheSize, 0));
  // Allocate 2× L2 to guarantee full eviction even with set-associativity
  // aliasing effects.  The extra cost is one extra memset pass (~2 ms on
  // a 72 MB L2), which is acceptable.
  const std::size_t l2_bytes = static_cast<std::size_t>(l2_bytes_int) * 2;

  std::fprintf(stderr,
               "[l2thrash] L2 cache size: %d bytes (%.1f MB); "
               "thrash buffer: %.1f MB\n",
               l2_bytes_int, l2_bytes_int / (1024.0 * 1024.0),
               l2_bytes / (1024.0 * 1024.0));
  std::fflush(stderr);

  // ── Thrash buffer (device) ────────────────────────────────────────────
  void* thrash_buf = nullptr;
  CCC(cudaMalloc(&thrash_buf, l2_bytes));

  // ── Device and host memory ────────────────────────────────────────────
  cuda_resource::device_pitched_memory<std::uint8_t> pk_d(
      params::publickeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> sk_d(
      params::secretkeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ct_d(
      params::ciphertextbytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ss_d(
      params::ssbytes, ninputs);

  cuda_resource::pinned_memory<std::uint8_t> pk_h(
      params::publickeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> sk_h(
      params::secretkeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ct_h(
      params::ciphertextbytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ss_h(
      params::ssbytes * ninputs);

  // ── Kernel objects ────────────────────────────────────────────────────
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

  // ── Step 1: keypair + encaps via CUDA Graph ───────────────────────────
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
          params::ciphertextbytes<variant>,
          params::ciphertextbytes<variant>, ninputs);
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

  // ── Step 2: class 1 → overwrite ct with random bytes ─────────────────
  if (ct_class == 1) {
    std::uint8_t* ct_ptr = ct_h.get_ptr();
    size_t ct_total = params::ciphertextbytes<variant> * ninputs;
    for (size_t i = 0; i < ct_total; i++)
      ct_ptr[i] = (std::uint8_t)((i * 6364136223846793005ULL +
                                   1442695040888963407ULL) >> 56);
  }

  // ── Step 3: Upload ct and sk; thrash L2 before measurement begins ─────
  CCC(cudaMemcpy2D(
      ct_d.get_ptr(), ct_d.get_pitch(),
      ct_h.get_ptr(), params::ciphertextbytes<variant>,
      params::ciphertextbytes<variant>, ninputs,
      cudaMemcpyHostToDevice));
  CCC(cudaMemcpy2D(
      sk_d.get_ptr(), sk_d.get_pitch(),
      sk_h.get_ptr(), params::secretkeybytes<variant>,
      params::secretkeybytes<variant>, ninputs,
      cudaMemcpyHostToDevice));
  CCC(cudaDeviceSynchronize());
  // Initial L2 flush so warmup and measurement both start from a clean state.
  CCC(cudaMemset(thrash_buf, 0xA5, l2_bytes));
  CCC(cudaDeviceSynchronize());
  CCC(cudaCtxResetPersistingL2Cache());

  // ── Convenience pointers ──────────────────────────────────────────────
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

  cudaEvent_t ev_start, ev_stop;
  CCC(cudaEventCreate(&ev_start));
  CCC(cudaEventCreate(&ev_stop));

  auto run_one = [&](unsigned trace_i, bool record) {

    LAUNCH_TIMED_THRASH("decompress_u", decompressu, stream, ev_start, ev_stop,
                        record, trace_i,
                        bp_dec, ct_d.get_ptr(), ct_d.get_pitch());

    LAUNCH_TIMED_THRASH("decompress_v", decompressv, stream, ev_start, ev_stop,
                        record, trace_i,
                        v_dec,
                        ct_d.get_ptr() + params::polyveccompressedbytes<variant>,
                        ct_d.get_pitch());

    LAUNCH_TIMED_THRASH("decode_s", decodes, stream, ev_start, ev_stop,
                        record, trace_i,
                        skpv_dec, sk_d.get_ptr(), sk_d.get_pitch());

    LAUNCH_TIMED_THRASH("fwdntt_u", fwdnttvec_u, stream, ev_start, ev_stop,
                        record, trace_i,
                        bp_dec);

    LAUNCH_TIMED_THRASH("s_times_u", stimesu, stream, ev_start, ev_stop,
                        record, trace_i,
                        mp_dec, skpv_dec, bp_dec);

    LAUNCH_TIMED_THRASH("intt_su", intt_su, stream, ev_start, ev_stop,
                        record, trace_i,
                        mp_dec);

    LAUNCH_TIMED_THRASH("v_minus_su", psub, stream, ev_start, ev_stop,
                        record, trace_i,
                        mp_dec, v_dec, mp_dec);

    LAUNCH_TIMED_THRASH("poly_tomsg", tomsg, stream, ev_start, ev_stop,
                        record, trace_i,
                        buf_ptr, buf_pitch, mp_dec);

    CCC(cudaMemcpy2DAsync(
        buf_ptr + params::symbytes, buf_pitch,
        sk_d.get_ptr() + (params::secretkeybytes<variant> - 2 * params::symbytes),
        sk_d.get_pitch(),
        params::symbytes, ninputs,
        cudaMemcpyDeviceToDevice, stream));
    CCC(cudaDeviceSynchronize());
    CCC(cudaMemsetAsync(thrash_buf, 0xA5, l2_bytes, stream));
    CCC(cudaDeviceSynchronize());
    CCC(cudaCtxResetPersistingL2Cache());

    LAUNCH_TIMED_THRASH("hash_coin", dec_hash_coin, stream, ev_start, ev_stop,
                        record, trace_i,
                        kr_ptr, kr_pitch, buf_ptr, buf_pitch, 2 * params::symbytes);

    LAUNCH_TIMED_THRASH("decode_t", decodet, stream, ev_start, ev_stop,
                        record, trace_i,
                        pkpv_enc, pk_in_sk, sk_d.get_pitch());

    LAUNCH_TIMED_THRASH("poly_frommsg", frommsg, stream, ev_start, ev_stop,
                        record, trace_i,
                        k_enc, buf_ptr, buf_pitch);

    LAUNCH_TIMED_THRASH("generate_at", generate_at, stream, ev_start, ev_stop,
                        record, trace_i,
                        at_enc,
                        pk_in_sk + params::polyvecbytes<variant>,
                        sk_d.get_pitch());

    LAUNCH_TIMED_THRASH("generate_r", generate_r, stream, ev_start, ev_stop,
                        record, trace_i,
                        sp_enc, kr_ptr + params::symbytes, kr_pitch,
                        static_cast<std::uint8_t>(0));

    LAUNCH_TIMED_THRASH("generate_e1", generate_e1, stream, ev_start, ev_stop,
                        record, trace_i,
                        ep_enc, kr_ptr + params::symbytes, kr_pitch,
                        static_cast<std::uint8_t>(params::k<variant>));

    LAUNCH_TIMED_THRASH("generate_e2", generate_e2, stream, ev_start, ev_stop,
                        record, trace_i,
                        epp_enc, kr_ptr + params::symbytes, kr_pitch,
                        static_cast<std::uint8_t>(params::k<variant> * 2));

    LAUNCH_TIMED_THRASH("fwdntt_r", fwdnttvec_r, stream, ev_start, ev_stop,
                        record, trace_i,
                        sp_enc);

    LAUNCH_TIMED_THRASH("a_times_r", mtv, stream, ev_start, ev_stop,
                        record, trace_i,
                        bp_enc, at_enc, sp_enc);

    LAUNCH_TIMED_THRASH("t_times_r", ttimesr, stream, ev_start, ev_stop,
                        record, trace_i,
                        v_enc, pkpv_enc, sp_enc);

    LAUNCH_TIMED_THRASH("intt_ar", intt_ar, stream, ev_start, ev_stop,
                        record, trace_i,
                        bp_enc);

    LAUNCH_TIMED_THRASH("intt_tr", intt_tr, stream, ev_start, ev_stop,
                        record, trace_i,
                        v_enc);

    LAUNCH_TIMED_THRASH("ar_plus_e1", vpv, stream, ev_start, ev_stop,
                        record, trace_i,
                        bp_enc, bp_enc, ep_enc);

    LAUNCH_TIMED_THRASH("tr_plus_e2_plus_m", padd3, stream, ev_start, ev_stop,
                        record, trace_i,
                        v_enc, v_enc, epp_enc, k_enc);

    LAUNCH_TIMED_THRASH("compress_u", compressu, stream, ev_start, ev_stop,
                        record, trace_i,
                        cmp_ptr, cmp_pitch, bp_enc);

    LAUNCH_TIMED_THRASH("compress_v", compressv, stream, ev_start, ev_stop,
                        record, trace_i,
                        cmp_ptr + params::polyveccompressedbytes<variant>,
                        cmp_pitch, v_enc);

    LAUNCH_TIMED_THRASH("hash_ct", dec_hash_ct, stream, ev_start, ev_stop,
                        record, trace_i,
                        kr_ptr + params::symbytes, kr_pitch,
                        ct_d.get_ptr(), ct_d.get_pitch(),
                        params::ciphertextbytes<variant>);

    LAUNCH_TIMED_THRASH("verify_cmov", dec_verify_cmov, stream, ev_start, ev_stop,
                        record, trace_i,
                        kr_ptr, kr_pitch,
                        z_in_sk, sk_d.get_pitch(), params::symbytes,
                        ct_d.get_ptr(), ct_d.get_pitch(),
                        cmp_ptr, cmp_pitch,
                        params::ciphertextbytes<variant>);

    LAUNCH_TIMED_THRASH("kdf", dec_kdf, stream, ev_start, ev_stop,
                        record, trace_i,
                        ss_d.get_ptr(), ss_d.get_pitch(),
                        kr_ptr, kr_pitch, 2 * params::symbytes);
  };

  // ── Warmup ────────────────────────────────────────────────────────────
  fprintf(stderr, "[l2thrash warmup] 50 iterations (each has 28 L2 thrashes)...\n");
  for (unsigned w = 0; w < 50; w++)
    run_one(0, /*record=*/false);
  fprintf(stderr, "[l2thrash warmup] done\n");

  // ── CSV header ────────────────────────────────────────────────────────
  const char* variant_name =
    params::ciphertextbytes<variant> == 768  ? "Kyber-512"  :
    params::ciphertextbytes<variant> == 1088 ? "Kyber-768"  : "Kyber-1024";
  std::printf("# mlkem-gpu-sec per-kernel serialized timing traces (L2 thrash)\n");
  std::printf("# variant: %s\n", variant_name);
  std::printf("# class: %d (%s)\n", ct_class, ct_class == 0 ? "valid" : "invalid");
  std::printf("# n_traces: %u\n", ntraces);
  std::printf("# mitigation: full L2 eviction (2x L2-sized cudaMemset) after every kernel\n");
  std::printf("# l2_size_bytes: %d\n", l2_bytes_int);
  std::printf("# thrash_buf_bytes: %zu\n", l2_bytes);
  std::printf("# unit: microseconds (elapsed_us)\n");
  std::printf("trace_id,kernel_name,elapsed_us\n");

  // ── Trace collection ──────────────────────────────────────────────────
  for (unsigned i = 0; i < ntraces; i++) {
    run_one(i, /*record=*/true);
    if (ntraces >= 10 && (i + 1) % (ntraces / 10) == 0)
      fprintf(stderr, "  %u%%\n", (i + 1) * 100 / ntraces);
  }

  CCC(cudaEventDestroy(ev_start));
  CCC(cudaEventDestroy(ev_stop));
  CCC(cudaFree(thrash_buf));
}

}  // namespace atpqc_cuda::kyber::trace_ser_l2thrash

int main() {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::trace_ser_l2thrash::trace_serialized_l2thrash();
    CCC(cuCtxSynchronize());
  }

  return 0;
}
