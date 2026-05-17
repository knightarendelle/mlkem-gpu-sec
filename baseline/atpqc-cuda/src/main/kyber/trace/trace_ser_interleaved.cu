//
// trace_ser_interleaved.cu
// Interleaved valid-vs-random TVLA, NO L2 mitigation.
//
// Same interleaving structure as trace_ser_l2thrash_interleaved.cu but with no
// L2 thrash between kernels.  Use this as the unmitigated baseline for
// side-by-side comparison against the L2-thrash mitigated run.
//
// Class 0: fresh Kyber enc per iteration → timed dec (valid ciphertext)
// Class 1: fresh random bytes per iteration → timed dec (invalid ciphertext)
// Both classes use the same keypair K0.  Both classes run in the same binary
// invocation, strictly alternating, so both see the same GPU thermal state.
//
// Output: two CSV files, format: trace_id,kernel_name,elapsed_us
// Each trace includes a total_decaps row (sum of all per-kernel times).
//
// Usage:
//   echo "1 4 4 4 4 10000" | ./target/trace_ser_interleaved_kyber1024.out \
//     experiments/traces/baseline/kyber1024_interleaved_class0_n10000.csv \
//     experiments/traces/baseline/kyber1024_interleaved_class1_n10000.csv
// Build:
//   make -C baseline/atpqc-cuda trace_ser_interleaved
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
static inline cudaError_t sil_check(cudaError_t e,
                                     const char* call, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[SER_IL CUDA ERROR] %s:%d\n  call: %s\n  error: %s\n",
                     file, line, call, cudaGetErrorString(e));
        std::fflush(stderr);
        std::abort();
    }
    return e;
}
static inline CUresult sil_check(CUresult e,
                                  const char* call, const char* file, int line) {
    if (e != CUDA_SUCCESS) {
        const char* name = nullptr; const char* str = nullptr;
        cuGetErrorName(e, &name); cuGetErrorString(e, &str);
        std::fprintf(stderr, "[SER_IL CUDA ERROR] %s:%d\n  call: %s\n  error: %s (%s)\n",
                     file, line, call, str ? str : "?", name ? name : "?");
        std::fflush(stderr);
        std::abort();
    }
    return e;
}
#define CCC(call) sil_check((call), #call, __FILE__, __LINE__)

// Timed launch, no L2 thrash.
// total_us and fout come from the enclosing run_one scope.
#define LAUNCH_TIMED(label, obj, stream, ev_a, ev_b, record, fout, tid, ...) \
  do {                                                                        \
    auto _a = (obj).generate_args(__VA_ARGS__);                               \
    CCC(cudaEventRecord((ev_a), (stream)));                                   \
    CCC(cudaLaunchKernel((obj).get_func(), (obj).get_grid_dim(),              \
                         (obj).get_block_dim(), _a->get_args_ptr(),           \
                         (obj).get_shared_bytes(), (stream)));                \
    CCC(cudaEventRecord((ev_b), (stream)));                                   \
    CCC(cudaDeviceSynchronize());                                             \
    if (record) {                                                             \
      float _ms = 0.f;                                                        \
      CCC(cudaEventElapsedTime(&_ms, (ev_a), (ev_b)));                        \
      std::fprintf((fout), "%u,%s,%.3f\n", (tid), (label), _ms * 1000.f);   \
      total_us += _ms * 1000.f;                                               \
    }                                                                         \
  } while (0)

namespace atpqc_cuda::kyber::trace_ser_interleaved {

using rng_type = rng::std_random_device;
using variant  = variants::KYBER_VARIANT;
constexpr variant variant_v;

void run(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr,
            "Usage: %s class0.csv class1.csv\n"
            "  stdin: ninputs genmat_nw genvec_nw genpoly_nw fips_nw ntraces_per_class\n",
            argv[0]);
    std::abort();
  }

  FILE* out0 = std::fopen(argv[1], "w");
  FILE* out1 = std::fopen(argv[2], "w");
  if (!out0 || !out1) {
    fprintf(stderr, "[ser_il] failed to open output files: %s %s\n",
            argv[1], argv[2]);
    std::abort();
  }

  unsigned ninputs, genmat_nw, genvec_nw, genpoly_nw, fips_nw;
  unsigned ntraces_per_class;
  std::cin >> ninputs >> genmat_nw >> genvec_nw >> genpoly_nw >> fips_nw
           >> ntraces_per_class;

  const char* variant_name =
    params::ciphertextbytes<variant> == 768  ? "Kyber-512"  :
    params::ciphertextbytes<variant> == 1088 ? "Kyber-768"  : "Kyber-1024";

  fprintf(stderr,
          "[ser_il] %s  ninputs=%u  ntraces_per_class=%u  (NO L2 mitigation)\n",
          variant_name, ninputs, ntraces_per_class);
  std::fflush(stderr);

  // ── Device memory ─────────────────────────────────────────────────────────
  cuda_resource::device_pitched_memory<std::uint8_t> pk0_d(params::publickeybytes<variant>,  ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> sk0_d(params::secretkeybytes<variant>,  ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ct_d (params::ciphertextbytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ss_d (params::ssbytes,                  ninputs);

  // ── Host memory ───────────────────────────────────────────────────────────
  cuda_resource::pinned_memory<std::uint8_t> sk0_h(params::secretkeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> rand_ct_h(params::ciphertextbytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ss_enc_h(params::ssbytes * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ss_dec_h(params::ssbytes * ninputs);

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
  primitive::ccakem_enc::mem_resource<variant>     enc0_mr(ninputs);
  primitive::ccakem_dec::mem_resource<variant>     dec_mr(ninputs);

  // ── Generate K0 ───────────────────────────────────────────────────────────
  fprintf(stderr, "[ser_il] generating K0...\n");
  {
    cuda_resource::graph kp_graph;
    cudaGraphNode_t dummy, pk_avail, sk_avail;
    CCC(cudaGraphAddEmptyNode(&dummy, kp_graph, nullptr, 0));
    randombytes(keypair_mr.pke_keypair_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    randombytes(keypair_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    auto kr_args = keypair.join_graph(
        kp_graph,
        pk0_d.get_ptr(), pk0_d.get_pitch(), dummy, &pk_avail,
        sk0_d.get_ptr(), sk0_d.get_pitch(), dummy, &sk_avail,
        keypair_mr);
    cudaGraphNode_t cpysk;
    {
      cudaMemcpy3DParms p = {};
      p.srcPtr = make_cudaPitchedPtr(sk0_d.get_ptr(), sk0_d.get_pitch(),
          params::secretkeybytes<variant>, ninputs);
      p.dstPtr = make_cudaPitchedPtr(sk0_h.get_ptr(),
          params::secretkeybytes<variant>,
          params::secretkeybytes<variant>, ninputs);
      p.extent = make_cudaExtent(params::secretkeybytes<variant>, ninputs, 1);
      p.kind = cudaMemcpyDeviceToHost;
      std::array dep{sk_avail};
      CCC(cudaGraphAddMemcpyNode(&cpysk, kp_graph, dep.data(), dep.size(), &p));
    }
    cuda_resource::graph_exec kp_exec(kp_graph);
    cuda_resource::stream     kp_stream(cudaStreamNonBlocking);
    CCC(cudaGraphLaunch(kp_exec, kp_stream));
    CCC(cudaStreamSynchronize(kp_stream));
    (void)kr_args;
  }
  fprintf(stderr, "[ser_il] K0 generated.\n");

  // ── Build reusable enc graph for K0 ──────────────────────────────────────
  cuda_resource::graph enc0_graph;
  cudaGraphNode_t dummy_e0, ct0_avail, ss0_avail, pk0_used;
  CCC(cudaGraphAddEmptyNode(&dummy_e0, enc0_graph, nullptr, 0));
  randombytes(enc0_mr.rand_host.get_ptr(), params::symbytes * ninputs);
  auto er_args0 = enc.join_graph(
      enc0_graph,
      ct_d.get_ptr(),   ct_d.get_pitch(),   dummy_e0, &ct0_avail,
      ss_d.get_ptr(),   ss_d.get_pitch(),   dummy_e0, &ss0_avail,
      pk0_d.get_ptr(),  pk0_d.get_pitch(),  dummy_e0, &pk0_used,
      enc0_mr);

  cuda_resource::graph_exec enc0_exec(enc0_graph);
  cuda_resource::stream     enc_stream(cudaStreamNonBlocking);

  CCC(cudaMemcpy2D(sk0_d.get_ptr(), sk0_d.get_pitch(),
      sk0_h.get_ptr(), params::secretkeybytes<variant>,
      params::secretkeybytes<variant>, ninputs, cudaMemcpyHostToDevice));
  CCC(cudaDeviceSynchronize());

  auto do_enc0 = [&]() {
    randombytes(enc0_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    CCC(cudaGraphLaunch(enc0_exec, enc_stream));
    CCC(cudaStreamSynchronize(enc_stream));
  };

  auto do_rand_ct = [&]() {
    randombytes(rand_ct_h.get_ptr(), params::ciphertextbytes<variant> * ninputs);
    CCC(cudaMemcpy2D(
        ct_d.get_ptr(), ct_d.get_pitch(),
        rand_ct_h.get_ptr(), params::ciphertextbytes<variant>,
        params::ciphertextbytes<variant>, ninputs, cudaMemcpyHostToDevice));
    CCC(cudaDeviceSynchronize());
  };

  // ── Verify K0 enc+dec ─────────────────────────────────────────────────────
  {
    do_enc0();
    CCC(cudaMemcpy2D(ss_enc_h.get_ptr(), params::ssbytes,
        ss_d.get_ptr(), ss_d.get_pitch(),
        params::ssbytes, ninputs, cudaMemcpyDeviceToHost));
    CCC(cudaDeviceSynchronize());

    cuda_resource::stream verify_stream(cudaStreamNonBlocking);
    const std::uint8_t* pk_in_sk_v = sk0_d.get_ptr() + params::indcpa_secretkeybytes<variant>;
    const std::uint8_t* z_in_sk_v  = sk0_d.get_ptr() + params::secretkeybytes<variant> - params::symbytes;
    std::uint8_t* buf_v  = dec_mr.buf.get_ptr(); std::size_t buf_vp = dec_mr.buf.get_pitch();
    std::uint8_t* kr_v   = dec_mr.kr.get_ptr();  std::size_t kr_vp  = dec_mr.kr.get_pitch();
    std::uint8_t* cmp_v  = dec_mr.cmp.get_ptr(); std::size_t cmp_vp = dec_mr.cmp.get_pitch();
    short2* bp_dv   = dec_mr.pke_dec_mr.bp.get_ptr();
    short2* skpv_dv = dec_mr.pke_dec_mr.skpv.get_ptr();
    short2* v_dv    = dec_mr.pke_dec_mr.v.get_ptr();
    short2* mp_dv   = dec_mr.pke_dec_mr.mp.get_ptr();
    short2* at_ev   = dec_mr.pke_enc_mr.at.get_ptr();
    short2* sp_ev   = dec_mr.pke_enc_mr.sp.get_ptr();
    short2* pkpv_ev = dec_mr.pke_enc_mr.pkpv.get_ptr();
    short2* ep_ev   = dec_mr.pke_enc_mr.ep.get_ptr();
    short2* bp_ev   = dec_mr.pke_enc_mr.bp.get_ptr();
    short2* v_ev    = dec_mr.pke_enc_mr.v.get_ptr();
    short2* k_ev    = dec_mr.pke_enc_mr.k.get_ptr();
    short2* epp_ev  = dec_mr.pke_enc_mr.epp.get_ptr();

    auto vdec = [&](auto& obj, auto... args) {
      auto _a = obj.generate_args(args...);
      CCC(cudaLaunchKernel(obj.get_func(), obj.get_grid_dim(), obj.get_block_dim(),
                           _a->get_args_ptr(), obj.get_shared_bytes(), verify_stream));
      CCC(cudaStreamSynchronize(verify_stream));
    };
    vdec(decompressu,  bp_dv, ct_d.get_ptr(), ct_d.get_pitch());
    vdec(decompressv,  v_dv,
         ct_d.get_ptr() + params::polyveccompressedbytes<variant>, ct_d.get_pitch());
    vdec(decodes,      skpv_dv, sk0_d.get_ptr(), sk0_d.get_pitch());
    vdec(fwdnttvec_u,  bp_dv);
    vdec(stimesu,      mp_dv, skpv_dv, bp_dv);
    vdec(intt_su,      mp_dv);
    vdec(psub,         mp_dv, v_dv, mp_dv);
    vdec(tomsg,        buf_v, buf_vp, mp_dv);
    CCC(cudaMemcpy2DAsync(buf_v + params::symbytes, buf_vp,
        sk0_d.get_ptr() + (params::secretkeybytes<variant> - 2 * params::symbytes),
        sk0_d.get_pitch(), params::symbytes, ninputs,
        cudaMemcpyDeviceToDevice, verify_stream));
    CCC(cudaStreamSynchronize(verify_stream));
    vdec(dec_hash_coin, kr_v, kr_vp, buf_v, buf_vp, 2 * params::symbytes);
    vdec(decodet,       pkpv_ev, pk_in_sk_v, sk0_d.get_pitch());
    vdec(frommsg,       k_ev, buf_v, buf_vp);
    vdec(generate_at,   at_ev,
         pk_in_sk_v + params::polyvecbytes<variant>, sk0_d.get_pitch());
    vdec(generate_r,    sp_ev, kr_v + params::symbytes, kr_vp,
         static_cast<std::uint8_t>(0));
    vdec(generate_e1,   ep_ev, kr_v + params::symbytes, kr_vp,
         static_cast<std::uint8_t>(params::k<variant>));
    vdec(generate_e2,   epp_ev, kr_v + params::symbytes, kr_vp,
         static_cast<std::uint8_t>(params::k<variant> * 2));
    vdec(fwdnttvec_r,   sp_ev);
    vdec(mtv,           bp_ev, at_ev, sp_ev);
    vdec(ttimesr,       v_ev, pkpv_ev, sp_ev);
    vdec(intt_ar,       bp_ev);
    vdec(intt_tr,       v_ev);
    vdec(vpv,           bp_ev, bp_ev, ep_ev);
    vdec(padd3,         v_ev, v_ev, epp_ev, k_ev);
    vdec(compressu,     cmp_v, cmp_vp, bp_ev);
    vdec(compressv,     cmp_v + params::polyveccompressedbytes<variant>, cmp_vp, v_ev);
    vdec(dec_hash_ct,   kr_v + params::symbytes, kr_vp,
         ct_d.get_ptr(), ct_d.get_pitch(), params::ciphertextbytes<variant>);
    vdec(dec_verify_cmov, kr_v, kr_vp, z_in_sk_v, sk0_d.get_pitch(),
         params::symbytes, ct_d.get_ptr(), ct_d.get_pitch(),
         cmp_v, cmp_vp, params::ciphertextbytes<variant>);
    vdec(dec_kdf,       ss_d.get_ptr(), ss_d.get_pitch(),
         kr_v, kr_vp, 2 * params::symbytes);

    CCC(cudaMemcpy2D(ss_dec_h.get_ptr(), params::ssbytes,
        ss_d.get_ptr(), ss_d.get_pitch(),
        params::ssbytes, ninputs, cudaMemcpyDeviceToHost));
    CCC(cudaDeviceSynchronize());

    bool match = (memcmp(ss_enc_h.get_ptr(), ss_dec_h.get_ptr(),
                         params::ssbytes * ninputs) == 0);
    fprintf(stderr, "[ser_il] K0 enc/dec shared secret match: %s\n",
            match ? "YES" : "NO");
    if (!match) {
      fprintf(stderr, "[ser_il] ABORT: enc/dec mismatch — check key setup\n");
      std::abort();
    }
    std::fflush(stderr);
  }

  // ── Convenience pointers ─────────────────────────────────────────────────
  std::uint8_t* buf_ptr   = dec_mr.buf.get_ptr();
  std::size_t   buf_pitch = dec_mr.buf.get_pitch();
  std::uint8_t* kr_ptr    = dec_mr.kr.get_ptr();
  std::size_t   kr_pitch  = dec_mr.kr.get_pitch();
  std::uint8_t* cmp_ptr   = dec_mr.cmp.get_ptr();
  std::size_t   cmp_pitch = dec_mr.cmp.get_pitch();

  const std::uint8_t* pk_in_sk =
      sk0_d.get_ptr() + params::indcpa_secretkeybytes<variant>;
  const std::uint8_t* z_in_sk =
      sk0_d.get_ptr() + params::secretkeybytes<variant> - params::symbytes;

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

  auto run_one = [&](unsigned trace_i, bool record, FILE* fout) {
    float total_us = 0.f;

    LAUNCH_TIMED("decompress_u", decompressu, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 bp_dec, ct_d.get_ptr(), ct_d.get_pitch());

    LAUNCH_TIMED("decompress_v", decompressv, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 v_dec,
                 ct_d.get_ptr() + params::polyveccompressedbytes<variant>,
                 ct_d.get_pitch());

    LAUNCH_TIMED("decode_s", decodes, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 skpv_dec, sk0_d.get_ptr(), sk0_d.get_pitch());

    LAUNCH_TIMED("fwdntt_u", fwdnttvec_u, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 bp_dec);

    LAUNCH_TIMED("s_times_u", stimesu, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 mp_dec, skpv_dec, bp_dec);

    LAUNCH_TIMED("intt_su", intt_su, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 mp_dec);

    LAUNCH_TIMED("v_minus_su", psub, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 mp_dec, v_dec, mp_dec);

    LAUNCH_TIMED("poly_tomsg", tomsg, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 buf_ptr, buf_pitch, mp_dec);

    CCC(cudaMemcpy2DAsync(
        buf_ptr + params::symbytes, buf_pitch,
        sk0_d.get_ptr() + (params::secretkeybytes<variant> - 2 * params::symbytes),
        sk0_d.get_pitch(),
        params::symbytes, ninputs,
        cudaMemcpyDeviceToDevice, stream));
    CCC(cudaDeviceSynchronize());

    LAUNCH_TIMED("hash_coin", dec_hash_coin, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 kr_ptr, kr_pitch, buf_ptr, buf_pitch, 2 * params::symbytes);

    LAUNCH_TIMED("decode_t", decodet, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 pkpv_enc, pk_in_sk, sk0_d.get_pitch());

    LAUNCH_TIMED("poly_frommsg", frommsg, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 k_enc, buf_ptr, buf_pitch);

    LAUNCH_TIMED("generate_at", generate_at, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 at_enc,
                 pk_in_sk + params::polyvecbytes<variant>,
                 sk0_d.get_pitch());

    LAUNCH_TIMED("generate_r", generate_r, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 sp_enc, kr_ptr + params::symbytes, kr_pitch,
                 static_cast<std::uint8_t>(0));

    LAUNCH_TIMED("generate_e1", generate_e1, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 ep_enc, kr_ptr + params::symbytes, kr_pitch,
                 static_cast<std::uint8_t>(params::k<variant>));

    LAUNCH_TIMED("generate_e2", generate_e2, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 epp_enc, kr_ptr + params::symbytes, kr_pitch,
                 static_cast<std::uint8_t>(params::k<variant> * 2));

    LAUNCH_TIMED("fwdntt_r", fwdnttvec_r, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 sp_enc);

    LAUNCH_TIMED("a_times_r", mtv, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 bp_enc, at_enc, sp_enc);

    LAUNCH_TIMED("t_times_r", ttimesr, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 v_enc, pkpv_enc, sp_enc);

    LAUNCH_TIMED("intt_ar", intt_ar, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 bp_enc);

    LAUNCH_TIMED("intt_tr", intt_tr, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 v_enc);

    LAUNCH_TIMED("ar_plus_e1", vpv, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 bp_enc, bp_enc, ep_enc);

    LAUNCH_TIMED("tr_plus_e2_plus_m", padd3, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 v_enc, v_enc, epp_enc, k_enc);

    LAUNCH_TIMED("compress_u", compressu, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 cmp_ptr, cmp_pitch, bp_enc);

    LAUNCH_TIMED("compress_v", compressv, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 cmp_ptr + params::polyveccompressedbytes<variant>,
                 cmp_pitch, v_enc);

    LAUNCH_TIMED("hash_ct", dec_hash_ct, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 kr_ptr + params::symbytes, kr_pitch,
                 ct_d.get_ptr(), ct_d.get_pitch(),
                 params::ciphertextbytes<variant>);

    LAUNCH_TIMED("verify_cmov", dec_verify_cmov, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 kr_ptr, kr_pitch,
                 z_in_sk, sk0_d.get_pitch(), params::symbytes,
                 ct_d.get_ptr(), ct_d.get_pitch(),
                 cmp_ptr, cmp_pitch,
                 params::ciphertextbytes<variant>);

    LAUNCH_TIMED("kdf", dec_kdf, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 ss_d.get_ptr(), ss_d.get_pitch(),
                 kr_ptr, kr_pitch, 2 * params::symbytes);

    if (record)
      std::fprintf(fout, "%u,total_decaps,%.3f\n", trace_i, total_us);
  };

  // ── Warmup: 50 interleaved iterations ────────────────────────────────────
  fprintf(stderr, "[ser_il warmup] 50 iterations (interleaved)...\n");
  for (unsigned w = 0; w < 50; w++) {
    if (w % 2 == 0) { do_enc0();    run_one(0, false, nullptr); }
    else             { do_rand_ct(); run_one(0, false, nullptr); }
  }
  fprintf(stderr, "[ser_il warmup] done\n");
  std::fflush(stderr);

  // ── CSV headers ───────────────────────────────────────────────────────────
  auto write_header = [&](FILE* f, int cls) {
    fprintf(f, "# mlkem-gpu-sec per-kernel serialized timing traces (unmitigated, interleaved)\n");
    fprintf(f, "# variant: %s\n", variant_name);
    fprintf(f, "# class: %d (%s)\n", cls, cls == 0 ? "valid ct (fresh enc per trace)" : "random bytes (fresh per trace)");
    fprintf(f, "# n_traces: %u\n", ntraces_per_class);
    fprintf(f, "# mitigation: none\n");
    fprintf(f, "# interleaved: yes (both classes in same run, alternating per iteration)\n");
    fprintf(f, "# unit: microseconds (elapsed_us)\n");
    fprintf(f, "trace_id,kernel_name,elapsed_us\n");
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
      do_enc0();
      run_one(cnt0, true, out0);
      cnt0++;
    }
    if (cnt1 < ntraces_per_class) {
      do_rand_ct();
      run_one(cnt1, true, out1);
      cnt1++;
    }
    unsigned total     = cnt0 + cnt1;
    unsigned total_max = ntraces_per_class * 2;
    unsigned pct = total * 100 / total_max;
    if (pct / 10 > last_pct / 10) {
      fprintf(stderr, "  %u%% (class0: %u  class1: %u)\n", pct, cnt0, cnt1);
      std::fflush(stderr);
      last_pct = pct;
    }
  }

  fprintf(stderr,
          "[ser_il] done — class0: %u traces  class1: %u traces\n",
          cnt0, cnt1);
  std::fflush(stderr);

  std::fclose(out0);
  std::fclose(out1);

  CCC(cudaEventDestroy(ev_start));
  CCC(cudaEventDestroy(ev_stop));
  (void)er_args0;
}

}  // namespace atpqc_cuda::kyber::trace_ser_interleaved

int main(int argc, char** argv) {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::trace_ser_interleaved::run(argc, argv);
    CCC(cuCtxSynchronize());
  }

  return 0;
}
