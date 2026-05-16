//
// trace_keydist.cu
// Interleaved key-distinguishing per-kernel TVLA for Kyber-1024.
//
// Both keypairs K0 and K1 are held in GPU memory simultaneously.
// A single loop of 2*ntraces_per_class iterations strictly alternates:
//   class 0: fresh enc under K0, timed dec under K0
//   class 1: fresh enc under K1, timed dec under K1
// Interleaving within one binary run eliminates temporal GPU clock/thermal
// drift that was producing the spurious 27/27 uniform-offset artifact.
//
// Output: two CSV files (one per class), same format as trace_serialized.
//
// Usage:
//   echo "1 4 4 4 4 100000" | ./target/trace_keydist_kyber1024.out \
//     experiments/traces/per_kernel/kyber1024_keydist_class0_n100000_per_kernel.csv \
//     experiments/traces/per_kernel/kyber1024_keydist_class1_n100000_per_kernel.csv
//
// Build:
//   make -C baseline/atpqc-cuda trace_keydist_kyber1024
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
#define KYBER_VARIANT kyber1024
#endif

#undef  CCC
static inline cudaError_t kd_check(cudaError_t e,
                                    const char* call, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[KEYDIST CUDA ERROR] %s:%d\n  call: %s\n  error: %s\n",
                     file, line, call, cudaGetErrorString(e));
        std::fflush(stderr);
        std::abort();
    }
    return e;
}
static inline CUresult kd_check(CUresult e,
                                 const char* call, const char* file, int line) {
    if (e != CUDA_SUCCESS) {
        const char* name = nullptr; const char* str = nullptr;
        cuGetErrorName(e, &name); cuGetErrorString(e, &str);
        std::fprintf(stderr, "[KEYDIST CUDA ERROR] %s:%d\n  call: %s\n  error: %s (%s)\n",
                     file, line, call, str ? str : "?", name ? name : "?");
        std::fflush(stderr);
        std::abort();
    }
    return e;
}
#define CCC(call) kd_check((call), #call, __FILE__, __LINE__)

namespace atpqc_cuda::kyber::trace_keydist {

using rng_type = rng::std_random_device;
using variant  = variants::KYBER_VARIANT;
constexpr variant variant_v;

// LAUNCH_TIMED: time one kernel, write CSV row to fout when record=true.
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

// LAUNCH_TIMED_CAPTURE: always writes elapsed µs into out_us regardless of record.
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

void trace_keydist(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s class0.csv class1.csv\n  stdin: ninputs genmat_nw genvec_nw genpoly_nw fips_nw ntraces_per_class\n", argv[0]);
    std::abort();
  }
  FILE* out0 = std::fopen(argv[1], "w");
  FILE* out1 = std::fopen(argv[2], "w");
  if (!out0 || !out1) {
    fprintf(stderr, "[keydist] failed to open output files: %s %s\n", argv[1], argv[2]);
    std::abort();
  }

  unsigned ninputs, genmat_nw, genvec_nw, genpoly_nw, fips_nw;
  unsigned ntraces_per_class;
  std::cin >> ninputs >> genmat_nw >> genvec_nw >> genpoly_nw >> fips_nw
           >> ntraces_per_class;

  // ── Diagnostic 1: parameter set ───────────────────────────────────────────
  fprintf(stderr, "[keydist] ntraces_per_class=%u  out0=%s  out1=%s\n",
          ntraces_per_class, argv[1], argv[2]);
  fprintf(stderr, "[keydist] ciphertextbytes : %u\n", params::ciphertextbytes<variant>);
  fprintf(stderr, "[keydist] publickeybytes  : %u\n", params::publickeybytes<variant>);
  fprintf(stderr, "[keydist] secretkeybytes  : %u\n", params::secretkeybytes<variant>);
  fprintf(stderr, "[keydist] ssbytes         : %u\n", params::ssbytes);
  fprintf(stderr, "[keydist] k               : %u\n", params::k<variant>);
  if (params::ciphertextbytes<variant> != 1568) {
    fprintf(stderr, "[keydist] WRONG VARIANT: expected ciphertextbytes=1568 (Kyber-1024)\n");
    std::abort();
  }
  std::fflush(stderr);

  // ── Device memory — K0 and K1 keys live simultaneously; ct/ss are shared scratch
  cuda_resource::device_pitched_memory<std::uint8_t> pk0_d(params::publickeybytes<variant>,  ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> sk0_d(params::secretkeybytes<variant>,  ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> pk1_d(params::publickeybytes<variant>,  ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> sk1_d(params::secretkeybytes<variant>,  ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ct_d (params::ciphertextbytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ss_d (params::ssbytes,                  ninputs);

  // ── Host memory ───────────────────────────────────────────────────────────
  cuda_resource::pinned_memory<std::uint8_t> sk0_h(params::secretkeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> sk1_h(params::secretkeybytes<variant> * ninputs);
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
  // Two separate enc mem_resources so each key's randomness buffer is independent.
  primitive::ccakem_enc::mem_resource<variant> enc0_mr(ninputs);
  primitive::ccakem_enc::mem_resource<variant> enc1_mr(ninputs);
  primitive::ccakem_dec::mem_resource<variant> dec_mr(ninputs);

  // ── Keypair generation helper ─────────────────────────────────────────────
  // Runs a one-shot graph; kr_args lifetime is confined to the block (graph
  // completes before the block exits and kr_args is destroyed).
  auto gen_keypair = [&](
      cuda_resource::device_pitched_memory<std::uint8_t>& pk_d,
      cuda_resource::device_pitched_memory<std::uint8_t>& sk_d,
      cuda_resource::pinned_memory<std::uint8_t>& sk_h) {
    cuda_resource::graph kp_graph;
    cudaGraphNode_t dummy, pk_avail, sk_avail;
    CCC(cudaGraphAddEmptyNode(&dummy, kp_graph, nullptr, 0));
    randombytes(keypair_mr.pke_keypair_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    randombytes(keypair_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    auto kr_args = keypair.join_graph(
        kp_graph,
        pk_d.get_ptr(), pk_d.get_pitch(), dummy, &pk_avail,
        sk_d.get_ptr(), sk_d.get_pitch(), dummy, &sk_avail,
        keypair_mr);
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
      CCC(cudaGraphAddMemcpyNode(&cpysk, kp_graph, dep.data(), dep.size(), &p));
    }
    cuda_resource::graph_exec kp_exec(kp_graph);
    cuda_resource::stream     kp_stream(cudaStreamNonBlocking);
    CCC(cudaGraphLaunch(kp_exec, kp_stream));
    CCC(cudaStreamSynchronize(kp_stream));
    // kr_args destroyed here; graph is done, safe.
  };

  // ── Generate K0 and K1 ────────────────────────────────────────────────────
  fprintf(stderr, "[keydist] generating K0...\n");
  gen_keypair(pk0_d, sk0_d, sk0_h);
  fprintf(stderr, "[keydist] generating K1...\n");
  gen_keypair(pk1_d, sk1_d, sk1_h);

  // ── Build reusable enc graphs (one per key) ───────────────────────────────
  // er_args0/er_args1 MUST remain alive for the entire lifetime of
  // enc0_exec/enc1_exec. Graph nodes hold pointers into these objects.
  // Declared here at function scope so they outlive all graph launches.
  cuda_resource::graph enc0_graph, enc1_graph;
  cudaGraphNode_t dummy_e0, ct0_avail, ss0_avail, pk0_used;
  cudaGraphNode_t dummy_e1, ct1_avail, ss1_avail, pk1_used;
  CCC(cudaGraphAddEmptyNode(&dummy_e0, enc0_graph, nullptr, 0));
  CCC(cudaGraphAddEmptyNode(&dummy_e1, enc1_graph, nullptr, 0));

  randombytes(enc0_mr.rand_host.get_ptr(), params::symbytes * ninputs);
  auto er_args0 = enc.join_graph(
      enc0_graph,
      ct_d.get_ptr(),  ct_d.get_pitch(),  dummy_e0, &ct0_avail,
      ss_d.get_ptr(),  ss_d.get_pitch(),  dummy_e0, &ss0_avail,
      pk0_d.get_ptr(), pk0_d.get_pitch(), dummy_e0, &pk0_used,
      enc0_mr);

  randombytes(enc1_mr.rand_host.get_ptr(), params::symbytes * ninputs);
  auto er_args1 = enc.join_graph(
      enc1_graph,
      ct_d.get_ptr(),  ct_d.get_pitch(),  dummy_e1, &ct1_avail,
      ss_d.get_ptr(),  ss_d.get_pitch(),  dummy_e1, &ss1_avail,
      pk1_d.get_ptr(), pk1_d.get_pitch(), dummy_e1, &pk1_used,
      enc1_mr);

  cuda_resource::graph_exec enc0_exec(enc0_graph), enc1_exec(enc1_graph);
  cuda_resource::stream     enc_stream(cudaStreamNonBlocking);

  // ── Upload sk keys to device ──────────────────────────────────────────────
  CCC(cudaMemcpy2D(sk0_d.get_ptr(), sk0_d.get_pitch(),
      sk0_h.get_ptr(), params::secretkeybytes<variant>,
      params::secretkeybytes<variant>, ninputs, cudaMemcpyHostToDevice));
  CCC(cudaMemcpy2D(sk1_d.get_ptr(), sk1_d.get_pitch(),
      sk1_h.get_ptr(), params::secretkeybytes<variant>,
      params::secretkeybytes<variant>, ninputs, cudaMemcpyHostToDevice));
  CCC(cudaDeviceSynchronize());

  // ── Dec working-memory convenience pointers ───────────────────────────────
  std::uint8_t* buf_ptr   = dec_mr.buf.get_ptr();
  std::size_t   buf_pitch = dec_mr.buf.get_pitch();
  std::uint8_t* kr_ptr    = dec_mr.kr.get_ptr();
  std::size_t   kr_pitch  = dec_mr.kr.get_pitch();
  std::uint8_t* cmp_ptr   = dec_mr.cmp.get_ptr();
  std::size_t   cmp_pitch = dec_mr.cmp.get_pitch();

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

  // Fresh enc: update randomness in the pinned buffer the graph reads, relaunch.
  auto do_enc0 = [&]() {
    randombytes(enc0_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    CCC(cudaGraphLaunch(enc0_exec, enc_stream));
    CCC(cudaStreamSynchronize(enc_stream));
  };
  auto do_enc1 = [&]() {
    randombytes(enc1_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    CCC(cudaGraphLaunch(enc1_exec, enc_stream));
    CCC(cudaStreamSynchronize(enc_stream));
  };

  // Serialized decaps timing. sk_ptr/sk_pitch select which key to use.
  // fout is only dereferenced when record=true (nullptr safe for record=false).
  // elapsed_first_us always receives the decompress_u timing for pilot checks.
  auto run_one = [&](unsigned trace_i, bool record, FILE* fout,
                     float* elapsed_first_us,
                     const std::uint8_t* sk_ptr, std::size_t sk_pitch) {
    const std::uint8_t* pk_in_sk =
        sk_ptr + params::indcpa_secretkeybytes<variant>;
    const std::uint8_t* z_in_sk =
        sk_ptr + params::secretkeybytes<variant> - params::symbytes;

    LAUNCH_TIMED_CAPTURE("decompress_u", *elapsed_first_us,
                 decompressu, stream, ev_start, ev_stop, record, fout, trace_i,
                 bp_dec, ct_d.get_ptr(), ct_d.get_pitch());

    LAUNCH_TIMED("decompress_v", decompressv, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 v_dec,
                 ct_d.get_ptr() + params::polyveccompressedbytes<variant>,
                 ct_d.get_pitch());

    LAUNCH_TIMED("decode_s", decodes, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 skpv_dec, sk_ptr, sk_pitch);

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
        sk_ptr + (params::secretkeybytes<variant> - 2 * params::symbytes),
        sk_pitch,
        params::symbytes, ninputs,
        cudaMemcpyDeviceToDevice, stream));
    CCC(cudaDeviceSynchronize());

    LAUNCH_TIMED("hash_coin", dec_hash_coin, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 kr_ptr, kr_pitch, buf_ptr, buf_pitch, 2 * params::symbytes);

    LAUNCH_TIMED("decode_t", decodet, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 pkpv_enc, pk_in_sk, sk_pitch);

    LAUNCH_TIMED("poly_frommsg", frommsg, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 k_enc, buf_ptr, buf_pitch);

    LAUNCH_TIMED("generate_at", generate_at, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 at_enc,
                 pk_in_sk + params::polyvecbytes<variant>,
                 sk_pitch);

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
                 z_in_sk, sk_pitch, params::symbytes,
                 ct_d.get_ptr(), ct_d.get_pitch(),
                 cmp_ptr, cmp_pitch,
                 params::ciphertextbytes<variant>);

    LAUNCH_TIMED("kdf", dec_kdf, stream, ev_start, ev_stop,
                 record, fout, trace_i,
                 ss_d.get_ptr(), ss_d.get_pitch(),
                 kr_ptr, kr_pitch, 2 * params::symbytes);
  };

  // ── Diagnostic 2: verify enc+dec produce matching shared secrets ──────────
  auto verify_key = [&](const char* label,
                        auto do_enc_fn,
                        const std::uint8_t* sk_ptr, std::size_t sk_pitch) {
    float dummy_us = 0.f;
    do_enc_fn();
    CCC(cudaMemcpy2D(ss_enc_h.get_ptr(), params::ssbytes,
        ss_d.get_ptr(), ss_d.get_pitch(),
        params::ssbytes, ninputs, cudaMemcpyDeviceToHost));
    CCC(cudaDeviceSynchronize());
    run_one(0, false, nullptr, &dummy_us, sk_ptr, sk_pitch);
    CCC(cudaMemcpy2D(ss_dec_h.get_ptr(), params::ssbytes,
        ss_d.get_ptr(), ss_d.get_pitch(),
        params::ssbytes, ninputs, cudaMemcpyDeviceToHost));
    CCC(cudaDeviceSynchronize());
    const auto* se = ss_enc_h.get_ptr();
    const auto* sd = ss_dec_h.get_ptr();
    fprintf(stderr, "[%s] enc ss[0..7]: %02x%02x%02x%02x%02x%02x%02x%02x\n",
            label, se[0],se[1],se[2],se[3],se[4],se[5],se[6],se[7]);
    fprintf(stderr, "[%s] dec ss[0..7]: %02x%02x%02x%02x%02x%02x%02x%02x\n",
            label, sd[0],sd[1],sd[2],sd[3],sd[4],sd[5],sd[6],sd[7]);
    bool match = (memcmp(se, sd, params::ssbytes * ninputs) == 0);
    fprintf(stderr, "[%s] shared secret match: %s\n", label, match ? "YES" : "NO");
    if (!match) {
      fprintf(stderr, "[%s] ABORT: enc/dec shared secrets do not match\n", label);
      std::abort();
    }
    std::fflush(stderr);
  };

  verify_key("K0", do_enc0, sk0_d.get_ptr(), sk0_d.get_pitch());
  verify_key("K1", do_enc1, sk1_d.get_ptr(), sk1_d.get_pitch());

  // ── Warmup: 200 iterations, strictly alternating ──────────────────────────
  fprintf(stderr, "[warmup] 200 iterations (interleaved)...\n");
  float _wu = 0.f;
  for (unsigned w = 0; w < 200; w++) {
    if (w % 2 == 0) {
      do_enc0();
      run_one(0, false, nullptr, &_wu, sk0_d.get_ptr(), sk0_d.get_pitch());
    } else {
      do_enc1();
      run_one(0, false, nullptr, &_wu, sk1_d.get_ptr(), sk1_d.get_pitch());
    }
  }
  fprintf(stderr, "[warmup] done\n");

  // ── Diagnostic 3: pilot — 1000 per class, interleaved, print means ────────
  {
    fprintf(stderr, "[pilot] collecting 1000 traces per class (interleaved)...\n");
    double sum0 = 0.0, sum1 = 0.0;
    unsigned cnt0 = 0, cnt1 = 0;
    float us = 0.f;
    while (cnt0 < 1000 || cnt1 < 1000) {
      if (cnt0 < 1000) {
        do_enc0();
        run_one(cnt0, false, nullptr, &us, sk0_d.get_ptr(), sk0_d.get_pitch());
        sum0 += us; cnt0++;
      }
      if (cnt1 < 1000) {
        do_enc1();
        run_one(cnt1, false, nullptr, &us, sk1_d.get_ptr(), sk1_d.get_pitch());
        sum1 += us; cnt1++;
      }
    }
    double mean0 = sum0 / 1000.0, mean1 = sum1 / 1000.0;
    double diff  = (mean0 > mean1) ? (mean0 - mean1) : (mean1 - mean0);
    fprintf(stderr, "[pilot] decompress_u class0 mean: %.3f µs\n", mean0);
    fprintf(stderr, "[pilot] decompress_u class1 mean: %.3f µs\n", mean1);
    fprintf(stderr, "[pilot] diff: %.3f µs\n", diff);
    if (diff > 2.0) {
      fprintf(stderr, "[pilot] STOP: means differ by %.3f µs (threshold 2 µs) — "
                      "diagnose before collecting full traces\n", diff);
      std::abort();
    }
    fprintf(stderr, "[pilot] OK\n");
    std::fflush(stderr);
  }

  // ── CSV headers ───────────────────────────────────────────────────────────
  auto write_header = [&](FILE* f, int cls) {
    fprintf(f, "# mlkem-gpu-sec per-kernel key-distinguishing timing traces\n");
    fprintf(f, "# variant: Kyber-1024  ciphertextbytes=%u k=%u\n",
            params::ciphertextbytes<variant>, params::k<variant>);
    fprintf(f, "# class: %d (keypair K%d, fresh ct per trace, interleaved with other class)\n",
            cls, cls);
    fprintf(f, "# n_traces: %u\n", ntraces_per_class);
    fprintf(f, "# unit: microseconds (elapsed_us)\n");
    fprintf(f, "trace_id,kernel_name,elapsed_us\n");
  };
  write_header(out0, 0);
  write_header(out1, 1);
  std::fflush(out0);
  std::fflush(out1);

  // ── Interleaved collection ────────────────────────────────────────────────
  // class 0 and class 1 alternate every iteration so both experience the same
  // GPU thermal/clock state throughout the run.
  unsigned cnt0 = 0, cnt1 = 0;
  float first_us = 0.f;
  unsigned last_pct = 0;
  while (cnt0 < ntraces_per_class || cnt1 < ntraces_per_class) {
    if (cnt0 < ntraces_per_class) {
      do_enc0();
      run_one(cnt0, true, out0, &first_us, sk0_d.get_ptr(), sk0_d.get_pitch());
      cnt0++;
    }
    if (cnt1 < ntraces_per_class) {
      do_enc1();
      run_one(cnt1, true, out1, &first_us, sk1_d.get_ptr(), sk1_d.get_pitch());
      cnt1++;
    }
    unsigned total = cnt0 + cnt1;
    unsigned total_max = ntraces_per_class * 2;
    unsigned pct = total * 100 / total_max;
    if (pct / 10 > last_pct / 10) {
      fprintf(stderr, "  %u%% (class0: %u  class1: %u)\n", pct, cnt0, cnt1);
      std::fflush(stderr);
      last_pct = pct;
    }
  }

  fprintf(stderr, "[keydist] collection done — class0: %u traces  class1: %u traces\n",
          cnt0, cnt1);
  std::fflush(stderr);

  std::fclose(out0);
  std::fclose(out1);

  CCC(cudaEventDestroy(ev_start));
  CCC(cudaEventDestroy(ev_stop));
}

}  // namespace atpqc_cuda::kyber::trace_keydist

int main(int argc, char** argv) {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::trace_keydist::trace_keydist(argc, argv);
    CCC(cuCtxSynchronize());
  }

  return 0;
}
