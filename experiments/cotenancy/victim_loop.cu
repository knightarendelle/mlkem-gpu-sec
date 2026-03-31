//
// experiments/cotenancy/victim_loop.cu
// Victim program for the co-tenancy side-channel experiment.
//
// Runs ccakem_dec in a loop for exactly DURATION_SEC seconds,
// alternating between class 0 (valid ciphertexts) and class 1
// (random ciphertexts) every CLASS_SWITCH_INTERVAL operations.
//
// Output: /logs/victim_log.csv
//   header: timestamp_us,class
//   rows:   <us since program start>,<0 or 1>
//
// Compile via Makefile target (from baseline/atpqc-cuda/):
//   make victim_loop          # kyber512 (default)
//   make victim_loop_kyber768
//   make victim_loop_kyber1024
//

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lib/cuda_debug.hpp"
#include "lib/cuda_resource.hpp"
#include "lib/fips202_ws/host.cuh"
#include "lib/kyber/arithmetic_mt/host.cuh"
#include "lib/kyber/endecode_mt/host.cuh"
#include "lib/kyber/genpoly_warp/host.cuh"
#include "lib/kyber/ntt_ctgs_128t/host.cuh"
#include "lib/kyber/ntt_ctgs_64t/host.cuh"
#include "lib/kyber/params.cuh"
#include "lib/kyber/primitive/ccakem_dec.cuh"
#include "lib/kyber/primitive/ccakem_enc.cuh"
#include "lib/kyber/primitive/ccakem_keypair.cuh"
#include "lib/kyber/primitive/cpapke_dec.cuh"
#include "lib/kyber/primitive/cpapke_enc.cuh"
#include "lib/kyber/primitive/cpapke_keypair.cuh"
#include "lib/kyber/symmetric_ws/host.cuh"
#include "lib/kyber/variants.cuh"
#include "lib/rng/std_random_device.hpp"
#include "lib/verify_cmov_ws/host.cuh"

#ifndef KYBER_VARIANT
#define KYBER_VARIANT kyber512
#endif

static constexpr unsigned DURATION_SEC         = 30;
static constexpr unsigned CLASS_SWITCH_INTERVAL = 500;
static constexpr unsigned WARMUP_ITERS          = 200;

namespace atpqc_cuda::kyber::victim_loop {

using rng_type = rng::std_random_device;
using variant  = variants::KYBER_VARIANT;
constexpr variant variant_v;

void run() {
  constexpr unsigned ninputs    = 1;
  constexpr unsigned genmat_nw  = 4;
  constexpr unsigned genvec_nw  = 4;
  constexpr unsigned genpoly_nw = 4;
  constexpr unsigned fips_nw    = 4;

  // ── Device memory ───────────────────────────────────────
  cuda_resource::device_pitched_memory<std::uint8_t> pk_d(
      params::publickeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> sk_d(
      params::secretkeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ct_d(
      params::ciphertextbytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ss_d(
      params::ssbytes, ninputs);

  // ── Pinned host memory ──────────────────────────────────
  cuda_resource::pinned_memory<std::uint8_t> pk_h(
      params::publickeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> sk_h(
      params::secretkeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ct_h(
      params::ciphertextbytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ss_h(
      params::ssbytes * ninputs);

  // ── Operation objects (mirrors trace_main.cu) ───────────
  rng_type randombytes;
  symmetric_ws::host::hash_g hash_seed(ninputs, fips_nw);
  genpoly_warp::host::gena<params::k<variant>> generate_a(ninputs, genmat_nw);
  genpoly_warp::host::genat<params::k<variant>> generate_at(ninputs, genmat_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>> generate_s(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>> generate_e(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta1<variant>> generate_r(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<params::k<variant>, params::eta2> generate_e1(ninputs, genvec_nw);
  genpoly_warp::host::gennoise<1, params::eta2> generate_e2(ninputs, genpoly_nw);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_s(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_e(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_r(ninputs);
  ntt_ctgs_64t::host::fwdntt<params::k<variant>> fwdnttvec_u(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<params::k<variant>> intt_ar(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<1> intt_tr(ninputs);
  ntt_ctgs_64t::host::invntt_tomont<1> intt_su(ninputs);
  arithmetic_mt::host::mattimesvec_tomont_plusvec<params::k<variant>> mtvpv(ninputs);
  arithmetic_mt::host::mattimesvec<params::k<variant>> mtv(ninputs);
  arithmetic_mt::host::vectimesvec<params::k<variant>> ttimesr(ninputs);
  arithmetic_mt::host::vectimesvec<params::k<variant>> stimesu(ninputs);
  arithmetic_mt::host::vecadd2<params::k<variant>> vpv(ninputs);
  arithmetic_mt::host::polyadd3 padd3(ninputs);
  arithmetic_mt::host::polysub psub(ninputs);
  endecode_mt::host::polyvec_tobytes<params::k<variant>> encodet(ninputs);
  endecode_mt::host::polyvec_tobytes<params::k<variant>> encodes(ninputs);
  endecode_mt::host::polyvec_frombytes<params::k<variant>> decodet(ninputs);
  endecode_mt::host::polyvec_frombytes<params::k<variant>> decodes(ninputs);
  endecode_mt::host::poly_frommsg frommsg(ninputs);
  endecode_mt::host::poly_tomsg tomsg(ninputs);
  endecode_mt::host::polyvec_compress<params::k<variant>, params::du<variant>> compressu(ninputs);
  endecode_mt::host::poly_compress<params::dv<variant>> compressv(ninputs);
  endecode_mt::host::polyvec_decompress<params::k<variant>, params::du<variant>> decompressu(ninputs);
  endecode_mt::host::poly_decompress<params::dv<variant>> decompressv(ninputs);
  symmetric_ws::host::hash_h keypair_hash_pk(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_rand(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_pk(ninputs, fips_nw);
  symmetric_ws::host::hash_h enc_hash_ct(ninputs, fips_nw);
  symmetric_ws::host::hash_g enc_hash_coin(ninputs, fips_nw);
  symmetric_ws::host::kdf enc_kdf(ninputs, fips_nw);
  symmetric_ws::host::hash_h dec_hash_ct(ninputs, fips_nw);
  symmetric_ws::host::hash_g dec_hash_coin(ninputs, fips_nw);
  symmetric_ws::host::kdf dec_kdf(ninputs, fips_nw);
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

  primitive::ccakem_dec::dec dec(
      ninputs, variant_v,
      primitive::cpapke_enc::cpapke_enc(
          ninputs, variant_v, generate_at, generate_r, generate_e1, generate_e2,
          fwdnttvec_r, intt_ar, intt_tr, mtv, ttimesr, vpv, padd3, decodet,
          frommsg, compressu, compressv),
      primitive::cpapke_dec::cpapke_dec(
          ninputs, variant_v, fwdnttvec_u, intt_su, stimesu, psub, decodes,
          decompressu, decompressv, tomsg),
      dec_hash_ct, dec_hash_coin, dec_kdf, dec_verify_cmov);

  primitive::ccakem_keypair::mem_resource<variant> keypair_mr(ninputs);
  primitive::ccakem_enc::mem_resource<variant> enc_mr(ninputs);
  primitive::ccakem_dec::mem_resource<variant> dec_mr(ninputs);

  // ── Step 1: keypair + encaps to get valid ct ───────────
  {
    cuda_resource::graph setup_graph;
    cudaGraphNode_t dummy, pk_avail, sk_avail, ct_avail, ssb_avail, pk_used;

    CCC(cudaGraphAddEmptyNode(&dummy, setup_graph, nullptr, 0));

    randombytes(keypair_mr.pke_keypair_mr.rand_host.get_ptr(),
                params::symbytes * ninputs);
    randombytes(keypair_mr.rand_host.get_ptr(), params::symbytes * ninputs);
    randombytes(enc_mr.rand_host.get_ptr(), params::symbytes * ninputs);

    auto kr = keypair.join_graph(
        setup_graph,
        pk_d.get_ptr(), pk_d.get_pitch(), dummy, &pk_avail,
        sk_d.get_ptr(), sk_d.get_pitch(), dummy, &sk_avail,
        keypair_mr);

    auto er = enc.join_graph(
        setup_graph,
        ct_d.get_ptr(), ct_d.get_pitch(), dummy, &ct_avail,
        ss_d.get_ptr(), ss_d.get_pitch(), dummy, &ssb_avail,
        pk_d.get_ptr(), pk_d.get_pitch(), pk_avail, &pk_used,
        enc_mr);

    // Copy ct and sk back to pinned host memory
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
    cuda_resource::stream setup_stream(cudaStreamNonBlocking);
    CCC(cudaGraphLaunch(setup_exec, setup_stream));
    CCC(cudaStreamSynchronize(setup_stream));

    (void)kr; (void)er;
  }

  // ── Step 2: Pre-compute both ciphertext types ──────────
  const std::size_t ct_size = params::ciphertextbytes<variant> * ninputs;
  std::vector<std::uint8_t> ct_valid(ct_size);
  std::vector<std::uint8_t> ct_random(ct_size);

  // Class 0: valid ct from real encapsulation (now in ct_h)
  std::memcpy(ct_valid.data(), ct_h.get_ptr(), ct_size);

  // Class 1: deterministic random bytes (same PRNG as trace_main.cu)
  for (std::size_t i = 0; i < ct_size; i++)
    ct_random[i] = static_cast<std::uint8_t>(
        (i * 6364136223846793005ULL + 1442695040888963407ULL) >> 56);

  // ── Step 3: Build decaps-only graph ────────────────────
  cuda_resource::graph dec_graph;
  cudaGraphNode_t cpyct_todev, cpysk_todev;
  {
    cudaMemcpy3DParms p = {};
    p.srcPtr = make_cudaPitchedPtr(ct_h.get_ptr(),
        params::ciphertextbytes<variant>,
        params::ciphertextbytes<variant>, ninputs);
    p.dstPtr = make_cudaPitchedPtr(ct_d.get_ptr(), ct_d.get_pitch(),
        params::ciphertextbytes<variant>, ninputs);
    p.extent = make_cudaExtent(params::ciphertextbytes<variant>, ninputs, 1);
    p.kind = cudaMemcpyHostToDevice;
    CCC(cudaGraphAddMemcpyNode(&cpyct_todev, dec_graph, nullptr, 0, &p));
  }
  {
    cudaMemcpy3DParms p = {};
    p.srcPtr = make_cudaPitchedPtr(sk_h.get_ptr(),
        params::secretkeybytes<variant>,
        params::secretkeybytes<variant>, ninputs);
    p.dstPtr = make_cudaPitchedPtr(sk_d.get_ptr(), sk_d.get_pitch(),
        params::secretkeybytes<variant>, ninputs);
    p.extent = make_cudaExtent(params::secretkeybytes<variant>, ninputs, 1);
    p.kind = cudaMemcpyHostToDevice;
    CCC(cudaGraphAddMemcpyNode(&cpysk_todev, dec_graph, nullptr, 0, &p));
  }

  cudaGraphNode_t data_ready;
  {
    std::array dep{cpyct_todev, cpysk_todev};
    CCC(cudaGraphAddEmptyNode(&data_ready, dec_graph, dep.data(), dep.size()));
  }

  cudaGraphNode_t ssa_avail, ct_used, sk_used;
  auto dr = dec.join_graph(
      dec_graph,
      ss_d.get_ptr(), ss_d.get_pitch(), data_ready, &ssa_avail,
      ct_d.get_ptr(), ct_d.get_pitch(), data_ready, &ct_used,
      sk_d.get_ptr(), sk_d.get_pitch(), data_ready, &sk_used,
      dec_mr);
  (void)dr;

  cuda_resource::graph_exec dec_exec(dec_graph);
  cuda_resource::stream dec_stream(cudaStreamNonBlocking);

  // ── Step 4: Warmup ─────────────────────────────────────
  std::memcpy(ct_h.get_ptr(), ct_valid.data(), ct_size);
  for (unsigned w = 0; w < WARMUP_ITERS; w++) {
    CCC(cudaGraphLaunch(dec_exec, dec_stream));
    CCC(cudaStreamSynchronize(dec_stream));
  }

  // ── Step 5: Main loop ──────────────────────────────────
  FILE* log = std::fopen("/logs/victim_log.csv", "w");
  if (!log) {
    std::perror("fopen /logs/victim_log.csv");
    return;
  }
  std::fprintf(log, "timestamp_us,class\n");

  auto t_start = std::chrono::high_resolution_clock::now();
  auto t_end   = t_start + std::chrono::seconds(DURATION_SEC);

  std::fprintf(stderr,
      "[victim] starting %us loop, switching class every %u ops\n",
      DURATION_SEC, CLASS_SWITCH_INTERVAL);

  unsigned op_count  = 0;
  int      cur_class = -1;  // force update on first iteration

  while (std::chrono::high_resolution_clock::now() < t_end) {
    const int new_class = static_cast<int>((op_count / CLASS_SWITCH_INTERVAL) % 2);
    if (new_class != cur_class) {
      cur_class = new_class;
      const auto& src = (cur_class == 0) ? ct_valid : ct_random;
      std::memcpy(ct_h.get_ptr(), src.data(), ct_size);
    }

    CCC(cudaGraphLaunch(dec_exec, dec_stream));
    CCC(cudaStreamSynchronize(dec_stream));

    auto now = std::chrono::high_resolution_clock::now();
    double ts = std::chrono::duration<double, std::micro>(now - t_start).count();
    std::fprintf(log, "%.3f,%d\n", ts, cur_class);

    op_count++;
  }

  std::fclose(log);
  std::fprintf(stderr, "[victim] done — %u operations logged\n", op_count);
}

}  // namespace atpqc_cuda::kyber::victim_loop

int main() {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::victim_loop::run();
    CCC(cuCtxSynchronize());
  }

  return 0;
}
