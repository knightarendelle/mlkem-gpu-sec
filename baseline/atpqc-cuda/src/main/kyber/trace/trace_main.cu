//
// trace_main.cu
// Per-iteration decapsulation timing for TVLA analysis.
//
// Based on atpqc-cuda bench/main.cu by Tatsuki Ono (MIT License)
// Extended for timing trace collection by mlkem-gpu-sec team.
//
// Usage:
//   echo "<ninputs> <genmat_nw> <genvec_nw> <genpoly_nw> <fips_nw> <ntraces> <class>" \
//     | ./target/trace_kyber512.out > traces_class0.csv
//
//   class 0 = valid ciphertexts (from real encaps)
//   class 1 = invalid ciphertexts (random bytes)
//
// Example:
//   echo "1 4 4 4 4 100000 0" | ./target/trace_kyber512.out \
//     > experiments/traces/kyber512_class0_n100000.csv
//

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "../../../lib/cuda_debug.hpp"
#include "../../../lib/cuda_resource.hpp"
#include "../../../lib/fips202_ws/host.cuh"
#include "../../../lib/timing_fence_ws/host.cuh"
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

namespace atpqc_cuda::kyber::trace_bench {

using rng_type = rng::std_random_device;
using variant  = variants::KYBER_VARIANT;
constexpr variant variant_v;

void trace_bench() {
  unsigned ninputs, genmat_nw, genvec_nw, genpoly_nw, fips_nw;
  unsigned ntraces;
  int ct_class;

  std::cin >> ninputs >> genmat_nw >> genvec_nw >> genpoly_nw >> fips_nw
           >> ntraces >> ct_class;

  // ── Allocate device memory ──────────────────────────────
  cuda_resource::device_pitched_memory<std::uint8_t> pk_d(
      params::publickeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> sk_d(
      params::secretkeybytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ct_d(
      params::ciphertextbytes<variant>, ninputs);
  cuda_resource::device_pitched_memory<std::uint8_t> ss_d(
      params::ssbytes, ninputs);

  // ── Allocate pinned host memory ─────────────────────────
  cuda_resource::pinned_memory<std::uint8_t> pk_h(
      params::publickeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> sk_h(
      params::secretkeybytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ct_h(
      params::ciphertextbytes<variant> * ninputs);
  cuda_resource::pinned_memory<std::uint8_t> ss_h(
      params::ssbytes * ninputs);

  // ── Build all operation objects (same as bench/main.cu) ─
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
  // Timing fence: pad decaps to a fixed floor so valid/invalid paths
  // become indistinguishable to a timing observer.
  // Floor is set per-variant above the observed worst-case execution time:
  //   Kyber-512:  800µs  (observed max ~628µs)
  //   Kyber-768:  900µs  (observed max ~683µs)
  //   Kyber-1024: 1000µs (observed max ~757µs)
#if KYBER_VARIANT == kyber512
  timing_fence_ws::host::timing_fence dec_tf(1200);
#elif KYBER_VARIANT == kyber768
  timing_fence_ws::host::timing_fence dec_tf(1200);
#else  // kyber1024
  timing_fence_ws::host::timing_fence dec_tf(1200);
#endif

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
      dec_hash_ct, dec_hash_coin, dec_kdf, dec_verify_cmov, dec_tf);

  primitive::ccakem_keypair::mem_resource<variant> keypair_mr(ninputs);
  primitive::ccakem_enc::mem_resource<variant> enc_mr(ninputs);
  primitive::ccakem_dec::mem_resource<variant> dec_mr(ninputs);

  // ── Step 1: keypair + encaps to get valid ct ────────────
  {
    cuda_resource::graph setup_graph;
    cudaGraphNode_t dummy, pk_avail, sk_avail, ct_avail, ssb_avail, pk_used;

    CCC(cudaGraphAddEmptyNode(&dummy, setup_graph, nullptr, 0));

    randombytes(keypair_mr.pke_keypair_mr.rand_host.get_ptr(), params::symbytes * ninputs);
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

    // Copy ct and sk to host
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
  }

  // ── Step 2: If class 1, overwrite ct with random bytes ──
  if (ct_class == 1) {
    std::uint8_t* ct_ptr = ct_h.get_ptr();
    size_t ct_total = params::ciphertextbytes<variant> * ninputs;
    for (size_t i = 0; i < ct_total; i++)
      ct_ptr[i] = (std::uint8_t)((i * 6364136223846793005ULL + 1442695040888963407ULL) >> 56);
  }

  // ── Step 3: Build decaps-only timing graph ──────────────
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

  // Decaps kernel — events are recorded on the stream, not in the graph
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

  cuda_resource::event dec_start_event(cudaEventDefault);
  cuda_resource::event dec_end_event(cudaEventDefault);

  cuda_resource::graph_exec dec_exec(dec_graph);
  cuda_resource::stream dec_stream(cudaStreamNonBlocking);

  // ── Step 4: Warm up 200 iterations ─────────────────────
  for (unsigned w = 0; w < 200; w++) {
    CCC(cudaGraphLaunch(dec_exec, dec_stream));
    CCC(cudaStreamSynchronize(dec_stream));
  }

  // ── Step 5: Collect and print traces ───────────────────
  // Determine variant name from ciphertextbytes
  const char* variant_name =
    params::ciphertextbytes<variant> == 768  ? "Kyber-512"  :
    params::ciphertextbytes<variant> == 1088 ? "Kyber-768"  : "Kyber-1024";

  std::printf("# mlkem-gpu-sec Phase 2 timing traces\n");
  std::printf("# variant: %s\n", variant_name);
  std::printf("# class: %d (%s)\n", ct_class, ct_class == 0 ? "valid" : "invalid");
  std::printf("# n_traces: %u\n", ntraces);
  std::printf("# unit: microseconds\n");
  std::printf("timing_us\n");

  for (unsigned i = 0; i < ntraces; i++) {
    CCC(cudaEventRecord(dec_start_event, dec_stream));
    CCC(cudaGraphLaunch(dec_exec, dec_stream));
    CCC(cudaEventRecord(dec_end_event, dec_stream));
    CCC(cudaStreamSynchronize(dec_stream));

    float ms = 0.0f;
    CCC(cudaEventElapsedTime(&ms, dec_start_event, dec_end_event));
    std::printf("%.6f\n", ms * 1000.0f);

    if (ntraces >= 10 && (i + 1) % (ntraces / 10) == 0) {
      fprintf(stderr, "  %u%%\n", (i + 1) * 100 / ntraces);
      fflush(stderr);
    }
  }
}

}  // namespace atpqc_cuda::kyber::trace_bench

int main() {
  CUDA_DEBUG_RESET();

  CCC(cuInit(0));
  CUdevice dev;
  CCC(cuDeviceGet(&dev, 0));

  {
    atpqc_cuda::cuda_resource::context ctx(dev);
    atpqc_cuda::kyber::trace_bench::trace_bench();
    CCC(cuCtxSynchronize());
  }

  return 0;
}
