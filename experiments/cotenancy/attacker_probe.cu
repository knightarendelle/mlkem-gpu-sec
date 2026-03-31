//
// experiments/cotenancy/attacker_probe.cu
// Attacker program for the co-tenancy side-channel experiment.
//
// Continuously runs a 256 MB DRAM bandwidth probe for exactly
// DURATION_SEC seconds, recording each probe's execution time
// using both CUDA Events and wall clock.
//
// Has NO access to victim code, victim memory, or victim timing.
// Does not know when victim class switches happen.
//
// Output: /logs/attacker_log.csv
//   header: timestamp_us,wall_us,gpu_us
//   rows:   <us since program start>,<wall-clock us>,<GPU-event us>
//
// Compile (standalone, no atpqc-cuda dependency):
//   nvcc -O3 -arch=compute_89 \
//     experiments/cotenancy/attacker_probe.cu \
//     -o attacker_probe
//

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

static constexpr int DURATION_SEC  = 35;
static constexpr int WARMUP_ITERS  = 20;
static constexpr int BLOCKS        = 1024;
static constexpr int THREADS       = 256;

// 256 MB read-write probe — forces DRAM access on all channels.
// 1024 blocks x 256 threads = 262 144 threads; on RTX 40xx (128 SMs)
// this is 2048 threads/SM = full occupancy.
__global__ void bandwidth_probe(float* __restrict__ buf, int n,
                                float* __restrict__ out) {
  int stride = blockDim.x * gridDim.x;
  int idx    = blockIdx.x * blockDim.x + threadIdx.x;
  float sum  = 0.0f;
  for (int i = idx; i < n; i += stride) {
    sum += buf[i];
    buf[i] = sum;
  }
  // Prevent the compiler from optimising away the loop.
  if (idx == 0) *out = sum;
}

int main() {
  const int n = 256 * 1024 * 1024 / (int)sizeof(float);
  float* d_buf;
  float* d_out;
  cudaMalloc(&d_buf, (size_t)n * sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  cudaMemset(d_buf, 0, (size_t)n * sizeof(float));

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  // Warmup: bring GPU to steady-state frequency, fill TLB/cache state.
  for (int i = 0; i < WARMUP_ITERS; i++) {
    bandwidth_probe<<<BLOCKS, THREADS>>>(d_buf, n, d_out);
    cudaDeviceSynchronize();
  }

  FILE* log = fopen("/logs/attacker_log.csv", "w");
  if (!log) {
    perror("fopen /logs/attacker_log.csv");
    return 1;
  }
  fprintf(log, "timestamp_us,wall_us,gpu_us\n");

  auto t_start = std::chrono::high_resolution_clock::now();
  auto t_end   = t_start + std::chrono::seconds(DURATION_SEC);

  fprintf(stderr, "[attacker] starting %d-second probe loop\n", DURATION_SEC);

  unsigned probe_count = 0;
  while (std::chrono::high_resolution_clock::now() < t_end) {
    auto w_start = std::chrono::high_resolution_clock::now();

    cudaEventRecord(ev_start);
    bandwidth_probe<<<BLOCKS, THREADS>>>(d_buf, n, d_out);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    auto w_end = std::chrono::high_resolution_clock::now();

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    double wall_us = std::chrono::duration<double, std::micro>(
        w_end - w_start).count();
    double ts = std::chrono::duration<double, std::micro>(
        w_start - t_start).count();

    fprintf(log, "%.3f,%.3f,%.3f\n", ts, wall_us,
            static_cast<double>(gpu_ms) * 1000.0);
    probe_count++;
  }

  fclose(log);
  fprintf(stderr, "[attacker] done — %u probes logged\n", probe_count);

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
  cudaFree(d_buf);
  cudaFree(d_out);
  return 0;
}
