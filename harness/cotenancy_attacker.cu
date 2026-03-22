// harness/cotenancy_attacker.cu
// Co-tenancy attacker: measures own memory bandwidth timing
// while victim runs Kyber decapsulation on the same GPU.
//
// Two timing modes are supported:
//
//   GPU-event timing (CUDA events):
//     Measures only the attacker kernel's GPU execution time.
//     Correct when NVIDIA MPS is enabled and both processes truly co-tenant
//     the GPU — victim memory pressure slows attacker's kernel execution.
//     Blind to contention in the without-MPS case (serialized execution).
//
//   Wall-clock timing (clock_gettime, printed to stderr as a second column):
//     Measures time from kernel submit to sync return on the host.
//     Captures victim-induced contention delays even without MPS, because
//     the host blocks on cudaDeviceSynchronize() while the victim holds
//     the GPU context lock.
//
// NOTE: For true concurrent co-tenancy (GPU-event timing meaningful),
//       enable NVIDIA MPS before running:
//         nvidia-cuda-mps-control -d
//       Disable after:
//         echo quit | nvidia-cuda-mps-control
//
// Usage:
//   ./cotenancy_attacker <ntraces> <output.csv>
//
// Compile (from repo root):
//   nvcc -O3 -arch=compute_89 -o harness/cotenancy_attacker \
//       harness/cotenancy_attacker.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Read+write a large buffer to saturate DRAM bandwidth.
// Strided access pattern across all threads hits independent cache lines,
// maximising concurrent DRAM transactions and stressing all memory channels.
__global__ void bandwidth_probe(float* buf, int n, float* out) {
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

static inline double timespec_to_us(struct timespec t) {
    return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <ntraces> <output.csv>\n", argv[0]);
        return 1;
    }

    int ntraces       = atoi(argv[1]);
    const char* outfile = argv[2];

    // 256 MB buffer — well above L2 cache on all current GPUs, forces DRAM.
    int n = 256 * 1024 * 1024 / (int)sizeof(float);
    float* d_buf;
    float* d_out;
    cudaMalloc(&d_buf, (size_t)n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_buf, 0, (size_t)n * sizeof(float));

    // 1024 blocks x 256 threads = 262 144 threads.
    // On RTX 40xx (128 SMs) this gives 2048 threads/SM — full occupancy.
    const int BLOCKS  = 1024;
    const int THREADS = 256;

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // Warmup: let the GPU reach steady-state frequency and fill caches.
    for (int i = 0; i < 20; i++) {
        bandwidth_probe<<<BLOCKS, THREADS>>>(d_buf, n, d_out);
        cudaDeviceSynchronize();
    }

    FILE* f = fopen(outfile, "w");
    if (!f) { perror("fopen"); return 1; }
    // Two columns: GPU-event time (µs), wall-clock time (µs).
    // GPU time is meaningful under MPS; wall-clock captures contention always.
    fprintf(f, "gpu_us,wall_us\n");

    struct timespec t0, t1;

    for (int i = 0; i < ntraces; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);

        cudaEventRecord(ev_start);
        bandwidth_probe<<<BLOCKS, THREADS>>>(d_buf, n, d_out);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        clock_gettime(CLOCK_MONOTONIC, &t1);

        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
        double wall_us = timespec_to_us(t1) - timespec_to_us(t0);

        fprintf(f, "%.6f,%.6f\n", gpu_ms * 1000.0f, wall_us);
    }

    fclose(f);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_buf);
    cudaFree(d_out);
    return 0;
}
