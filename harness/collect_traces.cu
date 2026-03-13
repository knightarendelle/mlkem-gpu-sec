/*
 * harness/collect_traces.cu
 * ============================================================
 * Phase 2: Timing Trace Collection Harness
 *
 * Collects CUDA Event timing traces for Kyber decapsulation
 * across two ciphertext classes:
 *   Class 0 — valid ciphertexts (honestly encapsulated)
 *   Class 1 — invalid ciphertexts (random bytes)
 *
 * Output: CSV files in experiments/traces/
 *   traces_class0.csv — one timing value per line (microseconds)
 *   traces_class1.csv — one timing value per line (microseconds)
 *
 * Usage:
 *   ./collect_traces <variant> <n_traces>
 *   ./collect_traces 512 100000
 *   ./collect_traces 768 100000
 *   ./collect_traces 1024 100000
 *
 * Build:
 *   See harness/Makefile
 *
 * Author: Person 1
 * Project: mlkem-gpu-sec
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

// ── Kyber parameter sets ──────────────────────────────────
// Kyber-512
#define KYBER512_K          2
#define KYBER512_SKBYTES    1632
#define KYBER512_PKBYTES    800
#define KYBER512_CTBYTES    768
#define KYBER512_SSBYTES    32

// Kyber-768
#define KYBER768_K          3
#define KYBER768_SKBYTES    2400
#define KYBER768_PKBYTES    1184
#define KYBER768_CTBYTES    1088
#define KYBER768_SSBYTES    32

// Kyber-1024
#define KYBER1024_K         4
#define KYBER1024_SKBYTES   3168
#define KYBER1024_PKBYTES   1568
#define KYBER1024_CTBYTES   1568
#define KYBER1024_SSBYTES   32

// ── CUDA error checking ───────────────────────────────────
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ── Simple PRNG for generating test data ─────────────────
static uint64_t rng_state = 0x123456789ABCDEF0ULL;

static inline uint8_t rand_byte(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint8_t)(rng_state & 0xFF);
}

static void rand_bytes(uint8_t *buf, size_t len) {
    for (size_t i = 0; i < len; i++)
        buf[i] = rand_byte();
}

// ── Seed PRNG from system time ────────────────────────────
static void seed_rng(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    rng_state = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    // Warm up
    for (int i = 0; i < 100; i++) rand_byte();
}

// ── Dummy decapsulation kernel ────────────────────────────
// NOTE: This is a timing harness placeholder.
// Replace this with the actual atpqc-cuda decapsulation kernel
// once we integrate the GPU Kyber implementation properly.
//
// For now this measures the timing infrastructure itself and
// establishes the harness is working correctly before integration.
__global__ void dummy_decaps_kernel(
    const uint8_t * __restrict__ ct,
    const uint8_t * __restrict__ sk,
    uint8_t       * __restrict__ ss,
    int ct_bytes,
    int sk_bytes,
    int ss_bytes,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Simulate decapsulation work — memory access pattern
    // This will be replaced with real Kyber decaps
    uint8_t acc = 0;
    for (int i = 0; i < ct_bytes; i++)
        acc ^= ct[tid * ct_bytes + i];
    for (int i = 0; i < sk_bytes; i++)
        acc ^= sk[tid * sk_bytes + i];
    for (int i = 0; i < ss_bytes; i++)
        ss[tid * ss_bytes + i] = acc ^ (uint8_t)i;
}

// ── Timing harness ────────────────────────────────────────
typedef struct {
    int variant;      // 512, 768, or 1024
    int ct_bytes;
    int sk_bytes;
    int ss_bytes;
    int n_traces;
} HarnessConfig;

static void run_timing_harness(
    HarnessConfig *cfg,
    int ciphertext_class,   // 0 = valid, 1 = invalid
    float *timings_us       // output: timing in microseconds, length cfg->n_traces
) {
    int n = 1;  // one decapsulation at a time for timing granularity

    // Allocate host buffers
    uint8_t *h_ct = (uint8_t*)malloc(cfg->ct_bytes);
    uint8_t *h_sk = (uint8_t*)malloc(cfg->sk_bytes);
    uint8_t *h_ss = (uint8_t*)malloc(cfg->ss_bytes);

    if (!h_ct || !h_sk || !h_ss) {
        fprintf(stderr, "[ERROR] Host allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Allocate device buffers
    uint8_t *d_ct, *d_sk, *d_ss;
    CUDA_CHECK(cudaMalloc(&d_ct, cfg->ct_bytes));
    CUDA_CHECK(cudaMalloc(&d_sk, cfg->sk_bytes));
    CUDA_CHECK(cudaMalloc(&d_ss, cfg->ss_bytes));

    // Generate fixed secret key (same for all traces)
    rand_bytes(h_sk, cfg->sk_bytes);
    CUDA_CHECK(cudaMemcpy(d_sk, h_sk, cfg->sk_bytes, cudaMemcpyHostToDevice));

    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up — run 100 iterations before collecting traces
    // This ensures GPU is in steady state (clocks ramped up)
    rand_bytes(h_ct, cfg->ct_bytes);
    CUDA_CHECK(cudaMemcpy(d_ct, h_ct, cfg->ct_bytes, cudaMemcpyHostToDevice));
    for (int w = 0; w < 100; w++) {
        dummy_decaps_kernel<<<1, 1>>>(d_ct, d_sk, d_ss,
            cfg->ct_bytes, cfg->sk_bytes, cfg->ss_bytes, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  Collecting %d traces for class %d (%s ciphertexts)...\n",
           cfg->n_traces,
           ciphertext_class,
           ciphertext_class == 0 ? "valid" : "invalid");

    // Collect traces
    for (int t = 0; t < cfg->n_traces; t++) {
        // Generate ciphertext for this class
        if (ciphertext_class == 0) {
            // Class 0: valid-looking ciphertext (structured random)
            // In real integration: use actual GPU keypair + encaps
            rand_bytes(h_ct, cfg->ct_bytes);
            // Mark as valid class — set first byte to known pattern
            h_ct[0] = 0xAB;
        } else {
            // Class 1: fully random (invalid) ciphertext
            rand_bytes(h_ct, cfg->ct_bytes);
            h_ct[0] = 0xCD;
        }

        CUDA_CHECK(cudaMemcpy(d_ct, h_ct, cfg->ct_bytes, cudaMemcpyHostToDevice));

        // Time the kernel
        CUDA_CHECK(cudaEventRecord(start));
        dummy_decaps_kernel<<<1, 1>>>(d_ct, d_sk, d_ss,
            cfg->ct_bytes, cfg->sk_bytes, cfg->ss_bytes, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        timings_us[t] = ms * 1000.0f;  // convert ms to microseconds

        // Progress indicator every 10%
        if ((t + 1) % (cfg->n_traces / 10) == 0) {
            printf("    %d%%\n", (t + 1) * 100 / cfg->n_traces);
            fflush(stdout);
        }
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_ct));
    CUDA_CHECK(cudaFree(d_sk));
    CUDA_CHECK(cudaFree(d_ss));
    free(h_ct);
    free(h_sk);
    free(h_ss);
}

// ── Save traces to CSV ────────────────────────────────────
static void save_traces(
    const char *filename,
    float *timings_us,
    int n_traces,
    int ciphertext_class,
    int variant
) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open output file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Header
    fprintf(f, "# mlkem-gpu-sec Phase 2 timing traces\n");
    fprintf(f, "# variant: Kyber-%d\n", variant);
    fprintf(f, "# class: %d (%s)\n", ciphertext_class,
            ciphertext_class == 0 ? "valid" : "invalid");
    fprintf(f, "# n_traces: %d\n", n_traces);
    fprintf(f, "# unit: microseconds\n");
    fprintf(f, "timing_us\n");

    for (int i = 0; i < n_traces; i++)
        fprintf(f, "%.6f\n", timings_us[i]);

    fclose(f);
    printf("  Saved: %s\n", filename);
}

// ── Main ──────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <variant> <n_traces>\n", argv[0]);
        fprintf(stderr, "  variant:  512, 768, or 1024\n");
        fprintf(stderr, "  n_traces: number of timing traces per class\n");
        fprintf(stderr, "  Example:  %s 512 100000\n", argv[0]);
        return EXIT_FAILURE;
    }

    int variant  = atoi(argv[1]);
    int n_traces = atoi(argv[2]);

    if (variant != 512 && variant != 768 && variant != 1024) {
        fprintf(stderr, "[ERROR] variant must be 512, 768, or 1024\n");
        return EXIT_FAILURE;
    }
    if (n_traces < 1000 || n_traces > 10000000) {
        fprintf(stderr, "[ERROR] n_traces must be between 1000 and 10000000\n");
        return EXIT_FAILURE;
    }

    seed_rng();

    // Set parameters for chosen variant
    HarnessConfig cfg;
    cfg.variant  = variant;
    cfg.n_traces = n_traces;
    switch (variant) {
        case 512:
            cfg.ct_bytes = KYBER512_CTBYTES;
            cfg.sk_bytes = KYBER512_SKBYTES;
            cfg.ss_bytes = KYBER512_SSBYTES;
            break;
        case 768:
            cfg.ct_bytes = KYBER768_CTBYTES;
            cfg.sk_bytes = KYBER768_SKBYTES;
            cfg.ss_bytes = KYBER768_SSBYTES;
            break;
        case 1024:
            cfg.ct_bytes = KYBER1024_CTBYTES;
            cfg.sk_bytes = KYBER1024_SKBYTES;
            cfg.ss_bytes = KYBER1024_SSBYTES;
            break;
    }

    printf("\n");
    printf("================================================\n");
    printf("  mlkem-gpu-sec Phase 2: Trace Collection\n");
    printf("================================================\n");
    printf("  Variant:  Kyber-%d\n", variant);
    printf("  Traces:   %d per class (%d total)\n", n_traces, n_traces * 2);
    printf("  CT bytes: %d\n", cfg.ct_bytes);
    printf("  SK bytes: %d\n", cfg.sk_bytes);
    printf("\n");

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU: %s\n", prop.name);
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("  VRAM: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("\n");

    // Allocate timing buffers
    float *timings_class0 = (float*)malloc(n_traces * sizeof(float));
    float *timings_class1 = (float*)malloc(n_traces * sizeof(float));
    if (!timings_class0 || !timings_class1) {
        fprintf(stderr, "[ERROR] Failed to allocate timing buffers\n");
        return EXIT_FAILURE;
    }

    // Collect traces for both classes
    printf("[Class 0 — valid ciphertexts]\n");
    run_timing_harness(&cfg, 0, timings_class0);
    printf("\n");

    printf("[Class 1 — invalid ciphertexts]\n");
    run_timing_harness(&cfg, 1, timings_class1);
    printf("\n");

    // Save to CSV
    printf("[Saving traces]\n");
    char fname0[256], fname1[256];
    snprintf(fname0, sizeof(fname0),
             "experiments/traces/kyber%d_class0_n%d.csv",
             variant, n_traces);
    snprintf(fname1, sizeof(fname1),
             "experiments/traces/kyber%d_class1_n%d.csv",
             variant, n_traces);

    // Create output directory if needed
    system("mkdir -p experiments/traces");

    save_traces(fname0, timings_class0, n_traces, 0, variant);
    save_traces(fname1, timings_class1, n_traces, 1, variant);

    // Print quick summary statistics
    printf("\n[Quick Statistics]\n");
    for (int cls = 0; cls < 2; cls++) {
        float *t = (cls == 0) ? timings_class0 : timings_class1;
        double sum = 0, sum2 = 0;
        float tmin = t[0], tmax = t[0];
        for (int i = 0; i < n_traces; i++) {
            sum  += t[i];
            sum2 += t[i] * t[i];
            if (t[i] < tmin) tmin = t[i];
            if (t[i] > tmax) tmax = t[i];
        }
        double mean   = sum / n_traces;
        double var    = sum2 / n_traces - mean * mean;
        double stddev = (var > 0) ? sqrt(var) : 0;

        printf("  Class %d (%s):\n", cls, cls == 0 ? "valid  " : "invalid");
        printf("    Mean:   %.4f us\n", mean);
        printf("    StdDev: %.4f us\n", stddev);
        printf("    Min:    %.4f us\n", tmin);
        printf("    Max:    %.4f us\n", tmax);
    }

    printf("\n[Next step]\n");
    printf("  Run TVLA analysis:\n");
    printf("  python3 harness/tvla_analysis.py \\\n");
    printf("    experiments/traces/kyber%d_class0_n%d.csv \\\n", variant, n_traces);
    printf("    experiments/traces/kyber%d_class1_n%d.csv\n", variant, n_traces);
    printf("\n");
    printf("================================================\n");
    printf("  Trace collection complete.\n");
    printf("================================================\n\n");

    free(timings_class0);
    free(timings_class1);
    return EXIT_SUCCESS;
}