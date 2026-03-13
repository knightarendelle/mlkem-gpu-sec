# Experiment Log

**Rule:** Every experiment must be logged here before closing your RunPod session.
No exceptions. Reproducibility is part of the paper's contribution.

---

## Log Format

Copy this template for every entry:

```
### [DATE] [YOUR NAME] — [EXPERIMENT NAME]
- **GPU:** RTX 4090 | Driver: XXX | CUDA: 12.X
- **Phase:** Phase 1 / 2 / 3 / 4 / 5
- **Command:** `bash scripts/...`
- **Key result:** ...
- **Trace file (if any):** experiments/traces/FILENAME
- **Notes:** ...
```

---

## Entries

### [DATE] [NAME] — Environment Verification
- **GPU:** (fill in from verify_env.sh output)
- **Driver:** (fill in)
- **CUDA:** (fill in)
- **Phase:** Setup
- **Command:** `bash scripts/verify_env.sh`
- **Key result:** All checks passed / N failed
- **Notes:** First run on RunPod instance.

---

<!-- Add new entries below this line -->

### 2026-03-02 — Phase 1: Baseline Build & Correctness Verification
- **GPU:** RTX 4050 Laptop | Driver: 591.86 | CUDA: 12.6
- **Phase:** Phase 1
- **Command:** `make test` in `/workspace/refs/atpqc-cuda`
- **Key result:** All three parameter sets pass correctness tests (exit code 0)
  - Kyber-512: PASS
  - Kyber-768: PASS
  - Kyber-1024: PASS
- **Notes:** Using atpqc-cuda (MIT license). liboqs-cupqc-meta has no standalone build system.

### 2026-03-02 — Phase 1: Throughput Baseline
- **GPU:** RTX 4050 Laptop | Driver: 591.86 | CUDA: 12.6
- **Phase:** Phase 1
- **Command:** `echo "1024 4 4 4 4" | ./target/bench_kyberXXX.out`
- **Results:**
  - Kyber-512:  KeyGen 683K ops/s | Encaps 653K ops/s | Decaps 725K ops/s
  - Kyber-768:  KeyGen 470K ops/s | Encaps 412K ops/s | Decaps 484K ops/s
  - Kyber-1024: KeyGen 322K ops/s | Encaps 295K ops/s | Decaps 323K ops/s
- **Notes:** Baseline with ninputs=1024, all warp params=4. Final paper numbers will use RTX 4090 on RunPod.

### 2026-03-11 — Phase 1: RTX 3080 Ti Throughput Baseline (Local Docker)
- **GPU:** RTX 3080 Ti (12GB) | Driver: 591.44 | CUDA: 12.6
- **Platform:** Local Docker + WSL2
- **Command:** `echo "1024 4 4 4 4" | ./target/bench_kyberXXX.out`
- **Results:**
  - Kyber-512:  KeyGen 1,784K ops/s | Encaps 1,740K ops/s | Decaps 1,653K ops/s
  - Kyber-768:  KeyGen 1,332K ops/s | Encaps 1,264K ops/s | Decaps 1,376K ops/s
  - Kyber-1024: KeyGen 1,028K ops/s | Encaps 935K ops/s  | Decaps 1,081K ops/s
- **Notes:** Dev machine. Not used in paper.

### 2026-03-11 — Phase 1: RTX 4090 Throughput Baseline (RunPod Secure Cloud)
- **GPU:** RTX 4090 (24GB) | Driver: 550.127.05 | CUDA: 12.4
- **Platform:** RunPod Secure Cloud (bare-metal Linux)
- **Command:** `echo "1024 4 4 4 4" | ./target/bench_kyberXXX.out`
- **Results:**
  - Kyber-512:  KeyGen 4,306K ops/s | Encaps 4,541K ops/s | Decaps 4,298K ops/s
  - Kyber-768:  KeyGen 3,200K ops/s | Encaps 2,947K ops/s | Decaps 2,970K ops/s
  - Kyber-1024: KeyGen 2,276K ops/s | Encaps 2,221K ops/s | Decaps 2,120K ops/s
- **Notes:** CANONICAL paper numbers. Phase 1 complete.

### 2026-03-13 — Phase 2: TVLA Timing Analysis (RTX 4090, RunPod Secure Cloud)
- **GPU:** RTX 4090 (24GB) | Driver: 550.127.05 | CUDA: 12.4
- **Platform:** RunPod Secure Cloud (bare-metal Linux)
- **Phase:** Phase 2
- **Command:** `echo "1 4 4 4 4 100000 <class>" | ./target/trace_kyberXXX.out > experiments/traces/kyberXXX_classN_n100000.csv`
- **Methodology:** CUDA Events timing, ninputs=1, 100,000 traces per class, outlier removal at z > 5.0
- **Results:**

| Variant | \|t-statistic\| | Mean diff | Direction | Sliding window max \|t\| | Windows above threshold |
|---------|----------------|-----------|-----------|--------------------------|-------------------------|
| Kyber-512 | 63.42 | +0.95 µs | valid faster | 96.09 | 68.5% |
| Kyber-768 | 20.83 | −0.25 µs | invalid faster | 92.08 | — |
| Kyber-1024 | 155.53 | −1.19 µs | invalid faster | 106.12 | 89.3% |

All three variants exceed the \|t\| ≥ 4.5 TVLA threshold with p ≈ 0. **Leakage confirmed across all Kyber parameter sets.**

- **Notable observations:**
  - Leakage direction flips between Kyber-512 (valid faster) and Kyber-768/1024 (invalid faster)
  - Kyber-1024 shows the strongest leakage signal despite being the largest variant
  - Sliding window analysis reveals leakage is not confined to specific trace regions — it is pervasive
- **Trace files:** `experiments/traces/kyber{512,768,1024}_class{0,1}_n100000.csv` (~1.4 MB each, gitignored)
- **Notes:** CANONICAL Phase 2 results. Phase 2 complete.