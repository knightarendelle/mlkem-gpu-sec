# mlkem-gpu-sec

> **GPU ML-KEM Security Stack: Timing Variance, Leakage Assessment & Hardened NTT Design**
>
> First systematic security evaluation of GPU-accelerated ML-KEM (FIPS 203).
> Target venue: TCHES 2026.

---

## What This Project Is

ML-KEM is now the global post-quantum key encapsulation standard (NIST FIPS 203), deployed in ~50% of HTTPS traffic. GPU implementations achieve millions of operations per second — but nobody has checked if they are safe against timing attacks.

We measure the timing leakage surface of GPU ML-KEM, attribute root causes to GPU microarchitecture, and build a hardened NTT variant that reduces leakage with minimal throughput overhead.

**This is the first rigorous security evaluation of GPU-accelerated ML-KEM.**

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| ✅ Phase 1 | Baseline GPU Kyber implementation + throughput benchmark | Complete |
| 🔄 Phase 2 | Timing trace harness + TVLA leakage analysis | In Progress |
| ⬜ Phase 3 | Nsight Compute root cause analysis | Pending |
| ⬜ Phase 4 | Hardened NTT design + evaluation | Pending |
| ⬜ Phase 5 | Co-tenancy experiments (MPS / MIG) | Pending |

---

## Baseline Results (Phase 1)

Measured on RTX 4050 Laptop GPU | Driver 591.86 | CUDA 12.6 | 1024 parallel inputs

| Variant | Operation | Latency | Throughput |
|---------|-----------|---------|------------|
| Kyber-512 | KeyGen | 1.499 ms | 683,177 ops/sec |
| Kyber-512 | Encaps | 1.567 ms | 653,271 ops/sec |
| Kyber-512 | Decaps | 1.412 ms | 725,171 ops/sec |
| Kyber-768 | KeyGen | 2.179 ms | 469,958 ops/sec |
| Kyber-768 | Encaps | 2.487 ms | 411,814 ops/sec |
| Kyber-768 | Decaps | 2.116 ms | 483,930 ops/sec |
| Kyber-1024 | KeyGen | 3.181 ms | 321,879 ops/sec |
| Kyber-1024 | Encaps | 3.466 ms | 295,455 ops/sec |
| Kyber-1024 | Decaps | 3.171 ms | 322,929 ops/sec |

> Final paper benchmarks will be re-run on RunPod RTX 4090 for comparability with prior work.

---

## Team

| Person | Role | GitHub |
|--------|------|--------|
| Person 1 | GPU Systems Engineer | @username |
| Person 2 | Security Analyst | @username |
| Person 3 | Crypto Systems Engineer | @username |
| Person 4 | Lead Writer & PM | @username |

> Replace @username with actual GitHub handles.

---

## Repo Structure

```
mlkem-gpu-sec/
├── baseline/
│   └── atpqc-cuda/         # Phase 1: GPU Kyber baseline (MIT license)
├── harness/                # Phase 2: Timing trace collection & TVLA analysis
├── rootcause/              # Phase 3: Nsight Compute root cause analysis
├── hardened_ntt/           # Phase 4: Hardened NTT design & evaluation
├── experiments/
│   ├── traces/             # Raw timing trace data (gitignored if large)
│   ├── nsight/             # Nsight Compute reports (.ncu-rep files)
│   └── results/            # Processed results, logs, plots
│       └── experiment_log.md
├── paper/                  # LaTeX paper (synced from Overleaf)
├── scripts/
│   ├── verify_env.sh       # Environment verification script
│   ├── docker_run.sh       # Docker wrapper script
│   └── runpod_install.sh   # RunPod setup script (for final experiments)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Quick Start (Docker — Local Development)

### Prerequisites
- Docker Desktop installed and running
- NVIDIA GPU (RTX 30xx or 40xx recommended)
- WSL2 enabled on Windows
- NVIDIA Control Panel → Developer → Allow access to GPU performance counters

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/mlkem-gpu-sec.git
cd mlkem-gpu-sec
```

### 2. Build the Docker image
```bash
docker build -t mlkem-gpu-sec:latest .
```
> First build takes 5-10 minutes. Subsequent builds use cache and are much faster.

### 3. Run the container
```bash
docker run -it --gpus all --cap-add=SYS_ADMIN -v ${PWD}:/workspace/gpu-mlkem-security mlkem-gpu-sec:latest
```

### 4. Verify environment
```bash
bash scripts/verify_env.sh
```
> Expect 17/18 checks to pass. The WSL2 warning is expected — see note below.

### 5. Build and test the baseline
```bash
cd baseline/atpqc-cuda
make test
```

### 6. Run throughput benchmark
```bash
echo "1024 4 4 4 4" | ./target/bench_kyber512.out
echo "1024 4 4 4 4" | ./target/bench_kyber768.out
echo "1024 4 4 4 4" | ./target/bench_kyber1024.out
```

---

## Environment Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| OS | Ubuntu 22.04 LTS | Inside Docker container |
| CUDA Toolkit | 12.6 | Pre-installed in Docker image |
| Nsight Compute | Latest | Pre-installed in Docker image |
| CMake | 3.20+ | Pre-installed in Docker image |
| Python | 3.10+ | Pre-installed in Docker image |
| GPU (dev) | RTX 4050+ | Local development |
| GPU (final) | RTX 4090 | RunPod — final paper experiments |
| Docker Desktop | Latest | Windows host |
| NVIDIA Driver | 591+ | Windows host |

### Timing Experiment Environments

| Environment | Development | TVLA Timing Traces | Paper Results |
|-------------|-------------|-------------------|---------------|
| Docker (WSL2) | ✅ Fine | ❌ WDDM adds noise | ❌ |
| RunPod (bare-metal) | ✅ Fine | ✅ Valid | ✅ |

**Never collect TVLA timing traces inside Docker on Windows.** WDDM jitter invalidates measurements. All final timing experiments run on RunPod.

---

## Implementation Notes

### Why atpqc-cuda instead of liboqs-cupqc-meta?

`liboqs-cupqc-meta` contains valid CUDA ML-KEM source files but has no standalone build system. It is designed to be integrated into the liboqs framework, not built independently.

`atpqc-cuda` is a standalone buildable CUDA Kyber implementation (MIT license) with a complete Makefile and benchmark harness. It implements pre-standardization Kyber, which is functionally equivalent to ML-KEM for timing analysis. Differences are noted in the paper.

### Makefile Configuration (atpqc-cuda)

Updated for our environment:
```makefile
CXX := g++                                          # was g++-9
CUDAPATH := /usr/local/cuda                         # was /usr/local/cuda-11.2
CUDA_GENCODE_FLAG := -arch=compute_89 -code=sm_89  # RTX 4050 = Ada Lovelace CC 8.9
```

---

## Key References

| Paper | Relevance | Link |
|-------|-----------|------|
| NIST FIPS 203 (ML-KEM) | The standard we are evaluating | [Free PDF](https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.203.pdf) |
| KyberSlash — TCHES 2025 | Timing attack on ARM ML-KEM — our motivation | [Free PDF](https://kyberslash.cr.yp.to/kyberslash-20250115.pdf) |
| PQShield compiler bug — 2025 | Compiler silently reintroduces timing vulnerability | [Blog post](https://pqshield.com/pqshield-plugs-timing-leaks-in-kyber-ml-kem-to-improve-pqc-implementation-maturity/) |
| HI-Kyber — TPDS 2024 | GPU ML-KEM throughput (zero security analysis) | [IACR ePrint](https://eprint.iacr.org/2023/1194) |
| GPU timing side-channel — DAC 2017 | Proves GPU timing attacks feasible for AES | [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3060403.3060462) |
| TVLA — Becker et al. 2013 | Statistical leakage test framework we use | Search: 'TVLA methodology in practice Becker 2013' |

---

## Codebases Used

| Repo | License | Purpose | Location |
|------|---------|---------|----------|
| [atpqc-cuda](https://github.com/tono-satolab/atpqc-cuda) | MIT | Primary GPU Kyber baseline | `baseline/atpqc-cuda/` |
| [liboqs-cupqc-meta](https://github.com/open-quantum-safe/liboqs-cupqc-meta) | Apache-2.0 | ML-KEM CUDA source reference | `/workspace/refs/` (container only) |
| [mlkem-native](https://github.com/pq-code-package/mlkem-native) | MIT/Apache | CPU reference for correctness | `/workspace/refs/` (container only) |
| [kyber-py](https://github.com/GiacomoPope/kyber-py) | MIT | Python reference for KAT | `/workspace/refs/` (container only) |

---

## Experiment Tracking

All experiments must be logged in `experiments/results/experiment_log.md` with:
- Date and time
- GPU model and driver version
- CUDA version
- Exact command run
- Key result or finding

This is non-negotiable. Reproducibility is part of the paper's contribution.

---

## Paper

- **Draft:** Overleaf (link shared privately with team)
- **Format:** TCHES / IACR
- **Target submission:** TCHES 2026

---

## Team Rules

1. **Never commit raw trace files > 100MB** — gitignored, store on RunPod volume
2. **Never run TVLA timing traces inside Docker** — use RunPod for all timing experiments
3. **Document every experiment** in `experiment_log.md` before closing your session
4. **Pin your CUDA and driver versions** — report them in every result
5. **Always push before ending a session** — never leave uncommitted work locally
6. **Do not start Phase 4 before Phase 3 is complete** — hardening without root cause is guesswork

---

## License

This research code is currently **private**. It will be made public as an artifact at time of paper submission under MIT License.
