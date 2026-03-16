# mlkem-gpu-sec

> **GPU ML-KEM Security Research** — Timing side-channel analysis of Kyber/ML-KEM on CUDA GPUs.
>
> First systematic security evaluation of GPU-accelerated ML-KEM (FIPS 203).
> Target venue: TCHES 2026.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Baseline GPU Kyber implementation + throughput benchmark | Complete |
| Phase 2 | Timing trace harness + TVLA leakage analysis | Complete |
| Phase 3/4 | Root cause analysis and mitigation | In progress |
| Phase 5 | Co-tenancy experiments (MPS / MIG) | Pending |

---

## Key Results

### Phase 2 — TVLA Leakage Detection (RTX 4090, 100K traces)

| Variant | \|t-statistic\| | Verdict |
|---------|-----------------|---------|
| Kyber-512 | 63.42 | Leakage detected |
| Kyber-768 | 20.83 | Leakage detected |
| Kyber-1024 | 155.53 | Leakage detected |

All three variants exceed the \|t\| >= 4.5 TVLA threshold (p ~ 0). Timing leakage confirmed across all Kyber parameter sets.

### Phase 4 — After Serialization Fix (CUDA Graph Dependency Hardening)

| Variant | Before \|t\| | After \|t\| | Status |
|---------|-------------|------------|--------|
| Kyber-512 | 63.42 | 2.10 | Fixed |
| Kyber-768 | 20.83 | 34.54 | Residual leakage |
| Kyber-1024 | 155.53 | 8.13 | Reduced |

Serializing the CUDA graph (forcing `cpapke_enc` to complete before `hash_h_ct` begins) eliminates Kyber-512 leakage entirely and substantially reduces Kyber-1024. Kyber-768 retains residual leakage under investigation.

---

## Baseline Throughput (Phase 1)

1024 parallel inputs | All throughput in thousands of ops/sec (K ops/s)

| Variant | Operation | RTX 4050 Laptop | RTX 3080 Ti | RTX 4090 (paper) |
|---------|-----------|-----------------|-------------|------------------|
| Kyber-512 | KeyGen | 683K | 1,784K | 4,306K |
| Kyber-512 | Encaps | 653K | 1,740K | 4,541K |
| Kyber-512 | Decaps | 725K | 1,653K | 4,298K |
| Kyber-768 | KeyGen | 470K | 1,332K | 3,200K |
| Kyber-768 | Encaps | 412K | 1,264K | 2,947K |
| Kyber-768 | Decaps | 484K | 1,376K | 2,970K |
| Kyber-1024 | KeyGen | 322K | 1,028K | 2,276K |
| Kyber-1024 | Encaps | 295K | 935K | 2,221K |
| Kyber-1024 | Decaps | 323K | 1,081K | 2,120K |

**Environments:** RTX 4050 Laptop (Driver 591.86, CUDA 12.6) · RTX 3080 Ti local Docker/WSL2 (Driver 591.44, CUDA 12.6) · RTX 4090 RunPod Secure Cloud (Driver 550.127.05, CUDA 12.4)

---

## Quick Start (Docker)

```bash
git clone https://github.com/knightarendelle/mlkem-gpu-sec.git
cd mlkem-gpu-sec
docker build -t mlkem-gpu-sec .
docker run --gpus all -it --rm -v $(pwd):/workspace/mlkem-gpu-sec mlkem-gpu-sec
bash scripts/runpod_install.sh
bash baseline/setup.sh
```

### Prerequisites
- Docker Desktop installed and running
- NVIDIA GPU (RTX 30xx or 40xx recommended)
- WSL2 enabled on Windows
- NVIDIA Control Panel -> Developer -> Allow access to GPU performance counters

### Run throughput benchmark
```bash
echo "1024 4 4 4 4" | ./target/bench_kyber512.out
echo "1024 4 4 4 4" | ./target/bench_kyber768.out
echo "1024 4 4 4 4" | ./target/bench_kyber1024.out
```

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
| Docker (WSL2) | Fine | WDDM adds noise | No |
| RunPod (bare-metal) | Fine | Valid | Yes |

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

## Team

4 members | TCHES 2026 target submission April 15

---

## Experiment Tracking

All experiments must be logged in `experiments/results/experiment_log.md` with:
- Date and time
- GPU model and driver version
- CUDA version
- Exact command run
- Key result or finding

Reproducibility is part of the paper's contribution.

---

## Team Rules

1. **Never commit raw trace files > 100MB** — gitignored, store on RunPod volume
2. **Never run TVLA timing traces inside Docker** — use RunPod for all timing experiments
3. **Document every experiment** in `experiment_log.md` before closing your session
4. **Pin your CUDA and driver versions** — report them in every result
5. **Always push before ending a session** — never leave uncommitted work locally

---

## License

This research code is currently **private**. It will be made public as an artifact at time of paper submission under MIT License.
