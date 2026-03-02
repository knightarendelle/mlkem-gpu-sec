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

## Team

| Person | Role | GitHub |
|--------|------|--------|
| Person 1 | GPU Systems Engineer | @username |
| Person 2 | Security Analyst | @username |
| Person 3 | Crypto Systems Engineer | @username |
| Person 4 | Lead Writer & PM | @username |

> Replace @username with actual GitHub handles after setup.

---

## Repo Structure

```
mlkem-gpu-sec/
├── baseline/           # Phase 1: GPU ML-KEM baseline implementation
├── harness/            # Phase 2: Timing trace collection & TVLA analysis
├── rootcause/          # Phase 3: Nsight Compute root cause analysis
├── hardened_ntt/       # Phase 4: Hardened NTT design & evaluation
├── experiments/
│   ├── traces/         # Raw timing trace data (gitignored if large)
│   ├── nsight/         # Nsight Compute reports (.ncu-rep files)
│   └── results/        # Processed results, CSVs, plots
├── paper/              # LaTeX paper (synced from Overleaf)
├── scripts/            # Shared utility scripts
├── .gitignore
└── README.md
```

---

## Quick Start (RunPod)

### 1. Launch a RunPod RTX 4090 pod
- Template: `RunPod PyTorch 2.1` (CUDA 12.1 pre-installed)
- Container disk: 20 GB | Volume disk: 50 GB

### 2. Connect and clone
```bash
git clone https://github.com/YOUR_USERNAME/mlkem-gpu-sec.git
cd mlkem-gpu-sec
```

### 3. Run environment verification
```bash
bash scripts/verify_env.sh
```

### 4. Build the ML-KEM baseline
```bash
cd baseline
bash setup.sh
```

---

## Environment Requirements

| Requirement | Version |
|-------------|---------|
| OS | Ubuntu 22.04 LTS |
| CUDA Toolkit | 12.x |
| Nsight Compute | Latest |
| CMake | 3.20+ |
| Python | 3.10+ |
| GPU | RTX 4090 (primary), RTX 30xx (secondary) |

> Do NOT run timing experiments on Windows (WDDM adds jitter).
> Do NOT run timing experiments inside WSL2 (virtualization layer corrupts measurements).

---

## Key References

| Paper | Relevance | Link |
|-------|-----------|------|
| NIST FIPS 203 (ML-KEM) | The standard we are evaluating | [Free PDF](https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.203.pdf) |
| KyberSlash — TCHES 2025 | Timing attack on ARM ML-KEM (our motivation) | [Free PDF](https://kyberslash.cr.yp.to/kyberslash-20250115.pdf) |
| PQShield compiler bug — 2025 | Compiler reintroduces timing leak | [Blog post](https://pqshield.com/pqshield-plugs-timing-leaks-in-kyber-ml-kem-to-improve-pqc-implementation-maturity/) |
| HI-Kyber — TPDS 2024 | GPU ML-KEM throughput (no security analysis) | [IACR ePrint](https://eprint.iacr.org/2023/1194) |
| GPU timing side-channel — DAC 2017 | Proves GPU timing attacks feasible for AES | [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3060403.3060462) |

---

## Baseline Codebases

| Repo | License | Purpose |
|------|---------|---------|
| [liboqs-cupqc-meta](https://github.com/open-quantum-safe/liboqs-cupqc-meta) | Apache-2.0 | Primary GPU ML-KEM baseline |
| [atpqc-cuda](https://github.com/tono-satolab/atpqc-cuda) | MIT | Fallback Kyber CUDA implementation |
| [mlkem-native](https://github.com/pq-code-package/mlkem-native) | MIT/Apache | CPU reference (correctness validation) |
| [kyber-py](https://github.com/GiacomoPope/kyber-py) | MIT | Python reference (KAT cross-check) |

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

## ⚠️ Important Rules

1. **Never commit raw trace files > 100MB** — use gitignore, store on RunPod volume
2. **Always stop your RunPod pod when done** — idle pods waste money
3. **Never run timing experiments on Windows or WSL2** — results will be invalid
4. **Document every experiment** in the experiment log before closing your session
5. **Pin your CUDA and driver versions** — report them in every result

---

## License

This research code is currently **private**. It will be made public as an artifact at time of paper submission under MIT License.
