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
