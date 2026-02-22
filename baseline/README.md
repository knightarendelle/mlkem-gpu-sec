# Phase 1 — Baseline GPU ML-KEM Implementation

**Owner:** Person 1
**Goal:** Working GPU ML-KEM (all 3 parameter sets) that passes KAT tests and matches published throughput numbers.

---

## Success Criteria

- [ ] ML-KEM-512 keygen + encaps + decaps runs on GPU without errors
- [ ] ML-KEM-768 passes all NIST KAT test vectors
- [ ] ML-KEM-1024 passes all NIST KAT test vectors
- [ ] Shared secret matches CPU reference (mlkem-native) for 10,000 random key pairs
- [ ] Throughput baseline recorded (kops/s) for all 3 parameter sets
- [ ] Results logged in experiments/results/experiment_log.md

---

## Setup

```bash
# From repo root
cd baseline
bash setup.sh
```

---

## What setup.sh Does

1. Attempts to build liboqs-cupqc-meta (primary)
2. If that fails after 3 hours, falls back to atpqc-cuda
3. Runs KAT validation against NIST test vectors
4. Records throughput baseline

---

## Notes for Person 1

- Use RTX 40xx only — not GTX 16xx
- Run on RunPod (Ubuntu 22.04), not Windows/WSL2
- If liboqs-cupqc-meta fails to build, do NOT spend more than 3 hours on it — switch to atpqc-cuda
- Document every build error you hit in this README under "Known Issues"

---

## Known Issues

(Document build errors here as you encounter them)
