# Phase 4 — Hardened NTT Design

**Owner:** Person 3
**Depends on:** Phase 3 complete (root cause identified first)
**Goal:** Build a hardened NTT variant that reduces leakage with minimal throughput overhead.

---

## Success Criteria

- [ ] Hardened NTT implemented with at least 2 hardening techniques
- [ ] Hardened version passes same KAT tests as baseline
- [ ] TVLA t-statistic reduced vs baseline (re-run Phase 2 harness on hardened version)
- [ ] Throughput overhead measured and reported
- [ ] Leakage vs throughput curve plotted
- [ ] Results logged in experiments/results/experiment_log.md

---

## Hardening Techniques (Implement in Priority Order)

| Priority | Technique | Addresses | Est. Cost |
|----------|-----------|-----------|-----------|
| 1 | Branchless conditional selection (CMOV-style PTX) | Secret-dependent branches | Negligible |
| 2 | Bank-conflict-free twiddle factor layout | Shared memory bank conflicts | Memory overhead |
| 3 | Fixed memory access schedule | Data-dependent coalescing | 5-15% throughput |
| 4 | Uniform work partitioning | Warp divergence | 10-20% throughput |
| 5 | Noise padding (optional) | Residual timing variance | Configurable |

---

## Important Rule

**Do NOT start Phase 4 until Phase 3 is complete.**
The hardening techniques you implement must be driven by what Phase 3 found as root causes.
Hardening without root cause analysis is guesswork.

---

## Evaluation Protocol

For each hardened variant:
1. Run KAT tests — must pass identically to baseline
2. Run timing harness (Phase 2) — collect 100K traces
3. Compute TVLA t-statistic — compare to baseline
4. Measure throughput (kops/s) — compute overhead %
5. Log result to experiment_log.md

---

## Portability (Stretch Goal)

If time allows, implement hardened NTT in HIP (AMD ROCm) as well:
- Demonstrates cross-vendor applicability
- Strengthens paper's contribution claim
- See /workspace/refs for ROCm documentation

---

## Notes for Person 3

- Always test correctness before measuring performance
- The goal is NOT "constant-time" — it is "reduced statistical distinguishability"
- Even a partial reduction in |t| is publishable if root cause is well-attributed
- Document every variant you try, including ones that didn't work
