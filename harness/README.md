# Phase 2 — Timing Trace Harness & TVLA Analysis

**Owner:** Person 2
**Goal:** Collect timing traces for GPU ML-KEM decapsulation and run TVLA statistical leakage test.

---

## Success Criteria

- [ ] Timing harness collects 100,000 CUDA Event timestamps for decapsulation
- [ ] Histogram of timing distribution plotted and saved
- [ ] TVLA t-statistic computed for fixed vs random ciphertext sets
- [ ] KS test and mutual information computed
- [ ] Results logged in experiments/results/experiment_log.md

---

## Key Rule

**NEVER mix Nsight profiling runs with timing collection runs.**
Nsight replay distorts timing. Path A (CUDA Events) and Path B (Nsight) are always separate.

---

## TVLA Pass/Fail Threshold

| t-statistic | Interpretation |
|-------------|---------------|
| \|t\| < 4.5 | No detectable leakage |
| \|t\| ≥ 4.5 | Statistically significant leakage detected |

---

## Files in This Folder

- `collect_traces.cu` — CUDA timing harness (to be written by Person 2)
- `tvla_analysis.py` — TVLA t-test + KS test + mutual information (to be written by Person 2)
- `plot_distributions.py` — Timing distribution plots

---

## Notes for Person 2

- Warm up: always discard first 1,000 decapsulations before collecting
- Fix the secret key (dk) for the entire experiment session
- Store (ciphertext_class, timing_us) tuples — not just timing alone
- Repeat across 5 different secret keys to check stability
- If \|t\| < 4.5: increase N to 500,000 before concluding no leakage
