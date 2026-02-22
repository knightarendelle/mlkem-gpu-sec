# Phase 3 — Root Cause Analysis

**Owner:** Person 3 (with Person 1 support)
**Goal:** Use Nsight Compute to attribute timing variance to specific GPU microarchitectural phenomena.

---

## Success Criteria

- [ ] Nsight Compute report collected for decapsulation kernel
- [ ] Bank conflict counts compared across ciphertext classes
- [ ] Warp divergence compared across ciphertext classes
- [ ] PTX/SASS inspected for secret-dependent branches (cuobjdump)
- [ ] Root cause identified and documented
- [ ] Results logged in experiments/results/experiment_log.md

---

## Key Nsight Metrics to Collect

| Metric | What It Reveals |
|--------|----------------|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Shared memory bank conflicts in NTT |
| `smsp__sass_branch_targets_threads_divergent.sum` | Warp divergence on secret-dependent paths |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` | Global memory access variation |
| `sm__sass_inst_executed_op_memory_128b.sum` | Coalescing efficiency |
| `gpu__time_duration.sum` | Total kernel time |

---

## Nsight Collect Command Template

```bash
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  smsp__sass_branch_targets_threads_divergent.sum,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  sm__sass_inst_executed_op_memory_128b.sum,\
  gpu__time_duration.sum \
  --export experiments/nsight/report_%i \
  ./baseline/mlkem_benchmark
```

---

## SASS Inspection Command

```bash
cuobjdump --dump-sass ./baseline/mlkem_benchmark > experiments/nsight/sass_dump.txt
grep -n "BRA\|EXIT\|SYNC" experiments/nsight/sass_dump.txt
```

---

## Notes for Person 3

- Run Nsight in SEPARATE runs from timing collection — never together
- Compare metrics between: (a) fixed ciphertext vs random, (b) legitimate vs invalid ciphertexts
- Any metric that differs between ciphertext classes is a potential leakage root cause
- Document every finding in this README under "Findings"

---

## Findings

(Document root cause findings here as you discover them)
