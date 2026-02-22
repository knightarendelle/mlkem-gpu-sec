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
