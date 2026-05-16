#!/usr/bin/env python3
"""
harness/ncu_isolated_profile.py
============================================================
NCU cache-metric profiling of hash_ct and generate_at in ISOLATION
(not embedded in the full decapsulation pipeline).

Each binary runs a SINGLE class only — clean sequential kernel launches
that NCU can profile without needing to split interleaved invocations.

Binaries:
  ncu_hashct_kyber1024.out   class 0 → valid ciphertext
                              class 1 → uniform-random bytes
  ncu_genmatrix_kyber1024.out class 0 → rho0 (random seed)
                              class 1 → rho1 (independent random seed)

Both binaries share the stdin format of the existing trace_* scripts:
  ninputs genmat_nw genvec_nw genpoly_nw fips_nw ntraces ct_class

Expected result (DCC dirty-line hypothesis):
  - hash_ct:    DRAM_wr_B nearly equal between classes in isolation
                (pipeline surplus was from upstream dirty lines, not hash itself)
  - generate_at: all metrics nearly equal (same SHAKE128 work regardless of class)

Usage:
  python3 harness/ncu_isolated_profile.py [--ntraces N] [--sudo] [--skip-run]
  python3 harness/ncu_isolated_profile.py --kernel genmatrix --ntraces 1000

Options:
  --kernel       hashct | genmatrix | all  (default: all)
  --ntraces      kernel invocations per class (default: 1000)
  --outdir       output directory (default: experiments/results)
  --skip-run     skip ncu runs; re-parse existing CSVs
  --sudo         prefix ncu with sudo

Build:
  make -C baseline/atpqc-cuda ncu_hashct_kyber1024 ncu_genmatrix_kyber1024
============================================================
"""

import argparse
import csv
import io
import os
import subprocess
import sys
from collections import defaultdict

METRICS = [
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
]

METRIC_SHORT = {
    "lts__t_sectors_op_read.sum":                      "L2_rd_sec",
    "lts__t_sectors_op_write.sum":                     "L2_wr_sec",
    "lts__t_sectors_lookup_hit.sum":                   "L2_hit",
    "lts__t_sectors_lookup_miss.sum":                  "L2_miss",
    "dram__bytes_read.sum":                            "DRAM_rd_B",
    "dram__bytes_write.sum":                           "DRAM_wr_B",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":  "L1_ld_sec",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum":  "L1_st_sec",
}

KERNELS = {
    "hashct": {
        "binary_default": "baseline/atpqc-cuda/target/ncu_hashct_kyber1024.out",
        "target_logical":  "hash_ct",
        "label":           "hash_ct (SHA3-256 over 1568-byte ciphertext)",
        "class0_desc":     "valid Kyber-1024 ciphertext",
        "class1_desc":     "uniform-random 1568 bytes",
    },
    "genmatrix": {
        "binary_default": "baseline/atpqc-cuda/target/ncu_genmatrix_kyber1024.out",
        "target_logical":  "generate_at",
        "label":           "generate_at (SHAKE128 matrix generation)",
        "class0_desc":     "rho0 (random 32-byte seed)",
        "class1_desc":     "rho1 (independent random 32-byte seed)",
    },
}


# ── CUDA name → logical name (shared with ncu_cache_profile.py) ───────────────
def cuda_name_to_logical(cuda_name: str) -> str:
    n = cuda_name.lower()
    if "sha3" in n:
        return "hash_coin" if "512" in cuda_name else "hash_ct"
    if "shake" in n:
        return "kdf"
    if "genmatrix" in n:
        return ("generate_at"
                if ("true" in n or ", true" in cuda_name or ",true" in cuda_name)
                else "generate_a(keypair)")
    if "gennoise" in n:
        if "<1" in cuda_name or ", 1," in cuda_name or "<1u" in cuda_name or "<1," in cuda_name:
            return "generate_e2"
        return "generate_r/e1"
    if "polyvec_decompress" in n:
        return "decompress_u"
    if "poly_decompress" in n:
        return "decompress_v"
    if "polyvec_frombytes" in n or ("polyvec" in n and "frombytes" in n):
        return "decode_s/t"
    if "poly_frombytes" in n:
        return "decode_s/t"
    if "polyvec_tobytes" in n or ("polyvec" in n and "tobytes" in n):
        return "encode(keypair)"
    if "poly_tomsg" in n:
        return "poly_tomsg"
    if "poly_frommsg" in n:
        return "poly_frommsg"
    if "polyvec_compress" in n:
        return "compress_u"
    if "poly_compress" in n:
        return "compress_v"
    if "fwdntt" in n:
        return "fwdntt_u/r"
    if "invntt" in n:
        if "<1" in cuda_name or "<1u" in cuda_name:
            return "intt_su/tr"
        return "intt_ar"
    if "mattimesvec_tomont" in n:
        return "mtvpv(keypair)"
    if "mattimesvec" in n:
        return "a_times_r"
    if "vectimesvec" in n:
        return "s/t_times_u/r"
    if "vecadd2" in n:
        return "ar_plus_e1"
    if "polyadd3" in n:
        return "tr_plus_e2_plus_m"
    if "polysub" in n:
        return "v_minus_su"
    if "verify" in n or "cmov" in n:
        return "verify_cmov"
    return cuda_name


# ── Run NCU ───────────────────────────────────────────────────────────────────
def run_ncu(binary: str, ct_class: int, ntraces: int,
            csv_out: str, use_sudo: bool) -> None:
    stdin_str = f"1 4 4 4 4 {ntraces} {ct_class}\n"
    ncu_bin = ["sudo", "ncu"] if use_sudo else ["ncu"]
    cmd = ncu_bin + [
        "--metrics", ",".join(METRICS),
        "--csv",
        "--target-processes", "all",
        "--kernel-name-base", "demangled",
        binary,
    ]
    print(f"\n[ncu class {ct_class}] {' '.join(cmd)}", flush=True)
    print(f"[ncu class {ct_class}] stdin: {stdin_str.strip()}", flush=True)
    print(f"[ncu class {ct_class}] output → {csv_out}", flush=True)
    print(f"[ncu class {ct_class}] (profiling {ntraces} invocations; may take several minutes...)",
          flush=True)

    with open(csv_out, "w") as fout:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=fout,
            stderr=sys.stderr, text=True)
        proc.communicate(input=stdin_str)

    if proc.returncode != 0:
        print(f"[ERROR] ncu exited with code {proc.returncode}", file=sys.stderr)
        sys.exit(1)
    print(f"[ncu class {ct_class}] done.", flush=True)


# ── Parse NCU CSV ─────────────────────────────────────────────────────────────
def parse_ncu_csv(filepath: str, target_logical: str) -> dict:
    """
    Parse NCU --csv output and return data for the target logical kernel only.
    Returns: metric_name → list[float]
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if '"ID"' in line or line.strip().strip('"').startswith("ID"):
            header_idx = i
            break

    if header_idx is None:
        print(f"[ERROR] No CSV header in {filepath}", file=sys.stderr)
        for ln in lines[:10]:
            print(f"  {ln!r}", file=sys.stderr)
        sys.exit(1)

    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text))

    data = defaultdict(list)  # metric → list[float]
    total_rows = 0
    matched_invocations = set()  # (kernel_name, invocation_id) for target

    for row in reader:
        total_rows += 1
        cuda_name = row.get("Kernel Name", "").strip()
        if not cuda_name:
            continue

        logical = cuda_name_to_logical(cuda_name)
        if logical != target_logical:
            continue

        metric = row.get("Metric Name", "").strip()
        if metric not in METRICS:
            continue

        val_str = row.get("Metric Value", "").strip().replace(",", "")
        try:
            val = float(val_str)
        except ValueError:
            continue

        inv_id = row.get("ID", "?")
        matched_invocations.add(inv_id)
        data[metric].append(val)

    n_inv = len(matched_invocations)
    n_metrics = len(data)
    print(f"  Parsed {total_rows} rows → {n_inv} '{target_logical}' invocations"
          f" × {n_metrics} metrics from {os.path.basename(filepath)}")

    if n_inv == 0:
        print(f"  [WARN] No invocations of '{target_logical}' found in {filepath}",
              file=sys.stderr)
        print(f"  [WARN] Logical names seen: {_sample_logical_names(csv_text)}",
              file=sys.stderr)

    return dict(data)


def _sample_logical_names(csv_text: str) -> list:
    seen = set()
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        name = row.get("Kernel Name", "").strip()
        if name:
            seen.add(cuda_name_to_logical(name))
        if len(seen) >= 10:
            break
    return sorted(seen)


# ── Format comparison table ───────────────────────────────────────────────────
def format_kernel_table(kernel_key: str, data0: dict, data1: dict,
                        ntraces: int) -> str:
    info = KERNELS[kernel_key]
    lines = []
    sep = "=" * 100

    lines.append(sep)
    lines.append(f"  ISOLATED NCU PROFILE: {info['label']}")
    lines.append(f"  class 0: {info['class0_desc']}")
    lines.append(f"  class 1: {info['class1_desc']}")
    lines.append(f"  ntraces per class: {ntraces}")
    lines.append(f"  DCC hypothesis: DRAM_wr_B should be nearly equal in isolation.")
    lines.append(sep)

    hdr = (f"  {'Metric':<12}  {'Class0_mean':>14}  {'Class1_mean':>14}"
           f"  {'%diff (C1-C0)':>14}  {'N0':>6}  {'N1':>6}")
    lines.append(hdr)
    lines.append("  " + "-" * 80)

    dram_wr_pct = None

    for metric in METRICS:
        vals0 = data0.get(metric, [])
        vals1 = data1.get(metric, [])
        if not vals0 and not vals1:
            continue
        mean0 = sum(vals0) / len(vals0) if vals0 else float("nan")
        mean1 = sum(vals1) / len(vals1) if vals1 else float("nan")
        n0, n1 = len(vals0), len(vals1)

        if mean0 != mean0 or mean1 != mean1:
            pct_str = "   n/a"
        elif mean0 == 0:
            pct_str = "    inf"
        else:
            pct = (mean1 - mean0) / mean0 * 100.0
            pct_str = f"{pct:+.2f}%"
            if metric == "dram__bytes_write.sum":
                dram_wr_pct = pct

        short = METRIC_SHORT.get(metric, metric[:12])
        m0_str = f"{mean0:,.1f}" if mean0 == mean0 else "n/a"
        m1_str = f"{mean1:,.1f}" if mean1 == mean1 else "n/a"
        lines.append(f"  {short:<12}  {m0_str:>14}  {m1_str:>14}"
                     f"  {pct_str:>14}  {n0:>6}  {n1:>6}")

    lines.append("")

    # Verdict
    if dram_wr_pct is not None:
        abs_pct = abs(dram_wr_pct)
        if abs_pct < 5.0:
            verdict = (f"  DRAM_wr_B difference: {dram_wr_pct:+.2f}%  → NEGLIGIBLE in isolation.\n"
                       f"  Pipeline surplus was from upstream dirty-line residue (DCC context bleed),\n"
                       f"  not from the kernel itself. Confirms DCC hypothesis.")
        elif abs_pct < 20.0:
            verdict = (f"  DRAM_wr_B difference: {dram_wr_pct:+.2f}%  → SMALL but non-zero in isolation.\n"
                       f"  Kernel has modest intrinsic sensitivity; pipeline surplus is likely mostly\n"
                       f"  DCC dirty-line residue from upstream.")
        else:
            verdict = (f"  DRAM_wr_B difference: {dram_wr_pct:+.2f}%  → SIGNIFICANT in isolation.\n"
                       f"  The kernel itself produces different DRAM write traffic per class.\n"
                       f"  DCC acts directly on this kernel's output, not just via upstream context.")
        lines.append(verdict)
    lines.append(sep)
    return "\n".join(lines)


# ── Pipeline comparison cross-reference ──────────────────────────────────────
PIPELINE_DRAM_WR = {
    "hash_ct":     {"pct": +161.05, "c0": 30,   "c1": 79},
    "generate_at": {"pct": None,    "c0": None,  "c1": None},  # not in pipeline NCU yet
}


def format_crossref(kernel_key: str, iso_data0: dict, iso_data1: dict) -> str:
    target = KERNELS[kernel_key]["target_logical"]
    if target not in PIPELINE_DRAM_WR:
        return ""
    pip = PIPELINE_DRAM_WR[target]
    if pip["pct"] is None:
        return ""

    vals0 = iso_data0.get("dram__bytes_write.sum", [])
    vals1 = iso_data1.get("dram__bytes_write.sum", [])
    if not vals0 or not vals1:
        return ""

    m0 = sum(vals0) / len(vals0)
    m1 = sum(vals1) / len(vals1)
    iso_pct = (m1 - m0) / m0 * 100.0 if m0 != 0 else float("inf")

    lines = [
        "",
        "  Cross-reference with full-pipeline NCU (experiments/results/dcc_ncu.txt):",
        f"  {'':25}  {'pipeline':>10}  {'isolated':>10}",
        f"  {'DRAM_wr_B class0 mean':25}  {pip['c0']:>10}  {m0:>10.1f}",
        f"  {'DRAM_wr_B class1 mean':25}  {pip['c1']:>10}  {m1:>10.1f}",
        f"  {'DRAM_wr_B %diff (C1-C0)':25}  {pip['pct']:>+9.2f}%  {iso_pct:>+9.2f}%",
        "",
        ("  If isolated %diff << pipeline %diff: dirty-line residue from upstream\n"
         "  kernels (not the kernel itself) is the dominant DCC leakage source."),
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="NCU isolated cache-metric profiling for hash_ct and generate_at",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kernel", choices=["hashct", "genmatrix", "all"], default="all",
        help="Which kernel(s) to profile (default: all)",
    )
    parser.add_argument(
        "--ntraces", type=int, default=1000,
        help="Kernel invocations per class (default: 1000)",
    )
    parser.add_argument(
        "--outdir", default="experiments/results",
        help="Directory for NCU CSV files and text report",
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip ncu runs; parse existing CSVs from --outdir",
    )
    parser.add_argument(
        "--sudo", action="store_true",
        help="Prefix ncu with sudo",
    )
    parser.add_argument(
        "--binary-hashct",
        default=None,
        help="Override path to ncu_hashct_kyber1024.out",
    )
    parser.add_argument(
        "--binary-genmatrix",
        default=None,
        help="Override path to ncu_genmatrix_kyber1024.out",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    which = ["hashct", "genmatrix"] if args.kernel == "all" else [args.kernel]
    all_sections = []

    for kk in which:
        info = KERNELS[kk]
        binary = (args.binary_hashct if kk == "hashct" else args.binary_genmatrix) \
                 or info["binary_default"]

        csv0 = os.path.join(args.outdir, f"ncu_iso_{kk}_class0_raw.csv")
        csv1 = os.path.join(args.outdir, f"ncu_iso_{kk}_class1_raw.csv")

        if not args.skip_run:
            if not os.path.isfile(binary):
                print(f"[ERROR] Binary not found: {binary}", file=sys.stderr)
                print(f"  Build with: make -C baseline/atpqc-cuda ncu_{kk}_kyber1024",
                      file=sys.stderr)
                return 1
            print(f"\n{'='*60}\n  Profiling: {info['label']}\n{'='*60}")
            run_ncu(binary, 0, args.ntraces, csv0, args.sudo)
            run_ncu(binary, 1, args.ntraces, csv1, args.sudo)
        else:
            for p in [csv0, csv1]:
                if not os.path.isfile(p):
                    print(f"[ERROR] --skip-run set but CSV not found: {p}",
                          file=sys.stderr)
                    return 1
            print(f"[skip-run] re-parsing {csv0}, {csv1}")

        print(f"\n[parse] {kk} class 0...", flush=True)
        data0 = parse_ncu_csv(csv0, info["target_logical"])
        print(f"[parse] {kk} class 1...", flush=True)
        data1 = parse_ncu_csv(csv1, info["target_logical"])

        section = format_kernel_table(kk, data0, data1, args.ntraces)
        xref = format_crossref(kk, data0, data1)
        all_sections.append(section + xref)

    report = "\n\n".join(all_sections) + "\n"
    print("\n" + report)

    report_path = os.path.join(args.outdir, "dcc_ncu_isolated.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[report] saved → {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
