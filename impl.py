"""
Vivado out-of-context place & route for every synthesised config in results/hw.csv.

HLS C-synthesis numbers are estimates (hls4ml designs are typically over-estimated on LUTs);
this script gets post-route LUT/FF/DSP/BRAM and timing slack, and writes results/impl.csv.
Configs already in impl.csv are skipped; configs whose HLS DSP estimate exceeds the device
are skipped too (placement would fail) unless --try-oversized.

Run in WSL after sourcing Vitis settings64.sh (needs `vivado` on PATH):
  python impl.py                 # all synthesised configs
  python impl.py --bits 4 3      # subset
"""
import argparse
import os
import re
import shutil
import subprocess
import time

import pandas as pd

from synth import CLOCK_NS, DEFAULT_CLK_UNC, HW_CSV, PART, hls_dir, read_hw

IMPL_CSV = "results/impl.csv"
TCL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vivado_impl.tcl")

# report_utilization row name -> column (UltraScale+ naming)
UTIL_ROWS = {"CLB LUTs": "LUT", "CLB Registers": "FF", "DSPs": "DSP", "Block RAM Tile": "BRAM_tile"}


def parse_util(path):
    out = {}
    with open(path) as f:
        for line in f:
            cells = [c.strip().rstrip("*").strip() for c in line.split("|")]
            if len(cells) > 6 and cells[1] in UTIL_ROWS and UTIL_ROWS[cells[1]] not in out:
                col = UTIL_ROWS[cells[1]]
                out[col] = float(cells[2])
                out[f"{col}_pct"] = float(cells[-2])  # last numeric column is Util%
    return out


def parse_summary(path):
    kv = dict(line.split() for line in open(path) if line.strip())
    wns, period = float(kv["WNS"]), float(kv["PERIOD"])
    return dict(wns_ns=wns, timing_met=wns >= 0, fmax_mhz=1000.0 / (period - wns),
                fully_routed=int(kv["BAD_NETS"]) == 0)


def read_impl():
    if not os.path.exists(IMPL_CSV):
        return pd.DataFrame()
    df = pd.read_csv(IMPL_CSV, dtype={"bits": str})
    if "clk_unc" not in df:                    # rows written before --clock-uncertainty existed
        df.insert(3, "clk_unc", DEFAULT_CLK_UNC)
    df["clk_unc"] = df["clk_unc"].fillna(DEFAULT_CLK_UNC).astype(str)
    if "routed" in df and "fully_routed" not in df:
        df = df.rename(columns={"routed": "fully_routed"})
    return df


def run(d):
    t0 = time.time()
    log = os.path.join(d, "impl", "vivado.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.call(["vivado", "-mode", "batch", "-nojournal", "-nolog", "-source", TCL,
                              "-tclargs", d, PART, str(CLOCK_NS)], stdout=f, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(os.path.join(d, "impl", "summary.txt")):
        raise RuntimeError(f"vivado failed (rc={rc}); see {log}")
    row = parse_util(os.path.join(d, "impl", "post_route_util.rpt"))
    row.update(parse_summary(os.path.join(d, "impl", "summary.txt")))
    row["impl_minutes"] = round((time.time() - t0) / 60, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--reuse", nargs="+", type=int, help="only these reuse factors")
    ap.add_argument("--clock-uncertainty", nargs="+", help="only these HLS clock uncertainties, e.g. 45%%")
    ap.add_argument("--outdir", default="hls")
    ap.add_argument("--try-oversized", action="store_true")
    args = ap.parse_args()
    if shutil.which("vivado") is None:
        raise SystemExit("vivado not on PATH (source <Xilinx>/2026.1/Vitis/settings64.sh)")

    hw = read_hw()
    hw = hw[hw.synthesised.astype(bool) & hw.seed.isin(args.seeds)]
    if args.bits:
        hw = hw[hw.bits.isin(args.bits)]
    if args.reuse:
        hw = hw[hw.reuse.isin(args.reuse)]
    if args.clock_uncertainty:
        hw = hw[hw.clk_unc.isin(args.clock_uncertainty)]
    done = read_impl()

    def match(df, k):
        return (df.bits == k[0]) & (df.seed == k[1]) & (df.reuse == k[2]) & (df.clk_unc == k[3])

    for _, h in hw.iterrows():
        key = (h.bits, int(h.seed), int(h.reuse), str(h.clk_unc))
        tag = f"{h.bits}b s{h.seed} rf{h.reuse} cu{h.clk_unc}"
        if len(done) and match(done, key).any():
            print(f"skip {tag} (in {IMPL_CSV})")
            continue
        if h.DSP_pct > 100 and not args.try_oversized:
            print(f"skip {tag}: HLS estimate needs {h.DSP:.0f} DSPs ({h.DSP_pct:.0f}% of device)")
            continue
        bits_dir = "float" if h.bits == "ptq16" else h.bits
        d = hls_dir(args.outdir, bits_dir, key[1], key[2], key[3])
        print(f"\n=== impl {tag} -> {d}/impl", flush=True)
        row = dict(bits=key[0], seed=key[1], reuse=key[2], clk_unc=key[3], **run(d))
        print({k: row[k] for k in ("LUT", "FF", "DSP", "wns_ns", "fmax_mhz", "fully_routed", "impl_minutes")},
              flush=True)
        done = pd.concat([done[~match(done, key)] if len(done) else done, pd.DataFrame([row])],
                         ignore_index=True)
        done.to_csv(IMPL_CSV, index=False)

    if len(done):
        print(done.to_string(index=False))


if __name__ == "__main__":
    main()
