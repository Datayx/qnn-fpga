"""
hls4ml synthesis of the sweep.py models for the AMD Kria KR260 (xck26).

For each (bits, seed) model in models/:
  1. convert with hls4ml (Vitis backend, io_parallel, Latency strategy, ReuseFactor 1)
  2. compile the bit-accurate C++ model and measure test accuracy on the same
     split as sweep.py (keras_acc vs hls_acc, plus argmax agreement)
  3. run Vitis HLS C synthesis at 200 MHz and parse latency + resources
Rows are appended to results/hw.csv after every config, so an interrupted run
resumes where it stopped (use --force to redo a config).

The float model is synthesised by post-training quantisation at ap_fixed<16,6>
and labelled "ptq16"; it is NOT a float model on hardware.

Needs Linux (WSL is fine): g++ for step 2, Vitis 2026.1 `vitis-run` on PATH for step 3
(hls4ml >= 1.3 drives `vitis-run --mode hls` directly; no `vitis_hls` binary needed).

Run:
  python synth.py                       # seed 0, bits 16 8 6 4 3 (add --bits float for ptq16)
  python synth.py --seeds 1 2           # error bars, e.g. overnight
  python synth.py --bits 8 4 --no-synth # convert + accuracy check only
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
import hls4ml
from qkeras.utils import _add_supported_quantized_objects

from sweep import load_data

PART = "xck26-sfvc784-2LV-c"     # KR260 SoM
CLOCK_NS = 5                     # 200 MHz
DEFAULT_PRECISION = "ap_fixed<16,6>"
INPUT_PRECISION = "ap_fixed<16,6>"   # standardised inputs; range +/-32, 10 fractional bits
ALL_BITS = ["float", "16", "8", "6", "4", "3"]
DEFAULT_BITS = ["16", "8", "6", "4", "3"]
N_TEST = 10_000                  # test samples for the hls4ml-vs-Keras accuracy check
HW_CSV = "results/hw.csv"


def label(bits):
    return "ptq16" if bits == "float" else bits


def load_model(bits, seed):
    co = {}
    _add_supported_quantized_objects(co)
    return tf.keras.models.load_model(f"models/mlp_{bits}b_s{seed}.h5", custom_objects=co, compile=False)


def make_config(model, reuse=1):
    cfg = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vitis", default_precision=DEFAULT_PRECISION)
    cfg["Model"]["Strategy"] = "Latency"
    cfg["Model"]["ReuseFactor"] = reuse
    for layer_cfg in cfg["LayerName"].values():   # granularity='name' copies RF per layer; override it
        if "ReuseFactor" in layer_cfg:
            layer_cfg["ReuseFactor"] = reuse
    cfg["LayerName"][model.layers[0].name]["Precision"] = {"result": INPUT_PRECISION}
    return cfg


DEFAULT_CLK_UNC = "27%"          # hls4ml Vitis default: HLS schedules for 5 ns * (1 - 0.27) = 3.65 ns


def convert(model, out_dir, reuse=1, clk_unc=DEFAULT_CLK_UNC):
    return hls4ml.converters.convert_from_keras_model(
        model, hls_config=make_config(model, reuse), output_dir=out_dir, part=PART,
        clock_period=CLOCK_NS, clock_uncertainty=clk_unc, io_type="io_parallel", backend="Vitis")


def _csynth_xml_fallback(out_dir):
    """Read <top>_csynth.xml directly (vitis-run 2025+ may not write the hls.app hls4ml looks for)."""
    import glob
    import xml.etree.ElementTree as ET
    hits = sorted(glob.glob(os.path.join(out_dir, "**", "syn", "report", "myproject_csynth.xml"), recursive=True))
    if not hits:
        return None
    root = ET.parse(hits[0]).getroot()
    t = lambda path: root.findtext(path)
    cs = {
        "BestLatency": t("PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"),
        "WorstLatency": t("PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"),
        "IntervalMax": t("PerformanceEstimates/SummaryOfOverallLatency/Interval-max"),
        "EstimatedClockPeriod": t("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"),
    }
    for r in ("LUT", "FF", "DSP", "BRAM_18K"):
        cs[r] = t(f"AreaEstimates/Resources/{r}")
        cs[f"Available{r}"] = t(f"AreaEstimates/AvailableResources/{r}")
    return {k: v for k, v in cs.items() if v is not None}


def parse_report(out_dir):
    try:
        rep = hls4ml.report.parse_vivado_report(out_dir) or {}
    except Exception as e:  # e.g. IndexError when no solution list is found
        print(f"hls4ml report parser failed ({e!r}); falling back to csynth.xml")
        rep = {}
    cs = rep.get("CSynthesisReport") or _csynth_xml_fallback(out_dir)
    if not cs:
        raise RuntimeError(f"no C-synthesis report in {out_dir}; check {out_dir}/build_stdout.log")

    def num(*keys):
        for k in keys:
            if k in cs:
                return float(cs[k])
        return np.nan

    worst = num("WorstLatency")
    est_clk = num("EstimatedClockPeriod")
    row = dict(
        latency_cycles_best=num("BestLatency"), latency_cycles_worst=worst,
        latency_ns=worst * CLOCK_NS, est_clock_ns=est_clk, timing_met=est_clk <= CLOCK_NS,
        II=num("IntervalMax"),
    )
    for res, keys in {"LUT": ["LUT"], "FF": ["FF"], "DSP": ["DSP", "DSP48E"], "BRAM_18K": ["BRAM_18K"]}.items():
        used, avail = num(*keys), num(*[f"Available{k}" for k in keys])
        row[res] = used
        row[f"{res}_pct"] = 100 * used / avail if avail else np.nan
    return row


def read_hw():
    df = pd.read_csv(HW_CSV, dtype={"bits": str})
    if "reuse" not in df:                      # rows written before --reuse-factor existed
        df.insert(2, "reuse", 1)
    df["reuse"] = df["reuse"].fillna(1).astype(int)
    if "clk_unc" not in df:                    # rows written before --clock-uncertainty existed
        df.insert(3, "clk_unc", DEFAULT_CLK_UNC)
    df["clk_unc"] = df["clk_unc"].fillna(DEFAULT_CLK_UNC).astype(str)
    return df


def _match(df, bits, seed, reuse, clk_unc):
    return (df.bits == bits) & (df.seed == seed) & (df.reuse == reuse) & (df.clk_unc == clk_unc)


def save_row(row):
    df = read_hw() if os.path.exists(HW_CSV) else pd.DataFrame()
    if len(df):
        df = df[~_match(df, row["bits"], row["seed"], row["reuse"], row["clk_unc"])]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HW_CSV, index=False)


def already_done(bits, seed, reuse, need_synth, clk_unc=DEFAULT_CLK_UNC):
    if not os.path.exists(HW_CSV):
        return False
    df = read_hw()
    hit = df[_match(df, label(bits), seed, reuse, clk_unc)]
    return len(hit) > 0 and (not need_synth or bool(hit.synthesised.iloc[0]))


def hls_dir(outdir, bits, seed, reuse, clk_unc=DEFAULT_CLK_UNC):
    name = (f"mlp_{bits}b" + (f"_s{seed}" if seed else "") + (f"_rf{reuse}" if reuse != 1 else "")
            + (f"_cu{clk_unc.rstrip('%')}" if clk_unc != DEFAULT_CLK_UNC else ""))
    return os.path.join(outdir, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", nargs="+", default=DEFAULT_BITS, choices=ALL_BITS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--reuse-factor", type=int, default=1,
                    help="hls4ml ReuseFactor; >1 shares multipliers (fewer DSPs, II = RF)")
    ap.add_argument("--clock-uncertainty", default=DEFAULT_CLK_UNC,
                    help="HLS clock uncertainty, e.g. 45%%; larger = HLS pipelines deeper to close timing")
    ap.add_argument("--outdir", default="hls")
    ap.add_argument("--no-synth", action="store_true", help="convert + bit-accurate accuracy only")
    ap.add_argument("--force", action="store_true", help="redo configs already in results/hw.csv")
    args = ap.parse_args()

    synth = not args.no_synth
    if synth and shutil.which("vitis-run") is None:
        raise SystemExit("vitis-run not on PATH (source <Xilinx>/2026.1/Vitis/settings64.sh), or use --no-synth")
    os.makedirs("results", exist_ok=True)

    _, X_te, _, y_te, _ = load_data()
    X_te = np.ascontiguousarray(X_te[:N_TEST], dtype=np.float32)
    y_te = y_te[:N_TEST]
    clipped = (np.abs(X_te) >= 32).mean()
    if clipped:
        print(f"warning: {clipped:.2e} of input values exceed {INPUT_PRECISION} range")

    for seed in args.seeds:
        for bits in args.bits:
            rf, cu = args.reuse_factor, args.clock_uncertainty
            if not args.force and already_done(bits, seed, rf, synth, cu):
                print(f"skip {label(bits)} s{seed} rf{rf} cu{cu} (in {HW_CSV})")
                continue
            out_dir = hls_dir(args.outdir, bits, seed, rf, cu)
            print(f"\n=== {label(bits)} seed {seed} rf {rf} cu {cu} -> {out_dir}")

            model = load_model(bits, seed)
            hls_model = convert(model, out_dir, rf, cu)
            hls_model.compile()
            k_pred = model.predict(X_te, batch_size=4096, verbose=0).argmax(1)
            h_pred = np.asarray(hls_model.predict(X_te)).reshape(len(X_te), -1).argmax(1)
            row = dict(bits=label(bits), seed=seed, reuse=rf, clk_unc=cu,
                       keras_acc=(k_pred == y_te).mean(), hls_acc=(h_pred == y_te).mean(),
                       hls_keras_agree=(h_pred == k_pred).mean(), synthesised=False)
            print(f"keras_acc {row['keras_acc']:.4f}  hls_acc {row['hls_acc']:.4f}  "
                  f"agree {row['hls_keras_agree']:.4f}")

            if synth:
                hls_model.build(csim=False, synth=True)
                row.update(parse_report(out_dir), synthesised=True)
                print({k: row[k] for k in ("latency_ns", "est_clock_ns", "LUT", "FF", "DSP", "BRAM_18K")})
            save_row(row)

    print(read_hw().to_string(index=False))


if __name__ == "__main__":
    main()
