"""
hls4ml softmax calibration study (run in WSL via run_softmax.sh).

Background: hls4ml 1.3.0's `stable` softmax indexes its exp and 1/x lookup tables with the top
log2(table_size) bits of the table-input types. With the default inv_inp_t = fixed<18,8> the 1/sum lookup
has 0.25-wide bins over a sum that is always in [1, 5], so outputs do not sum to 1 (~1.08 here) and the raw
hardware probabilities are miscalibrated.

Modes:
  configs [--synth]          4-bit seed 0, softmax configs A-G: csim outputs (+ HLS C-synthesis estimates)
  seeds --config C [--bits]  one config for seeds 0-9 (4-bit and/or float/ptq16)
  impl --configs A ...       Vivado out-of-context place & route of chosen configs (seed 0, 4-bit)
  report                     results/softmax_{configs,seeds}.csv, results/softmax_summary.md

Every config change is asserted in the generated firmware (defines.h / parameters.h): hls4ml silently
ignores unknown config keys.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

CACHE = os.path.join("cache", "softmax")
BUILD = os.path.expanduser("~/qnn_build/softmax")
SEEDS = list(range(10))
TOP = 2
N_BOOT = 10_000

# Softmax table types are given as (width, integer) of an unsigned RND/SAT fixed type. hls4ml 1.3.0 does NOT
# read these from the layer config (Precision: exp_table/inv_table/inv_inp/accum are silently ignored: the
# Softmax layer's initialize() never looks them up), so they are set on the converted graph node before the
# firmware is written. TableSize / Implementation / Skip are configurable attributes and go via the config.
D_TYPES = {"inv_inp": (18, 3), "exp_table": (18, 1), "inv_table": (18, 1), "accum": (18, 3)}
CONFIGS = {
    "A_default": ({}, "deployed: stable, tables 1024, all table types fixed<18,8>"),
    "B_table4096": ({"TableSize": 4096}, "A + table size 4096"),
    "C_invinp": ({"Types": {"inv_inp": (18, 3)}}, "A + inv_inp ufixed<18,3> (1/sum lookup input)"),
    "D_types": ({"Types": D_TYPES}, "C + exp/inv table values ufixed<18,1>, accum ufixed<18,3>"),
    "E_types_table4096": ({"TableSize": 4096, "Types": D_TYPES}, "D + table size 4096"),
    "F_latency": ({"Implementation": "latency"}, "A with the latency implementation"),
    "G_skip": ({"Skip": True}, "softmax skipped: hardware outputs fixed-point logits (softmax on the ARM)"),
}


def expectations(name, sm):
    """(file, regex) that must match in the generated firmware for each config."""
    u = lambda w, i: rf"ap_ufixed<{w},\s*{i}[,>]"
    types = [("defines.h", u(18, 3) + rf"[^\n]*\b{sm}_inv_inp_t"), ("defines.h", u(18, 1) + rf"[^\n]*\b{sm}_exp_table_t"),
             ("defines.h", u(18, 1) + rf"[^\n]*\b{sm}_inv_table_t"), ("defines.h", u(18, 3) + rf"[^\n]*\b{sm}_accum_t")]
    t4096 = [("parameters.h", r"exp_table_size = 4096;"), ("parameters.h", r"inv_table_size = 4096;")]
    return {
        "A_default": [("parameters.h", r"softmax_implementation::stable"), ("parameters.h", r"inv_table_size = 1024;")],
        "B_table4096": t4096,
        "C_invinp": types[:1],
        "D_types": types,
        "E_types_table4096": types + t4096,
        "F_latency": [("parameters.h", r"softmax_implementation::latency")],
        "G_skip": [],
    }[name]


def softmax_name(model):
    return model.layers[-1].name


def build_hls(model, name, out_dir):
    import hls4ml
    from hls4ml.model.types import RoundingMode, SaturationMode
    from synth import CLOCK_NS, DEFAULT_CLK_UNC, PART, make_config
    cfg = make_config(model)
    sm = softmax_name(model)
    change, _ = CONFIGS[name]
    for k, v in change.items():
        if k != "Types":
            cfg["LayerName"][sm][k] = v
    hm = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=out_dir, part=PART, clock_period=CLOCK_NS,
        clock_uncertainty=DEFAULT_CLK_UNC, io_type="io_parallel", backend="Vitis")
    for attr, (w, i) in change.get("Types", {}).items():        # set on the node; see note at CONFIGS
        p = hm.graph[sm].get_attr(f"{attr}_t").precision         # backend-converted type: edit in place
        p.width, p.integer, p.signed = w, i, False
        p.rounding_mode, p.saturation_mode = RoundingMode.RND, SaturationMode.SAT
    hm.compile()
    fw = os.path.join(out_dir, "firmware")
    text = {f: open(os.path.join(fw, f)).read() for f in ("defines.h", "parameters.h")}
    for f, rx in expectations(name, sm):
        assert re.search(rx, text[f]), f"{name}: setting not applied, /{rx}/ not found in {f}"
    if name == "G_skip":
        assert "softmax_config" not in text["parameters.h"], "G_skip: softmax still present"
    else:
        assert "softmax_config" in text["parameters.h"], f"{name}: softmax missing"
    return hm


def run_csim(model, name, out_dir, X, dst):
    hm = build_hls(model, name, out_dir)
    out = np.asarray(hm.predict(X)).reshape(len(X), -1).astype(np.float32)
    np.save(dst, out)
    return hm, out


def mode_configs(args):
    import tensorflow as tf
    from sweep import load_data
    from synth import load_model, parse_report
    _, X_te, _, _, _ = load_data()
    X = np.ascontiguousarray(X_te, dtype=np.float32)
    os.makedirs(os.path.join(CACHE, "configs"), exist_ok=True)
    model = load_model("4", 0)
    np.save(os.path.join(CACHE, "keras_4_s0_pred.npy"), model.predict(X, batch_size=8192, verbose=0).argmax(1))
    hw_csv = os.path.join(CACHE, "configs_hw.csv")
    hw = pd.read_csv(hw_csv) if os.path.exists(hw_csv) else pd.DataFrame(columns=["config"])
    for name in args.only or CONFIGS:
        out_dir = os.path.join(BUILD, f"mlp_4b_s0_{name}")
        dst = os.path.join(CACHE, "configs", f"{name}.npy")
        hm, out = run_csim(model, name, out_dir, X, dst)
        print(f"{name:18s} out sum mean {out.sum(1).mean():.4f} [{out.sum(1).min():.3f}, {out.sum(1).max():.3f}]",
              flush=True)
        if args.synth and name not in set(hw.config):
            hm.build(csim=False, synth=True)
            row = dict(config=name, **parse_report(out_dir))
            hw = pd.concat([hw[hw.config != name], pd.DataFrame([row])], ignore_index=True)
            hw.to_csv(hw_csv, index=False)
            print(f"   HLS: {row['latency_cycles_worst']:.0f} cycles, LUT {row['LUT']:.0f}, FF {row['FF']:.0f}, "
                  f"DSP {row['DSP']:.0f}, BRAM_18K {row['BRAM_18K']:.0f}", flush=True)
        tf.keras.backend.clear_session()


def mode_seeds(args):
    import tensorflow as tf
    from sweep import load_data
    from synth import load_model
    _, X_te, _, _, _ = load_data()
    X = np.ascontiguousarray(X_te, dtype=np.float32)
    os.makedirs(os.path.join(CACHE, "seeds"), exist_ok=True)
    for bits in args.bits:
        for seed in SEEDS:
            dst = os.path.join(CACHE, "seeds", f"{args.config}_{bits}_s{seed}.npy")
            if os.path.exists(dst):
                print(f"skip {dst}")
                continue
            model = load_model(bits, seed)
            np.save(os.path.join(CACHE, "seeds", f"keras_{bits}_s{seed}_pred.npy"),
                    model.predict(X, batch_size=8192, verbose=0).argmax(1))
            _, out = run_csim(model, args.config, os.path.join(BUILD, f"mlp_{bits}b_s{seed}_{args.config}"), X, dst)
            print(f"{args.config} {bits:>5} s{seed}: out sum mean {out.sum(1).mean():.4f}", flush=True)
            tf.keras.backend.clear_session()


def mode_impl(args):
    from impl import run
    csv = os.path.join(CACHE, "configs_impl.csv")
    df = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame(columns=["config"])
    for name in args.configs:
        if name in set(df.config):
            print(f"skip {name} (in {csv})")
            continue
        d = os.path.join(BUILD, f"mlp_4b_s0_{name}")
        print(f"=== impl {name} -> {d}/impl", flush=True)
        row = dict(config=name, **run(d))
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv, index=False)
        print({k: row[k] for k in ("LUT", "FF", "DSP", "BRAM_tile", "wns_ns", "fmax_mhz", "impl_minutes")}, flush=True)


# ------------------------------------------------------------------ report
def probs_metrics(p, y, mass, raw=None):
    from sklearn.metrics import roc_auc_score
    from sweep import ece, sculpting
    chi2, jsd, _ = sculpting(p[:, TOP], y, mass)
    m = dict(accuracy=float((p.argmax(1) == y).mean()), ece=float(ece(p, y)),
             auc_top=float(roc_auc_score(y == TOP, p[:, TOP])), sculpt_jsd=float(jsd))
    if raw is not None:
        s = raw.sum(1)
        m.update(out_sum_mean=float(s.mean()), out_sum_min=float(s.min()), out_sum_max=float(s.max()))
    return m


def views(name, out):
    """(label, probabilities) to evaluate for a csim output array."""
    out = out.astype(np.float64)
    if name == "G_skip":                                   # logits -> exact float softmax (on the ARM)
        z = out - out.max(1, keepdims=True)
        e = np.exp(z)
        return [("softmax_on_arm", e / e.sum(1, keepdims=True))]
    return [("raw", out), ("renorm", out / out.sum(1, keepdims=True))]


def holm(ps):
    order = np.argsort(ps)
    out, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def wilcox(d):
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    return 1.0 if not (d != 0).any() else float(wilcoxon(d).pvalue)


def boot(d, rng):
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mode_report(args):
    from sweep import load_data
    _, _, _, y, mass = load_data()
    sw = pd.read_csv("results/sweep.csv", dtype={"bits": str}).set_index(["bits", "seed"])

    # configs (4-bit seed 0)
    rows = []
    kpred = np.load(os.path.join(CACHE, "keras_4_s0_pred.npy"))
    for name, (_, desc) in CONFIGS.items():
        f = os.path.join(CACHE, "configs", f"{name}.npy")
        if not os.path.exists(f):
            continue
        out = np.load(f)
        for view, p in views(name, out):
            rows.append(dict(config=name, description=desc, view=view, agree_keras=float((p.argmax(1) == kpred).mean()),
                             **probs_metrics(p, y, mass, raw=out if view != "softmax_on_arm" else None)))
    cfg = pd.DataFrame(rows)
    for extra in ("configs_hw.csv", "configs_impl.csv"):
        path = os.path.join(CACHE, extra)
        if os.path.exists(path) and len(cfg):
            h = pd.read_csv(path)
            pref = "hls_" if extra == "configs_hw.csv" else "postroute_"
            keep = [c for c in h.columns if c != "config"]
            cfg = cfg.merge(h.rename(columns={c: pref + c for c in keep}), on="config", how="left")
    cfg.to_csv("results/softmax_configs.csv", index=False)

    # seeds
    srows = []
    for f in sorted(os.listdir(os.path.join(CACHE, "seeds"))) if os.path.isdir(os.path.join(CACHE, "seeds")) else []:
        m = re.match(r"(?!keras_)(.+)_(float|\d+)_s(\d+)\.npy$", f)
        if not m:
            continue
        name, bits, seed = m.group(1), m.group(2), int(m.group(3))
        out = np.load(os.path.join(CACHE, "seeds", f))
        kp = np.load(os.path.join(CACHE, "seeds", f"keras_{bits}_s{seed}_pred.npy"))
        for view, p in views(name, out):
            srows.append(dict(config=name, bits=bits, seed=seed, view=view, keras_ece=sw.loc[(bits, seed), "ece"],
                              agree_keras=float((p.argmax(1) == kp).mean()), **probs_metrics(p, y, mass, raw=out)))
    sdf = pd.DataFrame(srows)
    sdf.to_csv("results/softmax_seeds.csv", index=False)

    # tests across seeds
    tests, rng = [], np.random.default_rng(0)
    if len(sdf):
        piv = sdf.pivot_table(index=["config", "bits", "seed"], columns="view", values="ece")
        base = piv.loc["A_default"] if "A_default" in piv.index.get_level_values(0) else None
        fam = []
        if base is not None:
            for bits in sorted(base.index.get_level_values(0).unique()):
                b = base.loc[bits]
                d = (b["raw"] - b["renorm"]).values
                lo, hi = boot(d, rng)
                fam.append(dict(test="raw_minus_renorm (A_default)", bits=bits, n=len(d), mean_delta=d.mean(),
                                ci95_lo=lo, ci95_hi=hi, p=wilcox(d)))
            for r, ph in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"] = ph
            tests += fam
            fam = []
            for bits in sorted(base.index.get_level_values(0).unique()):
                k = sw.loc[bits, "ece"].reindex(base.loc[bits].index).values
                d = base.loc[bits]["renorm"].values - k
                lo, hi = boot(d, rng)
                fam.append(dict(test="renorm_minus_keras (A_default)", bits=bits, n=len(d), mean_delta=d.mean(),
                                ci95_lo=lo, ci95_hi=hi, p=wilcox(d)))
            for r, ph in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"] = ph
            tests += fam
        for name in [c for c in piv.index.get_level_values(0).unique() if c != "A_default"]:
            fam = []
            for bits in sorted(piv.loc[name].index.get_level_values(0).unique()):
                fixed = piv.loc[(name, bits)]["raw"]
                for ref_name, ref in (("A_default raw", base.loc[bits]["raw"] if base is not None else None),
                                      ("Keras", sw.loc[bits, "ece"])):
                    if ref is None:
                        continue
                    d = fixed.values - ref.reindex(fixed.index).values
                    lo, hi = boot(d, rng)
                    fam.append(dict(test=f"{name} raw minus {ref_name}", bits=bits, n=len(d), mean_delta=d.mean(),
                                    ci95_lo=lo, ci95_hi=hi, p=wilcox(d)))
            for r, ph in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"] = ph
            tests += fam
    tdf = pd.DataFrame(tests)
    tdf.to_csv("results/softmax_tests.csv", index=False)
    write_summary(cfg, sdf, tdf, sw)
    print("wrote results/softmax_{configs,seeds,tests}.csv, results/softmax_summary.md")


def write_summary(cfg, sdf, tdf, sw):
    L = ["# hls4ml softmax: calibration and cost", "",
         "4-bit seed-0 model unless noted; fixed-point C simulation of all 166,000 test events. raw = hardware "
         "softmax outputs as produced; renorm = raw / row sum; softmax_on_arm = exact float softmax of the "
         f"hardware's fixed-point logits. Keras ECE of this model: {sw.loc[('4', 0), 'ece']:.4f}.", ""]
    if len(cfg):
        L += ["## Softmax configurations", "",
              "| config | view | out sum mean [min, max] | ECE | accuracy | agrees w/ Keras | HLS cycles | HLS LUT | "
              "HLS FF | HLS BRAM_18K | post-route LUT | post-route FF | WNS (ns) |", "|---|" + "---|" * 12]
        for _, r in cfg.iterrows():
            g = lambda c, f="{:.0f}": (f.format(r[c]) if c in r and pd.notna(r[c]) else "–")
            s = (f"{r.out_sum_mean:.3f} [{r.out_sum_min:.3f}, {r.out_sum_max:.3f}]"
                 if "out_sum_mean" in r and pd.notna(r.out_sum_mean) else "– (logits)")
            L.append(f"| {r.config} | {r['view']} | {s} | {r.ece:.4f} | {r.accuracy:.4f} | {r.agree_keras:.4f} | "
                     f"{g('hls_latency_cycles_worst')} | {g('hls_LUT')} | {g('hls_FF')} | {g('hls_BRAM_18K')} | "
                     f"{g('postroute_LUT')} | {g('postroute_FF')} | {g('postroute_wns_ns', '{:+.3f}')} |")
        L += ["", "Configs: " + "; ".join(f"**{k}** = {v[1]}" for k, v in CONFIGS.items()) + "."]
    if len(sdf):
        L += ["", "## Across seeds (mean ± std over 10 seeds)", "",
              "| config | bits | view | ECE | Keras ECE | out sum mean | accuracy |", "|---|---|---|---|---|---|---|"]
        for (c, b, v), g in sdf.groupby(["config", "bits", "view"]):
            L.append(f"| {c} | {b} | {v} | {g.ece.mean():.4f} ± {g.ece.std():.4f} | {g.keras_ece.mean():.4f} ± "
                     f"{g.keras_ece.std():.4f} | {g.out_sum_mean.mean():.3f} | {g.accuracy.mean():.4f} |")
    if len(tdf):
        L += ["", "## Paired tests (Wilcoxon over seeds, Holm within family; 95% CI = seed bootstrap of the mean)", "",
              "| test | bits | n | mean Δ ECE | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|---|"]
        for _, r in tdf.iterrows():
            L.append(f"| {r.test} | {r.bits} | {r.n} | {r.mean_delta:+.4f} | [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}] | "
                     f"{r.p:.3g} | {r.p_holm:.3g} |")
    with open("results/softmax_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("configs")
    c.add_argument("--synth", action="store_true")
    c.add_argument("--only", nargs="+", choices=list(CONFIGS))
    s = sub.add_parser("seeds")
    s.add_argument("--config", default="A_default", choices=list(CONFIGS))
    s.add_argument("--bits", nargs="+", default=["4", "float"])
    i = sub.add_parser("impl")
    i.add_argument("--configs", nargs="+", required=True, choices=list(CONFIGS))
    sub.add_parser("report")
    args = ap.parse_args()
    {"configs": mode_configs, "seeds": mode_seeds, "impl": mode_impl, "report": mode_report}[args.mode](args)


if __name__ == "__main__":
    main()
