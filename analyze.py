"""
Join the training sweep (results/sweep.csv) with hardware results (results/hw.csv,
results/impl.csv) and make the README figures + a markdown summary table.

  python analyze.py          ->  results/summary.csv, results/summary.md, figures/*.png

Hardware numbers prefer post-route (impl.csv) and fall back to HLS estimates (hw.csv);
the `hw_source` column says which one each row uses. Only ReuseFactor-1 rows are used for
the bit-width comparison; other reuse factors are listed separately in the table.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["float", "16", "8", "6", "4", "3"]
CLOCK_NS = 5
DEFAULT_CLK_UNC = "27%"   # must match synth.DEFAULT_CLK_UNC (not imported: synth pulls in TF/hls4ml)
DEVICE = {"LUT": 117_120, "FF": 234_240, "DSP": 1_248}  # xck26 (KR260 SoM)

# reference palette (dataviz skill): one series -> slot-1 blue; ink/grid tokens for chrome
BLUE, INK, INK2, MUTED, GRID, AXIS, SURF = ("#2a78d6", "#0b0b0b", "#52514e", "#898781",
                                            "#e1e0d9", "#c3c2b7", "#fcfcfb")
CRIT = "#d03b3b"   # status: critical (device limit), always paired with a text label

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "lines.linewidth": 2, "lines.markersize": 7,
})


def mean_std(df, cols):
    g = df.groupby("bits")[cols].agg(["mean", "std", "count"])
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    return g


def load():
    sweep = pd.read_csv("results/sweep.csv", dtype={"bits": str})
    hw = pd.read_csv("results/hw.csv", dtype={"bits": str})
    for col, default in (("reuse", 1), ("clk_unc", DEFAULT_CLK_UNC)):
        if col not in hw:
            hw[col] = default
        hw[col] = hw[col].fillna(default)
    hw["reuse"] = hw["reuse"].astype(int)
    hw["clk_unc"] = hw["clk_unc"].astype(str)
    hw["hw_source"] = "hls"
    hw = hw.rename(columns={"LUT": "LUT_hls", "FF": "FF_hls", "DSP": "DSP_hls"})
    keys = ["bits", "seed", "reuse", "clk_unc"]
    if os.path.exists("results/impl.csv"):
        impl = pd.read_csv("results/impl.csv", dtype={"bits": str})
        if "clk_unc" not in impl:
            impl["clk_unc"] = DEFAULT_CLK_UNC
        impl["clk_unc"] = impl["clk_unc"].fillna(DEFAULT_CLK_UNC).astype(str)
        hw = hw.merge(impl[keys + ["LUT", "FF", "DSP", "wns_ns", "fmax_mhz"]], on=keys, how="left")
        hw.loc[hw.LUT.notna(), "hw_source"] = "post-route"
    else:
        hw[["LUT", "FF", "DSP", "wns_ns", "fmax_mhz"]] = np.nan
    for r in ("LUT", "FF", "DSP"):
        hw[r] = hw[r].fillna(hw[f"{r}_hls"])
        hw[f"{r}_pct"] = 100 * hw[r] / DEVICE[r]
    hw["fits"] = (hw[["LUT_pct", "FF_pct", "DSP_pct"]] <= 100).all(axis=1)
    # Post-route: clock at min(200 MHz, achieved Fmax); latency in ns at that clock.
    hw["clock_mhz"] = np.minimum(1000 / CLOCK_NS, hw["fmax_mhz"]).fillna(1000 / CLOCK_NS)
    hw["latency_ns_hls"] = hw["latency_ns"]
    hw["latency_ns"] = hw["latency_cycles_worst"] * 1000 / hw["clock_mhz"]
    hw["timing_met_routed"] = hw["wns_ns"] >= 0

    # One row per (bits, seed, reuse): the build with the lowest latency at its achieved clock.
    # A deeper-pipelined build that closes 200 MHz can be slower in ns than the default build at a
    # slightly lower clock (8-bit: 18 cyc @ 200 MHz = 90 ns vs 14 cyc @ 182 MHz = 77 ns), and the
    # KR260 PL clock can be set to any of these frequencies. All builds stay in results/hw_variants.csv.
    hw.sort_values(["bits", "seed", "reuse", "clk_unc"]).to_csv("results/hw_variants.csv", index=False)
    hw["_default"] = hw.clk_unc != DEFAULT_CLK_UNC
    hw = (hw.sort_values(["latency_ns", "_default"], na_position="last")
            .drop_duplicates(["bits", "seed", "reuse"]).drop(columns="_default"))
    return sweep, hw.sort_values(["reuse", "seed"])


# Headline sculpting metric (see HANDOFF.md): each model vs the float models of all OTHER seeds,
# so the float row is a like-for-like noise floor. Older sweep.csv files only have the next-seed version.
def sculpt_col(sweep):
    return "sculpt_jsd_vs_float_xseed" if "sculpt_jsd_vs_float_xseed" in sweep else "sculpt_jsd_vs_float"


def summary(sweep, hw):
    sc = sculpt_col(sweep)
    rel = mean_std(sweep, ["accuracy", "auc_top", "ece", sc]).rename(
        columns=lambda c: c.replace(sc, "sculpt"))
    # Hardware: seed 0 only, so post-route (seed 0) and HLS-only (seeds 1, 2) numbers are never averaged together.
    # HLS estimates for seeds 0-2 differ by <= ~5% in LUT (results/hw.csv).
    h1 = hw[(hw.reuse == 1) & (hw.seed == 0)]
    hwm = mean_std(h1, ["hls_acc", "latency_ns", "clock_mhz", "LUT", "FF", "DSP"])
    src = h1.groupby("bits").hw_source.agg(lambda s: "/".join(sorted(set(s))))
    fits = h1.groupby("bits").fits.all()
    out = rel.join(hwm, how="left").join(src).join(fits)
    out = out.reindex([b for b in ORDER if b in out.index])
    out.to_csv("results/summary.csv")

    def pm(row, c, fmt):
        m, s, n = row[f"{c}_mean"], row[f"{c}_std"], row[f"{c}_count"]
        if pd.isna(m):
            return "-"
        return f"{m:{fmt}}" if n < 2 or pd.isna(s) else f"{m:{fmt}} ± {s:{fmt}}"

    lines = ["| bits | accuracy | top AUC | ECE | sculpting JSD vs float (cross-seed) | latency (ns) | clock (MHz) | LUT | DSP | fits KR260 | hw source |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in out.iterrows():
        n_seeds = int(r["accuracy_count"])
        lines.append(" | ".join([
            f"| {b}", pm(r, "accuracy", ".4f"), pm(r, "auc_top", ".4f"), pm(r, "ece", ".4f"),
            pm(r, "sculpt", ".4f"), pm(r, "latency_ns", ".0f"), pm(r, "clock_mhz", ".0f"),
            pm(r, "LUT", ",.0f"),
            pm(r, "DSP", ",.0f"),
            "-" if pd.isna(r.fits) or b == "float" else ("yes" if r.fits else "**no**"),
            "-" if pd.isna(r.hw_source) else r.hw_source]) + " |")
    other = hw[hw.reuse != 1]
    if len(other):
        lines += ["", "Other reuse factors (seed-level rows):", "",
                  "| bits | seed | reuse | II | latency (ns) | clock (MHz) | LUT | DSP | fits | hw source |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in other.iterrows():
            lines.append(f"| {r.bits} | {r.seed} | {r.reuse} | {r.II:.0f} | {r.latency_ns:.0f} | {r.clock_mhz:.0f} | {r.LUT:,.0f} | "
                         f"{r.DSP:,.0f} | {'yes' if r.fits else 'no'} | {r.hw_source} |")
    variants = pd.read_csv("results/hw_variants.csv", dtype={"bits": str, "clk_unc": str})
    variants = variants[variants.hw_source == "post-route"]
    if len(variants):
        lines += ["", "Post-route builds (seed 0): default HLS schedule vs deeper pipeline (45% clock uncertainty):", "",
                  "| bits | reuse | HLS clock uncertainty | cycles | Fmax (MHz) | latency (ns) | LUT | FF | DSP |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in variants.sort_values(["reuse", "bits", "clk_unc"]).iterrows():
            lines.append(f"| {r.bits} | {r.reuse} | {r.clk_unc} | {r.latency_cycles_worst:.0f} | {r.fmax_mhz:.1f} | "
                         f"{r.latency_ns:.1f} | {r.LUT:,.0f} | {r.FF:,.0f} | {r.DSP:,.0f} |")
    note = (f"\nAccuracy/AUC/ECE/sculpting: mean ± std over {n_seeds} training seeds on the full test split "
            "(sweep.csv); float sculpting = cross-seed noise floor. Hardware: seed-0 model, ReuseFactor 1, "
            "part xck26-sfvc784-2LV-c, 200 MHz target; HLS estimates for seeds 0-2 differ by <= ~5% (hw.csv). Where post-route results exist, LUT/DSP are post-route and "
            "latency = HLS cycles at min(200 MHz, post-route Fmax); per config the lowest-latency build is used.\n")
    with open("results/summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" + note)
    return out


def paired_tests(sweep):
    """Per-seed paired comparison of each bit width against float trained with the same seed.
    Wilcoxon signed-rank (two-sided), n = number of seeds. Holm-corrected within each metric.
    sculpt_jsd_vs_float is skipped: it is already a distance *to* the same-seed float model."""
    from scipy.stats import wilcoxon
    rows = []
    fl = sweep[sweep.bits == "float"].set_index("seed")
    for metric, better in [("accuracy", "higher"), ("auc_top", "higher"), ("ece", "lower")]:
        res = []
        for b in [b for b in ORDER[1:] if b in set(sweep.bits)]:
            q = sweep[sweep.bits == b].set_index("seed")
            seeds = q.index.intersection(fl.index)
            d = q.loc[seeds, metric] - fl.loc[seeds, metric]
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            res.append(dict(metric=metric, bits=b, n=len(d), median_delta=d.median(),
                            rel_change_pct=100 * d.mean() / fl.loc[seeds, metric].mean(), p=p))
        # Holm step-down within metric
        order = sorted(range(len(res)), key=lambda i: res[i]["p"])
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (len(res) - rank) * res[i]["p"]))
            res[i]["p_holm"] = running
        rows += res
    df = pd.DataFrame(rows)
    df.to_csv("results/paired_tests.csv", index=False)
    return df


def errline(ax, x, m, s, labels=None):
    ax.errorbar(x, m, yerr=s, color=BLUE, marker="o", capsize=3, elinewidth=1.2,
                markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)


def fig_reliability(out):
    bits = [b for b in ORDER if b in out.index]
    x = np.arange(len(bits))
    panels = [("accuracy", "Test accuracy", ".3f"), ("ece", "Expected calibration error (lower is better)", ".3f"),
              ("sculpt", "QCD mass sculpting vs float\n(cross-seed JSD, lower is better)", ".3f")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, (c, title, _) in zip(axes, panels):
        errline(ax, x, out.loc[bits, f"{c}_mean"], out.loc[bits, f"{c}_std"].fillna(0))
        if c == "sculpt":  # float row = float vs float models of the other seeds
            floor = out.loc["float", f"{c}_mean"]
            ax.axhline(floor, color=MUTED, lw=1, ls="--", zorder=2)
            ax.text(len(bits) - 1.5, floor * 0.97, "float seed-to-seed\nnoise floor", color=INK2, fontsize=8.5,
                    va="top", ha="center")
        ax.set_xticks(x, [b if b == "float" else f"{b}-bit" for b in bits])
        ax.set_title(title, loc="left")
    n = int(out["accuracy_count"].max())
    fig.suptitle(f"Reliability vs weight/activation bit width  (mean ± std, {n} seeds)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
    fig.savefig("figures/reliability_vs_bits.png", dpi=180)
    plt.close(fig)


def fig_pareto(out):
    hwb = [b for b in ORDER if b in out.index and pd.notna(out.loc[b, "LUT_mean"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, (c, xlabel) in zip(axes, [("LUT", "LUTs (post-route where available)"),
                                      ("latency_ns", "Latency (ns, at achieved clock, max 200 MHz)")]):
        for b in hwb:
            r = out.loc[b]
            fits = bool(r.fits)
            ax.errorbar(r[f"{c}_mean"], r["accuracy_mean"],
                        xerr=None if pd.isna(r[f"{c}_std"]) else r[f"{c}_std"],
                        yerr=r["accuracy_std"],
                        fmt="o", color=BLUE, mfc=BLUE if fits else SURF, mec=BLUE, mew=2,
                        capsize=3, elinewidth=1.2, zorder=3)
            # 16- and 8-bit sit almost on top of each other: put the 16-bit label below-left
            off, ha = ((4, 12), "right") if b == "16" else ((8, 4), "left")
            ax.annotate(f"{b}-bit" + ("" if fits else " (exceeds DSPs)"), (r[f"{c}_mean"], r["accuracy_mean"]),
                        textcoords="offset points", xytext=off, ha=ha,
                        color=INK2 if fits else CRIT, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.margins(x=0.12)
        if c == "LUT":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    axes[0].set_ylabel("Test accuracy (mean ± std, 10 seeds)")
    fig.suptitle("Accuracy vs FPGA cost on the KR260 (xck26), ReuseFactor 1",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
    fig.text(0.01, -0.02, "Hollow marker = does not fit the device (needs more DSPs than the xck26 has). "
             "hls4ml bit-accurate model matches Keras accuracy within 0.002 at every bit width.",
             color=MUTED, fontsize=8.5)
    fig.savefig("figures/pareto.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_resources(out):
    hwb = [b for b in ORDER if b in out.index and pd.notna(out.loc[b, "LUT_mean"])]
    x = np.arange(len(hwb))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, (c, title) in zip(axes, [("LUT", "LUTs (% of device)"), ("DSP", "DSPs (% of device)"),
                                     ("latency_ns", "Latency (ns)")]):
        if c in DEVICE:
            m = 100 * out.loc[hwb, f"{c}_mean"] / DEVICE[c]
            s = 100 * out.loc[hwb, f"{c}_std"].fillna(0) / DEVICE[c]
            ax.axhline(100, color=CRIT, lw=1, ls="--", zorder=2)
            ax.text(len(hwb) - 0.6, 100, "device limit", color=CRIT, fontsize=8.5, va="bottom", ha="right")
        else:
            m, s = out.loc[hwb, f"{c}_mean"], out.loc[hwb, f"{c}_std"].fillna(0)
        errline(ax, x, m, s)
        ax.set_xticks(x, [f"{b}-bit" for b in hwb])
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left")
    fig.suptitle("FPGA cost vs bit width (ReuseFactor 1, 200 MHz)", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="semibold")
    fig.savefig("figures/resources_vs_bits.png", dpi=180)
    plt.close(fig)


def main():
    os.makedirs("figures", exist_ok=True)
    sweep, hw = load()
    out = summary(sweep, hw)
    tests = paired_tests(sweep)
    with open("results/summary.md", "a", encoding="utf-8") as f:
        f.write("\nPaired vs same-seed float (Wilcoxon signed-rank, Holm-corrected per metric):\n\n"
                "| metric | bits | n | median Δ | mean rel. change | p | p (Holm) |\n|---|---|---|---|---|---|---|\n")
        for _, t in tests.iterrows():
            f.write(f"| {t.metric} | {t.bits} | {t.n} | {t.median_delta:+.4f} | {t.rel_change_pct:+.1f}% | "
                    f"{t.p:.3g} | {t.p_holm:.3g} |\n")
    fig_reliability(out)
    fig_pareto(out)
    fig_resources(out)
    print(open("results/summary.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
