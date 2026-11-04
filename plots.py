"""
Figures from results/sweep.csv and results/mass_after_cut.npz (no TensorFlow needed).

  plots/relative_change.png  headline: % change vs float for accuracy, ECE, sculpting
  plots/metrics_vs_bits.png  absolute metrics vs bits, every seed + mean +/- std
  plots/mass_after_cut.png   QCD jet-mass shape after the top-tag cut, float vs 4 vs 3 bits
  results/summary_by_bits.csv  the table behind the figures

Run:  python plots.py
TODO once results/hw.csv exists: Pareto (accuracy vs latency vs LUTs).
TODO: reliability diagrams (need per-event probabilities; re-predict from models/).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["float", "16", "8", "6", "4", "3"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, fixed order
MARKERS = ["o", "s", "^"]                           # secondary encoding, not colour-only

METRICS = {   # column: (label, higher_is_better)
    "accuracy": ("Accuracy", True),
    "auc_top": ("Top-tag AUC", True),
    "ece": ("Expected calibration error", False),
    "sculpt_jsd_vs_float_xseed": ("Mass sculpting (JSD vs float)", False),
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 1.5,
})


def load():
    df = pd.read_csv("results/sweep.csv", dtype={"bits": str})
    df["bits"] = pd.Categorical(df.bits, ORDER, ordered=True)
    return df.sort_values(["bits", "seed"])


def xlabels(ax):
    ax.set_xticks(range(len(ORDER)), ["float"] + [f"{b}-bit" for b in ORDER[1:]])
    ax.set_xlabel("Weight / activation precision")


def summary(df):
    cols = ["accuracy", "auc_top", "ece", "sculpt_jsd", "sculpt_jsd_vs_float_xseed"]
    s = df.groupby("bits", observed=True)[cols].agg(["mean", "std"])
    s.columns = [f"{c}_{a}" for c, a in s.columns]
    s.insert(0, "n_seeds", df.groupby("bits", observed=True).size())
    s.round(5).to_csv("results/summary_by_bits.csv")


def relative_change(df):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(ORDER))
    show = ["accuracy", "ece", "sculpt_jsd_vs_float_xseed"]
    names = ["Accuracy", "Calibration error (ECE)", "Mass sculpting"]
    for i, (col, name) in enumerate(zip(show, names)):
        base = df.loc[df.bits == "float", col].mean()
        rel = (df[col] / base - 1) * 100
        g = rel.groupby(df.bits, observed=True)
        m, s = g.mean().reindex(ORDER), g.std().reindex(ORDER)
        ax.errorbar(x, m, yerr=s, color=SERIES[i], marker=MARKERS[i], ms=6, capsize=0,
                    elinewidth=1, label=name, zorder=3)
        end = m.iloc[-1]
        ax.annotate(f"{name}  {end:+.{1 if abs(end) < 10 else 0}f}%", (x[-1], end), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    xlabels(ax)
    ax.set_xlim(-0.3, len(ORDER) + 1.6)
    ax.set_ylabel("Change vs float baseline (%)")
    ax.set_title("Below 6 bits, calibration and sculpting shift far more than accuracy")
    ax.legend(loc="upper left")
    fig.text(0.01, 0.005, "Mean ± std over 10 seeds; float mean = 0%. Sculpting: JSD of QCD jet mass "
             "after a 10%-QCD-efficiency top-tag cut, vs float models of other seeds.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("plots/relative_change.png", dpi=200)
    plt.close(fig)


def metrics_vs_bits(df):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True)
    x = np.arange(len(ORDER))
    rng = np.random.default_rng(0)
    for ax, (col, (label, hib)) in zip(axes.flat, METRICS.items()):
        pos = df.bits.cat.codes.to_numpy() + rng.uniform(-0.12, 0.12, len(df))
        ax.scatter(pos, df[col], s=10, color=MUTED, alpha=0.55, lw=0, zorder=2)
        g = df.groupby("bits", observed=True)[col]
        ax.errorbar(x, g.mean().reindex(ORDER), yerr=g.std().reindex(ORDER), color=SERIES[0],
                    marker="o", ms=6, capsize=0, elinewidth=1.2, zorder=3)
        ax.set_title(f"{label} ({'higher' if hib else 'lower'} is better)", fontsize=10)
    for ax in axes[1]:
        xlabels(ax)
    fig.text(0.01, 0.005, "Grey dots: individual seeds (10 per precision). Blue: mean ± std.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("plots/metrics_vs_bits.png", dpi=200)
    plt.close(fig)


def mass_after_cut():
    npz = np.load("results/mass_after_cut.npz")
    bins = npz["bins"]
    centres = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, b in enumerate(["float", "4", "3"]):
        hs = np.array([npz[k] / npz[k].sum() for k in npz.files if k.startswith(f"{b}_s")])
        m, s = hs.mean(0), hs.std(0)
        name = "float" if b == "float" else f"{b}-bit"
        ax.stairs(m, bins, color=SERIES[i], lw=1.5, label=f"{name} (n={len(hs)} seeds)")
        ax.fill_between(centres, m - s, m + s, step="mid", color=SERIES[i], alpha=0.12, lw=0)
    ax.set_xlabel("Groomed jet mass, mass_mmdt (GeV)")
    ax.set_ylabel("Fraction of QCD jets passing cut")
    ax.set_title("QCD jet mass after the top-tag cut")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.text(0.01, 0.005, "Cut keeps 10% of QCD (g, q) jets. Line: mean over seeds; band: ± std.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("plots/mass_after_cut.png", dpi=200)
    plt.close(fig)


def main():
    os.makedirs("plots", exist_ok=True)
    df = load()
    summary(df)
    relative_change(df)
    metrics_vs_bits(df)
    mass_after_cut()
    print("wrote plots/*.png and results/summary_by_bits.csv")


if __name__ == "__main__":
    main()
