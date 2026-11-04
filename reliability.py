"""
Reliability diagrams from the saved sweep models (needs TensorFlow; ~1.5 GB RAM, one model at a time).

  results/reliability_bins.csv   per (bits, seed, bin): mean confidence, accuracy, count
  plots/reliability.png          accuracy - confidence per confidence bin, float vs 4-bit vs 3-bit
                                 (mean over 10 seeds, band = +/- std across seeds)
Also re-computes ECE from the bins and checks it against results/sweep.csv (same 15 equal-width bins).

Run:  python reliability.py            (add --plot-only to redraw from the CSV without TensorFlow)
"""
import argparse
import os

import numpy as np
import pandas as pd

N_BINS = 15
EDGES = np.linspace(0, 1, N_BINS + 1)
BITS = ["float", "16", "8", "6", "4", "3"]
SHOW = ["float", "4", "3"]
MIN_COUNT = 200          # per-seed bin count below which a bin is not plotted


def bin_stats(probs, y):
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for b, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        m = (conf > lo) & (conf <= hi)
        rows.append(dict(bin=b, lo=lo, hi=hi, n=int(m.sum()),
                         conf=float(conf[m].mean()) if m.any() else np.nan,
                         acc=float((pred[m] == y[m]).mean()) if m.any() else np.nan))
    return rows


def compute():
    import tensorflow as tf
    from qkeras.utils import _add_supported_quantized_objects
    from sweep import SEEDS, load_data

    _, X_te, _, y_te, _ = load_data()
    co = {}
    _add_supported_quantized_objects(co)
    rows = []
    for bits in BITS:
        for seed in SEEDS:
            model = tf.keras.models.load_model(f"models/mlp_{bits}b_s{seed}.h5", custom_objects=co, compile=False)
            probs = model.predict(X_te, batch_size=8192, verbose=0)
            rows += [dict(bits=bits, seed=seed, **r) for r in bin_stats(probs, y_te)]
            tf.keras.backend.clear_session()
        print(f"{bits}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/reliability_bins.csv", index=False)
    return df


def check_ece(df):
    ece = (df.assign(w=df.n * (df.acc - df.conf).abs()).groupby(["bits", "seed"])
             .apply(lambda g: g.w.sum() / g.n.sum(), include_groups=False).rename("ece_bins").reset_index())
    sw = pd.read_csv("results/sweep.csv", dtype={"bits": str})[["bits", "seed", "ece"]]
    j = ece.merge(sw, on=["bits", "seed"])
    print(f"ECE from bins vs sweep.csv: max |diff| = {(j.ece_bins - j.ece).abs().max():.2e} over {len(j)} models")


def plot(df):
    import matplotlib.pyplot as plt
    from plots import AXIS, INK2, MARKERS, MUTED, SERIES   # also applies the shared rcParams

    fig, ax = plt.subplots(figsize=(7, 4.2))
    d = df[df.n >= MIN_COUNT].assign(gap=lambda t: t.acc - t.conf)
    for i, b in enumerate(SHOW):
        g = d[d.bits == b].groupby("bin")
        keep = g.size()[g.size() == d[d.bits == b].seed.nunique()].index   # bins present for every seed
        conf = g.conf.mean().loc[keep]
        m, s = g.gap.mean().loc[keep], g.gap.std().loc[keep]
        name = "float" if b == "float" else f"{b}-bit"
        ax.fill_between(conf, m - s, m + s, color=SERIES[i], alpha=0.12, lw=0)
        ax.plot(conf, m, color=SERIES[i], marker=MARKERS[i], ms=5, label=name)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Predicted confidence (max softmax probability)")
    ax.set_ylabel("Accuracy − confidence")
    ax.set_title("Calibration gap by confidence bin")
    ax.legend(loc="lower left")
    ax.text(0.99, 0.97, "above 0: under-confident\nbelow 0: over-confident", transform=ax.transAxes,
            ha="right", va="top", color=INK2, fontsize=8.5)
    fig.text(0.01, 0.005, f"15 equal-width bins; mean over 10 seeds, band ± std; bins with < {MIN_COUNT} "
             "events in any seed omitted. Test set n = 166,000.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/reliability.png", dpi=200)
    plt.close(fig)
    print("wrote plots/reliability.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv("results/reliability_bins.csv", dtype={"bits": str}) if args.plot_only else compute()
    check_ece(df)
    plot(df)


if __name__ == "__main__":
    main()
