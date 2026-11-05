"""
Post-hoc temperature calibration of the 60 sweep models (6 precisions x 10 seeds).

Temperature T is fitted per model by minimising NLL on the Keras validation split: the last 20% of the
training arrays, exactly what model.fit(validation_split=0.2) held out in sweep.py (and also used for early
stopping). The test set is only used for evaluation.

Variants (logits z = pre-softmax output of the `out` layer):
  before  z
  cont    z / T                                  (needs a multiplier in hardware)
  pow2    z * 2^s, s = -round(log2 T)             folded into the output layer as a binary-point shift:
                                                  weights/biases x 2^s with quantized_bits(b, integer=s);
                                                  same bits, exact, free in hardware
  folded  output weights/biases / T, re-quantised to the same bit width with the smallest integer part
          that avoids clipping (float models: exact, so folded == cont in Keras)

Stages:
  python calibrate.py --keras     Keras logits, T fits, variants, test metrics; exports seed-0 variant models
  bash run_calib_hls.sh           (WSL) hls4ml csim of seed-0 before/pow2/folded -> cache/calib/csim/
  python calibrate.py --report    adds csim rows, paired tests, bootstrap, results/calib*.{csv,md}, figures
"""
import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import jensenshannon
from scipy.special import log_softmax, softmax
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

BITS = ["float", "16", "8", "6", "4", "3"]
SEEDS = list(range(10))
VARIANTS = ["before", "cont", "pow2", "folded"]
HW_VARIANTS = ["before", "pow2", "folded"]          # what can be built without a multiplier
VAL_SPLIT = 0.2
TOP = 2
CACHE = os.path.join("cache", "calib")
N_BOOT = 10_000


# ---------------------------------------------------------------- metrics
def nll(p_or_z, y, logits=False):
    lp = log_softmax(p_or_z.astype(np.float64), axis=1) if logits else np.log(np.clip(p_or_z, 1e-12, 1.0))
    return float(-lp[np.arange(len(y)), y].mean())


def brier(p, y):
    onehot = np.zeros_like(p, dtype=np.float64)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(1).mean())


def test_metrics(p, y, mass, normalised=True):
    from sweep import ece, sculpting
    chi2, jsd, after = sculpting(p[:, TOP], y, mass)
    return dict(accuracy=float((p.argmax(1) == y).mean()),
                auc_top=float(roc_auc_score(y == TOP, p[:, TOP])),
                ece=float(ece(p, y)),
                nll=nll(p, y) if normalised else np.nan,
                brier=brier(p, y) if normalised else np.nan,
                sculpt_chi2_dof=float(chi2), sculpt_jsd=float(jsd)), after


def bin_stats(p, y, n_bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf > lo) & (conf <= hi)
        rows.append(dict(bin=b, n=int(m.sum()), conf=float(conf[m].mean()) if m.any() else np.nan,
                         acc=float((pred[m] == y[m]).mean()) if m.any() else np.nan))
    return rows


def fit_temperature(z, y):
    res = minimize_scalar(lambda t: nll(z / math.exp(t), y, logits=True), bounds=(-4.0, 4.0),
                          method="bounded", options={"xatol": 1e-7})
    if abs(res.x) > 3.99:
        raise RuntimeError(f"temperature fit hit the search bound (log T = {res.x:.3f})")
    return math.exp(res.x)


# ---------------------------------------------------------------- models
def custom_objects():
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)
    return co


def build_variant(n_in, bits, out_integer=0):
    """sweep.build(), except the output layer's quantizer integer part is a parameter."""
    from qkeras import QActivation, QDense, quantized_bits, quantized_relu
    from tensorflow.keras.layers import Activation, Dense, Input, ReLU
    from tensorflow.keras.models import Model
    x_in = Input(shape=(n_in,))
    x = x_in
    for i, width in enumerate([32, 32]):
        if bits is None:
            x = ReLU()(Dense(width, name=f"fc{i}")(x))
        else:
            q = quantized_bits(bits, 0, alpha=1)
            x = QDense(width, kernel_quantizer=q, bias_quantizer=q, name=f"fc{i}")(x)
            x = QActivation(quantized_relu(bits), name=f"relu{i}")(x)
    if bits is None:
        x = Dense(5, name="out")(x)
    else:
        q = quantized_bits(bits, out_integer, alpha=1)
        x = QDense(5, kernel_quantizer=q, bias_quantizer=q, name="out")(x)
    return Model(x_in, Activation("softmax")(x))


def quantised_out_weights(model, nbits):
    layer = model.get_layer("out")
    if nbits is None:
        return [w.astype(np.float64) for w in layer.get_weights()]
    kq, bq = layer.get_quantizers()
    return [np.asarray(kq(layer.kernel)).astype(np.float64), np.asarray(bq(layer.bias)).astype(np.float64)]


def smallest_integer(vals, nbits):
    """Smallest integer part I so quantized_bits(nbits, I) holds vals without clipping (step 2^(I-nbits+1))."""
    m = 2 ** (nbits - 1)
    for i in range(-8, 16):
        if vals.max() <= 2.0 ** i * (m - 1) / m and vals.min() >= -(2.0 ** i):
            return i
    raise RuntimeError("no integer part fits")


def variant_model(orig, n_in, nbits, out_integer, out_w):
    m = build_variant(n_in, nbits, out_integer)
    for name in ("fc0", "fc1"):
        m.get_layer(name).set_weights(orig.get_layer(name).get_weights())
    m.get_layer("out").set_weights([w.astype(np.float32) for w in out_w])
    return m


def logits_of(model, X):
    import tensorflow as tf
    sub = tf.keras.Model(model.input, model.get_layer("out").output)
    return sub.predict(X, batch_size=8192, verbose=0).astype(np.float64)


# ---------------------------------------------------------------- stage 1: Keras
def run_keras():
    import tensorflow as tf
    from sweep import load_data

    X_tr, X_te, y_tr, y_te, mass_te = load_data()
    split_at = int(math.floor(len(X_tr) * (1.0 - VAL_SPLIT)))      # = Keras validation_split
    X_val, y_val = X_tr[split_at:], y_tr[split_at:]
    print(f"validation slice: {len(y_val)} events (X_tr[{split_at}:]), test: {len(y_te)}")
    sw = pd.read_csv("results/sweep.csv", dtype={"bits": str}).set_index(["bits", "seed"])
    os.makedirs(os.path.join(CACHE, "models"), exist_ok=True)
    os.makedirs(os.path.join(CACHE, "keras_seed0"), exist_ok=True)
    co = custom_objects()
    n_in = X_te.shape[1]
    rows, hists, rel = [], {}, []

    for bits in BITS:
        nbits = None if bits == "float" else int(bits)
        for seed in SEEDS:
            orig = tf.keras.models.load_model(f"models/mlp_{bits}b_s{seed}.h5", custom_objects=co, compile=False)
            zv, zt = logits_of(orig, X_val), logits_of(orig, X_te)

            # check: logits -> softmax reproduces the ECE in sweep.csv
            d_ece = abs(test_metrics(softmax(zt, axis=1), y_te, mass_te)[0]["ece"] - sw.loc[(bits, seed), "ece"])
            assert d_ece < 1e-4, f"{bits} s{seed}: ECE from logits differs from sweep.csv by {d_ece:.2e}"

            T = fit_temperature(zv, y_val)
            s = -int(round(math.log2(T)))                      # pow2: multiply logits by 2^s
            Wq = quantised_out_weights(orig, nbits)

            # check: rebuilding with the original latent weights reproduces the logits exactly
            same = variant_model(orig, n_in, nbits, 0, orig.get_layer("out").get_weights())
            assert np.array_equal(logits_of(same, X_te[:20000]), zt[:20000]), f"{bits} s{seed}: rebuild differs"

            # pow2: weights x 2^s, binary point shifted by s (integer part 0 -> s), same bit width
            m_pow2 = variant_model(orig, n_in, nbits, s, [w * 2.0 ** s for w in Wq])
            zv_p, zt_p = logits_of(m_pow2, X_val), logits_of(m_pow2, X_te)
            assert np.allclose(zt_p, zt * 2.0 ** s, rtol=1e-6, atol=1e-9), f"{bits} s{seed}: pow2 fold not exact"

            # folded: weights / T re-quantised; integer part widened just enough to avoid clipping
            Wf = [w / T for w in Wq]
            I_f = None if nbits is None else smallest_integer(np.concatenate([w.ravel() for w in Wf]), nbits)
            m_fold = variant_model(orig, n_in, nbits, 0 if I_f is None else I_f, Wf)
            zv_f, zt_f = logits_of(m_fold, X_val), logits_of(m_fold, X_te)

            logit_sets = {"before": (zv, zt, 1.0), "cont": (zv / T, zt / T, T),
                          "pow2": (zv_p, zt_p, 2.0 ** -s), "folded": (zv_f, zt_f, T)}
            pred0 = zt.argmax(1)
            for variant, (lv, lt, t_used) in logit_sets.items():
                p = softmax(lt, axis=1)
                changed = float((p.argmax(1) != pred0).mean())
                if variant in ("cont", "pow2"):
                    assert changed == 0.0, f"{bits} s{seed} {variant}: argmax changed for {changed:.2e} of events"
                met, after = test_metrics(p, y_te, mass_te)
                rows.append(dict(source="keras", variant=variant, bits=bits, seed=seed, T_fit=T, T_used=t_used,
                                 pow2_shift=s, fold_integer_bits=I_f, val_nll=nll(lv, y_val, logits=True),
                                 pred_changed_frac=changed, **met))
                hists[("keras", variant, bits, seed)] = after
                if variant in ("before", "cont"):
                    rel += [dict(variant=variant, bits=bits, seed=seed, **r) for r in bin_stats(p, y_te)]
                if seed == 0 and variant in HW_VARIANTS:
                    np.save(os.path.join(CACHE, "keras_seed0", f"{bits}_{variant}_test_logits.npy"), lt)

            if seed == 0:                                     # models for the hls4ml csim stage
                m_pow2.save(os.path.join(CACHE, "models", f"mlp_{bits}b_s0_pow2.h5"))
                m_fold.save(os.path.join(CACHE, "models", f"mlp_{bits}b_s0_folded.h5"))
            print(f"{bits:>5} s{seed}: T={T:.4f} pow2 shift={s:+d} fold int={I_f} "
                  f"ECE {rows[-4]['ece']:.4f} -> cont {rows[-3]['ece']:.4f} pow2 {rows[-2]['ece']:.4f} "
                  f"folded {rows[-1]['ece']:.4f} (folded pred changed {rows[-1]['pred_changed_frac']:.1e})",
                  flush=True)
            tf.keras.backend.clear_session()

    pd.DataFrame(rows).to_csv(os.path.join(CACHE, "keras_rows.csv"), index=False)
    pd.DataFrame(rel).to_csv("results/calib_reliability_bins.csv", index=False)
    np.savez(os.path.join(CACHE, "hists.npz"), **{"|".join(map(str, k)): v for k, v in hists.items()})
    print("stage 1 done")


# ---------------------------------------------------------------- stage 3: report
def holm(ps):
    order = np.argsort(ps)
    out, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def wilcox(d):
    d = np.asarray(d, dtype=float)
    return 1.0 if not (d != 0).any() else float(wilcoxon(d).pvalue)


def boot_ci(d, rng):
    d = np.asarray(d, dtype=float)
    means = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def add_xseed(df, hists):
    """Cross-seed sculpting per (source, variant): vs the float models of the other seeds, same variant."""
    vals = []
    for _, r in df.iterrows():
        refs = [hists.get((r.source, r.variant, "float", o)) for o in SEEDS if o != r.seed]
        refs = [h for h in refs if h is not None]
        h = hists.get((r.source, r.variant, r.bits, r.seed))
        vals.append(np.mean([jensenshannon(h, f) for f in refs]) if (h is not None and refs) else np.nan)
    df["sculpt_jsd_vs_float_xseed"] = vals
    return df


def csim_rows(y_te, mass_te):
    rows, approx = [], []
    for f in sorted(glob.glob(os.path.join(CACHE, "csim", "*_s0_*.npy"))):
        bits, _, variant = os.path.basename(f)[:-4].split("_")
        raw = np.load(f).astype(np.float64)
        ren = raw / raw.sum(1, keepdims=True)
        for mode, p in (("raw", raw), ("renorm", ren)):
            met, _ = test_metrics(p, y_te, mass_te, normalised=(mode == "renorm"))
            rows.append(dict(source=f"hls_{mode}", variant=variant, bits=bits, seed=0, **met))
        kl = os.path.join(CACHE, "keras_seed0", f"{bits}_{variant}_test_logits.npy")
        if os.path.exists(kl):
            pk = softmax(np.load(kl), axis=1)
            approx.append(dict(bits=bits, variant=variant, out_sum_mean=float(raw.sum(1).mean()),
                               out_sum_min=float(raw.sum(1).min()), out_sum_max=float(raw.sum(1).max()),
                               argmax_agree_keras=float((raw.argmax(1) == pk.argmax(1)).mean()),
                               maxprob_absdiff_mean=float(np.abs(ren.max(1) - pk.max(1)).mean()),
                               frac_maxprob_diff_gt_0p05=float((np.abs(ren.max(1) - pk.max(1)) > 0.05).mean())))
    approx = pd.DataFrame(approx)
    if len(approx):
        approx = approx.sort_values(["bits", "variant"], key=lambda c: c.map(
            {**{b: i for i, b in enumerate(BITS)}, **{v: i for i, v in enumerate(HW_VARIANTS)}})).reset_index(drop=True)
    return pd.DataFrame(rows), approx


def run_report():
    from sweep import load_data
    _, _, _, y_te, mass_te = load_data()
    kdf = pd.read_csv(os.path.join(CACHE, "keras_rows.csv"), dtype={"bits": str})
    npz = np.load(os.path.join(CACHE, "hists.npz"))
    hists = {(k.split("|")[0], k.split("|")[1], k.split("|")[2], int(k.split("|")[3])): npz[k] for k in npz.files}
    kdf = add_xseed(kdf, hists)
    cdf, approx = csim_rows(y_te, mass_te)
    if len(cdf):
        cdf["sculpt_jsd_vs_float_xseed"] = np.nan                     # only seed 0 exists in csim
    df = pd.concat([kdf, cdf], ignore_index=True)
    df.to_csv("results/calib.csv", index=False)

    # paired tests on Keras rows, Holm within each family
    k = kdf.set_index(["variant", "bits", "seed"])
    tests, rng = [], np.random.default_rng(0)
    for variant in ["cont", "pow2", "folded"]:
        for metric in ["ece", "nll", "brier"]:                       # before vs after
            fam = []
            for b in BITS:
                d = k.loc[(variant, b), metric].values - k.loc[("before", b), metric].values
                lo, hi = boot_ci(d, rng) if metric == "ece" else (np.nan, np.nan)
                fam.append(dict(test="after_vs_before", variant=variant, metric=metric, bits=b, n=len(d),
                                mean_delta=d.mean(), median_delta=np.median(d), ci95_lo=lo, ci95_hi=hi,
                                p=wilcox(d)))
            for r, ph in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"] = ph
            tests += fam
        for test in ["quant_vs_float_after", "gap_change"]:            # does the gap to float close?
            fam = []
            for b in BITS[1:]:
                gap_after = k.loc[(variant, b), "ece"].values - k.loc[(variant, "float"), "ece"].values
                gap_before = k.loc[("before", b), "ece"].values - k.loc[("before", "float"), "ece"].values
                d = gap_after if test == "quant_vs_float_after" else gap_after - gap_before
                lo, hi = boot_ci(d, rng)
                fam.append(dict(test=test, variant=variant, metric="ece", bits=b, n=len(d), mean_delta=d.mean(),
                                median_delta=np.median(d), ci95_lo=lo, ci95_hi=hi, p=wilcox(d)))
            for r, ph in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"] = ph
            tests += fam
    tdf = pd.DataFrame(tests)
    tdf.to_csv("results/calib_tests.csv", index=False)
    write_summary(kdf, cdf, approx, tdf)
    plot_ece(kdf)
    plot_reliability()
    print("wrote results/calib.csv, calib_tests.csv, calib_summary.md, plots/calib_*.png")


def write_summary(kdf, cdf, approx, tdf):
    def pm(s):
        return f"{s.mean():.4f} ± {s.std():.4f}"

    L = ["# Post-hoc temperature calibration", "",
         "T fitted per model by NLL on the Keras validation split (last 20% of the training arrays; the same "
         "split was used for early stopping). Test set n = 166,000. Mean ± std over 10 seeds unless noted.", ""]
    b = kdf[kdf.variant == "before"].groupby("bits")
    L += ["## Fitted temperatures", "", "| bits | T (mean ± std) | pow2 shift s (logits × 2^s) | folded integer bits |",
          "|---|---|---|---|"]
    for bits in BITS:
        g = kdf[(kdf.variant == "before") & (kdf.bits == bits)]
        shifts = ", ".join(f"{v:+d}×{n}" for v, n in g.pow2_shift.astype(int).value_counts().sort_index().items())
        fib = g.fold_integer_bits.dropna().astype(int)
        fi = ", ".join(f"{v}×{n}" for v, n in fib.value_counts().sort_index().items()) if len(fib) else "– (float)"
        L.append(f"| {bits} | {pm(g.T_fit)} | {shifts} | {fi} |")
    for metric, fmt in [("ece", "ECE"), ("nll", "NLL"), ("brier", "Brier"), ("accuracy", "accuracy"),
                        ("auc_top", "top-tag AUC"), ("sculpt_jsd_vs_float_xseed", "sculpting JSD vs float (x-seed)")]:
        L += ["", f"## {fmt} (Keras)", "", "| bits | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
        for bits in BITS:
            L.append(f"| {bits} | " + " | ".join(pm(kdf[(kdf.variant == v) & (kdf.bits == bits)][metric])
                                                 for v in VARIANTS) + " |")
    fold = kdf[kdf.variant == "folded"].groupby("bits").pred_changed_frac.agg(["mean", "max"])
    L += ["", "Folded variant, fraction of test predictions changed by re-quantisation: " +
          ", ".join(f"{b}: mean {fold.loc[b, 'mean']:.1e} / max {fold.loc[b, 'max']:.1e}" for b in BITS) +
          ". cont and pow2: 0 by construction (asserted)."]
    L += ["", "## Paired tests (Wilcoxon signed-rank over 10 seeds, Holm within family; 95% CI = seed bootstrap of "
          "the mean, 10,000 resamples)", "",
          "| test | variant | metric | bits | mean Δ | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|---|---|"]
    for _, r in tdf.iterrows():
        ci = f"[{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}]" if not np.isnan(r.ci95_lo) else "–"
        L.append(f"| {r.test} | {r.variant} | {r.metric} | {r.bits} | {r.mean_delta:+.4f} | {ci} | {r.p:.3g} | "
                 f"{r.p_holm:.3g} |")
    if len(cdf):
        L += ["", "## hls4ml fixed point (csim), seed 0", "",
              "raw = hardware softmax outputs as produced (LUT approximation, rows need not sum to 1); "
              "renorm = raw / row sum. Float = post-training ap_fixed<16,6> (ptq16). Continuous T needs a "
              "multiplier and is not built.", "",
              "| bits | variant | ECE raw | ECE renorm | accuracy | NLL renorm | Keras ECE (seed 0) |",
              "|---|---|---|---|---|---|---|"]
        for bits in BITS:
            for v in HW_VARIANTS:
                r_raw = cdf[(cdf.source == "hls_raw") & (cdf.bits == bits) & (cdf.variant == v)]
                r_ren = cdf[(cdf.source == "hls_renorm") & (cdf.bits == bits) & (cdf.variant == v)]
                kk = kdf[(kdf.bits == bits) & (kdf.variant == v) & (kdf.seed == 0)]
                if len(r_raw):
                    L.append(f"| {bits} | {v} | {r_raw.ece.iloc[0]:.4f} | {r_ren.ece.iloc[0]:.4f} | "
                             f"{r_raw.accuracy.iloc[0]:.4f} | {r_ren.nll.iloc[0]:.4f} | {kk.ece.iloc[0]:.4f} |")
        if len(approx):
            L += ["", "Softmax LUT approximation check (vs exact softmax of the Keras logits, same variant):", "",
                  "| bits | variant | output sum mean [min, max] | argmax agrees with Keras | mean abs Δ max-prob "
                  "(renorm) | events with Δ > 0.05 |", "|---|---|---|---|---|---|"]
            for _, a in approx.iterrows():
                L.append(f"| {a.bits} | {a.variant} | {a.out_sum_mean:.3f} [{a.out_sum_min:.3f}, {a.out_sum_max:.3f}]"
                         f" | {a.argmax_agree_keras:.4f} | {a.maxprob_absdiff_mean:.4f} | "
                         f"{a.frac_maxprob_diff_gt_0p05:.4f} |")
    with open("results/calib_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def plot_ece(kdf):
    import matplotlib.pyplot as plt
    from plots import ORDER, INK2, MUTED, SERIES, xlabels
    colors = SERIES + ["#eda100"]                          # categorical slots 1-4 (validated palette)
    markers = ["o", "s", "^", "D"]
    names = ["before", "continuous T", "power-of-two T (= before: T rounds to 1)", "T folded + re-quantised"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(ORDER))
    ends = []
    for i, v in enumerate(VARIANTS):
        g = kdf[kdf.variant == v].groupby("bits").ece
        m, s = g.mean().reindex(ORDER), g.std().reindex(ORDER)
        ax.errorbar(x + (i - 1.5) * 0.08, m, yerr=s, color=colors[i], marker=markers[i], ms=5, capsize=0,
                    elinewidth=1, label=names[i], zorder=3)
        ends.append([m.iloc[-1], names[i]])
    ends.sort()
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for j in range(1, len(ends)):                           # keep end labels >= 4% of the axis apart
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.04 * span)
    for yv, name in ends:
        ax.annotate(name, (x[-1] + 0.2, yv), va="center", color=INK2, fontsize=8.5, annotation_clip=False)
    xlabels(ax)
    ax.set_xlim(-0.4, len(ORDER) + 1.4)
    ax.set_ylabel("Expected calibration error (lower is better)")
    ax.set_title("Temperature scaling on the validation split, evaluated on test")
    ax.legend(loc="upper left")
    fig.text(0.01, 0.005, "Mean ± std over 10 seeds; Keras models; 15-bin ECE; test n = 166,000.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("plots/calib_ece_vs_bits.png", dpi=200)
    plt.close(fig)


def plot_reliability(min_count=200):
    import matplotlib.pyplot as plt
    from plots import AXIS, MARKERS, MUTED, SERIES
    df = pd.read_csv("results/calib_reliability_bins.csv", dtype={"bits": str})
    df = df[df.n >= min_count].assign(gap=lambda t: t.acc - t.conf)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    for ax, bits in zip(axes, ["float", "4", "3"]):
        for i, (v, name) in enumerate([("before", "before"), ("cont", "after (continuous T)")]):
            g = df[(df.bits == bits) & (df.variant == v)].groupby("bin")
            keep = g.size()[g.size() == len(SEEDS)].index
            conf, m, s = g.conf.mean().loc[keep], g.gap.mean().loc[keep], g.gap.std().loc[keep]
            ax.fill_between(conf, m - s, m + s, color=SERIES[i], alpha=0.12, lw=0)
            ax.plot(conf, m, color=SERIES[i], marker=MARKERS[i], ms=4, label=name)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title("float" if bits == "float" else f"{bits}-bit", fontsize=10)
        ax.set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy − confidence")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.text(0.01, 0.01, f"Mean over 10 seeds, band ± std; bins with < {min_count} events in any seed omitted. "
             "Above 0: under-confident.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("plots/calib_reliability.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", action="store_true", help="stage 1: Keras logits, T fits, variants")
    ap.add_argument("--report", action="store_true", help="stage 3: csim rows, tests, summary, figures")
    args = ap.parse_args()
    if not (args.keras or args.report):
        ap.error("choose --keras and/or --report")
    if args.keras:
        run_keras()
    if args.report:
        run_report()


if __name__ == "__main__":
    main()
