"""
Does quantisation for FPGA deployment break reliability before it breaks accuracy?

Trains float and QKeras MLPs at several bit widths on the hls4ml jet-tagging
dataset and records, per (bits, seed):
  - accuracy, top-tag AUC
  - top-label expected calibration error (ECE)
  - background mass sculpting: chi2/dof and Jensen-Shannon distance between the
    QCD jet-mass distribution before and after a top-tag score cut
Jet mass (j_mass_mmdt) is held out of the inputs and used only as the protected
variable.

Run:  python sweep.py            -> results/sweep.csv, models/*.h5, results/mass_after_cut.npz
      python sweep.py --resculpt -> recompute vs-float sculpting columns from the saved histograms
Saved models are reused (resume); delete models/ to retrain from scratch.
"""
import itertools
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import jensenshannon
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Activation, Dense, Input, ReLU
from tensorflow.keras.models import Model
from qkeras import QActivation, QDense, quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects

BITS = [None, 16, 8, 6, 4, 3]       # None = float baseline
SEEDS = list(range(10))
CLASSES = ["g", "q", "t", "w", "z"]
TOP = CLASSES.index("t")
QCD = [CLASSES.index("g"), CLASSES.index("q")]
PROTECTED = "mass_mmdt"             # OpenML name for j_mass_mmdt
BKG_EFF = 0.10                      # keep 10% of QCD after the score cut
MASS_BINS = np.linspace(0, 250, 31)


def load_data(seed=42):
    df = fetch_openml("hls4ml_lhc_jets_hlf", as_frame=True).frame
    y = pd.Categorical(df.pop("class"), categories=CLASSES).codes
    mass = df[PROTECTED].to_numpy()
    X = df.drop(columns=[PROTECTED]).to_numpy(np.float32)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X[tr])
    return scaler.transform(X[tr]), scaler.transform(X[te]), y[tr], y[te], mass[te]


def build(n_in, bits):
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
        x = Dense(len(CLASSES), name="out")(x)
    else:
        q = quantized_bits(bits, 0, alpha=1)
        x = QDense(len(CLASSES), kernel_quantizer=q, bias_quantizer=q, name="out")(x)
    return Model(x_in, Activation("softmax")(x))


def ece(probs, y, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def sculpting(score, y, mass):
    """Absolute sculpting: QCD mass before vs after the cut. Returns (chi2/dof, JSD, after-cut counts)."""
    bkg = np.isin(y, QCD)
    cut = np.quantile(score[bkg], 1 - BKG_EFF)
    before, _ = np.histogram(mass[bkg], MASS_BINS)
    after, _ = np.histogram(mass[bkg & (score > cut)], MASS_BINS)
    p, q = before / before.sum(), after / after.sum()
    expected = p * after.sum()
    ok = expected > 0
    chi2 = ((after[ok] - expected[ok]) ** 2 / expected[ok]).sum() / max(ok.sum() - 1, 1)
    return chi2, jensenshannon(p, q), after


def add_vs_float(rows, hists):
    """Quantisation-induced sculpting, from after-cut QCD mass shapes.
    sculpt_jsd_vs_float:       JSD vs the float model of the same seed (paired; NaN for float rows).
    sculpt_jsd_vs_float_xseed: mean JSD vs the float models of all *other* seeds. Every
                               bits value, float included, is measured the same way, so
                               the float row is a like-for-like noise floor."""
    for r in rows:
        h, s = hists[(str(r["bits"]), r["seed"])], r["seed"]
        r["sculpt_jsd_vs_float"] = (np.nan if str(r["bits"]) == "float"
                                    else jensenshannon(h, hists[("float", s)]))
        r["sculpt_jsd_vs_float_xseed"] = np.mean(
            [jensenshannon(h, hists[("float", o)]) for o in SEEDS if o != s])


def resculpt():
    """Recompute the vs-float columns of results/sweep.csv from results/mass_after_cut.npz."""
    df = pd.read_csv("results/sweep.csv", dtype={"bits": str})
    npz = np.load("results/mass_after_cut.npz")
    hists = {tuple(k.rsplit("_s", 1)): npz[k] for k in npz.files if k != "bins"}
    hists = {(b, int(s)): h for (b, s), h in hists.items()}
    rows = df.to_dict("records")
    add_vs_float(rows, hists)
    df = pd.DataFrame(rows)
    df.to_csv("results/sweep.csv", index=False)
    print(df.drop(columns="seed").groupby("bits").agg(["mean", "std"]).round(4))


def load_saved(path):
    """Saved model, or None if missing/unreadable (e.g. killed mid-save). Delete models/ to retrain all."""
    if not os.path.exists(path):
        return None
    co = {}
    _add_supported_quantized_objects(co)
    try:
        return tf.keras.models.load_model(path, custom_objects=co, compile=False)
    except Exception as e:  # noqa: BLE001
        print(f"retraining {path}: could not load ({e})")
        return None


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    X_tr, X_te, y_tr, y_te, mass_te = load_data()
    rows, hists = [], {}
    for bits, seed in itertools.product(BITS, SEEDS):
        path = f"models/mlp_{'float' if bits is None else bits}b_s{seed}.h5"
        model = load_saved(path)            # resume: reuse models from an interrupted run
        if model is None:
            tf.keras.utils.set_random_seed(seed)
            model = build(X_tr.shape[1], bits)
            model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            model.fit(X_tr, y_tr, batch_size=1024, epochs=30, validation_split=0.2, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
            model.save(path)
        probs = model.predict(X_te, batch_size=4096, verbose=0)
        top_score = probs[:, TOP]
        chi2, jsd, after = sculpting(top_score, y_te, mass_te)
        rows.append(dict(
            bits="float" if bits is None else bits, seed=seed,
            accuracy=(probs.argmax(1) == y_te).mean(),
            auc_top=roc_auc_score(y_te == TOP, top_score),
            ece=ece(probs, y_te),
            sculpt_chi2_dof=chi2, sculpt_jsd=jsd,
        ))
        hists[(rows[-1]["bits"], seed)] = after
        print(rows[-1])
    add_vs_float(rows, hists)
    np.savez("results/mass_after_cut.npz", bins=MASS_BINS,
             **{f"{b}_s{s}": h for (b, s), h in hists.items()})
    df = pd.DataFrame(rows)
    df.to_csv("results/sweep.csv", index=False)
    print(df.groupby("bits").agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    import sys
    resculpt() if "--resculpt" in sys.argv else main()
