"""
Quantised MLP -> hls4ml -> Vitis HLS synthesis for the AMD Kria KR260 (xck26).
Produces real latency (cycles, ns) and resource numbers (LUT, FF, DSP, BRAM)
plus a CPU latency baseline.

Requirements:
  pip install hls4ml qkeras tensorflow scikit-learn
  Vitis HLS (2023.x or 2024.x) on PATH for the synthesis step.

Run:
  python kr260_hls4ml_mlp.py            # train, convert, C-sim check, synthesise
  python kr260_hls4ml_mlp.py --no-synth # skip Vitis HLS (no hardware numbers)
"""
import argparse
import time

import numpy as np
import hls4ml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from qkeras import QActivation, QDense, quantized_bits, quantized_relu

PART = "xck26-sfvc784-2LV-c"   # KR260 SoM
CLOCK_NS = 5                   # 200 MHz target
BITS = 8


def load_data():
    data = fetch_openml("hls4ml_lhc_jets_hlf", as_frame=False)
    X, y = data["data"], to_categorical(LabelEncoder().fit_transform(data["target"]), 5)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te


def build_model():
    q = quantized_bits(BITS, 0, alpha=1)
    x_in = Input(shape=(16,), name="input")
    x = QDense(32, kernel_quantizer=q, bias_quantizer=q, name="fc1")(x_in)
    x = QActivation(quantized_relu(BITS), name="relu1")(x)
    x = QDense(32, kernel_quantizer=q, bias_quantizer=q, name="fc2")(x)
    x = QActivation(quantized_relu(BITS), name="relu2")(x)
    x = QDense(5, kernel_quantizer=q, bias_quantizer=q, name="output")(x)
    out = Activation("softmax", name="softmax")(x)
    return Model(x_in, out)


def cpu_latency_us(model, n=2000):
    sample = np.random.randn(1, 16).astype(np.float32)
    for _ in range(50):
        model(sample, training=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model(sample, training=False)
    return (time.perf_counter() - t0) / n * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-synth", action="store_true")
    args = ap.parse_args()

    X_tr, X_te, y_tr, y_te = load_data()
    model = build_model()
    model.compile(Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, batch_size=1024, epochs=30, validation_split=0.2, verbose=2)
    keras_acc = np.mean(model.predict(X_te).argmax(1) == y_te.argmax(1))

    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
    cfg["Model"]["Strategy"] = "Latency"
    cfg["Model"]["ReuseFactor"] = 1
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir="hls_kr260_mlp",
        part=PART, clock_period=CLOCK_NS, io_type="io_parallel", backend="Vitis",
    )
    hls_model.compile()
    hls_acc = np.mean(hls_model.predict(np.ascontiguousarray(X_te)).argmax(1) == y_te.argmax(1))

    print(f"\nKeras accuracy: {keras_acc:.4f} | hls4ml fixed-point accuracy: {hls_acc:.4f}")
    print(f"CPU latency (this machine, batch 1): {cpu_latency_us(model):.1f} us")

    if not args.no_synth:
        hls_model.build(csim=False, synth=True)
        hls4ml.report.read_vivado_report("hls_kr260_mlp")  # prints latency + resources


if __name__ == "__main__":
    main()
