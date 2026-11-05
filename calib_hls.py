"""
Stage 2 of calibrate.py (run in WSL): hls4ml fixed-point (csim) outputs of the seed-0 models, before and after
the two hardware-free calibrations. Same hls4ml configuration as synth.py (the deployed hardware), so the float
model is post-training-quantised to ap_fixed<16,6> (ptq16).

  before  models/mlp_{bits}b_s0.h5
  pow2    cache/calib/models/mlp_{bits}b_s0_pow2.h5     (written by `python calibrate.py --keras`)
  folded  cache/calib/models/mlp_{bits}b_s0_folded.h5

Writes cache/calib/csim/{bits}_s0_{variant}.npy: raw hls4ml softmax outputs for every test event.
Run:  bash run_calib_hls.sh
"""
import os

import numpy as np
import tensorflow as tf

from synth import convert, load_model
from sweep import load_data
from calibrate import BITS, CACHE, HW_VARIANTS, custom_objects

BUILD = os.path.expanduser("~/qnn_build/calib_hls")     # WSL ext4: much faster than /mnt/c


def main():
    _, X_te, _, _, _ = load_data()
    X_te = np.ascontiguousarray(X_te, dtype=np.float32)
    out_dir = os.path.join(CACHE, "csim")
    os.makedirs(out_dir, exist_ok=True)
    co = custom_objects()
    for bits in BITS:
        for variant in HW_VARIANTS:
            dst = os.path.join(out_dir, f"{bits}_s0_{variant}.npy")
            if os.path.exists(dst):
                print(f"skip {bits} {variant} (exists)")
                continue
            if variant == "before":
                model = load_model(bits, 0)
            else:
                model = tf.keras.models.load_model(os.path.join(CACHE, "models", f"mlp_{bits}b_s0_{variant}.h5"),
                                                   custom_objects=co, compile=False)
            hls_model = convert(model, os.path.join(BUILD, f"mlp_{bits}b_s0_{variant}"))
            hls_model.compile()
            out = np.asarray(hls_model.predict(X_te)).reshape(len(X_te), -1).astype(np.float32)
            np.save(dst, out)
            keras_pred = model.predict(X_te, batch_size=8192, verbose=0).argmax(1)
            print(f"{bits:>5} {variant:>6}: out sum mean {out.sum(1).mean():.3f}, "
                  f"argmax agrees with Keras {np.mean(out.argmax(1) == keras_pred):.4f}", flush=True)
            tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
