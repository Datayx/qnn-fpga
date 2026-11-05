# Quantised neural networks on an FPGA

I trained a small jet-tagging network, quantised it down to 3 bits, and ran it on an AMD Kria KR260. The question: when you shrink a model for hardware, does it stop being trustworthy before it stops being accurate?

## Key results

- **Calibration breaks harder than accuracy.** At 3 bits, accuracy drops 1.5% but calibration error rises 2.4× (p_Holm = 0.03). Down to 6 bits, calibration doesn't change.
- **hls4ml's softmax hurts calibration more than quantisation does.** Its lookup tables make outputs sum to 1.08 instead of 1, which raises on-chip calibration error about 6×. Changing one fixed-point type fixes it, at no hardware cost.
- **It runs on the board.** The 4-bit model is bit-exact on all 166,000 test events and takes 75 ns on chip, 27× faster than the board's ARM CPU. End to end, fed one event at a time from the ARM, it's only 1.2× faster, because bus access dominates.

![Relative change vs float](plots/relative_change.png)

## Setup

- **Model:** MLP, 15 → 32 → 32 → 5, on the hls4ml jet-tagging dataset (gluon, quark, top, W, Z).
- **Precisions:** float, plus QKeras quantisation-aware training at 16, 8, 6, 4 and 3 bits. 10 seeds each, 60 models in total.
- **Metrics:** accuracy, top-tag AUC, calibration (ECE), and background mass sculpting. Jet mass is kept out of the inputs, and I measure how much a top-tag cut distorts the QCD mass distribution.
- **Hardware:** hls4ml → Vitis HLS → Vivado, KR260 (xck26) at 200 MHz.
- **Statistics:** paired Wilcoxon tests against same-seed float, Holm-corrected. With 10 seeds, the smallest possible p is 0.002.

## Results

Mean ± std over 10 seeds, 166,000 test events. Hardware numbers are for the seed-0 model, post-route where available.

| bits | accuracy | ECE | sculpting (JSD vs float) | latency | clock | LUT | DSP | fits? |
|---|---|---|---|---|---|---|---|---|
| float | 0.7282 ± 0.0003 | 0.0049 ± 0.0011 | 0.0233 ± 0.0040 | – | – | – | – | – |
| 16 | 0.7282 ± 0.0004 | 0.0047 ± 0.0018 | 0.0230 ± 0.0051 | 85 ns | 200 MHz | 82,268 (HLS est.) | 1,617 | no, 130% of DSPs |
| 8 | 0.7280 ± 0.0004 | 0.0047 ± 0.0022 | 0.0234 ± 0.0047 | 77 ns | 182 MHz | 35,429 | 280 | yes |
| 6 | 0.7274 ± 0.0004 | 0.0046 ± 0.0019 | 0.0244 ± 0.0052 | 71 ns | 198 MHz | 25,903 | 59 | yes |
| 4 | 0.7236 ± 0.0006 | 0.0092 ± 0.0063 | 0.0304 ± 0.0076 | 65 ns | 200 MHz | 16,114 | 5 | yes |
| 3 | 0.7175 ± 0.0022 | 0.0117 ± 0.0054 | 0.0406 ± 0.0141 | 60 ns | 200 MHz | 11,854 | 5 | yes |

What changes, and when:

- **Accuracy** moves first, but barely: −0.09 points at 6 bits, −0.45 at 4 and −1.05 at 3 (all p_Holm = 0.010).
- **Calibration** only moves at 4 bits and below. +88% at 4 bits is borderline (p_Holm = 0.055); +140% at 3 bits is significant (p_Holm = 0.029).
- **Sculpting:** +74% at 3 bits, only just significant (p_Holm = 0.049). Treat it as a trend.

So reliability doesn't break *first*, but when it breaks, it breaks much harder. At low precision it also becomes strongly seed-dependent.

The low-bit models aren't over-confident; they're *under*-confident at high scores. Above 0.8 confidence the 3-bit models are about 1.3 points more accurate than they claim (4-bit: 0.4).

![Accuracy vs FPGA cost](figures/pareto.png)
![Reliability diagrams](plots/reliability.png)
![QCD mass after cut](plots/mass_after_cut.png)

Full tables: [results/summary.md](results/summary.md), [results/paired_tests.csv](results/paired_tests.csv).

## Does temperature scaling fix it?

No. I fitted one temperature per model on the validation split. That split was also used for early stopping, so it isn't fully independent.

- **It helps a little,** but float improves too, so after calibration 4 and 3 bits are still worse than float (p_Holm = 0.020).
- **A power-of-two temperature, free in hardware, does nothing.** Every fitted T is between 0.93 and 1.03, so it rounds to 1.
- **Folding T into the weights and re-quantising makes it worse:** 3-bit ECE goes from 0.012 to 0.051.

A single temperature shifts the whole confidence curve. It can't fix miscalibration that depends on confidence. Details: [results/calib_summary.md](results/calib_summary.md).

![ECE before/after temperature scaling](plots/calib_ece_vs_bits.png)

## The hls4ml softmax problem

On hardware, the biggest calibration error isn't quantisation. It's the softmax.

**The cause.** hls4ml's stable softmax looks up 1/Σexp in a table indexed by the top 10 bits of `inv_inp_t`, which defaults to `fixed<18,8>`. That leaves only 2 fractional bits, so the lookup works in steps of 0.25 over a sum that's always between 1 and 5. The outputs sum to 1.08 on average (0.996–1.249), and every probability gets inflated.

**The effect, over 10 seeds:** raw hardware ECE is 0.057 at 4 bits and 0.058 for a float model quantised after training, against 0.009 and 0.005 in Keras. Dividing by the sum brings it back to Keras level (raw vs divided: p_Holm = 0.004).

**The fix.** Setting `inv_inp_t = ufixed<18,3,RND,SAT>` makes the sums 1.000 and restores Keras-level ECE over 10 seeds (0.0093 vs 0.0092, difference 95% CI [−0.0009, +0.0011]). Latency and resources don't change: 16,070 vs 16,114 LUTs post-route, timing met.

| softmax config (4-bit, seed 0) | output sum | ECE | cycles | post-route LUT |
|---|---|---|---|---|
| default | 1.085 [0.996, 1.249] | 0.0735 | 13 | 16,114 |
| table size 4096 | 1.021 | 0.0300 | 13 | – (3× the BRAM) |
| **`inv_inp_t = ufixed<18,3>`** | **1.000 [0.995, 1.006]** | **0.0105** | **13** | **16,070** |
| latency implementation | 1.009 | 0.0189 | 11 | – |
| skip softmax, compute on ARM | – | 0.0126 | 6 | – |

Which fix to use:

- **Change `inv_inp_t` and rebuild.** Free, and it works even when the data never leaves the FPGA, as in a trigger.
- **Renormalise on the ARM.** No rebuild needed, and it costs 19 ns per event on the A53.
- **Skip the softmax on chip.** Saves 35 ns on the FPGA, but the ARM then spends about 240 ns computing it.

**Watch out:** in hls4ml 1.3.0, setting softmax table types through the layer config is silently ignored. I applied the fix by editing the converted model before `compile()`, and checked it in the generated `defines.h`. Details: [softmax_study.py](softmax_study.py), [results/softmax_summary.md](results/softmax_summary.md).

**Upstream:** related softmax issues are [#1512](https://github.com/fastmachinelearning/hls4ml/issues/1512) (argmax flips) and PR [#1542](https://github.com/fastmachinelearning/hls4ml/pull/1542) (inferred table types, config fix; open). As of 2026-11-05 I couldn't find any report of the sums not being 1 or of the calibration effect.

## On the board

The 4-bit model (seed 0) keeps accuracy within 0.5 points of float using only 5 DSPs, so that's the one I deployed. It sits behind an AXI-Lite wrapper in a full KR260 design, with an on-chip cycle counter. See [DEPLOY.md](DEPLOY.md).

| | result |
|---|---|
| bit-exact vs hls4ml model | 166,000 / 166,000 events |
| board accuracy | 0.7237 (Keras 0.7241) |
| on-chip latency | 15 cycles = 75 ns, identical for every event |
| end to end, ARM → FPGA → ARM | median 1.65 µs, p99 1.84 µs |
| same network in C on the ARM (A53, 1.33 GHz) | median 2.01 µs, p99 2.35 µs |
| full design | timing met at 200 MHz; 15,479 LUT (13%), 5 DSP |

The 75 ns figure is the one that matters when data arrives in the fabric. When the ARM feeds one event at a time, about 11 register accesses dominate, and streaming (AXI-Stream + DMA) would be needed to close that gap. Raw output: [results/board.csv](results/board.csv).

## Reproduce

Python 3.10, `pip install -r requirements.txt` (TensorFlow 2.14, QKeras, hls4ml 1.3). Synthesis: Vivado/Vitis 2026.1 on Linux/WSL.

```bash
python sweep.py                # train 60 models (~2 h CPU)
python analyze.py              # stats and figures
python plots.py && python reliability.py
python calibrate.py --keras && bash run_calib_hls.sh && python calibrate.py --report
bash run_softmax.sh configs --synth
bash run_softmax.sh seeds --config A_default --bits 4 float
bash run_softmax.sh seeds --config C_invinp --bits 4
bash run_softmax.sh impl --configs A_default C_invinp
bash run_softmax.sh report
bash run_synth.sh && bash run_impl.sh   # HLS + place & route
bash deploy/run_axi.sh && bash deploy/run_bd.sh   # bitstream
python deploy/export_testvec.py && bash deploy/run_golden.sh
# on the board: deploy/board/README.md, then:
python deploy/parse_board.py results/board_*.txt
```

## Limitations

- One architecture, one dataset, one QKeras recipe.
- 10 seeds is enough for accuracy, borderline for calibration at 4 bits.
- Hardware numbers are for seed 0 only (HLS LUT estimates for seeds 0–2 differ by up to ~5%).
- HLS overestimates LUTs about 2.4×, so I quote post-route numbers.
- 16-bit doesn't fit at ReuseFactor 1, and didn't route at ReuseFactor 2 on a 14 GB laptop.
