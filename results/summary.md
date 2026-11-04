| bits | accuracy | top AUC | ECE | sculpting JSD vs float (cross-seed) | latency (ns) | clock (MHz) | LUT | DSP | fits KR260 | hw source |
|---|---|---|---|---|---|---|---|---|---|---|
| float | 0.7282 ± 0.0003 | 0.9415 ± 0.0002 | 0.0049 ± 0.0011 | 0.0233 ± 0.0040 | - | - | - | - | - | - |
| 16 | 0.7282 ± 0.0004 | 0.9414 ± 0.0002 | 0.0047 ± 0.0018 | 0.0230 ± 0.0051 | 85 | 200 | 82,268 | 1,617 | **no** | hls |
| 8 | 0.7280 ± 0.0004 | 0.9414 ± 0.0002 | 0.0047 ± 0.0022 | 0.0234 ± 0.0047 | 77 | 182 | 35,429 | 280 | yes | post-route |
| 6 | 0.7274 ± 0.0004 | 0.9409 ± 0.0003 | 0.0046 ± 0.0019 | 0.0244 ± 0.0052 | 71 | 198 | 25,903 | 59 | yes | post-route |
| 4 | 0.7236 ± 0.0006 | 0.9387 ± 0.0006 | 0.0092 ± 0.0063 | 0.0304 ± 0.0076 | 65 | 200 | 16,114 | 5 | yes | post-route |
| 3 | 0.7175 ± 0.0022 | 0.9351 ± 0.0008 | 0.0117 ± 0.0054 | 0.0406 ± 0.0141 | 60 | 200 | 11,854 | 5 | yes | post-route |

Other reuse factors (seed-level rows):

| bits | seed | reuse | II | latency (ns) | clock (MHz) | LUT | DSP | fits | hw source |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0 | 2 | 2 | 90 | 200 | 100,212 | 837 | yes | hls |

Post-route builds (seed 0): default HLS schedule vs deeper pipeline (45% clock uncertainty):

| bits | reuse | HLS clock uncertainty | cycles | Fmax (MHz) | latency (ns) | LUT | FF | DSP |
|---|---|---|---|---|---|---|---|---|
| 3 | 1 | 27% | 12 | 204.4 | 60.0 | 11,854 | 2,409 | 5 |
| 4 | 1 | 27% | 13 | 201.7 | 65.0 | 16,114 | 3,566 | 5 |
| 6 | 1 | 27% | 14 | 197.5 | 70.9 | 25,903 | 6,906 | 59 |
| 6 | 1 | 45% | 18 | 192.3 | 93.6 | 24,916 | 10,294 | 61 |
| 8 | 1 | 27% | 14 | 182.1 | 76.9 | 35,429 | 9,040 | 280 |
| 8 | 1 | 45% | 18 | 201.1 | 90.0 | 36,065 | 13,800 | 288 |

Accuracy/AUC/ECE/sculpting: mean ± std over 10 training seeds on the full test split (sweep.csv); float sculpting = cross-seed noise floor. Hardware: seed-0 model, ReuseFactor 1, part xck26-sfvc784-2LV-c, 200 MHz target; HLS estimates for seeds 0-2 differ by <= ~5% (hw.csv). Where post-route results exist, LUT/DSP are post-route and latency = HLS cycles at min(200 MHz, post-route Fmax); per config the lowest-latency build is used.

Paired vs same-seed float (Wilcoxon signed-rank, Holm-corrected per metric):

| metric | bits | n | median Δ | mean rel. change | p | p (Holm) |
|---|---|---|---|---|---|---|
| accuracy | 16 | 10 | +0.0002 | +0.0% | 0.846 | 0.846 |
| accuracy | 8 | 10 | -0.0001 | -0.0% | 0.334 | 0.668 |
| accuracy | 6 | 10 | -0.0009 | -0.1% | 0.00195 | 0.00977 |
| accuracy | 4 | 10 | -0.0045 | -0.6% | 0.00195 | 0.00977 |
| accuracy | 3 | 10 | -0.0105 | -1.5% | 0.00195 | 0.00977 |
| auc_top | 16 | 10 | -0.0001 | -0.0% | 0.492 | 0.645 |
| auc_top | 8 | 10 | -0.0002 | -0.0% | 0.322 | 0.645 |
| auc_top | 6 | 10 | -0.0006 | -0.1% | 0.00977 | 0.0293 |
| auc_top | 4 | 10 | -0.0028 | -0.3% | 0.00195 | 0.00977 |
| auc_top | 3 | 10 | -0.0064 | -0.7% | 0.00195 | 0.00977 |
| ece | 16 | 10 | -0.0009 | -4.2% | 0.846 | 1 |
| ece | 8 | 10 | -0.0001 | -2.8% | 0.922 | 1 |
| ece | 6 | 10 | -0.0003 | -5.0% | 0.77 | 1 |
| ece | 4 | 10 | +0.0029 | +87.6% | 0.0137 | 0.0547 |
| ece | 3 | 10 | +0.0063 | +140.1% | 0.00586 | 0.0293 |
