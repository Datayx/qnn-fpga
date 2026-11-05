# hls4ml softmax: calibration and cost

4-bit seed-0 model unless noted; fixed-point C simulation of all 166,000 test events. raw = hardware softmax outputs as produced; renorm = raw / row sum; softmax_on_arm = exact float softmax of the hardware's fixed-point logits. Keras ECE of this model: 0.0128.

## Softmax configurations

| config | view | out sum mean [min, max] | ECE | accuracy | agrees w/ Keras | HLS cycles | HLS LUT | HLS FF | HLS BRAM_18K | post-route LUT | post-route FF | WNS (ns) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_default | raw | 1.085 [0.996, 1.249] | 0.0735 | 0.7237 | 0.9924 | 13 | 38360 | 4074 | 4 | 16114 | 3566 | +0.041 |
| A_default | renorm | 1.085 [0.996, 1.249] | 0.0098 | 0.7237 | 0.9924 | 13 | 38360 | 4074 | 4 | 16114 | 3566 | +0.041 |
| B_table4096 | raw | 1.021 [0.996, 1.062] | 0.0300 | 0.7240 | 0.9978 | 13 | 38392 | 4117 | 13 | – | – | – |
| B_table4096 | renorm | 1.021 [0.996, 1.062] | 0.0131 | 0.7240 | 0.9978 | 13 | 38392 | 4117 | 13 | – | – | – |
| C_invinp | raw | 1.000 [0.995, 1.006] | 0.0105 | 0.7237 | 0.9924 | 13 | 38360 | 4073 | 4 | 16070 | 3566 | +0.211 |
| C_invinp | renorm | 1.000 [0.995, 1.006] | 0.0101 | 0.7237 | 0.9924 | 13 | 38360 | 4073 | 4 | 16070 | 3566 | +0.211 |
| D_types | raw | 1.000 [0.996, 1.007] | 0.0102 | 0.7237 | 0.9924 | 13 | 38519 | 4122 | 4 | – | – | – |
| D_types | renorm | 1.000 [0.996, 1.007] | 0.0097 | 0.7237 | 0.9924 | 13 | 38519 | 4122 | 4 | – | – | – |
| E_types_table4096 | raw | 0.998 [0.996, 1.001] | 0.0120 | 0.7240 | 0.9978 | 13 | 38519 | 4122 | 16 | – | – | – |
| E_types_table4096 | renorm | 0.998 [0.996, 1.001] | 0.0132 | 0.7240 | 0.9978 | 13 | 38519 | 4122 | 16 | – | – | – |
| F_latency | raw | 1.009 [0.938, 1.079] | 0.0189 | 0.7237 | 0.9917 | 11 | 38327 | 4015 | 4 | – | – | – |
| F_latency | renorm | 1.009 [0.938, 1.079] | 0.0138 | 0.7237 | 0.9917 | 11 | 38327 | 4015 | 4 | – | – | – |
| G_skip | softmax_on_arm | – (logits) | 0.0126 | 0.7241 | 0.9988 | 6 | 37817 | 3418 | 0 | – | – | – |

Configs: **A_default** = deployed: stable, tables 1024, all table types fixed<18,8>; **B_table4096** = A + table size 4096; **C_invinp** = A + inv_inp ufixed<18,3> (1/sum lookup input); **D_types** = C + exp/inv table values ufixed<18,1>, accum ufixed<18,3>; **E_types_table4096** = D + table size 4096; **F_latency** = A with the latency implementation; **G_skip** = softmax skipped: hardware outputs fixed-point logits (softmax on the ARM).

## Across seeds (mean ± std over 10 seeds)

| config | bits | view | ECE | Keras ECE | out sum mean | accuracy |
|---|---|---|---|---|---|---|
| A_default | 4 | raw | 0.0572 ± 0.0092 | 0.0092 ± 0.0063 | 1.084 | 0.7235 |
| A_default | 4 | renorm | 0.0099 ± 0.0074 | 0.0092 ± 0.0063 | 1.084 | 0.7235 |
| A_default | float | raw | 0.0576 ± 0.0052 | 0.0049 ± 0.0011 | 1.082 | 0.7282 |
| A_default | float | renorm | 0.0053 ± 0.0030 | 0.0049 ± 0.0011 | 1.082 | 0.7282 |
| C_invinp | 4 | raw | 0.0093 ± 0.0071 | 0.0092 ± 0.0063 | 1.000 | 0.7235 |
| C_invinp | 4 | renorm | 0.0093 ± 0.0073 | 0.0092 ± 0.0063 | 1.000 | 0.7235 |

## Paired tests (Wilcoxon over seeds, Holm within family; 95% CI = seed bootstrap of the mean)

| test | bits | n | mean Δ ECE | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|---|
| raw_minus_renorm (A_default) | 4 | 10 | +0.0473 | [+0.0372, +0.0552] | 0.00195 | 0.00391 |
| raw_minus_renorm (A_default) | float | 10 | +0.0523 | [+0.0475, +0.0567] | 0.00195 | 0.00391 |
| renorm_minus_keras (A_default) | 4 | 10 | +0.0007 | [-0.0006, +0.0020] | 0.375 | 0.75 |
| renorm_minus_keras (A_default) | float | 10 | +0.0004 | [-0.0011, +0.0020] | 0.77 | 0.77 |
| C_invinp raw minus A_default raw | 4 | 10 | -0.0480 | [-0.0555, -0.0385] | 0.00195 | 0.00391 |
| C_invinp raw minus Keras | 4 | 10 | +0.0001 | [-0.0009, +0.0011] | 0.846 | 0.846 |
