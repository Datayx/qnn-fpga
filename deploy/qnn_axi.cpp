#include "qnn_axi.h"

void qnn_axi(word_t in0, word_t in1, word_t in2, word_t in3,
             word_t in4, word_t in5, word_t in6, word_t in7,
             word_t &out0, word_t &out1, word_t &out2) {
#pragma HLS INTERFACE mode=s_axilite port=in0 bundle=ctrl
#pragma HLS INTERFACE mode=s_axilite port=in1 bundle=ctrl
#pragma HLS INTERFACE mode=s_axilite port=in2 bundle=ctrl
#pragma HLS INTERFACE mode=s_axilite port=in3 bundle=ctrl
#pragma HLS INTERFACE mode=s_axilite port=in4 bundle=ctrl
#pragma HLS INTERFACE mode=s_axilite port=in5 bundle=ctrl
#pragma HLS INTERFACE mode=s_axilite port=in6 bundle=ctrl
#pragma HLS INTERFACE mode=s_axilite port=in7 bundle=ctrl
#pragma HLS INTERFACE mode=s_axilite port=out0 bundle=ctrl
#pragma HLS INTERFACE mode=s_axilite port=out1 bundle=ctrl
#pragma HLS INTERFACE mode=s_axilite port=out2 bundle=ctrl
// Block control as plain ap_start/ap_done/ap_idle/ap_ready wires (not AXI-Lite) so that
// deploy/latency_ctrl.v can start the core and count cycles to ap_done on chip.
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
#pragma HLS PIPELINE

    const word_t w_in[8] = {in0, in1, in2, in3, in4, in5, in6, in7};
    input_t x[QNN_N_IN];
    result_t y[QNN_N_OUT];
#pragma HLS ARRAY_PARTITION variable=x complete
#pragma HLS ARRAY_PARTITION variable=y complete

    for (int i = 0; i < QNN_N_IN; i++) {
#pragma HLS UNROLL
        x[i].range() = w_in[i / 2].range(16 * (i % 2) + 15, 16 * (i % 2));
    }

    myproject(x, y);

    word_t w_out[3] = {0, 0, 0};
    for (int j = 0; j < QNN_N_OUT; j++) {
#pragma HLS UNROLL
        w_out[j / 2].range(16 * (j % 2) + 15, 16 * (j % 2)) = y[j].range();
    }
    out0 = w_out[0];
    out1 = w_out[1];
    out2 = w_out[2];
}
