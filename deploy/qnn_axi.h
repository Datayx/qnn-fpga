// AXI-Lite wrapper around the hls4ml core `myproject` for the KR260 (see DEPLOY.md step 1).
//
// Register packing: two 16-bit ap_fixed<16,6> raw values per 32-bit word, element 2k in bits [15:0]
// and element 2k+1 in bits [31:16]. 15 inputs -> 8 words (in7[31:16] unused), 5 outputs -> 3 words
// (out2[31:16] = 0). Raw value = floor(x * 2^10) as int16 (two's complement), i.e. the same
// quantisation as input_t = ap_fixed<16,6> with its default AP_TRN (truncate toward -inf) / AP_WRAP.
// Outputs decode as int16 / 2^10.
#ifndef QNN_AXI_H_
#define QNN_AXI_H_

#include "ap_int.h"
#include "myproject.h"

static const int QNN_N_IN = 15;
static const int QNN_N_OUT = 5;
typedef ap_uint<32> word_t;

void qnn_axi(word_t in0, word_t in1, word_t in2, word_t in3,
             word_t in4, word_t in5, word_t in6, word_t in7,
             word_t &out0, word_t &out1, word_t &out2);

#endif
