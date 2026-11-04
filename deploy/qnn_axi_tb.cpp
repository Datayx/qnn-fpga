// C-sim check: the wrapper must be bit-exact with calling myproject directly.
#include <cstdio>
#include <cstdlib>
#include "qnn_axi.h"

int main() {
    srand(1234);
    const int N = 2000;
    int bad = 0;
    for (int n = 0; n < N; n++) {
        input_t x[QNN_N_IN];
        result_t y_ref[QNN_N_OUT];
        word_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < QNN_N_IN; i++) {
            // standardised inputs in sweep.py stay within |x| < 8
            x[i] = (rand() % 16001) / 1000.0f - 8.0f;
            w[i / 2].range(16 * (i % 2) + 15, 16 * (i % 2)) = x[i].range();
        }
        myproject(x, y_ref);

        word_t o0, o1, o2;
        qnn_axi(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], o0, o1, o2);
        const word_t o[3] = {o0, o1, o2};
        for (int j = 0; j < QNN_N_OUT; j++) {
            ap_uint<16> got = o[j / 2].range(16 * (j % 2) + 15, 16 * (j % 2));
            if (got != y_ref[j].range()) {
                if (bad < 5) printf("mismatch event %d out %d: got %04x want %04x\n",
                                    n, j, (unsigned)got, (unsigned)y_ref[j].range());
                bad++;
            }
        }
        if (o2.range(31, 16) != 0) bad++;
    }
    printf("qnn_axi C-sim: %d events, %d mismatches -> %s\n", N, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
