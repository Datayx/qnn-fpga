// Bit-exact reference: run hls4ml's C++ model (myproject) on every exported test event.
// Reads testvec/x_q.bin (int16 [N,15]), writes testvec/golden_q.bin (int16 [N,5] raw result_t).
// Must run with cwd = <hls project>/firmware (hls4ml loads weights/*.txt at run time).
// Build/run: bash deploy/run_golden.sh
#include <cstdio>
#include <cstdint>
#include <vector>
#include "myproject.h"

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: golden x_q.bin golden_q.bin\n"); return 2; }
    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { perror(argv[1]); return 1; }
    std::vector<int16_t> xq;
    int16_t buf[15];
    while (fread(buf, sizeof(int16_t), 15, fi) == 15) xq.insert(xq.end(), buf, buf + 15);
    fclose(fi);
    const size_t n = xq.size() / 15;
    std::vector<int16_t> out(n * 5);
    for (size_t e = 0; e < n; e++) {
        input_t x[15];
        result_t y[5];
        for (int i = 0; i < 15; i++) x[i].range() = (ap_uint<16>)(uint16_t)xq[e * 15 + i];
        myproject(x, y);
        for (int j = 0; j < 5; j++) out[e * 5 + j] = (int16_t)(uint16_t)y[j].range().to_uint();
    }
    FILE *fo = fopen(argv[2], "wb");
    fwrite(out.data(), sizeof(int16_t), out.size(), fo);
    fclose(fo);
    printf("golden: %zu events -> %s\n", n, argv[2]);
    return 0;
}
