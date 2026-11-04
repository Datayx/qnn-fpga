// ARM Cortex-A53 baseline: the same 15-32-32-5 network in plain C, batch 1.
// Exact quantised weights from hardware (cpu_weights.h); float32 arithmetic; ReLU quantised like the
// hardware (ufixed<4,0>: round-half-even to 1/16, saturate to 15/16); float softmax.
//   gcc -O3 -mcpu=cortex-a53 -o cpu_bench cpu_bench.c -lm && taskset -c 3 ./cpu_bench ../testvec
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "cpu_weights.h"

#define N_IN 15
#define H 32
#define N_OUT 5

static inline float qrelu(float v) {
    float r = nearbyintf(v * 16.0f) / 16.0f;   // default rounding mode = round half to even (RND_CONV)
    return r < 0.0f ? 0.0f : (r > 0.9375f ? 0.9375f : r);
}

static void dense(const float *x, int n_in, int n_out, const float *w, const float *b, float *y) {
    for (int j = 0; j < n_out; j++) y[j] = b[j];
    for (int i = 0; i < n_in; i++) {
        const float xi = x[i];
        const float *wr = w + i * n_out;
        for (int j = 0; j < n_out; j++) y[j] += xi * wr[j];
    }
}

static int predict(const float *x, float *p) {
    float h1[H], h2[H], z[N_OUT];
    dense(x, N_IN, H, W2, B2, h1);
    for (int j = 0; j < H; j++) h1[j] = qrelu(h1[j]);
    dense(h1, H, H, W5, B5, h2);
    for (int j = 0; j < H; j++) h2[j] = qrelu(h2[j]);
    dense(h2, H, N_OUT, W8, B8, z);
    float m = z[0], s = 0.0f;
    int best = 0;
    for (int j = 1; j < N_OUT; j++) if (z[j] > m) { m = z[j]; best = j; }
    for (int j = 0; j < N_OUT; j++) { p[j] = expf(z[j] - m); s += p[j]; }
    for (int j = 0; j < N_OUT; j++) p[j] /= s;
    return best;
}

static void *load(const char *dir, const char *name, size_t *bytes) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *bytes = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(*bytes);
    if (fread(buf, 1, *bytes, f) != *bytes) { perror("read"); exit(1); }
    fclose(f);
    return buf;
}
static double ns(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec);
}
static int cmp(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b; return (d > 0) - (d < 0);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "../testvec";
    int n_time = argc > 2 ? atoi(argv[2]) : 100000;
    size_t bx, bg, by;
    float *x = load(dir, "x_float.bin", &bx);
    int16_t *gold = load(dir, "golden_q.bin", &bg);
    uint8_t *y = load(dir, "y.bin", &by);
    size_t n = by;
    if (bx != n * N_IN * 4 || bg != n * N_OUT * 2) { fprintf(stderr, "size mismatch\n"); return 1; }

    size_t correct = 0, agree = 0;
    float p[N_OUT];
    for (size_t e = 0; e < n; e++) {
        int c = predict(x + e * N_IN, p);
        int g = 0;
        for (int j = 1; j < N_OUT; j++) if (gold[e * N_OUT + j] > gold[e * N_OUT + g]) g = j;
        correct += (c == y[e]);
        agree += (c == g);
    }
    printf("CPU accuracy: %.4f, argmax agreement with FPGA golden: %.4f (n=%zu)\n",
           (double)correct / n, (double)agree / n, n);

    if ((size_t)n_time > n) n_time = n;
    double *t = malloc(sizeof(double) * n_time);
    volatile int sink = 0;
    for (int i = 0; i < 1000; i++) sink += predict(x + (i % n) * N_IN, p);
    struct timespec a, b, a0, b0;
    for (int i = 0; i < n_time; i++) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &a);
        sink += predict(x + i * N_IN, p);
        clock_gettime(CLOCK_MONOTONIC_RAW, &b);
        t[i] = ns(a, b);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &a0);
    for (int i = 0; i < n_time; i++) sink += predict(x + i * N_IN, p);
    clock_gettime(CLOCK_MONOTONIC_RAW, &b0);
    clock_gettime(CLOCK_MONOTONIC_RAW, &a);
    for (int i = 0; i < 100000; i++) clock_gettime(CLOCK_MONOTONIC_RAW, &b);
    double timer_ns = ns(a, b) / 100000;
    qsort(t, n_time, sizeof(double), cmp);
    printf("CPU batch-1 (n=%d): median %.0f ns, p99 %.0f ns, loop mean %.0f ns, timer overhead ~%.0f ns\n",
           n_time, t[n_time / 2], t[(int)(n_time * 0.99)], ns(a0, b0) / n_time, timer_ns);
    return 0;
}
