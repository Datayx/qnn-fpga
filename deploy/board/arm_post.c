// ARM-side cost of the two software fixes for hls4ml's softmax approximation, per event, on the A53.
//   (i)   renormalise the hardware's 5 softmax outputs (int16 raw, ap_fixed<16,6>: value = raw / 2^10)
//   (iii) exact float softmax of the hardware's 5 logits (int16 raw, ap_fixed<14,7>: value = raw / 2^7),
//         i.e. the design built with the softmax skipped
//   base  only converting 5 int16 to float (the work any consumer does anyway)
// Inputs: testvec/golden_q.bin (deployed outputs) and testvec/logits_q.bin (skip-softmax csim logits).
//   gcc -O3 -mcpu=cortex-a53 -o arm_post arm_post.c -lm && taskset -c 3 ./arm_post ../testvec
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_OUT 5

static void *load(const char *dir, const char *name, size_t *bytes) {
    char p[512];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *bytes = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*bytes);
    if (fread(b, 1, *bytes, f) != *bytes) { perror("read"); exit(1); }
    fclose(f);
    return b;
}
static double ns(struct timespec a, struct timespec b) { return (b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec); }
static int cmp(const void *a, const void *b) { double d = *(const double *)a - *(const double *)b; return (d > 0) - (d < 0); }

static inline void base(const int16_t *q, float *p) {
    for (int j = 0; j < N_OUT; j++) p[j] = q[j] * (1.0f / 1024.0f);
}
static inline void renorm(const int16_t *q, float *p) {
    float s = 0.0f;
    for (int j = 0; j < N_OUT; j++) { p[j] = (float)q[j]; s += p[j]; }
    const float inv = 1.0f / s;
    for (int j = 0; j < N_OUT; j++) p[j] *= inv;
}
static inline void softmax_logits(const int16_t *q, float *p) {
    int16_t m = q[0];
    for (int j = 1; j < N_OUT; j++) if (q[j] > m) m = q[j];
    float s = 0.0f;
    for (int j = 0; j < N_OUT; j++) { p[j] = expf((q[j] - m) * (1.0f / 128.0f)); s += p[j]; }
    const float inv = 1.0f / s;
    for (int j = 0; j < N_OUT; j++) p[j] *= inv;
}

typedef void (*fn_t)(const int16_t *, float *);

static void bench(const char *label, fn_t f, const int16_t *data, size_t n, int n_time) {
    float p[N_OUT];
    volatile float sink = 0.0f;
    double *t = malloc(sizeof(double) * n_time);
    struct timespec a, b, a0, b0;
    for (int i = 0; i < 1000; i++) { f(data + (i % n) * N_OUT, p); sink += p[0]; }
    for (int i = 0; i < n_time; i++) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &a);
        f(data + (i % n) * N_OUT, p);
        clock_gettime(CLOCK_MONOTONIC_RAW, &b);
        sink += p[0];
        t[i] = ns(a, b);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &a0);
    for (int i = 0; i < n_time; i++) { f(data + (i % n) * N_OUT, p); sink += p[0]; }
    clock_gettime(CLOCK_MONOTONIC_RAW, &b0);
    qsort(t, n_time, sizeof(double), cmp);
    printf("%-14s (n=%d): median %.0f ns, p99 %.0f ns, loop mean %.1f ns\n", label, n_time, t[n_time / 2],
           t[(int)(n_time * 0.99)], ns(a0, b0) / n_time);
    free(t);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "../testvec";
    int n_time = argc > 2 ? atoi(argv[2]) : 100000;
    size_t bo, bl;
    int16_t *outs = load(dir, "golden_q.bin", &bo);
    int16_t *logits = load(dir, "logits_q.bin", &bl);
    size_t n = bo / (2 * N_OUT);
    if (bl != bo) { fprintf(stderr, "size mismatch golden_q vs logits_q\n"); return 1; }
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC_RAW, &a);
    for (int i = 0; i < 100000; i++) clock_gettime(CLOCK_MONOTONIC_RAW, &b);
    printf("timer overhead ~%.0f ns per clock_gettime\n", ns(a, b) / 100000);
    bench("base", base, outs, n, n_time);
    bench("renorm", renorm, outs, n, n_time);
    bench("softmax_logits", softmax_logits, logits, n, n_time);
    return 0;
}
