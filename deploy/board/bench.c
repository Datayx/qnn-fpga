// FPGA end-to-end benchmark + full bit-exact validation on the KR260 (run as root, overlay loaded).
//   gcc -O2 -o bench bench.c && sudo taskset -c 3 ./bench ../testvec
// For each event: write 8 input words, rising edge on go, poll done, read 3 output words, drop go.
// Reports: mismatches vs golden (must be 0), accuracy, on-chip cycles, end-to-end latency
// (per call: median/p99 incl. ~clock_gettime overhead; and loop mean without per-call timers).
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "qnn_regs.h"

static volatile uint32_t *gpio, *core;

static void *load(const char *dir, const char *name, size_t *bytes) {
    char p[512];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *bytes = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(*bytes);
    if (fread(buf, 1, *bytes, f) != *bytes) { perror("read"); exit(1); }
    fclose(f);
    return buf;
}

static inline uint32_t infer(const int16_t *x, uint32_t out[3]) {
    for (int k = 0; k < 8; k++) {
        uint32_t lo = (uint16_t)x[2 * k];
        uint32_t hi = (2 * k + 1 < N_IN) ? (uint16_t)x[2 * k + 1] : 0;
        core[QNN_IN(k)] = lo | (hi << 16);
    }
    __sync_synchronize();
    gpio[GPIO_GO] = 1;
    __sync_synchronize();
    uint32_t s;
    while (!((s = gpio[GPIO_STATUS]) & 0x80000000u)) {}
    out[0] = core[QNN_OUT(0)]; out[1] = core[QNN_OUT(1)]; out[2] = core[QNN_OUT(2)];
    gpio[GPIO_GO] = 0;
    __sync_synchronize();
    return s & 0x7fffffffu;
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
    int16_t *xq = load(dir, "x_q.bin", &bx);
    int16_t *gold = load(dir, "golden_q.bin", &bg);
    uint8_t *y = load(dir, "y.bin", &by);
    size_t n = by;
    if (bx != n * N_IN * 2 || bg != n * N_OUT * 2) { fprintf(stderr, "size mismatch\n"); return 1; }

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem (run as root)"); return 1; }
    gpio = mmap(NULL, QNN_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, QNN_GPIO_BASE);
    core = mmap(NULL, QNN_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, QNN_CORE_BASE);
    if (gpio == MAP_FAILED || core == MAP_FAILED) { perror("mmap"); return 1; }
    gpio[GPIO_GO] = 0;

    // 1) validation over every event
    size_t bad = 0, correct = 0;
    uint32_t cmin = ~0u, cmax = 0;
    for (size_t e = 0; e < n; e++) {
        uint32_t o[3];
        uint32_t c = infer(xq + e * N_IN, o);
        if (c < cmin) cmin = c;
        if (c > cmax) cmax = c;
        int16_t v[N_OUT];
        int best = 0;
        for (int j = 0; j < N_OUT; j++) {
            v[j] = (int16_t)((o[j / 2] >> (16 * (j % 2))) & 0xffff);
            if (v[j] != gold[e * N_OUT + j]) bad++;
            if (v[j] > v[best]) best = j;
        }
        correct += (best == y[e]);
    }
    printf("validation: %zu events, %zu output mismatches vs golden -> %s\n", n, bad, bad ? "FAIL" : "PASS");
    printf("board accuracy: %.4f\n", (double)correct / n);
    printf("on-chip cycles (ap_start..ap_done incl.): min %u max %u -> %.1f ns at %.3f MHz\n",
           cmin, cmax, cmin * 1e9 / CORE_CLK_HZ, CORE_CLK_HZ / 1e6);

    // 2) end-to-end latency, batch 1
    if ((size_t)n_time > n) n_time = n;
    double *t = malloc(sizeof(double) * n_time);
    uint32_t o[3];
    for (int i = 0; i < 1000; i++) infer(xq + (i % n) * N_IN, o);           // warm-up
    struct timespec a, b, a0, b0;
    for (int i = 0; i < n_time; i++) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &a);
        infer(xq + i * N_IN, o);
        clock_gettime(CLOCK_MONOTONIC_RAW, &b);
        t[i] = ns(a, b);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &a0);
    for (int i = 0; i < n_time; i++) infer(xq + i * N_IN, o);
    clock_gettime(CLOCK_MONOTONIC_RAW, &b0);
    clock_gettime(CLOCK_MONOTONIC_RAW, &a);                                  // timer overhead
    for (int i = 0; i < 100000; i++) clock_gettime(CLOCK_MONOTONIC_RAW, &b);
    double timer_ns = ns(a, b) / 100000;
    qsort(t, n_time, sizeof(double), cmp);
    printf("FPGA end-to-end (n=%d): median %.0f ns, p99 %.0f ns, loop mean %.0f ns, timer overhead ~%.0f ns\n",
           n_time, t[n_time / 2], t[(int)(n_time * 0.99)], ns(a0, b0) / n_time, timer_ns);
    return 0;
}
