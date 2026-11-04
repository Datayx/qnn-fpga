// Register map of the KR260 overlay (deploy/overlay/address_map.txt, HLS xqnn_axi_hw.h).
#pragma once
#include <stdint.h>

#define QNN_GPIO_BASE 0xA0000000u  // axi_gpio_0: ch1 data (0x0) = go, ch2 data (0x8) = {done, count[30:0]}
#define QNN_CORE_BASE 0xA0010000u  // qnn_axi_0 s_axi_ctrl
#define QNN_SPAN      0x10000u

#define GPIO_GO     (0x0 / 4)
#define GPIO_STATUS (0x8 / 4)
#define QNN_IN(k)   ((0x10 + 8 * (k)) / 4)   // k = 0..7, two int16 per word (elem 2k low, 2k+1 high)
#define QNN_OUT(k)  ((0x50 + 0x10 * (k)) / 4) // k = 0..2

#define N_IN  15
#define N_OUT 5
#define CORE_CLK_HZ 199998000.0              // clk_wiz output (pl_clk0 is 99.999 MHz)
