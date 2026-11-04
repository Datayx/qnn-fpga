# KR260 deployment plan

**Status (2026-11-04):** steps 1-2 done on the PC, step 3 needs the board.
- Step 1: `deploy/run_axi.sh` -> qnn_axi IP, C-sim bit-exact 2000/2000, 14 cycles (70 ns), ap_ctrl_hs wires.
- Step 2: `deploy/run_bd.sh` -> `deploy/overlay/qnn.{bit,hwh}`, timing met at 200 MHz (WNS +0.306 ns),
  15.5k LUT / 4.7k FF / 5 DSP / 4 RAMB18 total (core 14.9k LUT). Rerun export only: `export_overlay.tcl`.
- Test vectors + golden: `deploy/export_testvec.py` + `deploy/run_golden.sh` (166k events; golden acc 0.7237).
- Board code + instructions: `deploy/board/` (driver.py, bench.c, cpu_bench.c, README.md).
  cpu_bench verified on the PC: acc 0.7241, 99.24% argmax agreement with golden.
- Design change vs below: start/done are wires driven by `deploy/latency_ctrl.v` via AXI GPIO, so the
  on-chip latency is measured on the board (cycle counter), not only in simulation.

Goal: run one quantised MLP on the KR260 PL, check it is bit-exact against hls4ml, and measure
latency vs the board's Cortex-A53, producing the CV headline number honestly.
Estimate: ~3 working days (day 1 build, day 2 bring-up + validation, day 3 measurement).

## 0. Which model
**4-bit, seed 0, RF 1** (primary).
- Accuracy -0.5 pt vs float, ECE ~2x: exactly the "kept accuracy within Z% but degraded calibration
  by W%" story in the CV template.
- Post-route OOC: 16.1k LUT (13.8%), 5 DSP, WNS +0.041 ns at 5 ns. Slack is thin: inside a full design
  timing may fail at 200 MHz (see step 2).
- Fallback: **3-bit** (11.9k LUT, WNS +0.107 ns) if 4-bit won't close; accuracy -1.1 pt.

## 1. AXI wrapper (HLS, WSL)
The hls4ml core has plain `ap_vld` array ports; the ARM needs a bus interface.
- Write `deploy/qnn_axi.cpp`: top function `qnn_axi(in[15], out[5])` that calls hls4ml's `myproject`,
  with `#pragma HLS INTERFACE s_axilite` on `in`, `out` and `return` (ap_ctrl_hs via AXI-Lite).
  Use the same `input_t`/`result_t` types as the synthesised project (`ap_fixed<16,6>`).
- Synthesise + export IP (`export_design -format ip_catalog`) with the same part / 5 ns clock.
- Expect a few extra cycles vs the bare core (65 ns) for AXI-Lite handshaking; record both.
- Why AXI-Lite, not AXI-Stream + DMA: batch-1 latency is the measurement; DMA setup costs more than it
  saves for 15 inputs. Stream + DMA only if a throughput number is wanted later.

## 2. Vivado block design (script it: `deploy/kr260_bd.tcl`)
- Board: KR260 (install the Kria KR260 board files from the Vivado board store).
- Zynq UltraScale+ MPSoC with board preset; master `M_AXI_HPM0_FPD` -> SmartConnect -> `qnn_axi`.
- Clock: `pl_clk0` 100 MHz -> Clocking Wizard -> 200 MHz for `qnn_axi` (you've done this part already),
  `proc_sys_reset` per clock domain. Simplest first: run the AXI bus and `qnn_axi` on the same 200 MHz clock.
- **Latency counter**: small Verilog module counting cycles from `ap_start` rising to `ap_done`,
  read through an AXI GPIO. This gives the on-chip compute latency, separate from bus overhead.
- Implement, check WNS. If negative at 200 MHz: try 187.5 / 150 MHz and **report the frequency that
  closes**, don't hide it. Export `.bit` + `.hwh` (same basename) for PYNQ.

## 3. Board bring-up
- Ubuntu 22.04 for Kria + Kria-PYNQ (`git clone Xilinx/Kria-PYNQ; sudo bash install.sh -b KR260`).
- Copy overlay to the board, `Overlay("qnn.bit")`, find `qnn_axi` in `ol.ip_dict`.
- Python driver: quantise inputs to `ap_fixed<16,6>` (round to 2^-10, saturate to +/-32, write as int16),
  write registers, set ap_start, poll ap_done, read 5 outputs.

## 4. Validation (before any timing)
- Run 10k test events (same split as sweep.py, saved as `.npy` on the PC).
- Must match hls4ml C-sim outputs bit-for-bit (or argmax agreement ~1.0); report board accuracy.
  If not bit-exact: check input rounding/saturation and register byte order first.

## 5. Latency measurement (report all three; never only the flattering one)
| Measurement | How | Expected order |
|---|---|---|
| PL compute latency | latency counter cycles x clock period | ~65-80 ns |
| FPGA end-to-end, batch 1 | **C driver** via UIO or `/dev/mem` mmap: write inputs, start, poll, read; 10k calls, median + p99 | ~1-3 us (AXI-Lite dominated) |
| ARM A53 baseline, batch 1 | same weights, float32, plain C loops (`-O3`), 10k calls, median + p99; also numpy for reference | ~1-10 us |
- Python/PYNQ end-to-end will be tens of us (interpreter overhead); quote it only as a footnote.
- Pin to one core, warm up, disable frequency scaling if possible (`cpufreq` performance governor).
- Save raw timings to `results/board.csv`.

## 6. Write-up
- README: block diagram, the three latencies, resources of the deployed design (post-route, in-context).
- CV bullet: "X ns on-chip latency (Y us end-to-end) vs Z us on the board's ARM CPU", real numbers only.

## Risks
- Kria-PYNQ vs Ubuntu/kernel version mismatch: follow the Kria-PYNQ README exactly.
- Timing at 200 MHz in-context (thin slack): fallback frequency or 3-bit.
- Board files / preset for KR260 in Vivado 2026.1: if missing, configure the MPSoC manually from the K26 SOM preset.
