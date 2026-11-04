# On the KR260

## One-time setup
1. Flash **Ubuntu 22.04 for Kria** to the SD card, boot, update the boot firmware if the Kria docs ask.
2. Install Kria-PYNQ (follow its README exactly):
   `git clone https://github.com/Xilinx/Kria-PYNQ.git && cd Kria-PYNQ && sudo bash install.sh -b KR260`

## Copy files (from the PC, board on the network)
```
scp -r deploy/overlay deploy/testvec deploy/board ubuntu@<board-ip>:~/qnn/
```
(`testvec/` is ~10 MB: x_q, x_float, y, golden_q for all 166,000 test events.)

## Run, in this order
```
cd ~/qnn/board
sudo -E python3 driver.py                       # loads qnn.bit; quick check on 2,000 events; cycle count
gcc -O2 -o bench bench.c
sudo taskset -c 3 ./bench ../testvec            # all events bit-exact + FPGA end-to-end latency
gcc -O3 -mcpu=cortex-a53 -o cpu_bench cpu_bench.c -lm
taskset -c 3 ./cpu_bench ../testvec             # ARM baseline, same network
```
Optional for steadier numbers: `sudo cpupower frequency-set -g performance` before benchmarking.

## Expected
- driver.py / bench: **0 mismatches** vs golden, accuracy 0.7237, on-chip cycles = 15
  (HLS latency 14 + the start cycle; = 75 ns at 200 MHz).
- bench end-to-end: a few us (11 AXI-Lite accesses + polling dominate, not the 70 ns compute).
- cpu_bench: accuracy ~0.72, argmax agreement with FPGA ~0.99; latency to be measured.

If validation fails: check the overlay is loaded (driver.py first), input packing (element 2k in bits
[15:0]), and that testvec/ was made by deploy/export_testvec.py + run_golden.sh from the same hls/mlp_4b.

Save the printed output to results/board_*.txt on the PC; the CV number comes from these.
