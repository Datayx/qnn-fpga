#!/bin/bash
# Build the qnn_axi IP in WSL. Usage: bash deploy/run_axi.sh [hls4ml project dir, default hls/mlp_4b]
# Output: deploy/build/qnn_axi_prj/solution1/impl/ip/ (Vivado IP repo), csynth report in .../syn/report/
set -e
source /tools/Xilinx/2026.1/Vitis/settings64.sh
here="$(cd "$(dirname "$0")" && pwd)"
export QNN_HLS_DIR="$(realpath "${1:-$here/../hls/mlp_4b}")"
mkdir -p "$here/build" && cd "$here/build"
vitis-run --mode hls --tcl "$here/build_axi.tcl"
echo "AXI_EXIT $?"
grep -A3 "Latency (cycles)" qnn_axi_prj/solution1/syn/report/qnn_axi_csynth.rpt | head -8 || true
