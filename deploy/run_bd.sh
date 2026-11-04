#!/bin/bash
# Build the KR260 bitstream in WSL. Needs the qnn_axi IP first: bash deploy/run_axi.sh
# Usage: QNN_FREQ_MHZ=200 VIVADO_THREADS=2 bash deploy/run_bd.sh
set -e
source /tools/Xilinx/2026.1/Vitis/settings64.sh
here="$(cd "$(dirname "$0")" && pwd)"
export QNN_BUILD="${QNN_BUILD:-$HOME/qnn_build/vivado}"
mkdir -p "$QNN_BUILD" && cd "$QNN_BUILD/.."
vivado -mode batch -nojournal -log "$here/run_bd_vivado.log" -source "$here/kr260_bd.tcl"
echo "BD_EXIT $?"
cat "$here/overlay/address_map.txt"
