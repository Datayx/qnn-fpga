#!/bin/bash
# Build and run the bit-exact golden model in WSL. Usage: bash deploy/run_golden.sh [hls dir, default hls/mlp_4b]
set -e
here="$(cd "$(dirname "$0")" && pwd)"
hls="$(realpath "${1:-$here/../hls/mlp_4b}")"
inc="$(dirname "$(find /tools/Xilinx/2026.1 -name ap_fixed.h -path '*include/ap_fixed.h' | head -1)")"
mkdir -p "$here/build"
g++ -O2 -std=c++14 -I"$hls/firmware" -I"$inc" "$here/golden.cpp" "$hls/firmware/myproject.cpp" -o "$here/build/golden"
cd "$hls/firmware"
"$here/build/golden" "$here/testvec/x_q.bin" "$here/testvec/golden_q.bin"
