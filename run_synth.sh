#!/bin/bash
# Run synth.py in WSL with the qnn env and Vitis 2026.1. Args are passed to synth.py.
# Example: nohup bash run_synth.sh --bits 16 > synth_16.log 2>&1 &
source ~/miniforge3/etc/profile.d/conda.sh
conda activate qnn
source /tools/Xilinx/2026.1/Vitis/settings64.sh
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2
python -u synth.py "$@"
echo "SYNTH_EXIT $?"
