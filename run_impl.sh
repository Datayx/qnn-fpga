#!/bin/bash
# Run impl.py in WSL with the qnn env and Vivado 2026.1. Args are passed to impl.py.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate qnn
source /tools/Xilinx/2026.1/Vitis/settings64.sh
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2
python -u impl.py "$@"
echo "IMPL_EXIT $?"
