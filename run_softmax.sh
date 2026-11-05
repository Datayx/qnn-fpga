#!/bin/bash
# softmax_study.py in WSL with the qnn env and Vitis/Vivado 2026.1 (needed for --synth and impl).
# Usage: bash run_softmax.sh configs --synth | seeds --config A_default --bits 4 float | impl --configs A_default D_types | report
source ~/miniforge3/etc/profile.d/conda.sh
conda activate qnn
source /tools/Xilinx/2026.1/Vitis/settings64.sh
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2
export VIVADO_THREADS=${VIVADO_THREADS:-2}
python -u softmax_study.py "$@"
echo "SOFTMAX_EXIT $?"
