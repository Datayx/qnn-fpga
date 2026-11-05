#!/bin/bash
# Stage 2 of calibrate.py: hls4ml csim of the seed-0 calibration variants, in WSL with the qnn env.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate qnn
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2
python -u calib_hls.py "$@"
echo "CALIB_HLS_EXIT $?"
