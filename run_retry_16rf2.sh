#!/bin/bash
# Retry place & route of the 16-bit RF2 build (OOM-killed at 5.6 GB with default threads), after the
# overnight pipeline, with 1 Vivado thread to cut peak memory. Then re-run the analysis.
cd "$(dirname "$0")"
while pgrep -f "run_all_hw.sh|run_overnight.sh" > /dev/null; do sleep 60; done
echo "=== retry 16b rf2 $(date -Is)"
VIVADO_THREADS=1 bash run_impl.sh --bits 16 --reuse 2
source ~/miniforge3/etc/profile.d/conda.sh && conda activate qnn
MPLBACKEND=Agg python analyze.py > /dev/null && echo "ANALYZE_OK"
echo "RETRY_DONE $(date -Is)"
