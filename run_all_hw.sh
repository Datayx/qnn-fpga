#!/bin/bash
# Full hardware pipeline, sequential (Vivado is memory-hungry; don't parallelise on a 16 GB machine).
#   nohup bash run_all_hw.sh > hw_all.log 2>&1 &
cd "$(dirname "$0")"
bash run_synth.sh --bits 16 --reuse-factor 2     # deployable 16-bit point (RF1 needs 130% of DSPs)
bash run_impl.sh                                 # post-route for every seed-0 config that fits
bash run_synth.sh --seeds 1 2                    # HLS for seeds 1, 2 -> error bars on latency/resources
echo "ALL_DONE"
