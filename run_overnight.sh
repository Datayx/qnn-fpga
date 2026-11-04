#!/bin/bash
# Follow-up after run_all_hw.sh: timing closure + analysis. Idempotent (synth/impl skip finished configs).
#   1. wait for run_all_hw.sh to finish
#   2. every seed-0 build that missed 5 ns post-route -> re-synthesise with 45% HLS clock uncertainty
#      (HLS schedules for 2.75 ns, i.e. pipelines deeper), then place & route again
#   3. analyze.py -> results/summary.md, results/paired_tests.csv, figures/*.png
cd "$(dirname "$0")"
PY=~/miniforge3/envs/qnn/bin/python
CU="45%"

while pgrep -f run_all_hw.sh > /dev/null; do sleep 60; done
echo "=== overnight follow-up start $(date -Is)"

FAILING=$($PY - <<'EOF'
import pandas as pd
d = pd.read_csv("results/impl.csv", dtype={"bits": str})
if "clk_unc" not in d:
    d["clk_unc"] = "27%"
d["clk_unc"] = d["clk_unc"].fillna("27%")
bad = d[(d.seed == 0) & (d.clk_unc == "27%") & (d.wns_ns < 0)]
for _, r in bad.iterrows():
    print(r.bits, int(r.reuse))
EOF
)
echo "timing-closure candidates (bits reuse): ${FAILING:-none}"

while read -r bits rf; do
    [ -z "$bits" ] && continue
    bash run_synth.sh --bits "$bits" --reuse-factor "$rf" --clock-uncertainty "$CU"
done <<< "$FAILING"
[ -n "$FAILING" ] && bash run_impl.sh --clock-uncertainty "$CU"

source ~/miniforge3/etc/profile.d/conda.sh && conda activate qnn
MPLBACKEND=Agg python analyze.py && echo "ANALYZE_OK"
echo "OVERNIGHT_DONE $(date -Is)"
touch results/.overnight_done
