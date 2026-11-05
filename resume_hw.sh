#!/bin/bash
# Resume the hardware pipeline after a reboot/shutdown. Safe to call any number of times:
#  - exits if the pipeline already finished (results/.hw_done) or is already running (flock);
#  - synth.py / impl.py skip configs already in results/hw.csv / results/impl.csv.
# Called at Windows logon by a local scheduled task ("qnn-fpga resume"), if one is set up.
cd "$(dirname "$0")"
[ -f results/.overnight_done ] && exit 0
exec 9> /tmp/qnn_hw.lock
flock -n 9 || exit 0
pgrep -f "run_all_hw.sh|run_overnight.sh" > /dev/null && exit 0   # already running
echo "=== resume $(date -Is)" >> hw_all.log
bash run_all_hw.sh >> hw_all.log 2>&1
bash run_overnight.sh >> hw_all.log 2>&1
