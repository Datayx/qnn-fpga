"""
Turn the board programs' printed output into results/board.csv.

Save each program's terminal output on the PC as text, e.g.
  results/board_driver.txt   (driver.py)
  results/board_bench.txt    (bench)
  results/board_cpu.txt      (cpu_bench)
then:  python deploy/parse_board.py results/board_*.txt
Only numbers actually printed by the board are written; nothing is filled in or estimated.
"""
import re
import sys

import pandas as pd

PATTERNS = [  # (metric, unit, regex) matched against the printf formats in deploy/board/*.{c,py}
    ("validation_mismatches", "outputs", r"validation: \d+ events, (\d+) output mismatches"),
    ("validation_events", "events", r"validation: (\d+) events"),
    ("board_accuracy", "", r"board accuracy: ([\d.]+)"),
    ("onchip_cycles_min", "cycles", r"on-chip cycles[^:]*: min (\d+)"),
    ("onchip_cycles_max", "cycles", r"on-chip cycles[^:]*: min \d+ max (\d+)"),
    ("onchip_latency", "ns", r"on-chip cycles[^>]*-> ([\d.]+) ns"),
    ("fpga_e2e_median", "ns", r"FPGA end-to-end[^:]*: median ([\d.]+) ns"),
    ("fpga_e2e_p99", "ns", r"FPGA end-to-end[^:]*: median [\d.]+ ns, p99 ([\d.]+) ns"),
    ("fpga_e2e_loop_mean", "ns", r"FPGA end-to-end[^:]*:.*loop mean ([\d.]+) ns"),
    ("cpu_accuracy", "", r"CPU accuracy: ([\d.]+)"),
    ("cpu_fpga_argmax_agreement", "", r"argmax agreement with FPGA golden: ([\d.]+)"),
    ("cpu_median", "ns", r"CPU batch-1[^:]*: median ([\d.]+) ns"),
    ("cpu_p99", "ns", r"CPU batch-1[^:]*: median [\d.]+ ns, p99 ([\d.]+) ns"),
    ("cpu_loop_mean", "ns", r"CPU batch-1[^:]*:.*loop mean ([\d.]+) ns"),
    ("driver_events_differ", "events", r"(\d+) events differ from golden"),
]


def main(paths):
    rows = []
    for p in paths:
        text = open(p, encoding="utf-8", errors="replace").read()
        for metric, unit, rx in PATTERNS:
            m = re.search(rx, text)
            if m:
                rows.append(dict(metric=metric, value=float(m.group(1)), unit=unit, source=p))
    if not rows:
        sys.exit("no board results recognised in the given files")
    df = pd.DataFrame(rows).drop_duplicates("metric", keep="last")
    df.to_csv("results/board.csv", index=False)
    print(df.to_string(index=False))
    v = dict(zip(df.metric, df.value))
    if v.get("validation_mismatches", 1) != 0:
        print("\nWARNING: board output is not bit-exact with the golden model; do not quote latency yet.")
    if "onchip_latency" in v and "cpu_median" in v:
        print(f"\non-chip {v['onchip_latency']:.0f} ns vs ARM A53 median {v['cpu_median'] / 1000:.2f} us "
              f"({v['cpu_median'] / v['onchip_latency']:.0f}x)")
    if "fpga_e2e_median" in v and "cpu_median" in v:
        print(f"end-to-end FPGA {v['fpga_e2e_median'] / 1000:.2f} us vs ARM {v['cpu_median'] / 1000:.2f} us")


if __name__ == "__main__":
    main(sys.argv[1:])
