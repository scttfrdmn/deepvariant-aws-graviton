#!/usr/bin/env bash
# dnn with more than 8 processes would exceed antares RAM (3.2 GB plan + 1.3 GB images each): skipped.
# sat.sh ENGINE PROCS THREADS [PIN=split|shared] [REPS]: PROCS concurrent call_variants processes, 8192 images
# each, THREADS threads each, on cores 0-15 (split: disjoint core sets; shared: all on 0-15). Every process
# loads and preprocesses first, then all start their timed loop at one epoch time; throughput =
# PROCS x 8192 / (last end - first start). ENGINE tf (pip TF 2.13.1, run.py) or dnn (cvdnn).
set -u
if [[ $1 == dnn && $2 -gt 8 ]]; then echo "skip dnn procs=$2: RAM"; exit 0; fi
E=$1; P=$2; T=$3; PIN=${4:-split}; REPS=${5:-3}
D=$HOME/dv-dev; cd $D/ox/p2
LEAD=${LEAD:-60}  # seconds for every process to load before the common start
for r in $(seq $REPS); do
  at=$(( $(date +%s) + LEAD )); pids=(); : > sat-$E.out
  for p in $(seq 0 $((P - 1))); do
    if [[ $PIN == split ]]; then c="$((p * 16 / P))-$(((p + 1) * 16 / P - 1))"; else c=0-15; fi
    if [[ $E == tf ]]; then
      TF_CPP_MIN_LOG_LEVEL=2 taskset -c $c $D/ox/tf2.13.1/bin/python $D/ie/run.py $D/ie/dump sat-$p tf $D/ie/models/wgs --limit 8192 --threads $T --start-at $at 2>/dev/null | grep "^{" >> sat-$E.out &
    else
      OMP_NUM_THREADS=$T OMP_PROC_BIND=${BIND:-close} OMP_PLACES=cores taskset -c $c ./cvdnn net --images $D/ie/dump --n 8192 --start-at $at 2>/dev/null | grep network >> sat-$E.out &
    fi
    pids+=($!)
  done
  wait "${pids[@]}"
  python3 - "$E" "$P" "$T" "$PIN" "$at" <<'PY'
import json, re, sys
E, P, T, PIN, at = sys.argv[1:]
L = open(f"sat-{E}.out").read().splitlines()
if E == "tf":
    r = [json.loads(l) for l in L]; s = [x["t_start"] for x in r]; e = [x["t_end"] for x in r]; c = [x["cpu_s"] for x in r]
else:
    g = lambda k, l: float(re.search(k + r"=([\d.]+)", l).group(1))
    s = [g("t_start", l) for l in L]; e = [g("t_end", l) for l in L]; c = [g("cpu", l) for l in L]
ok = "" if E == "tf" else (" all IDENTICAL" if all("IDENTICAL" in l for l in L) else " NOT IDENTICAL")
late = max(s) - float(at)
print(f"{E} procs={P} threads={T} pin={PIN} n={len(L)} window={max(e) - min(s):.2f}s start_skew={max(s) - min(s):.2f}s late={late:.2f}s "
      f"sum_cpu={sum(c):.1f}s ex/s={len(L) * 8192 / (max(e) - min(s)):.0f}{ok}")
PY
done
