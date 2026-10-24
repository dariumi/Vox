#!/usr/bin/env bash
# A/B of two vox builds on the CPU, interleaved in one session (so heat, background load and the page cache hit both alike).
#   tools/ab_cpu.sh OLD/build/vox NEW/build/vox --model q8.tmod --voice v.tvoice [--threads 8] [--rounds 5]
# Prints the warm RTF of every round and the median of each build.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,4p' "$0"; exit 2; }
A=$1; B=$2; shift 2
ROUNDS=5; THREADS=8; ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --rounds) ROUNDS=$2; shift 2 ;;
    --threads) THREADS=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
run() { VOX_TUNE=0 "$1" bench "${ARGS[@]}" --device cpu --threads "$THREADS" --runs 3 2>/dev/null | tail -n 1 |
        python3 -c 'import json,sys; print("%.3f" % json.load(sys.stdin)["warm"]["rtf"])'; }
ra=(); rb=()
echo "A: $A"
echo "B: $B"
for i in $(seq 1 "$ROUNDS"); do
  if [ $((i % 2)) -eq 1 ]; then x=$(run "$A"); y=$(run "$B"); else y=$(run "$B"); x=$(run "$A"); fi
  ra+=("$x"); rb+=("$y")
  echo "round $i: A $x   B $y"
done
med() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END {print (NR % 2) ? a[(NR+1)/2] : (a[NR/2] + a[NR/2+1]) / 2}'; }
echo "median RTF: A $(med "${ra[@]}")   B $(med "${rb[@]}")"
