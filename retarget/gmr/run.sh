#!/bin/bash
# Retarget LAFAN1 BVH clips onto Q1 with GMR and convert them to motion npz.
#   retarget/gmr/run.sh walk1_subject1 [dance1_subject2 ...]
# Env: PY, GMR (checkout), LAFAN1 (dir of .bvh), OUT (default runs/motions; files get a gmr_ prefix)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python}
GMR=${GMR:-$ROOT/_ref/GMR}
LAFAN1=${LAFAN1:-$ROOT/_ref/lafan1}
OUT=${OUT:-$ROOT/runs/motions}
"$PY" "$ROOT/retarget/make_model.py" >/dev/null          # the Q1 model GMR loads
"$PY" "$ROOT/retarget/gmr/make_config.py" --gmr "$GMR" | tail -1
for m in "$@"; do
  bvh=$LAFAN1/$m.bvh; [ -f "$bvh" ] || bvh=$m
  name=$(basename "$bvh" .bvh)
  "$PY" "$ROOT/retarget/gmr/bvh_to_q1.py" "$bvh" --gmr "$GMR" -o "$ROOT/runs/gmr/${name}_q1.csv"
  "$PY" "$ROOT/retarget/to_motion.py" "$ROOT/runs/gmr/${name}_q1.csv" --fps 30 -o "$OUT/gmr_$name.npz" --name "gmr_$name"
done
