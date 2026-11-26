#!/bin/bash
# Retarget another robot's motion onto Q1 with robot_retargeter and convert it to motion npz.
#   retarget/run.sh walk1_subject1 [dance1_subject2 ...]          # G1 clips (LAFAN1 on G1)
#   SOURCE=t800 SOURCE_FPS=50 retarget/run.sh path/to/clip_t800.npz   # T800 motion npz (Isaac / bvh_tool)
# Env: PY (python with mink/mujoco), RETARGETER (checkout), SOURCE (g1 | t800, default g1),
#      SOURCE_FPS (default 30; a T800 npz carries its own fps, pass the same here),
#      START (seconds to skip, default 0; the first IK frames are not converged),
#      OUT (default runs/motions)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
RT=${RETARGETER:-$ROOT/_ref/robot_retargeter}
SOURCE=${SOURCE:-g1}
FPS=${SOURCE_FPS:-30}
START=${START:-0}
OUT=${OUT:-$ROOT/runs/motions}
"$PY" "$ROOT/retarget/make_model.py" --retargeter "$RT"
for m in "$@"; do
  if [ "$SOURCE" = t800 ]; then
    name=$(basename "$m" .npz)
    src=$ROOT/runs/t800/$name.csv
    "$PY" "$ROOT/retarget/t800_to_csv.py" "$m" -o "$src"
  else
    src=$RT/dataset/lafan1_g1/$m.csv
    [ -f "$src" ] || src=$m
    name=$(basename "$src" .csv)
  fi
  (cd "$RT" && "$PY" scripts/robot_replay.py --no-viewer --motion-file "$src" \
      --source-robot-config "config/robot/$SOURCE.yaml" --target-robot-config config/robot/q1.yaml --fps "$FPS" >/dev/null \
   && MUJOCO_GL=egl "$PY" scripts/robot_retarget.py --config config/robot/q1.yaml \
      --keypoints-name "${name}_from_$SOURCE" --no-render-debug >/dev/null)
  "$PY" "$ROOT/retarget/to_motion.py" "$RT/output_data/robot_motion/${name}_from_${SOURCE}_q1.csv" \
      --fps "$FPS" --start "$START" -o "$OUT/$name.npz"
done
