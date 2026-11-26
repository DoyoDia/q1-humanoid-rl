#!/bin/bash
# Undo install.sh: restore the original motion_list.yaml and mimic_opensource_v2_config.yaml
# and delete our files. Runs ON the robot. Takes effect after the next mc restart.
set -euo pipefail
CFG=${Q1_CFG_DIR:-/robot/software/mc/bin/cfg/robot/${ROBOT_CFG:-q1_t2d0}}
for f in "$CFG/motion_list.yaml" "$CFG/rl/mimic_opensource_v2_config.yaml"; do
  [ -f "$f.q1rl_orig" ] && cp -p "$f.q1rl_orig" "$f" && rm -f "$f.q1rl_orig" && echo "restored $f"
done
rm -fv "$CFG"/rl_models/q1rl_*.onnx "$CFG"/res/mimic_motion/npz/q1rl_*.npz "$CFG"/res/mimic_gravity_guard/q1rl_*.csv
