#!/bin/bash
# Register our policies as MIMIC_OPENSOURCE_V2 motions on the robot. Runs ON the robot.
#   install.sh DIR NAME[=LABEL] [NAME[=LABEL] ...]
# DIR holds q1rl_<name>.onnx/.npz/.csv for each NAME (tools/make_deploy_package.py), and
# optionally q1rl_<name>.detector.onnx (tools/train_lift_detector.py). LABEL is the display
# name the robot web page shows. Example:
#   install.sh /tmp/q1rl Q1RL_DANCE=自训舞蹈 Q1RL_BT=广播体操
# For each motion: copies the files, adds it to motion_list.yaml and a CustomConfigs entry to
# rl/mimic_opensource_v2_config.yaml that enables the gravity guard and, with a detector
# file, the lift detector (mc damps the robot when it sees it lifted off the floor). The
# guard is not optional: the runtime reads obs[44:47] (reference gravity) from the guard's
# csv, and for a motion without one that slot is never written. It also damps the robot
# when its tilt leaves the reference's by more than 30 deg.
# Every install starts from the original config files (backed up on first run) and removes
# all our earlier files, so list every motion that should stay installed. Nothing moves
# until mc is restarted (it reads both files only at Init) and SetMcMotion is called.
# Undo: rollback.sh.
#
# SDK v4 firmware keeps rl/*.yaml inside the encrypted rl/<robot>_rl_bundle.enc, and mc reads
# ./rl/ paths only from it, so CustomConfigs cannot be edited: our motions then run with no
# gravity guard and no lift detector, and obs[44:47] (reference gravity) is never written
# (the runner logs "Gravity guard config not found ... disabled"). install.sh refuses that
# unless Q1RL_NO_GUARD=1: only for a policy checked to ignore those 3 values (a SuperMimic
# 119-d policy with them held at 0 or upright tracks the same in sim), and the remote is
# then the only fall protection.
set -euo pipefail
DIR=$1; shift
[ $# -ge 1 ] || { echo "usage: install.sh DIR NAME[=LABEL] ..." >&2; exit 2; }
CFG=${Q1_CFG_DIR:-/robot/software/mc/bin/cfg/robot/${ROBOT_CFG:-q1_t2d0}}
LIST=$CFG/motion_list.yaml
RL=$CFG/rl/mimic_opensource_v2_config.yaml
# lift detector settings. The vendor's are EMA 0.35, threshold 0.6, debounce 1; ours are
# stricter (tools/train_lift_detector.py sweeps them): with 0.6/1 broadcast gymnastics damped
# itself 0.22 times a minute in simulated normal play, with 0.9/3 never in 108 min
LIFT_EMA=${LIFT_EMA:-0.35}; LIFT_THRESHOLD=${LIFT_THRESHOLD:-0.9}; LIFT_DEBOUNCE=${LIFT_DEBOUNCE:-3}
for spec in "$@"; do                       # check everything before touching anything
  NAME=${spec%%=*}; STEM=q1rl_$(tr '[:upper:]' '[:lower:]' <<<"${NAME#Q1RL_}")
  for f in "$DIR/$STEM.onnx" "$DIR/$STEM.npz" "$DIR/$STEM.csv"; do
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
  done
done
GUARD=1
if [ ! -f "$RL" ] && ls "$CFG"/rl/*_rl_bundle.enc >/dev/null 2>&1; then
  if [ "${Q1RL_NO_GUARD:-0}" != 1 ]; then
    echo "$RL is inside the encrypted rl bundle: no gravity guard / lift detector possible, and" >&2
    echo "obs[44:47] stays unwritten. Re-run with Q1RL_NO_GUARD=1 only for a policy that ignores it." >&2
    exit 1
  fi
  GUARD=0; echo "encrypted rl bundle: registering in motion_list.yaml only (no gravity guard, no lift detector)"
fi
FILES=$LIST; [ $GUARD = 1 ] && FILES="$LIST $RL"
for f in $FILES; do
  [ -f "$f.q1rl_orig" ] || cp -p "$f" "$f.q1rl_orig"
  cp -p "$f.q1rl_orig" "$f"
done
rm -f "$CFG"/rl_models/q1rl_*.onnx "$CFG"/rl_models/detectors/q1rl_*.onnx \
      "$CFG"/res/mimic_motion/npz/q1rl_*.npz "$CFG"/res/mimic_gravity_guard/q1rl_*.csv
for spec in "$@"; do
  NAME=${spec%%=*}; LABEL=${spec#*=}; [ "$spec" = "$NAME" ] && LABEL=$NAME
  STEM=q1rl_$(tr '[:upper:]' '[:lower:]' <<<"${NAME#Q1RL_}")
  install -m 644 "$DIR/$STEM.onnx" "$CFG/rl_models/$STEM.onnx"
  install -m 644 "$DIR/$STEM.npz"  "$CFG/res/mimic_motion/npz/$STEM.npz"
  install -m 644 "$DIR/$STEM.csv"  "$CFG/res/mimic_gravity_guard/$STEM.csv"
  cat >> "$LIST" <<EOT

  ###################### q1_rl: $LABEL ######################
  - name: $NAME
    type: MIMIC_OPENSOURCE_V2
    motion_area: 0
    interrupt_type: 0
    policy_path: "./rl_models/$STEM.onnx"
    custom:
      start_count: 0
      end_count: 10
      trajectory_path: "./res/mimic_motion/npz/$STEM.npz"
EOT
  [ $GUARD = 1 ] || continue
  cat >> "$RL" <<EOT

  # q1_rl: $NAME
  - motion_name: "$NAME"
    gravity_guard:
      enable: true
      path: "./res/mimic_gravity_guard/$STEM.csv"
      threshold_deg: 30
      count_threshold: 1
EOT
  if [ -f "$DIR/$STEM.detector.onnx" ]; then
    mkdir -p "$CFG/rl_models/detectors"
    install -m 644 "$DIR/$STEM.detector.onnx" "$CFG/rl_models/detectors/$STEM.detector.onnx"
    cat >> "$RL" <<EOT

    lift_detector:
      enabled: true
      detector_policy_path: "./rl_models/detectors/$STEM.detector.onnx"
      obs_start_index: 44
      obs_nums: 75
      history_len: 10
      logit_ema_alpha: $LIFT_EMA
      lift_threshold: $LIFT_THRESHOLD
      lift_debounce: $LIFT_DEBOUNCE
      skip_frame:
        start: 25
        end: 25
EOT
  fi
done
python3 - "$LIST" "$RL" "$CFG" "$GUARD" "$@" <<'PY'
import sys, yaml, os
lst, rl, cfg, guard, *specs = sys.argv[1:]
motions = yaml.safe_load(open(lst))["motion_configs"]
customs = yaml.safe_load(open(rl))["CustomConfigs"] if guard == "1" else []
for spec in specs:
    name = spec.split("=", 1)[0]
    m = [x for x in motions if x["name"] == name]
    c = [x for x in customs if x["motion_name"] == name]
    assert len(m) == 1 and len(c) == int(guard), (name, len(m), len(c))
    paths = [m[0]["policy_path"], m[0]["custom"]["trajectory_path"]]
    if not c:
        for p in paths:
            assert os.path.isfile(os.path.join(cfg, p)), p
        print(f"registered {name}: gravity guard OFF, lift detector OFF (encrypted rl bundle)")
        continue
    paths.append(c[0]["gravity_guard"]["path"])
    lift = c[0].get("lift_detector")
    if lift:
        paths.append(lift["detector_policy_path"])
    for p in paths:
        assert os.path.isfile(os.path.join(cfg, p)), p
    rows = sum(1 for _ in open(os.path.join(cfg, c[0]["gravity_guard"]["path"]))) - 1
    print(f"registered {name}: {rows} gravity rows, gravity guard on, lift detector "
          + ("on" if lift else "off"))
print("yaml ok")
PY
