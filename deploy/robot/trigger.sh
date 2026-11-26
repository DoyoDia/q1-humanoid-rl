#!/bin/bash
# Play a registered motion. Runs ON the robot:  trigger.sh NAME
# Refuses unless the robot is in BIPED_WALK_RUN with status RUNNING (100), as the SDK
# requires: standing, walk policy in control, no transition under way.
NAME=$1
source "$(dirname "$0")/env.sh"      # before set -u: ROS's setup scripts read unset variables
set -euo pipefail
S=$(timeout 20 ros2 service call /aimdk_5Fmsgs/srv/GetMcAction aimdk_msgs/srv/GetMcAction "{}" 2>&1)
DESC=$(grep -o "action_desc='[^']*'" <<<"$S" | cut -d"'" -f2)
STAT=$(grep -o "McActionStatus(value=[0-9]*)" <<<"$S" | grep -o "[0-9]*")
echo "current: $DESC status $STAT"
if [ "$DESC" != "BIPED_WALK_RUN" ] || [ "$STAT" != "100" ]; then
  echo "NOT triggering: need BIPED_WALK_RUN / RUNNING(100). Stand the robot up in walk mode first."; exit 1
fi
echo "triggering $NAME"
timeout 20 ros2 service call /aimdk_5Fmsgs/srv/SetMcMotion aimdk_msgs/srv/SetMcMotion \
  "{source: 'q1_rl', type: 1, motion: '$NAME', interrupt: false}" 2>&1 | tail -2
