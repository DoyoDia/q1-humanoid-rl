#!/bin/bash
# Print the current motion-control action and its status (read-only). Runs ON the robot.
source "$(dirname "$0")/env.sh"
timeout 20 ros2 service call /aimdk_5Fmsgs/srv/GetMcAction aimdk_msgs/srv/GetMcAction "{}" 2>&1 \
  | grep -o "action_desc='[^']*'\|status=aimdk_msgs.msg.McActionStatus(value=[0-9]*)"
