#!/bin/bash
# Start the dance page and keep it up. Runs ON the robot, from cron @reboot (install_web.sh).
# Waits for motion control, sets up the robot's ROS 2 environment, restarts the server if
# it ever exits.
DIR=$(cd "$(dirname "$0")" && pwd)
PORT=${Q1RL_WEB_PORT:-8090}
source "$DIR/env.sh"
sleep 20                                    # let mc and the ROS graph come up after boot
while true; do
  python3 "$DIR/q1rl_web.py" --port "$PORT" --key "$(cat "$DIR/key")" >> "$DIR/web.log" 2>&1
  echo "$(date '+%F %T') q1rl_web exited ($?), restarting in 5 s" >> "$DIR/web.log"
  sleep 5
done
