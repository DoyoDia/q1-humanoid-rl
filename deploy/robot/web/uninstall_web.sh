#!/bin/bash
# Stop the dance page, remove it from boot and delete ~/q1rl_web. Runs ON the robot.
DST=$(readlink -f "$HOME/q1rl_web")             # /home/user/q1rl_web on SDK v4 robots
if [ -f /etc/systemd/system/q1rl-web.service ]; then
  sudo systemctl disable --now q1rl-web.service; sudo rm -f /etc/systemd/system/q1rl-web.service; sudo systemctl daemon-reload
fi
CUR=$(crontab -l 2>/dev/null || true)
{ grep -v -F "$DST/run.sh" <<<"$CUR" || true; } | crontab -
for p in $(pgrep -f "^/bin/bash $DST/run.sh") $(pgrep -f "^python3 $DST/q1rl_web.py"); do kill "$p" 2>/dev/null || true; done
rm -rf "$DST"; rm -f "$HOME/q1rl_web"
echo "removed $DST and its boot entry"
