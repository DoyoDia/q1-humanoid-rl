#!/bin/bash
# Install the motion page into /home/user/q1rl_web (linked from ~/q1rl_web) and start it on every boot. Runs ON the robot:
#   bash install_web.sh SRC_DIR [MEDIA_DIR]
# SRC_DIR holds q1rl_web.py, index.html, run.sh, env.sh, q1rl-web.service, media.json (which motion plays
# which music); MEDIA_DIR, if given, holds the sound files media.json names (.wav for the
# robot's speaker, .mp3 for a phone) and is copied to ~/q1rl_web/media.
# Re-running updates the files and restarts the server; the access key is kept.
set -euo pipefail
SRC=$1; MEDIA=${2:-}
# /home/run is on the overlay root (an OTA or factory reset may wipe it); /home/user is its
# own partition on SDK v4 robots. Install there when it exists, and link ~/q1rl_web to it.
if [ -n "${Q1RL_WEB_DIR:-}" ]; then DST=$Q1RL_WEB_DIR
elif [ -L "$HOME/q1rl_web" ]; then DST=$(readlink -f "$HOME/q1rl_web")
elif [ ! -e "$HOME/q1rl_web" ] && [ -d /home/user ] && [ -w /home/user ]; then DST=/home/user/q1rl_web
else DST=$HOME/q1rl_web; fi
mkdir -p "$DST/media"
[ "$DST" = "$HOME/q1rl_web" ] || ln -sfn "$DST" "$HOME/q1rl_web"
install -m 644 "$SRC/q1rl_web.py" "$SRC/index.html" "$SRC/env.sh" "$SRC/media.json" "$DST/"
[ -z "$MEDIA" ] || install -m 644 "$MEDIA"/* "$DST/media/"
install -m 755 "$SRC/run.sh" "$DST/"
[ -f "$DST/key" ] || (tr -dc 'a-z0-9' </dev/urandom | head -c 8 > "$DST/key"; chmod 600 "$DST/key")
install -m 644 "$SRC/q1rl-web.service" "$DST/"
for p in $(pgrep -f "^/bin/bash $DST/run.sh") $(pgrep -f "^python3 $DST/q1rl_web.py"); do kill "$p" 2>/dev/null || true; done
if command -v systemctl >/dev/null && sudo -n true 2>/dev/null; then
  # systemd: SDK v4 robots refuse run's crontab
  sed -e "s|@DST@|$DST|g" -e "s|@USER@|$(id -un)|g" -e "s|@HOME@|$HOME|g" "$DST/q1rl-web.service" \
    | sudo tee /etc/systemd/system/q1rl-web.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable q1rl-web.service >/dev/null 2>&1
  sudo systemctl restart q1rl-web.service
  CUR=$(crontab -l 2>/dev/null || true)          # drop an old @reboot line, if any
  if grep -q -F "$DST/run.sh" <<<"$CUR"; then grep -v -F "$DST/run.sh" <<<"$CUR" | crontab - || true; fi
  echo "installed in $DST, systemd q1rl-web.service (enabled, started)"
else
  LINE="@reboot $DST/run.sh"
  CUR=$(crontab -l 2>/dev/null || true)          # no crontab yet is not an error
  { grep -v -F "$DST/run.sh" <<<"$CUR" || true; echo "$LINE"; } | grep -v '^$' | crontab -
  sleep 1
  setsid nohup "$DST/run.sh" >/dev/null 2>&1 < /dev/null &
  echo "installed in $DST, starts on boot (crontab: $LINE)"
fi
