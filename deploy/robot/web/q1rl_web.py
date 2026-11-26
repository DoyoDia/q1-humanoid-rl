"""A small web page on the robot for playing its motions over the LAN. Runs ON the robot.

    python3 q1rl_web.py --port 8090 --key <access key>

    GET  /                 the page
    GET  /api/status       current motion-control action and its status
    GET  /api/motions      every motion, as official / custom and whole-body / upper-body
    POST /api/play         {"id": ...}  play one, if the robot is in the mode it needs
    POST /api/mode         {"mode": "BIPED_WALK_RUN" | "BIPED_WHOLE_BODY_CTRL"}
    POST /api/stop_music   stop the music a motion started
    GET  /media/<file>?key  a motion's music, for playing it on the phone instead

Music: media.json next to this file binds a motion to a sound file in media/, e.g.
  {"Q1RL_BT": {"wav": "bt.wav", "mp3": "bt.mp3", "offset_s": 0.0}}
When that motion starts, the robot's speaker plays the wav (hal_audio's PlayAudioFile; mono
16-bit wav) -- or, if the page asks for it, the phone plays the mp3. offset_s > 0 starts the
music that much after the motion, < 0 before it. If the motion ends early (remote, guard,
lift detector) the music is stopped.

Sharing the robot with a navigation stack: if something publishes /cmd_vel_bridge/enabled
(std_msgs/Bool, latched) and it is true, that stack is driving the walk with velocity commands,
and this page neither plays a motion nor switches mode -- a whole-body motion started under a
navigating robot would fight it for the walk. No message for NAV_STALE_S counts as off. Other
clients may switch the mode too: we act only from BIPED_WALK_RUN / BIPED_WHOLE_BODY_CTRL, and
only once the action has stayed the same for SETTLE_S. Set Q1RL_AFTER_MOTION_NOTE to show a
reminder after every motion (e.g. to restart an odometry node). No 50 Hz subscriptions here:
the robot's CPU is shared and full.

Two kinds of motion, two modes (the SDK's preconditions):
  whole body  motion_list.yaml entries of type MIMIC, MIMIC_V2, MIMIC_OPENSOURCE_V2 --
              SetMcMotion, from BIPED_WALK_RUN
  upper body  type ANIMATION entries (by number) and the user's taught motions
              (GetCustomMotionList, by csv path) -- SetMcPresetMotion, from
              BIPED_WHOLE_BODY_CTRL
Official = the vendor's motion_list entries; custom = ours (Q1RL_*) and the taught ones.
The server checks the mode before every play, and only switches between those two standing
modes. Every /api call needs the access key (header X-Key). There is deliberately no stop
or damping button: the remote is the emergency stop.

Standard library plus the robot's own rclpy, aimdk_msgs and PyYAML; start it through run.sh.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import rclpy
import yaml
from aimdk_msgs.msg import AudioFile, AudioInfo, McActionCommand, McPresetMotion, RequestHeader
from aimdk_msgs.srv import (GetCustomMotionList, GetMcAction, PlayAudioFile, SetMcAction, SetMcMotion,
                            SetMcPresetMotion, StopAudioPlay)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

CFG = Path(os.environ.get("Q1_CFG_DIR", "/robot/software/mc/bin/cfg/robot/q1_t2d0"))
HERE = Path(__file__).resolve().parent

WALK, UPPER = "BIPED_WALK_RUN", "BIPED_WHOLE_BODY_CTRL"
RUNNING = 100
WHOLE_TYPES = ("MIMIC", "MIMIC_V2", "MIMIC_OPENSOURCE_V2")
# Motions that end on the floor, sitting, or that throw the robot around: shown, but flagged.
RISKY = re.compile(r"LYING|LIE|SIT|FOLD|CRAWL|FLIP|KICK|JUMP|STANDING_TO|CRY")
# Motions that start on the floor (getting up, folding from lying or crawling). From walk
# mode they would be played on a standing robot; the page lists them but will not play them.
FROM_FLOOR = re.compile(r"^LYING|GET_UP|LIE_TO|CRAWL_TO")
MEDIA = HERE / "media"
NAV_TOPIC = "/cmd_vel_bridge/enabled"               # a navigation stack's velocity bridge, if any
NAV_STALE_S = 3.0
SETTLE_S = 3.0                                     # the action must be this old before we act
ODOM_NOTE = os.environ.get("Q1RL_AFTER_MOTION_NOTE") or None


def load_media():
    try:
        return json.loads((HERE / "media.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def wav_info(path):
    import wave
    with wave.open(str(path)) as w:
        return w.getnchannels(), w.getframerate(), w.getnframes() / w.getframerate()


class Robot(Node):
    def __init__(self):
        super().__init__("q1rl_web")
        self.c_get = self.create_client(GetMcAction, "/aimdk_5Fmsgs/srv/GetMcAction")
        self.c_action = self.create_client(SetMcAction, "/aimdk_5Fmsgs/srv/SetMcAction")
        self.c_motion = self.create_client(SetMcMotion, "/aimdk_5Fmsgs/srv/SetMcMotion")
        self.c_preset = self.create_client(SetMcPresetMotion, "/aimdk_5Fmsgs/srv/SetMcPresetMotion")
        self.c_custom = self.create_client(GetCustomMotionList, "/aimdk_5Fmsgs/srv/GetCustomMotionList")
        self.c_audio = self.create_client(PlayAudioFile, "/aimdk_5Fmsgs/srv/PlayAudioFile")
        self.c_stop_audio = self.create_client(StopAudioPlay, "/aimdk_5Fmsgs/srv/StopAudioPlay")
        self.lock = threading.Lock()
        self.music = None                                # the motion whose music is playing
        self.nav = (False, 0.0)                          # (bridge enabled, when we heard it)
        self.seen = (None, 0.0)                          # (last action seen, when it changed)
        self.create_subscription(Bool, NAV_TOPIC, self._on_nav, QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST))

    def _on_nav(self, msg):
        self.nav = (bool(msg.data), time.time())

    def nav_active(self):
        on, t = self.nav
        return on and time.time() - t < NAV_STALE_S

    def _no_nav(self):
        if self.nav_active():
            raise PermissionError("导航 / 跟随正在控制走路（cmd_vel_bridge 已使能），先把它关掉再来")

    def _call(self, client, req, timeout=3.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("运控服务不可用")
        fut = client.call_async(req)
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                raise RuntimeError("运控没有响应")
            time.sleep(0.01)
        return fut.result()

    def _header(self):
        h = RequestHeader()
        h.stamp = self.get_clock().now().to_msg()
        return h

    def status(self):
        r = self._call(self.c_get, GetMcAction.Request())
        action, status = r.info.action_desc, int(r.info.status.value)
        if (action, status) != self.seen[0]:
            self.seen = ((action, status), time.time())
        return {"action": action, "status": status, "nav": self.nav_active(),
                "settling": time.time() - self.seen[1] < SETTLE_S}

    def custom_motions(self):
        try:
            r = self._call(self.c_custom, GetCustomMotionList.Request())
            return list(r.motion_list)
        except Exception:                                # noqa: BLE001 -- the list is optional
            return []

    def _require(self, mode):
        self._no_nav()
        st = self.status()
        if st["action"] != mode or st["status"] != RUNNING:
            name = "走跑模式" if mode == WALK else "全身控制（上肢动作）模式"
            raise PermissionError(f"需要先切到{name}，机器人当前是 {st['action']}（{st['status']}）")
        if st["settling"]:
            raise PermissionError("机器人模式刚变过（可能是别的面板在切），等几秒再试")

    def play_music(self, wav):
        ch, rate, dur = wav_info(wav)
        req = PlayAudioFile.Request()
        req.request.header = self._header()
        f = AudioFile()
        f.pkg_name = "q1rl_web"
        f.file_path = str(wav.parent)
        f.file_name = wav.name
        f.info = AudioInfo()
        f.info.channels, f.info.sample_rate = ch, rate
        f.info.sample_format, f.info.coding_format = "S16_LE", "wave"
        f.priority = 6
        req.file = f
        r = self._call(self.c_audio, req).response
        if int(r.header.code) != 0:
            raise RuntimeError(f"音频服务拒绝了请求（code {int(r.header.code)}）{r.message}")
        return dur

    def stop_music(self):
        self.music = None
        req = StopAudioPlay.Request()
        req.request.header = self._header()
        self._call(self.c_stop_audio, req)

    def _music_thread(self, motion, wav, offset):
        """Start the music offset_s after the motion (it was started before this thread),
        then stop it if the motion ends early."""
        token = object()
        self.music = token
        try:
            if offset > 0:
                time.sleep(offset)
            if self.music is not token:
                return
            dur = self.play_music(wav)
            t0, seen = time.time(), False
            while self.music is token and time.time() - t0 < dur:
                time.sleep(0.5)
                action = self.status()["action"]
                if action == motion or "MIMIC" in action:
                    seen = True
                elif seen or time.time() - t0 > 5.0:     # done, damped, or never started
                    if time.time() - t0 < dur - 2.0:
                        self.stop_music()
                    break
        except Exception as e:                           # noqa: BLE001 -- music is best effort
            print(f"music for {motion}: {e}", flush=True)
        finally:
            if self.music is token:
                self.music = None

    def play(self, m, phone_music=False):
        media = load_media().get(m.get("motion", ""))
        wav = MEDIA / media["wav"] if media and media.get("wav") else None
        offset = float(media.get("offset_s", 0.0)) if media else 0.0
        if wav is not None and not phone_music and offset < 0:
            self.play_music(wav)                         # music leads: start it, then the motion
            time.sleep(-offset)
        r = self._play(m)
        if r["ok"] and wav is not None and not phone_music and offset >= 0:
            threading.Thread(target=self._music_thread, args=(m["motion"], wav, offset), daemon=True).start()
        if media and media.get("mp3"):
            r["music"] = {"url": f"/media/{media['mp3']}", "offset_s": offset}
        return r

    def _play(self, m):
        with self.lock:                                  # check and trigger as one step
            self._require(m["mode"])
            if m["mode"] == WALK:
                req = SetMcMotion.Request()
                req.header = self._header()
                req.source = "q1rl_web"
                req.type = SetMcMotion.Request.MIMIC_QY
                req.motion = m["motion"]
                req.interrupt = False
                r = self._call(self.c_motion, req).response
            else:
                req = SetMcPresetMotion.Request()
                req.header = self._header()
                req.source = "q1rl_web"
                req.motion = McPresetMotion()
                req.motion.value = int(m.get("preset", 0))
                req.ani_path = m.get("path", "")
                req.interrupt = False
                r = self._call(self.c_preset, req).response
            code, state = int(r.header.code), int(r.state.value)
            ok = code == 0
            return {"ok": ok, "code": code, "state": state, "note": ODOM_NOTE if ok else None,
                    "error": None if ok else f"运控拒绝了请求（code {code}，state {state}），可能有别的动作还没做完"}

    def set_mode(self, mode):
        with self.lock:
            self._no_nav()
            st = self.status()
            other = UPPER if mode == WALK else WALK
            if st["action"] == mode:
                return {"ok": True}
            if st["action"] != other or st["status"] != RUNNING:
                raise PermissionError(f"只能在机器人站着、处于走跑或全身控制模式时切换，当前是 {st['action']}（{st['status']}）")
            if st["settling"]:
                raise PermissionError("机器人模式刚变过（可能是别的面板在切），等几秒再试")
            req = SetMcAction.Request()
            req.header = self._header()
            req.source = "q1rl_web"
            req.command = McActionCommand()
            req.command.action_desc = mode
            r = self._call(self.c_action, req).response
            ok = int(r.header.code) == 0
            return {"ok": ok, "error": None if ok else f"切换被拒绝（code {int(r.header.code)}）",
                    "note": ODOM_NOTE if ok and mode == UPPER else None}


def banners(text):
    """name -> the comment banner above its entry in motion_list.yaml (`###### 锁舞 ######`)."""
    out, banner = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*#{3,}\s*(.*?)\s*#{3,}\s*$", line)
        if m:
            banner = m.group(1)
            continue
        m = re.match(r"\s*-\s*name:\s*([A-Za-z0-9_]+)", line)
        if m:
            out[m.group(1)] = banner
            banner = None
    return out


def load_motions(robot):
    text = (CFG / "motion_list.yaml").read_text(encoding="utf-8")
    label = banners(text)
    media = load_media()
    out = []
    for e in yaml.safe_load(text)["motion_configs"]:
        name, typ = str(e.get("name")), e.get("type")
        lab = (label.get(name) or name).replace("q1_rl: ", "")
        ours = name.startswith("Q1RL_")
        if typ in WHOLE_TYPES:
            floor = bool(FROM_FLOOR.search(name))
            mp3 = (media.get(name) or {}).get("mp3")
            music = f"/media/{mp3}" if mp3 else (name in media or None)
            out.append({"id": f"m:{name}", "motion": name, "mode": WALK, "kind": "whole",
                        "group": "custom" if ours else "official", "label": lab, "name": name,
                        "risky": bool(RISKY.search(name)), "unsupported": floor, "music": music,
                        "note": "从躺姿开始，请用遥控器" if floor else ("带音乐 ♪" if music else None)})
        elif typ == "ANIMATION" and name.isdigit():
            out.append({"id": f"p:{name}", "preset": int(name), "mode": UPPER, "kind": "upper",
                        "group": "official", "label": lab.replace("上肢动作-", ""), "name": name, "risky": False})
    for c in robot.custom_motions():
        teach = int(c.resource_type) == 2
        out.append({"id": f"c:{c.resource_path}", "path": c.resource_path, "mode": UPPER, "kind": "upper",
                    "group": "custom", "label": c.resource_key, "name": f"示教 · {int(c.duration)} 秒" if teach else "灵创",
                    "risky": False, "unsupported": not teach, "note": None if teach else "灵创动作暂不支持"})
    return out


def make_handler(robot, key, page):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self):
            return hmac.compare_digest((self.headers.get("X-Key") or "").encode(), key.encode())

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _media(self, name, qs):
            if not hmac.compare_digest((parse_qs(qs).get("key", [""])[0]).encode(), key.encode()):
                return self._send(403, {"error": "口令不对"})
            f = (MEDIA / name).resolve()
            if f.parent != MEDIA.resolve() or not f.is_file():
                return self._send(404, {"error": "not found"})
            size = f.stat().st_size
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start, end = int(m.group(1)), int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
            ctype = "audio/mpeg" if f.suffix == ".mp3" else "audio/wav"
            self.send_response(206 if m else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(f, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            if path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if path.startswith("/media/"):
                return self._media(path[len("/media/"):], url.query)
            if not self._authed():
                return self._send(403, {"error": "口令不对"})
            try:
                if path == "/api/status":
                    return self._send(200, robot.status())
                if path == "/api/motions":
                    return self._send(200, load_motions(robot))
            except Exception as e:                       # noqa: BLE001 -- report, keep serving
                return self._send(503, {"error": str(e)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            if not self._authed():
                return self._send(403, {"ok": False, "error": "口令不对"})
            try:
                body = self._body()
                if path == "/api/play":
                    m = {x["id"]: x for x in load_motions(robot)}.get(body.get("id"))
                    if m is None or m.get("unsupported"):
                        return self._send(400, {"ok": False, "error": "这个动作不能从这里播放"})
                    return self._send(200, robot.play(m, bool(body.get("phone_music"))))
                if path == "/api/stop_music":
                    robot.stop_music()
                    return self._send(200, {"ok": True})
                if path == "/api/mode":
                    mode = body.get("mode")
                    if mode not in (WALK, UPPER):
                        return self._send(400, {"ok": False, "error": "未知模式"})
                    return self._send(200, robot.set_mode(mode))
            except PermissionError as e:
                return self._send(409, {"ok": False, "error": str(e)})
            except Exception as e:                       # noqa: BLE001
                return self._send(503, {"ok": False, "error": str(e)})
            self._send(404, {"error": "not found"})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--key", required=True, help="access key the page asks for")
    a = ap.parse_args()
    # rclpy's own SIGINT/SIGTERM handler only shuts the ROS context down: the HTTP server then
    # kept answering "rcl node's context is invalid" until systemd's stop timeout SIGKILLed it,
    # which also held up every reboot. Handle the signals here and stop the whole process.
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    robot = Robot()

    def spin():
        try:
            rclpy.spin(robot)
        except Exception as e:                           # noqa: BLE001
            print(f"ROS spin ended: {e!r}", flush=True)
        finally:
            os._exit(3)                                  # run.sh starts a fresh one in 5 s

    threading.Thread(target=spin, daemon=True).start()
    page = (HERE / "index.html").read_bytes()
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(robot, a.key, page))
    except OSError as e:                                 # someone else's page owns the port
        raise SystemExit(f"port {a.port} is taken ({e}); set Q1RL_WEB_PORT in run.sh's environment")
    import signal
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: os._exit(0))       # nothing to flush: no state on disk
    print(f"q1rl_web on :{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
