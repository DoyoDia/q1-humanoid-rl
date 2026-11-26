"""Render a Q1 motion npz: a contact sheet of evenly spaced frames, and optionally a video.

    python tools/render_motion.py runs/motions/walk.npz -o walk.png [--mp4 walk.mp4] [--frames 12]

For eyeballing a reference before training on it -- a retarget that maps an arm
backwards or sinks a foot is obvious in a picture and invisible in summary numbers.
Uses the visual model (python robots/q1/build_model.py --keep-meshes --out robots/q1/q1_visual.xml).
The camera follows the pelvis from a fixed angle.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# glvnd otherwise hands EGL to Mesa, whose llvmpipe renders on the CPU at ~6 fps (1300 on the GPU)
_NV_EGL = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.isfile(_NV_EGL):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NV_EGL)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl.contract import Contract  # noqa: E402
from q1_rl.motion import Motion  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def scene(xml: Path, w: int, h: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.meshdir = str((xml.parent / spec.meshdir).resolve()) + "/"
    spec.visual.global_.offwidth, spec.visual.global_.offheight = w, h
    spec.visual.headlight.ambient = [0.35, 0.35, 0.38]
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.rgba = [0.86, 0.87, 0.89, 1.0]
        elif g.type != mujoco.mjtGeom.mjGEOM_PLANE:
            g.rgba[3] = 0.0
    return spec.compile()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True, help="contact sheet .png")
    p.add_argument("--mp4", type=Path, default=None)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--model", type=Path, default=ROOT / "robots/q1/q1_visual.xml")
    p.add_argument("--azimuth", type=float, default=135.0)
    a = p.parse_args()

    mo = Motion.load_npz(a.npz)
    c = Contract()
    w, h = 480, 600
    m = scene(a.model, w, h)
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in c.joint_names]
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 1.6, a.azimuth, -12

    def frame(t):
        d.qpos[0:3] = mo.body_pos_w[t, 0]
        d.qpos[3:7] = mo.body_quat_w[t, 0]
        d.qpos[qadr] = mo.joint_pos[t]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [d.xpos[1][0], d.xpos[1][1], 0.30]
        r.update_scene(d, cam)
        return r.render()

    t0 = int(a.start * mo.fps)
    t1 = mo.n_frames - 1 if a.end is None else min(mo.n_frames - 1, int(a.end * mo.fps))
    picks = np.linspace(t0, t1, a.frames).astype(int)
    cols = min(6, a.frames)
    rows = int(np.ceil(a.frames / cols))
    sheet = Image.new("RGB", (cols * w, rows * h + 50), (250, 250, 252))
    font = ImageFont.truetype(FONT, 22) if Path(FONT).exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"{mo.name}  {mo.n_frames} frames @ {mo.fps:g} fps  ({mo.duration:.1f} s)",
              fill=(30, 30, 30), font=font)
    for k, t in enumerate(picks):
        img = Image.fromarray(frame(t))
        sheet.paste(img, ((k % cols) * w, 50 + (k // cols) * h))
        draw.text(((k % cols) * w + 10, 50 + (k // cols) * h + 8), f"{t / mo.fps:.1f}s",
                  fill=(60, 60, 60), font=font)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(f"wrote {a.out}")

    if a.mp4:
        import imageio.v2 as imageio
        with imageio.get_writer(a.mp4, fps=mo.fps, codec="libx264", quality=7) as vid:
            for t in range(t0, t1 + 1):
                vid.append_data(frame(t))
        print(f"wrote {a.mp4}")


if __name__ == "__main__":
    main()
