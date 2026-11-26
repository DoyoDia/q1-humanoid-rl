"""Physical-plausibility numbers for motion npz files, side by side.

    python tools/motion_quality.py runs/motions/a.npz runs/motions/b.npz ...

  jumps        frames where some joint moves > 0.3 rad in one 20 ms step (IK flips)
  over vmax    frames where some joint exceeds its URDF velocity limit
  worst joint  the joint with the most over-limit frames
  pen >5mm     frames whose lowest sole point is > 5 mm under the floor; max depth
  float        median height of the lowest contact point over all frames, mm. Something
               should be on the floor almost always, so this should be near 0;
               tens of mm means the clip hovers and the robot would drop onto it.
These say whether a reference is something the robot could physically follow -- not
whether it looks like the source; for that, render it (tools/render_motion.py).
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl.contract import Contract  # noqa: E402
from q1_rl.geometry import lowest_contact_z  # noqa: E402
from q1_rl.motion import Motion  # noqa: E402

c = Contract()
model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/q1/q1_physics.xml"))
data = mujoco.MjData(model)
qadr = [model.jnt_qposadr[model.joint(n).id] for n in c.joint_names]

print(f"{'motion':28s} {'secs':>5} {'jumps':>6} {'over vmax':>9}  {'worst joint':24s} {'pen>5mm':>8} {'max pen':>8} {'float':>6}")
for path in sys.argv[1:]:
    mo = Motion.load_npz(path)
    q, v = mo.joint_pos, np.abs(mo.joint_vel)
    jumps = int((np.abs(np.diff(q, axis=0)).max(axis=1) > 0.3).sum())
    over = v > c.max_vel
    worst = c.joint_names[int(over.sum(0).argmax())] if over.any() else "-"
    low = np.zeros(mo.n_frames)
    for t in range(mo.n_frames):
        data.qpos[0:3], data.qpos[3:7], data.qpos[qadr] = mo.body_pos_w[t, 0], mo.body_quat_w[t, 0], q[t]
        mujoco.mj_forward(model, data)
        low[t] = lowest_contact_z(model, data)
    print(f"{Path(path).stem:28s} {mo.duration:5.0f} {jumps:6d} {over.any(1).mean() * 100:8.1f}%  {worst:24s} "
          f"{(low < -0.005).mean() * 100:7.1f}% {-low.min() * 1000:6.1f}mm {np.median(low) * 1000:4.1f}mm")
