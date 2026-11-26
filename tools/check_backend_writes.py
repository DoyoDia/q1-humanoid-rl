#!/usr/bin/env python3
"""Regression guard: every write an environment makes must reach the simulation.

The warp backend once marked a mirror dirty only in bind() and cleared the mark on the
first upload, so every later write was dropped unless a reset happened to re-mark the
field in the same step. In large batches resets happen almost every step, which hid it:
the backend comparison looked behaviourally equivalent while commands were being ignored.

This test writes WITHOUT any reset in between -- a joint command, a per-world model
parameter, a velocity push -- and checks each one lands, on every available backend.

    python3 tools/check_backend_writes.py      # exit 1 on any failure
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from q1_rl.backends import available, make_backend  # noqa: E402


def check(name, model):
    n, key = 32, model.key("stand").id
    knee = [i for i in range(model.nu)
            if "left_knee" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)][0]
    b = make_backend(name, model, n, expand_fields=["geom_friction"])
    qpos, qvel, ctrl = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl")
    fr = b.expand("geom_friction")
    b.reset(np.arange(n), keyframe=key)
    b.forward()
    ctrl[:] = qpos[:, 7:]
    b.step(nstep=20)                                   # the first upload happens here

    k0 = float(qpos[:, 7 + knee].mean())
    ctrl[:, knee] += 0.6
    fr[:, :, 0] = 0.123
    for _ in range(10):
        b.step(nstep=20)
    moved = float(qpos[:, 7 + knee].mean()) - k0
    held = (float(b._src["geom_friction"].numpy()[:, :, 0].mean()) if hasattr(b, "_src")
            else float(fr[:, :, 0].mean()))
    vx0 = float(qvel[:, 0].mean())
    qvel[:, 0] += 1.0
    b.step(nstep=1)
    pushed = float(qvel[:, 0].mean()) - vx0

    results = [("joint command reaches the sim", moved > 0.4, f"knee moved {moved:+.3f} rad of +0.600"),
               ("model parameter reaches the sim", abs(held - 0.123) < 1e-6, f"friction {held:.3f}, wrote 0.123"),
               ("velocity push reaches the sim", pushed > 0.8, f"base vx changed {pushed:+.2f}, pushed +1.00")]
    return results


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots" / "q1" / "q1_physics.xml"))
    failed = 0
    for name, ok in available().items():
        if not ok:
            print(f"{name}: not available here, skipped")
            continue
        print(name)
        for label, passed, detail in check(name, model):
            print(f"  {'ok  ' if passed else 'FAIL'} {label}: {detail}")
            failed += not passed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
