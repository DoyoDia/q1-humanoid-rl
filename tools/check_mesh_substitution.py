#!/usr/bin/env python3
"""Measure what replacing the colliding foot meshes with boxes actually costs.

Background: q1_physics.xml drops the vendor's visual meshes and converts its *colliding*
meshes to bounding boxes. An earlier version of the builder deleted the colliding meshes
along with the visual ones. A standing rollout could not tell the difference -- the robot
rests on four 2 mm contact spheres per foot, so the foot mesh never touches the ground --
and that is exactly why the bug survived a 500-step check. Tumbling tells the difference
immediately.

This builds three models that differ ONLY in how mesh geoms are handled, with identical
actuators, integrator and timestep, and reports trajectory divergence from the faithful one.

    python3 tools/check_mesh_substitution.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "robots" / "q1"))
sys.path.insert(0, str(ROOT))

from build_model import (demote_undercutting_boxes, handle_meshes,  # noqa: E402
                         load_contract, to_position_actuators)

SRC = ROOT / "_ref" / "q1_v3" / "xml" / "q1_v3_mc.xml"


def build(mode: str):
    joints, gains, rates, _ = load_contract()
    spec = mujoco.MjSpec.from_file(str(SRC))
    spec.option.timestep = rates["sim"]["timestep"]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if mode == "keep":                       # q1_physics.xml: colliding meshes kept
        pass
    elif mode == "box":                      # q1_web.xml: colliding meshes boxed
        handle_meshes(spec, self_contained=True, meshdir=str(ROOT / "robots" / "q1" / "meshes"))
        demote_undercutting_boxes(spec)
    elif mode == "delete":                   # the bug this tool exists to catch
        for geom in list(spec.geoms):
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH or geom.meshname:
                spec.delete(geom)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
    to_position_actuators(spec, joints, gains["mimic_opensource_v2"])
    return spec.compile()


def rollout(model, seed, steps, stand):
    d = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    d.qpos[2] = 0.30
    q = np.array([0.9, 0.3, 0.2, 0.1]); d.qpos[3:7] = q / np.linalg.norm(q)
    d.qpos[7:] = stand
    d.qvel[:3] = [0.8, -0.5, -1.0]
    d.qvel[3:6] = [2.0, 1.0, -1.5]
    ctrl = rng.uniform(-0.4, 0.4, (steps, model.nu)) + stand
    out = np.empty((steps, model.nq))
    for i in range(steps):
        d.ctrl[:] = ctrl[i]
        mujoco.mj_step(model, d)
        out[i] = d.qpos
    return out


def main():
    _, gains, _, _ = load_contract()
    stand = np.array(gains["mimic_opensource_v2"]["default_dof_pos"])
    models = {m: build(m) for m in ("keep", "box", "delete")}
    for name, m in models.items():
        coll = sum(1 for g in range(m.ngeom)
                   if m.geom_bodyid[g] != 0 and m.geom_conaffinity[g])
        print(f"  {name:7s} ngeom={m.ngeom:3d}  floor-touching geoms={coll}")

    print("\n  3 s tumbling rollouts, divergence from 'keep' (max |qpos| over the trajectory)")
    print(f"  {'seed':>4}  {'box':>12}  {'delete':>12}")
    worst = {"box": 0.0, "delete": 0.0}
    for seed in range(4):
        ref = rollout(models["keep"], seed, 3000, stand)
        row = {}
        for name in ("box", "delete"):
            r = rollout(models[name], seed, 3000, stand)
            row[name] = float(np.abs(r - ref).max())
            worst[name] = max(worst[name], row[name])
        print(f"  {seed:>4}  {row['box']:>12.4f}  {row['delete']:>12.4f}")

    print(f"\n  worst:  box {worst['box']:.4f} rad   delete {worst['delete']:.4f} rad")
    print("""
  Standing is identical under all three: the feet rest on the vendor's four contact
  spheres per foot, which none of these variants touches. That is precisely why a
  standing-only check cannot catch a wrong collision model, and why this tool tumbles.

  q1_physics.xml is the 'keep' column -- the training and deployment model carries the
  real colliding meshes, so it has no substitution error at all. q1_web.xml is the 'box'
  column: the browser needs a model with no external assets, and the number above is what
  that costs. 'delete' is the bug an earlier build had, kept here as a regression guard.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
