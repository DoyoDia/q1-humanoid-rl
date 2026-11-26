"""Write the Q1 model robot_retargeter needs: our visual MJCF plus its marker bodies.

    python retarget/make_model.py --retargeter _ref/robot_retargeter

robot_retargeter scales a source skeleton link by link onto the target robot and solves
IK against named bodies. Besides the robot's own links it expects a few massless,
collision-free marker bodies -- the same ones its G1 model carries:

    hips_sphere                 pelvis, midway between the hip-roll joints
    neck_sphere, head_sphere    torso, between the shoulders; head centre
    {left,right}_foot_end_link  heel, at the sole
    {left,right}_toe_link       toe, at the sole

Positions come from Q1's own geometry (the sole contact spheres, the head mesh), not from
G1's numbers. Q1's link frames are world-aligned at zero pose like G1's, so the
retargeter can use the G1 orientation maps unchanged (see q1.yaml).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent

# (parent body, marker name, position in the parent's frame)
MARKERS = [
    ("pelvis", "hips_sphere", (0.002, 0.0, -0.023)),        # hip-roll joint height
    ("torso_link", "neck_sphere", (-0.010, 0.0, 0.178)),    # shoulder-roll height
    ("head_link", "head_sphere", (0.004, 0.0, 0.011)),      # head mesh centre
    ("left_ankle_roll_link", "left_foot_end_link", (-0.05, 0.0, -0.041)),   # sole spheres'
    ("left_ankle_roll_link", "left_toe_link", (0.095, 0.0, -0.041)),        # bottom, heel/toe
    ("right_ankle_roll_link", "right_foot_end_link", (-0.05, 0.0, -0.041)),
    ("right_ankle_roll_link", "right_toe_link", (0.095, 0.0, -0.041)),
]


def build(src: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(src))
    spec.meshdir = str((src.parent / spec.meshdir).resolve()) + "/"
    for g in list(spec.worldbody.geoms):        # the retargeter adds its own ground
        spec.delete(g)
    for parent, name, pos in MARKERS:
        b = spec.body(parent).add_body(name=name, pos=list(pos))
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0,
                   conaffinity=0, group=1, density=0, rgba=[1, 0, 0, 0.6])
    return spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=ROOT / "robots/q1/q1_visual.xml",
                   help="build with: python robots/q1/build_model.py --keep-meshes --out robots/q1/q1_visual.xml")
    p.add_argument("--retargeter", type=Path, default=ROOT / "_ref/robot_retargeter")
    a = p.parse_args()

    spec = build(a.src)
    model = spec.compile()
    out = a.retargeter / "asset/robot/q1/q1.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    cfg = a.retargeter / "config/robot/q1.yaml"
    shutil.copy(Path(__file__).with_name("q1.yaml"), cfg)
    print(f"wrote {out} ({model.nbody - 1} bodies, {model.njnt} joints) and {cfg}")


if __name__ == "__main__":
    main()
