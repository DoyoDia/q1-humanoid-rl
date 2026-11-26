"""Derive GMR's human -> Q1 IK configs from its human -> G1 ones.

    python retarget/gmr/make_config.py [--source bvh_lafan1|smplx] [--gmr _ref/GMR]
        ->  retarget/gmr/<source>_to_q1.json

bvh_lafan1 is for LAFAN1 BVH clips; smplx for SMPL-X from video pose estimation (PromptHMR).

Nothing is hand-tuned. Two things change between robots, and both are measured here:

  * Scale. GMR shrinks each human body's offset from the hips by a per-body factor so the
    target lands where the robot can reach it. G1's factors were tuned for G1; Q1's are
    G1's times the Q1/G1 length ratio of the matching limb (legs: hip-yaw to ankle;
    arms and torso: the shoulder chain), measured at zero pose.
  * Frame offsets. GMR asks a robot body to take the orientation human_bone * rot_offset.
    If a Q1 body's frame differs from G1's by a constant D at equal joint angles (D read
    off zero-pose FK), Q1's offset is G1's * D. Only hip-yaw differs (10 deg pitch).

Body pairings stay G1's (same link layout). One deliberate change to the weights, measured
on LAFAN1 dance1_subject2 (tools/motion_quality.py; "jumps" = frames where a joint moves
> 0.3 rad in one 20 ms step, i.e. the IK switching arm solutions):

  * arm orientation weights down: hand 10 -> ~0 (Q1 has no wrist joints, so a human hand's
    orientation is not reachable), upper arm 100 -> 1, forearm 10/5 -> 1. Q1's shoulder
    yaw spans +-90 deg against G1's +-150, so heavily weighted arm orientations kept
    asking for twists Q1 cannot make and the IK flipped between solutions.
    590 jumps -> 180, frames over a joint's speed limit 2.5% -> 0.1%; walk1 unaffected.
    (Exactly 0 makes GMR drop the entry and then fail looking it up, hence 1e-4.)
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]

LEG = ("left_hip_yaw_link", "left_ankle_roll_link")
ARM = ("left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
       "left_elbow_link", "left_wrist_yaw_link")
# human bones scaled by the leg ratio (the rest take the arm/torso one), per source skeleton
LEG_BONES = {
    "bvh_lafan1": ("Hips", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFootMod", "RightFootMod"),
    "smplx": ("pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_foot", "right_foot"),
}


def fk(xml):
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return m, d


def chain_length(m, d, bodies):
    p = [d.xpos[m.body(b).id] for b in bodies]
    return float(sum(np.linalg.norm(a - b) for a, b in zip(p, p[1:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(LEG_BONES), default="bvh_lafan1")
    ap.add_argument("--gmr", type=Path, default=ROOT / "_ref/GMR")
    ap.add_argument("--q1-model", type=Path, default=ROOT / "_ref/robot_retargeter/asset/robot/q1/q1.xml",
                    help="made by retarget/make_model.py")
    ap.add_argument("--out", type=Path, default=None, help="default: <source>_to_q1.json next to this script")
    a = ap.parse_args()
    out = a.out or Path(__file__).with_name(f"{a.source}_to_q1.json")

    g1_cfg = json.loads((a.gmr / f"general_motion_retargeting/ik_configs/{a.source}_to_g1.json").read_text())
    G, gd = fk(a.gmr / "assets/unitree_g1/g1_mocap_29dof.xml")
    Q, qd = fk(a.q1_model)

    leg = chain_length(Q, qd, LEG) / chain_length(G, gd, LEG)
    arm = chain_length(Q, qd, ARM) / chain_length(G, gd, ARM)
    cfg = copy.deepcopy(g1_cfg)
    for bone, s in g1_cfg["human_scale_table"].items():
        cfg["human_scale_table"][bone] = round(s * (leg if bone in LEG_BONES[a.source] else arm), 4)

    offsets = {}
    for table in ("ik_match_table1", "ik_match_table2"):
        for body, entry in cfg[table].items():
            D = (R.from_quat(gd.xquat[G.body(body).id], scalar_first=True).inv()
                 * R.from_quat(qd.xquat[Q.body(body).id], scalar_first=True))
            off = R.from_quat(entry[4], scalar_first=True) * D
            entry[4] = [round(float(x), 8) for x in off.as_quat(scalar_first=True)]
            offsets[body] = np.degrees(D.magnitude())

    for table in ("ik_match_table1", "ik_match_table2"):
        for body, entry in cfg[table].items():
            if "wrist_yaw" in body:
                entry[2] = 1e-4
            elif "shoulder_yaw" in body or "elbow" in body:
                entry[2] = 1

    out.write_text(json.dumps(cfg, indent=4) + "\n")
    print(f"limb ratios Q1/G1: legs {leg:.3f}, arms {arm:.3f}")
    print("scales:", cfg["human_scale_table"])
    print("frame offsets applied (deg):", {k: round(v, 1) for k, v in offsets.items() if v > 0.05})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
