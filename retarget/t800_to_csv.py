"""T800 motion npz (Isaac / bvh_tool format) -> the robot-motion CSV robot_retargeter replays.

    python retarget/t800_to_csv.py clip_t800.npz -o runs/t800/clip_t800.csv

The npz holds joint_pos in Isaac joint order (joint_names), body_pos_w / body_quat_w (wxyz)
in Isaac body order (body 0 = LINK_BASE) and fps. The CSV is one row per frame: base xyz,
base quaternion xyzw, then the joints in the retargeter T800 model's qpos order. As a check,
forward kinematics of the written rows on that model is compared with the npz body poses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ISAAC_BODY_ORDER = [
    "LINK_BASE", "LINK_HIP_PITCH_L", "LINK_HIP_PITCH_R", "LINK_WAIST_YAW", "LINK_HIP_ROLL_L",
    "LINK_HIP_ROLL_R", "LINK_SHOULDER_PITCH_L", "LINK_SHOULDER_PITCH_R", "LINK_HEAD_PITCH",
    "LINK_HIP_YAW_L", "LINK_HIP_YAW_R", "LINK_SHOULDER_ROLL_L", "LINK_SHOULDER_ROLL_R",
    "LINK_HEAD_YAW", "LINK_KNEE_PITCH_L", "LINK_KNEE_PITCH_R", "LINK_SHOULDER_YAW_L",
    "LINK_SHOULDER_YAW_R", "LINK_ANKLE_PITCH_L", "LINK_ANKLE_PITCH_R", "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_PITCH_R", "LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R", "LINK_ELBOW_YAW_L",
    "LINK_ELBOW_YAW_R", "LINK_FOOT_L", "LINK_FOOT_R", "LINK_WRIST_END_L", "LINK_WRIST_END_R",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "_ref/robot_retargeter/asset/robot/t800/xml/serial_t800.xml")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=False)
    names = [str(n) for n in z["joint_names"]]
    jp = z["joint_pos"].astype(np.float64)
    bpos, bquat = z["body_pos_w"].astype(np.float64), z["body_quat_w"].astype(np.float64)
    fps = float(np.asarray(z["fps"]).ravel()[0])

    m = mujoco.MjModel.from_xml_path(str(a.model))
    hinge = sorted((j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE),
                   key=lambda j: m.jnt_qposadr[j])
    model_names = [m.jnt(j).name for j in hinge]
    missing = set(model_names) ^ set(names)
    if missing:
        raise SystemExit(f"joint names differ between npz and model: {sorted(missing)}")
    cols = [names.index(n) for n in model_names]

    q = bquat[:, 0] / np.linalg.norm(bquat[:, 0], axis=1, keepdims=True)
    rows = np.concatenate([bpos[:, 0], q[:, [1, 2, 3, 0]], jp[:, cols]], axis=1)

    # check: FK of the rows against the npz bodies (LINK_FOOT_* is a dummy point that sits
    # 64.5 mm apart in the two T800 models, so it is left out)
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[j] for j in hinge]
    bid = [m.body(n).id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0 else -1 for n in ISAAC_BODY_ORDER]
    bid = [-1 if n.startswith("LINK_FOOT_") else b for n, b in zip(ISAAC_BODY_ORDER, bid)]
    err = []
    for t in range(len(rows)):
        d.qpos[0:3], d.qpos[3:7], d.qpos[adr] = bpos[t, 0], q[t], jp[t, cols]
        mujoco.mj_kinematics(m, d)
        err.append(max(np.linalg.norm(d.xpos[b] - bpos[t, i]) for i, b in enumerate(bid) if b >= 0))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(a.out, rows, delimiter=",", fmt="%.6f")
    print(f"wrote {a.out}: {len(rows)} frames at {fps:g} fps; FK vs npz bodies max {1000 * max(err):.2f} mm "
          f"(median {1000 * np.median(err):.2f})")


if __name__ == "__main__":
    main()
