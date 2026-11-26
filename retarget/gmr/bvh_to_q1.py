"""Retarget a LAFAN1 BVH onto Q1 with GMR, headless, and write the same CSV robot_retargeter
does (base xyz, base quat xyzw, joints in model order) so retarget/to_motion.py takes it.

    python retarget/gmr/bvh_to_q1.py _ref/lafan1/walk1_subject1.bvh -o out.csv

GMR's own scripts open a viewer; this drives its GeneralMotionRetargeting class directly
and registers Q1 in its tables at runtime, so the GMR checkout stays unmodified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bvh", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--gmr", type=Path, default=ROOT / "_ref/GMR")
    ap.add_argument("--q1-model", type=Path, default=ROOT / "_ref/robot_retargeter/asset/robot/q1/q1.xml")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("bvh_lafan1_to_q1.json"))
    ap.add_argument("--no-velocity-limit", action="store_true",
                    help="let the IK jump between solutions; by default each joint is capped at its URDF speed")
    a = ap.parse_args()

    sys.path.insert(0, str(a.gmr))
    import general_motion_retargeting.params as params
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    from tqdm import tqdm

    params.ROBOT_XML_DICT["q1"] = a.q1_model
    params.IK_CONFIG_DICT["bvh_lafan1"]["q1"] = a.config
    params.ROBOT_BASE_DICT["q1"] = "pelvis"
    params.VIEWER_CAM_DISTANCE_DICT["q1"] = 1.2

    frames, height = load_bvh_file(str(a.bvh), format="lafan1")
    gmr = GeneralMotionRetargeting(src_human="bvh_lafan1", tgt_robot="q1",
                                   actual_human_height=height, verbose=False)
    if not a.no_velocity_limit:
        # GMR's own use_velocity_limit names G1's motors, so add Q1's here: the URDF speed
        # of every joint. Without it the IK hops between arm solutions -- 590 frames of
        # dance1 moved some joint > 0.3 rad in a single 20 ms step.
        import mink
        sys.path.insert(0, str(ROOT))
        from q1_rl.contract import Contract
        c = Contract()
        gmr.ik_limits.append(mink.VelocityLimit(gmr.model, dict(zip(c.joint_names, c.max_vel))))
    rows = []
    for f in tqdm(frames, desc=a.bvh.stem):
        q = np.asarray(gmr.retarget(f), np.float64).copy()
        q[3:7] = q[[4, 5, 6, 3]]                        # wxyz -> xyzw, as robot_retargeter writes
        rows.append(q)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(a.out, np.asarray(rows), delimiter=",", fmt="%.6f")
    print(f"wrote {a.out} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
