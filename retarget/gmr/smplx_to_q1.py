"""Retarget an SMPL-X motion (from video pose estimation) onto Q1 with GMR, headless, and write
the same CSV robot_retargeter does (base xyz, base quat xyzw, joints in model order) so
retarget/to_motion.py takes it.

    python retarget/gmr/smplx_to_q1.py motion.npz -o out.csv --body-models <dir with smplx/>

motion.npz is AMASS-style (pose_body, root_orient, trans, betas, gender, mocap_frame_rate),
which is what video2robot's extract step writes from PromptHMR. The CSV keeps the source
frame rate: GMR's own resampler only skips whole frames (int(src/tgt)), which is 0 for a
24 fps video asked for 30; to_motion.py resamples to 50 Hz properly.
Like bvh_to_q1.py, Q1 is registered in GMR's tables at runtime and each joint is capped at
its URDF speed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_smplx(path, body_models):
    """GMR's load_smplx_file, but with every per-frame input given the batch size: newer smplx
    builds a default expression of batch 1 and fails to concatenate it with N frames of betas."""
    import smplx
    import torch
    d = dict(np.load(path, allow_pickle=False))
    n = d["pose_body"].shape[0]
    model = smplx.create(str(body_models), "smplx", gender=str(d["gender"]), use_pca=False)
    betas = np.asarray(d["betas"], np.float32).reshape(-1)[:model.num_betas]
    z = lambda k: torch.zeros(n, k)
    out = model(betas=torch.from_numpy(betas)[None].repeat(n, 1),
                global_orient=torch.from_numpy(d["root_orient"]).float(),
                body_pose=torch.from_numpy(d["pose_body"]).float(),
                transl=torch.from_numpy(d["trans"]).float(),
                left_hand_pose=z(45), right_hand_pose=z(45), jaw_pose=z(3), leye_pose=z(3), reye_pose=z(3),
                expression=z(model.num_expression_coeffs), return_full_pose=True)
    height = 1.66 + 0.1 * float(betas[0])
    return d, model, out, height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smplx", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--body-models", type=Path, required=True, help="directory holding smplx/SMPLX_*.pkl")
    ap.add_argument("--gmr", type=Path, default=ROOT / "_ref/GMR")
    ap.add_argument("--q1-model", type=Path, default=ROOT / "_ref/robot_retargeter/asset/robot/q1/q1.xml")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("smplx_to_q1.json"))
    ap.add_argument("--no-velocity-limit", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(a.gmr))
    import general_motion_retargeting.params as params
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
    from tqdm import tqdm

    params.ROBOT_XML_DICT["q1"] = a.q1_model
    params.IK_CONFIG_DICT["smplx"]["q1"] = a.config
    params.ROBOT_BASE_DICT["q1"] = "pelvis"
    params.VIEWER_CAM_DISTANCE_DICT["q1"] = 1.2

    data, body_model, out, height = load_smplx(a.smplx, a.body_models)
    fps = float(np.asarray(data["mocap_frame_rate"]).reshape(-1)[0])
    frames, fps = get_smplx_data_offline_fast(data, body_model, out, tgt_fps=fps)
    gmr = GeneralMotionRetargeting(src_human="smplx", tgt_robot="q1", actual_human_height=height, verbose=False)
    if not a.no_velocity_limit:
        import mink
        sys.path.insert(0, str(ROOT))
        from q1_rl.contract import Contract
        c = Contract()
        gmr.ik_limits.append(mink.VelocityLimit(gmr.model, dict(zip(c.joint_names, c.max_vel))))

    rows = []
    for f in tqdm(frames, desc=a.smplx.stem):
        q = np.asarray(gmr.retarget(f), np.float64).copy()
        q[3:7] = q[[4, 5, 6, 3]]                        # wxyz -> xyzw, as robot_retargeter writes
        rows.append(q)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(a.out, np.asarray(rows), delimiter=",", fmt="%.6f")
    print(f"wrote {a.out} ({len(rows)} frames @ {fps:g} fps, human height {height:.2f} m)")


if __name__ == "__main__":
    main()
