"""Deployment gate: the vendor's own policies must work through OUR observation builder.

    python tools/check_vendor_policies.py [--robot-dir _ref/robot_119/q1_t2d0]

Runs each vendor MIMIC_OPENSOURCE_V2 policy (rl_models/*.onnx with its npz) in our sim, with
observations from q1_rl.contract.Contract.build_obs -- the function our training env uses.
Those policies were trained on what the robot's runtime actually feeds them, so if our
builder disagrees with the runtime in any field, order, offset or scale, they flail here.

Why this exists: the layout in obs_spec.json was once transcribed from a comment in the
robot's config (116 valid + 3 padding). The runtime builds something else (119, with the
reference motion's gravity at 44). Every check we had compared our code with itself and
passed; a policy trained on that layout convulsed and fell on the robot on 2026-10-05.
This check would have caught it in seconds: with the old layout the vendor's FLY policy
fell in 4 of 4 runs, with the right one 0 of 4.

A policy passes if it ends where its clip ends (base height within 5 cm of the clip's last
frame -- the env's fall flag means nothing for a get-up, which starts lying down) and tracks
the joints within --max-rmse. Exit code 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl.contract import Contract  # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv  # noqa: E402
from q1_rl.motion import Motion  # noqa: E402

QUIET = dict(noise_ang_vel=0.0, noise_ori=0.0, imu_bias=0.0, noise_joint_pos=0.0, noise_joint_vel=0.0,
             act_delay_steps=(0, 0), obs_delay_steps=(0, 0), encoder_offset=0.0)
# policy onnx -> its reference npz, as paired in the robot's motion_list.yaml
PAIRS = {
    "q1_fly_0604.onnx": "q1_fly_0604.npz",
    "q1_getUp_v4.0_20260708.onnx": "q1_getUp_v4.0_20260708.npz",
}


def run(c, model, onnx_path, npz_path, n=4):
    motion = Motion.load_npz(npz_path)
    cfg = replace(EnvConfig(), num_envs=n, backend="mjbatch", random_phase=False, episode_seconds=motion.duration,
                  push_vel=0.0, init_joint_noise=0.0, init_vel_noise=0.0, **QUIET)
    env = MotionTrackingEnv(model, motion, c, cfg, seed=0)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    err = []
    obs = env.obs()
    for _ in range(env.max_steps):
        x = c.pad_for_onnx(obs).astype(np.float32)
        a = np.concatenate([sess.run(None, {name: x[i:i + 1]})[0] for i in range(n)])
        obs, _, _, _, _ = env.step(a)
        err.append(np.sqrt(np.mean((env.qpos[:, 7:] - env.ref_qpos[env.phase]) ** 2)))
    miss = np.abs(env.qpos[:, 2] - motion.body_pos_w[-1, 0, 2])        # final base height vs the clip's
    return int((miss > 0.05).sum()), float(np.mean(err)), motion.duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-dir", type=Path, default=ROOT / "_ref/robot_119/q1_t2d0",
                    help="a copy of the robot's cfg/robot/q1_t2d0 (rl_models/, res/mimic_motion/npz/)")
    ap.add_argument("--max-rmse", type=float, default=0.25, help="rad, mean joint tracking error")
    a = ap.parse_args()

    c = Contract()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/q1/q1_physics.xml"))
    ok = True
    for onnx_name, npz_name in PAIRS.items():
        onnx_path = a.robot_dir / "rl_models" / onnx_name
        npz_path = a.robot_dir / "res/mimic_motion/npz" / npz_name
        if not onnx_path.exists() or not npz_path.exists():
            print(f"  skip {onnx_name}: not in {a.robot_dir}")
            continue
        fell, rmse, dur = run(c, model, onnx_path, npz_path)
        good = fell == 0 and rmse <= a.max_rmse
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {onnx_name:32s} {dur:4.1f}s  ended off the clip in {fell}/4  joint RMSE {rmse:.3f} rad")
    if not ok:
        print("vendor policies do not work through our observation builder: the contract disagrees with the runtime")
        sys.exit(1)
    print("vendor policies track through our observation builder")


if __name__ == "__main__":
    main()
