"""Deployment gate for the Walk contract: a walking policy must follow commands through OUR env.

    python tools/check_vendor_walk.py [--onnx runs/walk_ppo/policy.onnx] [--backend warp]

Default --onnx is the vendor's own q1_walkRun_upperPosCtrl_v3.2. Its observations come from
q1_rl.envs.walk.WalkEnv -- Contract.build_obs, the 5-frame history -- the code our walk
policies train on. The vendor's policy was trained on what the robot's runtime feeds it, so
if WalkEnv disagrees with the runtime in a field, scale, order or the history, it stumbles
or falls here (with a wrong pattern, absolute joint angles or a newest-first history it falls
within 12 s; with unscaled commands it walks at half speed -- obs_spec.json layout_source).

The same script then gates our own exports: any ONNX with the runtime's two inputs.

Script: forward 0.3, sideways 0.3, turn 0.6 rad/s, back 0.2, stand -- 6 s each, ideal sensors,
no pushes. --cpg runtime (default): the robot's gait clock and target handling
(WalkConfig.cpg_law "runtime", no target clamp); --cpg random: the fixed 0.6 s
clock and clamped targets this check used before. Passes if no env falls and the mean velocity over the
second half of every segment is within --tol of the command (RMS over segments and axes).
Exit code 1 otherwise.
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
from q1_rl.envs.motion_tracking import _rotate_local_to_world, _to_heading, _yaw  # noqa: E402
from q1_rl.envs.walk import WalkConfig, WalkEnv  # noqa: E402
from q1_rl.motion import generate  # noqa: E402

VENDOR = ROOT / "_ref/robot_119/q1_t2d0/rl_models/q1_walkRun_upperPosCtrl_v3.2_20260816.onnx"
QUIET = dict(noise_ang_vel=0.0, noise_ori=0.0, imu_bias=0.0, noise_joint_pos=0.0, noise_joint_vel=0.0,
             act_delay_steps=(0, 0), obs_delay_steps=(0, 0), encoder_offset=0.0, push_vel=0.0,
             init_joint_noise=0.0, init_vel_noise=0.0, lift_prob=0.0, mass_scale_range=(1.0, 1.0),
             com_offset=0.0, kp_scale_range=(1.0, 1.0), kd_scale_range=(1.0, 1.0),
             cycle_time_s=(0.6, 0.6), cpg_idle_running=1.0, arm_swing=0.0)
SCRIPT = [((0.3, 0.0, 0.0), 6), ((0.0, 0.3, 0.0), 6), ((0.0, 0.0, 0.6), 6), ((-0.2, 0.0, 0.0), 6), ((0.0, 0.0, 0.0), 6)]


def run(onnx_path, backend="mjbatch", n=4, seed=0, cpg="runtime"):
    c = Contract("walk")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/q1/q1_physics.xml"))
    cfg = replace(WalkConfig(), num_envs=n, backend=backend, episode_seconds=1e4, cpg_law=cpg,
                  clamp_targets=cpg != "runtime", **QUIET)
    env = WalkEnv(model, generate("stand", model, c), c, cfg, seed=seed)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    fd, na, hz = env.frame_dim, env.na, c.policy_hz
    errs, fell = [], np.zeros(n, bool)
    for cmd, secs in SCRIPT:
        env.command[:] = cmd
        env.cmd_left[:] = 10 ** 9
        got = []
        for k in range(secs * hz):
            h = env.obs()
            a = np.concatenate([sess.run(None, {"observations": h[i:i + 1, -fd:], "observation_history": h[i:i + 1]})[0]
                                for i in range(n)])
            _, _, done, f, _ = env.step(a[:, :na])
            fell |= f
            q = env.qpos[:, 3:7]
            if k >= secs * hz // 2:
                v = _to_heading(env.qvel[:, 0:2], _yaw(q))
                got.append(np.concatenate([v, _rotate_local_to_world(q, env.qvel[:, 3:6])[:, 2:3]], axis=1))
        mean = np.mean(got, axis=0)                                    # (n, 3)
        errs.append(mean - np.asarray(cmd))
        print(f"  cmd {cmd}: got {np.round(mean.mean(0).astype(float), 3).tolist()}  (spread {np.round(mean.std(0).astype(float), 3).tolist()})")
    errs = np.stack(errs, axis=1)                                      # (n, segments, 3)
    return float(np.sqrt(np.mean(errs ** 2))), fell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(VENDOR))
    ap.add_argument("--backend", default="mjbatch")
    ap.add_argument("--cpg", default="runtime", choices=("runtime", "random"))
    ap.add_argument("--tol", type=float, default=0.08, help="m/s, rad/s RMS")
    a = ap.parse_args()
    print(f"{a.onnx} through WalkEnv ({a.backend}, cpg {a.cpg})")
    rms, fell = run(a.onnx, a.backend, cpg=a.cpg)
    ok = not fell.any() and rms <= a.tol
    print(f"{'PASS' if ok else 'FAIL'}: velocity error RMS {rms:.3f} (tol {a.tol}), falls {int(fell.sum())}/{fell.size}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
