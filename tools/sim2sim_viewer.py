"""Watch a policy in an interactive MuJoCo window, in real time: sim2sim of the deployment file.

    python tools/sim2sim_viewer.py runs/bt_v3            # runs/<run>/deploy/q1rl_*.onnx if there is one
    python tools/sim2sim_viewer.py runs/bt_v3 --pt       # policy.pt instead
    python tools/sim2sim_viewer.py runs/bt_v3 --start 200 --clean

Same physics, sensor noise and latency as the training env and tools/rollout_video.py, the
robot solid and the reference as a translucent ghost beside it. Plays the clip from --start
to its end, then again from --start. Mouse: rotate/pan/zoom as in MuJoCo's viewer; the
camera follows the robot unless you grab it. Space pauses (MuJoCo's own key).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ["MUJOCO_GL"] = "glfw"                 # a window, not the offscreen EGL renderers use

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl.algos.ppo import ActorCritic  # noqa: E402
from q1_rl.contract import Contract  # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv, _quat_mul, _yaw  # noqa: E402
from q1_rl.evaluate import deterministic_policy  # noqa: E402
from q1_rl.motion import Motion  # noqa: E402
from tools.rollout_video import QUIET, visual_scene  # noqa: E402


def policy_fn(run: Path, c: Contract, use_pt: bool):
    onnx = sorted((run / "deploy").glob("q1rl_*.onnx")) if (run / "deploy").exists() else []
    onnx = [p for p in onnx if not p.name.endswith(".detector.onnx")]
    if onnx and not use_pt:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx[0]), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return (lambda o: sess.run(None, {name: o.astype(np.float32)})[0]), f"{onnx[0].name} (onnxruntime)"
    net = ActorCritic.from_state_dict(torch.load(run / "policy.pt", map_location="cpu"))
    act = deterministic_policy(net.eval())
    return (lambda o: act(torch.as_tensor(o)).numpy()), "policy.pt (torch)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--pt", action="store_true", help="policy.pt instead of the deployment onnx")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the clip")
    ap.add_argument("--clean", action="store_true", help="no sensor noise or latency")
    ap.add_argument("--no-ghost", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    a = ap.parse_args()

    c = Contract()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/q1/q1_physics.xml"))
    motion = Motion.load_npz(a.run / "motion.npz")
    act, what = policy_fn(a.run, c, a.pt)
    start = int(a.start * c.policy_hz)
    if start:                                    # play from there: the clip itself starts there
        motion = replace(motion, **{k: getattr(motion, k)[start:] for k in (
            "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")})
    clip = motion
    cfg = replace(EnvConfig(), num_envs=1, random_phase=False, episode_seconds=motion.duration, push_vel=0.0,
                  init_joint_noise=0.0, init_vel_noise=0.0, **(QUIET if a.clean else {}))

    vm = visual_scene(ROOT / "robots/q1/q1_visual.xml", 1280, 720)
    vd = mujoco.MjData(vm)
    nq = model.nq
    qadr_ref = [vm.jnt_qposadr[vm.joint("ref_" + n).id] for n in c.joint_names]
    ref_free = vm.jnt_qposadr[vm.joint("ref_floating_base_joint").id]

    print(f"{a.run.name}: {what}, clip {motion.duration:.1f} s, from {a.start:.0f} s"
          f"{' (no noise/latency)' if a.clean else ', with sensor noise and latency'}", flush=True)
    with mujoco.viewer.launch_passive(vm, vd) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 5.0, -12
        episode = 0
        while viewer.is_running():
            env = MotionTrackingEnv(model, clip, c, cfg, seed=episode)
            obs = env.obs()
            t_wall = time.perf_counter()
            for t in range(env.max_steps):
                if not viewer.is_running():
                    break
                ph = int(env.phase[0])
                obs, _, done, fell, _ = env.step(act(obs))
                vd.qpos[:nq] = env.qpos[0]
                if a.no_ghost:
                    vd.qpos[ref_free + 2] = -50.0
                else:
                    rob_q, ref_q = env.qpos[0, 3:7], motion.body_quat_w[ph, 0]
                    dyaw = _yaw(rob_q) - _yaw(ref_q)
                    turn = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
                    vd.qpos[ref_free:ref_free + 2] = env.qpos[0, 0:2] + np.array([0.0, -0.8])
                    vd.qpos[ref_free + 2] = motion.body_pos_w[ph, 0, 2]
                    vd.qpos[ref_free + 3:ref_free + 7] = _quat_mul(turn, ref_q)
                    vd.qpos[qadr_ref] = motion.joint_pos[ph]
                mujoco.mj_forward(vm, vd)
                with viewer.lock():
                    viewer.cam.lookat[:] = [env.qpos[0, 0], env.qpos[0, 1] - (0 if a.no_ghost else 0.4), 0.35]
                viewer.sync()
                if fell[0]:
                    print(f"  fell at {a.start + (ph + 1) / c.policy_hz:.1f} s", flush=True)
                    time.sleep(1.5)
                    break
                t_wall += 1.0 / (c.policy_hz * a.speed)
                time.sleep(max(0.0, t_wall - time.perf_counter()))
                if done[0]:
                    break
            episode += 1


if __name__ == "__main__":
    main()
