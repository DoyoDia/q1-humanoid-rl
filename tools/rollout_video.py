"""Play a trained policy through a whole motion in physics, the way the robot would run it,
and render it with the reference pose as a ghost walking alongside.

    python tools/rollout_video.py runs/walk1_30s_ppo -o walk.mp4 [--seconds 30] [--clean]
    python tools/rollout_video.py runs/bt_v2 --onnx runs/bt_v2/deploy/q1rl_bt.onnx --no-ghost \
        --azimuth 5 --size 600x720 -o sim2sim.mp4

--onnx runs the exported deployment file (tools/make_deploy_package.py) through onnxruntime
instead of policy.pt -- sim2sim of exactly what goes on the robot. --no-ghost shows the
robot alone (e.g. beside a separate reference render); --azimuth and --size set the camera.

Deployment-shaped: starts at frame 0 (not a random phase), plays the clip once, no resets,
with the env's sensor noise and latency unless --clean. Prints whether it fell and the joint
tracking error; writes an mp4 with the policy's robot solid and the reference pose as a
translucent ghost beside it (same heading), so the poses can be compared frame by frame.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# glvnd otherwise hands EGL to Mesa, whose llvmpipe renders on the CPU at ~6 fps (1300 on the GPU)
_NV_EGL = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.isfile(_NV_EGL):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NV_EGL)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl.algos.ppo import ActorCritic  # noqa: E402
from q1_rl.contract import Contract  # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv, _quat_mul, _yaw  # noqa: E402
from q1_rl.evaluate import deterministic_policy  # noqa: E402
from q1_rl.motion import Motion  # noqa: E402

QUIET = dict(noise_ang_vel=0.0, noise_ori=0.0, imu_bias=0.0, noise_joint_pos=0.0, noise_joint_vel=0.0,
             act_delay_steps=(0, 0), obs_delay_steps=(0, 0), encoder_offset=0.0)


def visual_scene(xml: Path, w: int, h: int):
    """Two robots in one scene: the policy's, and a translucent copy for the reference."""
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.meshdir = str((xml.parent / spec.meshdir).resolve()) + "/"
    spec.visual.global_.offwidth, spec.visual.global_.offheight = w, h
    ghost = mujoco.MjSpec.from_file(str(xml))
    ghost.meshdir = spec.meshdir
    for g in list(ghost.worldbody.geoms):
        ghost.delete(g)
    for g in ghost.geoms:
        g.contype = g.conaffinity = 0
        g.rgba = [0.3, 0.55, 0.95, 0.25] if g.type == mujoco.mjtGeom.mjGEOM_MESH else [0, 0, 0, 0]
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.rgba = [0.86, 0.87, 0.89, 1.0]
        elif g.type != mujoco.mjtGeom.mjGEOM_PLANE:
            g.rgba[3] = 0.0
    frame = spec.worldbody.add_frame()
    frame.attach_body(ghost.worldbody.first_body(), "ref_", "")
    return spec.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="a train.py output dir (policy.pt + motion.npz); with --sm, any name")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=None, help="default: the whole clip")
    ap.add_argument("--clean", action="store_true", help="no sensor noise or latency")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visual", type=Path, default=ROOT / "robots/q1/q1_visual.xml")
    ap.add_argument("--onnx", type=Path, default=None, help="run this deployment onnx instead of policy.pt")
    ap.add_argument("--no-ghost", action="store_true", help="do not draw the reference beside the robot")
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--distance", type=float, default=2.0)
    ap.add_argument("--size", default="960x720", help="video WxH")
    ap.add_argument("--motion", type=Path, default=None, help="motion npz (default: RUN/motion.npz)")
    ap.add_argument("--sm", type=Path, default=None, help="play a SuperMimic checkpoint instead (tools/eval_supermimic.py)")
    ap.add_argument("--sm-npz", type=Path, default=None, help="the clip in SuperMimic's npz format, with --sm")
    a = ap.parse_args()

    c = Contract()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/q1/q1_physics.xml"))
    motion = Motion.load_npz(a.motion or a.run / "motion.npz")
    secs = a.seconds or motion.duration
    cfg = replace(EnvConfig(), num_envs=1, random_phase=False, episode_seconds=secs, push_vel=0.0,
                  init_joint_noise=0.0, init_vel_noise=0.0, **(QUIET if a.clean else {}))
    env = MotionTrackingEnv(model, motion, c, cfg, seed=a.seed)
    if a.sm:
        sys.path.insert(0, str(ROOT / "tools"))
        from eval_supermimic import SMAgent
        agent = SMAgent(a.sm, a.sm_npz, c, 1, noise=not a.clean, seed=a.seed,
                        device="cuda" if torch.cuda.is_available() else "cpu")

        def act(o):
            return torch.as_tensor(agent.act(env))
    elif a.onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(a.onnx))
        iname = sess.get_inputs()[0].name

        def act(o):
            return torch.as_tensor(sess.run(None, {iname: o.numpy().astype(np.float32).reshape(1, -1)})[0]).reshape(1, -1)
    else:
        net = ActorCritic.from_state_dict(torch.load(a.run / "policy.pt", map_location="cpu"))
        act = deterministic_policy(net.eval())

    w, h = (int(x) for x in a.size.split("x"))
    vm = visual_scene(a.visual, w, h) if a.out else None
    vd = mujoco.MjData(vm) if vm else None
    r = mujoco.Renderer(vm, h, w) if vm else None
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = a.distance, a.azimuth, -12
    az = np.radians(cam.azimuth)                 # the camera looks along (cos az, sin az)
    screen_right = np.array([np.sin(az), -np.cos(az)])
    nq = model.nq
    qadr_ref = [vm.jnt_qposadr[vm.joint("ref_" + n).id] for n in c.joint_names] if vm else None
    ref_free = vm.jnt_qposadr[vm.joint("ref_floating_base_joint").id] if vm else None

    frames, errs, fell_at = [], [], None
    went = should = 0.0         # path lengths: LAFAN walks loop back, so start-to-end says nothing
    obs = env.obs()
    with torch.no_grad():
        for t in range(env.max_steps):
            ph = int(env.phase[0])
            before = env.qpos[0, 0:2].copy()
            obs, _, done, fell, _ = env.step(act(torch.as_tensor(obs)).numpy())
            errs.append(float(np.sqrt(np.mean((env.qpos[0, 7:] - env.ref_qpos[env.phase[0]]) ** 2))))
            went += float(np.linalg.norm(env.qpos[0, 0:2] - before))
            should += float(np.linalg.norm(motion.body_pos_w[int(env.phase[0]), 0, :2] - motion.body_pos_w[ph, 0, :2]))
            if vm is not None:
                vd.qpos[:nq] = env.qpos[0]
                # The ghost stands to the right of the robot as the camera sees it, turned
                # to the robot's heading, so the two poses stay side by side. Placed at the
                # reference's absolute position it drifted out of frame within seconds (over
                # 30 s the two headings diverge); placed to the robot's own left it orbited
                # the robot whenever the robot turned, passing in front of the camera and
                # through the robot -- a dance turns a lot.
                rob_q, ref_q = env.qpos[0, 3:7], motion.body_quat_w[ph, 0]
                dyaw = _yaw(rob_q) - _yaw(ref_q)
                turn = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
                vd.qpos[ref_free:ref_free + 2] = env.qpos[0, 0:2] + 0.7 * screen_right
                vd.qpos[ref_free + 2] = motion.body_pos_w[ph, 0, 2] - (50.0 if a.no_ghost else 0.0)
                vd.qpos[ref_free + 3:ref_free + 7] = _quat_mul(turn, ref_q)
                vd.qpos[qadr_ref] = motion.joint_pos[ph]
                mujoco.mj_forward(vm, vd)
                off = 0.0 if a.no_ghost else 0.35
                cam.lookat[:] = [*(env.qpos[0, 0:2] + off * screen_right), 0.32]   # between the two
                r.update_scene(vd, cam)
                frames.append(r.render())
            if fell[0]:
                fell_at = (t + 1) / c.policy_hz
                break

    tag = "clean" if a.clean else "noise+latency"
    print(f"{a.run.name}  {tag}  {secs:.1f}s from frame 0: "
          + (f"FELL at {fell_at:.1f}s" if fell_at else "no fall")
          + f", joint RMSE {np.mean(errs):.3f} rad, walked {went:.1f} m of the reference's {should:.1f} m")
    if a.out:
        import imageio.v2 as imageio
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(a.out, fps=c.policy_hz, codec="libx264", quality=7) as vid:
            for f in frames:
                vid.append_data(f)
        print(f"wrote {a.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
