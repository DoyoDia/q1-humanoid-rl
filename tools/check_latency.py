"""Check the env's latency, noise and randomisation model does what EnvConfig says.

    python tools/check_latency.py [--backend mjbatch|warp]

1. With every delay and noise at zero, the split control period must reproduce a plain
   step(decimation) exactly -- the event schedule must not change the physics.
2. With an action delay of d substeps, the new target must reach ctrl only after d
   substeps: run the same command with d=0 and d=10 and compare a knee's motion.
3. With an observation delay of k substeps, the observed joint positions must equal the
   true ones from k substeps before the end of the period.
4. Noise must reach the observation and only the observation: true state untouched.
Exit code 0 when all hold.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402

from q1_rl.contract import Contract  # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv  # noqa: E402
from q1_rl.motion import generate  # noqa: E402

QUIET = dict(noise_ang_vel=0.0, noise_ori=0.0, imu_bias=0.0, noise_joint_pos=0.0,
             noise_joint_vel=0.0, act_delay_steps=(0, 0), obs_delay_steps=(0, 0),
             push_vel=0.0, init_joint_noise=0.0, init_vel_noise=0.0, random_phase=False,
             encoder_offset=0.0, mass_scale_range=(1.0, 1.0), com_offset=0.0)

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def make(backend, n=4, **kw):
    c = Contract()
    model = mujoco.MjModel.from_xml_path(str(c.dir.parent / "q1_physics.xml"))
    motion = generate("squat", model, c)
    cfg = replace(EnvConfig(), num_envs=n, backend=backend, **{**QUIET, **kw})
    return MotionTrackingEnv(model, motion, c, cfg, seed=0), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mjbatch")
    backend = ap.parse_args().backend
    knee = Contract().joint_names.index("left_knee_joint")
    act = np.zeros((4, 22))
    act[:, knee] = 0.8          # 0.2 rad target jump after scaling -- a visible step

    print("1. zero latency == one plain step(decimation)")
    env, c = make(backend)
    q0, v0 = env.qpos.copy(), env.qvel.copy()
    env.step(act)
    split = env.qpos.copy()
    env2, _ = make(backend)
    env2.qpos[:], env2.qvel[:] = q0, v0
    target = c.action_to_target(act, env2.prev_target)
    env2.ctrl[:] = target
    env2.backend.step(nstep=c.decimation)
    check(np.allclose(split, env2.qpos, atol=1e-9), f"split schedule matches (max diff {np.abs(split - env2.qpos).max():.1e})")

    print("2. action delay holds the old target for d substeps")
    moved = {}
    for d in (0, 10):
        env, c = make(backend, act_delay_steps=(d, d))
        start = env.qpos[0, 7 + knee]
        env.step(act)
        moved[d] = env.qpos[0, 7 + knee] - start
    check(moved[0] > moved[10] > 0, f"knee moved {moved[0]:.4f} rad with d=0, {moved[10]:.4f} with d=10")

    print("3. observation delay reports the state k substeps before the period ends")
    env, c = make(backend, obs_delay_steps=(2, 2))
    env.step(act)
    seen = env.obs_qpos[:, 7:].copy()
    check(not np.allclose(seen, env.qpos[:, 7:]), "observed joints differ from the final state")
    env2, _ = make(backend)                       # replay without delay, stop 2 substeps early
    target = c.action_to_target(act, env2.prev_target)
    env2.ctrl[:] = target
    env2.backend.step(nstep=c.decimation - 2)
    check(np.allclose(seen, env2.qpos[:, 7:], atol=1e-9),
          f"observed == state at substep {c.decimation - 2} (max diff {np.abs(seen - env2.qpos[:, 7:]).max():.1e})")

    print("4. noise reaches the observation only")
    env, c = make(backend, noise_joint_pos=0.01, noise_ang_vel=0.2, noise_ori=0.02, imu_bias=0.03)
    true_q = env.qpos.copy()
    o1, o2 = env.obs(), env.obs()
    check(not np.allclose(o1, o2), "two observations of the same state differ")
    check(np.array_equal(true_q, env.qpos), "true state untouched by observing")
    jp = c.fields["joint_pos_err"].sl
    err = np.abs(o1[:, jp] - (true_q[:, 7:] - c.default_dof_pos)).max()
    check(0 < err <= 0.01 + 1e-6, f"joint position noise within +-0.01 rad (max {err:.4f})")

    print("5. encoder offset: the observation reads q + b, the motors drive the joint to target - b")
    env, c = make(backend, encoder_offset=0.02)
    b = env.enc_bias.copy()
    check(np.abs(b).max() > 0, f"offsets drawn (max {np.abs(b).max():.4f} rad)")
    o = env.obs()
    seen = o[:, jp] + c.default_dof_pos
    check(np.allclose(seen, env.qpos[:, 7:] + b, atol=1e-5), "observed joints == true + offset")
    zero = np.zeros((4, 22))
    env.step(zero)
    tgt = c.action_to_target(zero, env.prev_target)
    check(np.allclose(env.ctrl, env.prev_target - b, atol=1e-9) and np.allclose(env.prev_target, tgt),
          "motor command == target - offset")

    print("6. inertial randomisation reaches the simulation")
    env, c = make(backend, mass_scale_range=(0.9, 1.1), com_offset=0.02)
    m = env.body_mass[:, 1:]
    check(np.ptp(m, axis=0).max() > 0, "link masses differ between envs")
    check(0.9 - 1e-9 <= (m / env.mass0[1:]).min() and (m / env.mass0[1:]).max() <= 1.1 + 1e-9,
          f"masses within x0.9-1.1 (total {env.mass_s.min():.3f}-{env.mass_s.max():.3f} of nominal)")
    check(np.abs(env.com_off).max() <= 0.02 + 1e-12 and np.abs(env.com_off).max() > 0, "COM offsets within +-2 cm")

    print("FAILED: " + "; ".join(FAIL) if FAIL else "all latency/noise checks passed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
