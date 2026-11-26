"""How a trained policy actually moves: jitter, self-collision, falls, distance.

    python tools/policy_quality.py runs/<run> [runs/<other> ...] [--clean]

Plays each policy through its whole clip from frame 0 (no resets, deterministic), with
the env's noise and latency unless --clean, and reports what reward curves hide:

  target RMS     joint-target change per 20 ms step, RMS over joints and steps (rad);
                 printed next to the reference's own, so a policy that tracks with
                 high-frequency effort stands out
  at limit       share of joint-steps hitting the firmware's 0.2 rad/frame rate limit
  zig-zag        share of joint-steps whose target reverses direction every step by
                 more than 0.01 rad -- chatter, not motion
  self-contact   share of steps with any robot-robot contact (MuJoCo's contacts), and
                 the deepest one
  fell / walked  as tools/rollout_video.py

A run dir needs policy.pt and motion.npz (train.py writes both).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl.algos.ppo import ActorCritic  # noqa: E402
from q1_rl.contract import Contract  # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv  # noqa: E402
from q1_rl.evaluate import deterministic_policy  # noqa: E402
from q1_rl.motion import Motion  # noqa: E402

QUIET = dict(noise_ang_vel=0.0, noise_ori=0.0, imu_bias=0.0, noise_joint_pos=0.0, noise_joint_vel=0.0,
             act_delay_steps=(0, 0), obs_delay_steps=(0, 0), encoder_offset=0.0)


def self_contacts(m, d):
    out = []
    for con in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
        if b1 and b2:
            out.append((tuple(sorted((m.body(b1).name, m.body(b2).name))), -con.dist))
    return out


def measure(run: Path, clean: bool, seed: int = 0):
    c = Contract()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/q1/q1_physics.xml"))
    motion = Motion.load_npz(run / "motion.npz")
    cfg = replace(EnvConfig(), num_envs=1, random_phase=False, episode_seconds=motion.duration,
                  push_vel=0.0, init_joint_noise=0.0, init_vel_noise=0.0, **(QUIET if clean else {}))
    env = MotionTrackingEnv(model, motion, c, cfg, seed=seed)
    net = ActorCritic.from_state_dict(torch.load(run / "policy.pt", map_location="cpu"))
    act = deterministic_policy(net.eval())
    d = mujoco.MjData(model)

    deltas, hit, deepest, pairs, fell_at, walked, ref_walked = [], 0, 0.0, Counter(), None, 0.0, 0.0
    obs = env.obs()
    with torch.no_grad():
        for t in range(env.max_steps):
            ph, before = int(env.phase[0]), env.qpos[0, 0:2].copy()
            obs, _, _, fell, _ = env.step(act(torch.as_tensor(obs)).numpy())
            deltas.append(env.target_delta[0].copy())
            walked += float(np.linalg.norm(env.qpos[0, 0:2] - before))
            ref_walked += float(np.linalg.norm(motion.body_pos_w[int(env.phase[0]), 0, :2] - motion.body_pos_w[ph, 0, :2]))
            d.qpos[:] = env.qpos[0]
            mujoco.mj_forward(model, d)
            sc = self_contacts(model, d)
            hit += bool(sc)
            for p, depth in sc:
                pairs[p] += 1
                deepest = max(deepest, depth)
            if fell[0]:
                fell_at = (t + 1) / c.policy_hz
                break
    D = np.array(deltas)
    ref = np.diff(motion.joint_pos, axis=0)
    flips = (np.sign(D[1:]) * np.sign(D[:-1]) < 0) & (np.abs(D[1:]) > 0.01) & (np.abs(D[:-1]) > 0.01)
    return dict(target_rms=float(np.sqrt((D ** 2).mean())), ref_rms=float(np.sqrt((ref ** 2).mean())),
                at_limit=float((np.abs(D) > c.max_step_rad - 1e-4).mean()), zigzag=float(flips.mean()),
                self_contact=hit / len(D), deepest=deepest, pairs=pairs.most_common(3),
                fell_at=fell_at, walked=walked, ref_walked=ref_walked, steps=len(D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    print(f"{'run':28s} {'target RMS':>14s} {'at limit':>8s} {'zig-zag':>8s} {'self-contact':>14s} "
          f"{'fell':>6s} {'walked':>12s}")
    for run in a.runs:
        r = measure(run, a.clean)
        print(f"{run.name:28s} {r['target_rms']:.3f} (ref {r['ref_rms']:.3f}) {r['at_limit'] * 100:7.1f}% "
              f"{r['zigzag'] * 100:7.1f}% {r['self_contact'] * 100:6.0f}% {r['deepest'] * 1000:4.0f} mm "
              f"{('%.1fs' % r['fell_at']) if r['fell_at'] else 'no':>6s} {r['walked']:5.1f}/{r['ref_walked']:.1f} m")
        if r["pairs"]:
            print("      contacts: " + ", ".join(f"{p[0].replace('_link', '')}<->{p[1].replace('_link', '')} {n}" for p, n in r["pairs"]))


if __name__ == "__main__":
    main()
