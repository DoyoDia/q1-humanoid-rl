#!/usr/bin/env python3
"""Run the same environment on every available backend and report whether they agree.

This is the test that decides whether the backend abstraction is real. The claim is not
that mjbatch and mujoco_warp produce identical trajectories -- they cannot: mjbatch steps
MuJoCo in float64 on the CPU, warp in float32 on the GPU, and a contact-rich humanoid
amplifies that difference within a second. The claim is that the same environment code,
unmodified, produces the same *behaviour*: comparable rewards, comparable fall rates,
comparable tracking error.

    python3 tools/compare_backends.py [--envs 512] [--steps 200]
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl import motion as M                                      # noqa: E402
from q1_rl.backends import available                               # noqa: E402
from q1_rl.contract import Contract                                # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv  # noqa: E402


def run(backend, model, mo, contract, envs, steps, seed):
    env = MotionTrackingEnv(model, mo, contract,
                            EnvConfig(num_envs=envs, backend=backend), seed=seed)
    rng = np.random.default_rng(seed)
    rewards, falls, terms = [], 0, []
    t0 = time.perf_counter()
    for t in range(steps):
        # a fixed, state-independent action sequence: identical inputs to both backends
        a = 0.3 * np.sin(0.1 * t + np.arange(env.act_dim) * 0.4)[None, :] \
            + 0.05 * rng.standard_normal((envs, env.act_dim))
        _, r, done, fell, _ = env.step(a)
        rewards.append(r.mean())
        terms.append({k: float(v.mean()) for k, v in env.reward_terms.items()})
        falls += int(fell.sum())
        ids = np.flatnonzero(done)
        if ids.size:
            env.reset(ids)
    wall = time.perf_counter() - t0
    return dict(
        backend=backend,
        reward=float(np.mean(rewards)),
        fall_rate=falls / (envs * steps),
        joint_pos=float(np.mean([x["joint_pos"] for x in terms])),
        root_ori=float(np.mean([x["root_ori"] for x in terms])),
        body_pos=float(np.mean([x["body_pos"] for x in terms])),
        steps_per_s=envs * steps / wall,
        wall=wall,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--motion", default="squat")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    av = available()
    print("backends:", ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in av.items()))
    names = [k for k, v in av.items() if v]
    if len(names) < 2:
        print("\nonly one backend available here -- nothing to compare.")
        print("the warp backend needs an NVIDIA GPU and:")
        print("  pip install 'mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp.git'")

    contract = Contract()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots" / "q1" / "q1_physics.xml"))
    mo = M.generate(a.motion, model, contract)
    print(f"\n{a.envs} envs x {a.steps} policy steps, motion={mo.name}, identical action sequence\n")

    rows = [run(n, model, mo, contract, a.envs, a.steps, a.seed) for n in names]
    hdr = f"{'backend':>10} {'reward':>8} {'fall%':>7} {'joint_pos':>10} {'body_pos':>9} {'root_ori':>9} {'steps/s':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['backend']:>10} {r['reward']:>8.3f} {100*r['fall_rate']:>7.2f} "
              f"{r['joint_pos']:>10.4f} {r['body_pos']:>9.4f} {r['root_ori']:>9.4f} "
              f"{r['steps_per_s']:>10,.0f}")

    if len(rows) > 1:
        base = rows[0]
        print()
        for r in rows[1:]:
            d = {k: abs(r[k] - base[k]) for k in ("reward", "joint_pos", "body_pos", "root_ori")}
            rel = d["reward"] / max(abs(base["reward"]), 1e-9)
            print(f"  {r['backend']} vs {base['backend']}: "
                  f"reward differs by {d['reward']:.4f} ({100*rel:.2f}%), "
                  f"fall rate by {100*abs(r['fall_rate']-base['fall_rate']):.2f} pp, "
                  f"speed {r['steps_per_s']/base['steps_per_s']:.2f}x")
        print("""
  Bitwise agreement is not the goal and is not achievable: mjbatch steps MuJoCo in
  float64 on the CPU, mujoco_warp in float32 on the GPU. What matters is that the same
  environment file, unmodified, behaves the same on both.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
