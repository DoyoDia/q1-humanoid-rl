#!/usr/bin/env python3
"""PPO vs SAC on the same environment, same motion, same wall-clock budget, several seeds,
scored by deterministic evaluation.

Comparing by iteration count would be meaningless -- an off-policy iteration does far more
gradient work than an on-policy one. What a developer actually has is time, so that is the
budget. Scoring is by evaluate(): the mean action on a fixed-seed environment, one full
episode per robot. An earlier version scored by training reward, which carries each
algorithm's own exploration noise and so compared noise levels as much as skill; it also
ran one seed.

Evaluation time is excluded from the budget (the clock pauses), so both algorithms get the
same training time however long their evaluations take.

    python3 tools/ab_algos.py --seconds 420 --seeds 0 1 2 --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl import motion as M                                        # noqa: E402
from q1_rl.algos.ppo import PPOConfig, Trainer                       # noqa: E402
from q1_rl.algos.sac import SACConfig, SACTrainer                    # noqa: E402
from q1_rl.contract import Contract                                  # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv  # noqa: E402
from q1_rl.evaluate import evaluate                                  # noqa: E402


class Budget(Exception):
    pass


def run(algo, seconds, envs, motion_name, seed, backend, device, eval_every, eval_envs, save_dir):
    contract = Contract()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots" / "q1" / "q1_physics.xml"))
    mo = M.generate(motion_name, model, contract)
    env = MotionTrackingEnv(model, mo, contract, EnvConfig(num_envs=envs, backend=backend), seed=seed)
    if algo == "ppo":
        trainer = Trainer(env, PPOConfig(iters=10 ** 6, horizon=24, seed=seed, device=device))
    else:
        trainer = SACTrainer(env, SACConfig(iters=10 ** 6, steps_per_iter=24, seed=seed,
                                            device=device), contract)

    hist, evals = [], []
    t0 = time.perf_counter()
    paused = 0.0
    next_eval = 0.0

    def clock():
        return time.perf_counter() - t0 - paused

    def do_eval(t):
        nonlocal paused
        e0 = time.perf_counter()
        r = evaluate(trainer.actor, model, mo, contract, num_envs=eval_envs, seed=10_000 + seed,
                     backend=backend, device=trainer.actor_device)
        paused += time.perf_counter() - e0
        evals.append(dict(t=t, steps=hist[-1]["steps"] if hist else 0, **r.as_dict()))
        print(f"    [{algo} seed {seed}] t={t:5.0f}s  eval fall {100*r.fall_rate:5.1f}%  "
              f"joint rmse {r.joint_rmse:.4f}  reward {r.reward:.3f}", flush=True)

    def on_iter(s):
        nonlocal next_eval
        t = clock()
        hist.append(dict(t=t, reward=s["reward"], fall_rate=s["fall_rate"], ep_len=s["ep_len"],
                         steps=s["steps"], joint_pos=s["joint_pos"], root_ori=s["root_ori"],
                         body_pos=s["body_pos"]))
        if t >= next_eval:
            do_eval(t)
            next_eval += eval_every
        if t >= seconds:
            raise Budget

    try:
        trainer.train(on_iter=on_iter)
    except Budget:
        pass
    if not evals or evals[-1]["t"] < hist[-1]["t"]:
        do_eval(hist[-1]["t"])
    torch.save(trainer.net.state_dict(), save_dir / f"{algo}_seed{seed}.pt")
    return dict(train=hist, eval=evals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=420)
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--motion", default="squat")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--algos", nargs="+", default=["ppo", "sac"])
    p.add_argument("--backend", default="mjbatch")
    p.add_argument("--device", default="auto", help="learner device for both algorithms")
    p.add_argument("--eval-every", type=float, default=60.0, help="seconds of training between evals")
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--out", default=ROOT / "runs" / "ab_algos.json")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dir = out.with_suffix("")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"{' vs '.join(x.upper() for x in a.algos)}: {a.seconds:.0f}s each, {a.envs} envs, "
          f"motion={a.motion}, seeds={a.seeds}, learner={a.device}", flush=True)

    results = json.loads(out.read_text()) if out.exists() else {}
    for seed in a.seeds:
        for algo in a.algos:
            key = f"{algo}/seed{seed}"
            if key in results:
                print(f"  {key}: already in {out}, skipped", flush=True)
                continue
            print(f"  {key}", flush=True)
            results[key] = run(algo, a.seconds, a.envs, a.motion, seed, a.backend, a.device,
                               a.eval_every, a.eval_envs, save_dir)
            out.write_text(json.dumps(results), encoding="utf-8")       # checkpoint per run
    print(f"\n  curves written to {out}; score them with tools/analyse_ab.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
