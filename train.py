#!/usr/bin/env python3
"""Train a motion-tracking policy for Q1 and export it in the robot's contract.

    python3 train.py --motion squat --iters 300
    python3 train.py --motion step_in_place --envs 4096 --backend auto

The default settings are sized for a laptop: 2048 environments on mjbatch reaches roughly
25k policy steps/s on 8 threads, so 300 iterations of horizon 24 is a few minutes.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from q1_rl import motion as M                              # noqa: E402
from q1_rl.algos.ppo import PPOConfig, Trainer             # noqa: E402
from q1_rl.algos.sac import SACConfig, SACTrainer          # noqa: E402
from q1_rl.contract import Contract                        # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv  # noqa: E402
from q1_rl.envs.velocity import VelocityConfig, VelocityEnv, VelocityWeights  # noqa: E402
from q1_rl.envs.walk import WalkConfig, WalkEnv            # noqa: E402

LOCOMOTION = ("velocity", "walk")


def env_config(a):
    """EnvConfig from the command line: --envs/--backend plus any --env KEY=VALUE overrides."""
    import ast
    from dataclasses import fields
    Cfg = {"velocity": VelocityConfig, "walk": WalkConfig}.get(a.task, EnvConfig)
    known = {f.name for f in fields(Cfg)}
    over = {}
    for kv in a.env:
        k, _, v = kv.partition("=")
        if k not in known:
            raise SystemExit(f"--env {k}: not an EnvConfig field ({', '.join(sorted(known))})")
        over[k] = ast.literal_eval(v)
    from q1_rl.envs.motion_tracking import RewardWeights
    if a.task in LOCOMOTION:
        RewardWeights = VelocityWeights          # noqa: N806
    wknown = {f.name for f in fields(RewardWeights)}
    wover = {}
    for kv in a.reward:
        k, _, v = kv.partition("=")
        if k not in wknown:
            raise SystemExit(f"--reward {k}: not a RewardWeights field ({', '.join(sorted(wknown))})")
        wover[k] = float(v)
    if wover:
        over["weights"] = RewardWeights(**wover)
    if a.algo == "ppo" and a.asym_critic:
        over["critic_obs"] = True
    return Cfg(num_envs=a.envs, backend=a.backend, obs_mode=a.obs_mode, **over)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", default="squat", choices=sorted(M.CATALOG) + ["npz"])
    p.add_argument("--npz", help="a reference motion file, when --motion npz")
    p.add_argument("--task", default="tracking", choices=["tracking", "velocity", "walk"],
                   help="tracking: follow --motion; velocity: walk on (vx, vy, yaw rate) commands "
                        "(q1_rl/envs/velocity.py, contract `velocity`), the browser viewer's WASD; "
                        "walk: the same on the robot's Walk contract (5-frame history, 13 actions; "
                        "q1_rl/envs/walk.py)")
    p.add_argument("--contract", default=None, help="default: mimic_opensource_v2, or the task's name with --task velocity / walk")
    p.add_argument("--model", default=ROOT / "robots" / "q1" / "q1_physics.xml")
    p.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--backend", default="mjbatch")
    p.add_argument("--gpus", default=None, metavar="0,1",
                   help="spread the envs over these GPUs, one worker process each (warp backend; "
                        "physical ids, the learner stays where --device puts it)")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="learner device: auto|cpu|cuda|mps")
    p.add_argument("--actor-device", default="auto", help="rollout actor: auto (GPU at >= 4096 envs)|cpu|cuda")
    p.add_argument("--sac-updates", type=int, default=None, help="gradient steps per iteration")
    p.add_argument("--sac-batch", type=int, default=None)
    p.add_argument("--out", default=ROOT / "runs")
    p.add_argument("--save-every", type=int, default=100, help="checkpoint policy.pt every N iterations (0: only at the end)")
    p.add_argument("--run-name", default=None, help="output folder name (default: <motion>_<algo>)")
    p.add_argument("--resume", default=None,
                   help="continue from a checkpoint.pt (weights, optimiser, LR schedule) or a bare policy.pt; "
                        "--iters is the total, so --resume at 500 with --iters 600 trains 100 more")
    p.add_argument("--resume-iter", type=int, default=None,
                   help="for a bare policy.pt: the iteration it was saved at, so the LR schedule picks up there")
    p.add_argument("--no-wandb", dest="wandb", action="store_false",
                   help="do not log to Weights & Biases (on by default when wandb is installed and logged in)")
    p.add_argument("--obs-mode", default="contract", choices=["contract"],
                   help="contract: MIMIC_OPENSOURCE_V2's 119-d frame")
    p.add_argument("--asym-critic", action=argparse.BooleanOptionalAction, default=True,
                   help="asymmetric actor-critic (PPO; on by default): the critic sees the env's "
                        "privileged critic_obs() -- training only, the exported policy is unchanged. "
                        "Against the symmetric one on bt_v3 (1500 iters): falls in training 3%% vs 12%%, "
                        "heading drift over the clip less than half, joint RMSE 0.28 vs 0.24. "
                        "--no-asym-critic for the old symmetric critic; resuming a symmetric "
                        "checkpoint switches it off by itself")
    p.add_argument("--reward", action="append", default=[], metavar="TERM=WEIGHT",
                   help="override a reward weight, e.g. --reward action_rate=-3")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="override an EnvConfig field, e.g. --env lift_prob=0.2 --env push_vel=0.6 "
                        "--env 'lift_height=(0.05, 0.25)' (values are Python literals); repeatable")
    p.add_argument("--wandb-project", default="q1_rl")
    p.add_argument("--wandb-entity", default=None, help="team or user (default: your wandb default)")
    a = p.parse_args()

    if a.resume and a.asym_critic and a.algo == "ppo":
        import torch
        ck = torch.load(a.resume, map_location="cpu")
        if "critic_norm.mean" not in (ck["net"] if "net" in ck else ck):
            a.asym_critic = False
            print(f"{a.resume} has a symmetric critic: continuing it symmetric")

    contract = Contract(a.contract or (a.task if a.task in LOCOMOTION else "mimic_opensource_v2"))
    model = mujoco.MjModel.from_xml_path(str(a.model))
    if a.task in LOCOMOTION:
        a.motion = "stand"                      # only the reset pose
    motion = (M.Motion.load_npz(a.npz) if a.motion == "npz"
              else M.generate(a.motion, model, contract))
    if a.task in LOCOMOTION:
        if a.gpus:
            raise SystemExit(f"--task {a.task} runs on one device (no --gpus)")
        env = (WalkEnv if a.task == "walk" else VelocityEnv)(model, motion, contract, env_config(a), seed=a.seed)
    elif a.gpus:
        from q1_rl.envs.multi_gpu import MultiGPUEnv
        env = MultiGPUEnv(a.model, motion, contract.name, env_config(a), seed=a.seed,
                          gpus=[int(g) for g in a.gpus.split(",")])
    else:
        env = MotionTrackingEnv(model, motion, contract,
                                env_config(a), seed=a.seed)
    if a.algo == "ppo":
        cfg = PPOConfig(iters=a.iters, horizon=a.horizon, seed=a.seed, device=a.device,
                        actor_device=a.actor_device)
        trainer = Trainer(env, cfg)
    else:
        cfg = SACConfig(iters=a.iters, steps_per_iter=a.horizon, seed=a.seed, device=a.device)
        if a.sac_updates:
            cfg.updates_per_iter = a.sac_updates
        if a.sac_batch:
            cfg.batch = a.sac_batch
        trainer = SACTrainer(env, cfg, contract)

    if a.resume:
        import torch
        ckpt = torch.load(a.resume, map_location=trainer.device)
        if hasattr(trainer, "load"):
            print(f"resumed from {a.resume}: {trainer.load(ckpt, a.resume_iter)}")
        else:                                   # SAC: weights only for now
            trainer.net.load_state_dict(ckpt["net"] if "net" in ckpt else ckpt)
            trainer.actor.load_state_dict(trainer.net.state_dict())
            print(f"resumed from {a.resume}: weights only (SAC optimisers restart)")
        if getattr(trainer, "start_iter", 0) >= cfg.iters:
            raise SystemExit(f"--iters {cfg.iters} is the total and the checkpoint is already at "
                             f"{trainer.start_iter}; ask for more")
    out = Path(a.out) / (a.run_name or (f"{a.task}_{a.algo}" if a.task in LOCOMOTION else f"{motion.name}_{a.algo}"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"{contract}\n{env}\n{motion}")
    print(f"learner on {trainer.device}, rollout actor on {trainer.actor_device}")
    print(f"algo {a.algo}")
    todo = a.iters - getattr(trainer, "start_iter", 0)
    print(f"{todo} iters x {a.horizon} steps x {a.envs} envs "
          f"= {todo * a.horizon * a.envs / 1e6:.1f}M policy steps -> {out}\n")

    run = None
    if a.wandb:
        from q1_rl import tracking
        try:
            run = tracking.start(out.name, dict(args=vars(a), env=env.cfg, algo=cfg, motion=dict(
                name=motion.name, frames=motion.n_frames, fps=motion.fps, cyclic=motion.cyclic)),
                project=a.wandb_project, entity=a.wandb_entity, tags=[a.algo, a.backend, motion.name])
            print(f"wandb: {run.url}")
        except Exception as e:                  # not installed, not logged in, offline
            run = None
            print(f"wandb: not logging ({type(e).__name__}: {e}); pass --no-wandb to silence this")

    hdr = f"{'iter':>5} {'steps':>9} {'k/s':>6} {'reward':>7} {'ep_len':>7} {'fall%':>6}  terms"
    print(hdr)

    def save():
        import torch
        torch.save(trainer.net.state_dict(), out / "policy.pt")      # what the tools load
        if hasattr(trainer, "checkpoint"):
            torch.save(trainer.checkpoint(), out / "checkpoint.pt")  # what --resume wants

    def report(s):
        if run is not None:
            tracking.log(run, s)
        if a.save_every and s["iter"] % a.save_every == 0:
            save()
            (out / "history.json").write_text(json.dumps(trainer.history), encoding="utf-8")
        if s["iter"] % 10 and s["iter"] != 1 and s["iter"] != cfg.iters:
            return
        keys = ("lin_vel", "ang_vel", "feet_swing") if a.task in LOCOMOTION else ("joint_pos", "body_pos", "root_ori")
        if "alpha" in s:                     # SAC also reports its entropy bookkeeping
            keys = keys + ("alpha", "entropy")
        terms = "  ".join(f"{k} {s[k]:.2f}" for k in keys)
        print(f"{s['iter']:>5} {s['steps']:>9} {s['steps_per_s']/1e3:>6.1f} "
              f"{s['reward']:>7.3f} {s['ep_len']:>7.1f} {100*s['fall_rate']:>6.1f}  {terms}")

    hist = trainer.train(on_iter=report)

    save()
    (out / "history.json").write_text(json.dumps(hist), encoding="utf-8")
    from q1_rl.export import export_onnx
    export_onnx(trainer.net, contract, out / "policy.onnx",
                motion_name=a.task if a.task in LOCOMOTION else motion.name, algo=a.algo, obs_dim=env.obs_dim)
    info = dict(obs_mode=env.cfg.obs_mode, obs_dim=env.obs_dim, critic_dim=getattr(env, "critic_dim", None))
    if a.task in LOCOMOTION:                       # what the viewer drives it with (tools/export_web_bundle.py)
        info["walk"] = dict(vx=list(env.cfg.cmd_vx), vy=list(env.cfg.cmd_vy), wz=list(env.cfg.cmd_wz),
                            cycle=list(getattr(env.cfg, "cycle_time_s", ())) or None,
                            stride=getattr(env.cfg, "cycle_stride_m", 0.0),
                            cpg=getattr(env.cfg, "cpg_law", "random"))
    (out / "env.json").write_text(json.dumps(info), encoding="utf-8")
    motion.save_npz(out / "motion.npz")
    print(f"\nsaved policy.pt / policy.onnx / motion.npz / history.json in {out}")
    if run is not None:
        run.summary["out_dir"] = str(out)
        run.finish()
    # The first iteration usually has no completed episode, so its fall rate is NaN and
    # its episode length is the cap rather than a measurement. Start the summary from the
    # first iteration that actually measured something.
    real = next((h for h in hist if h["fall_rate"] == h["fall_rate"]), hist[0])
    last = hist[-1]
    print(f"reward {real['reward']:.3f} -> {last['reward']:.3f}   "
          f"episode length {real['ep_len']:.0f} -> {last['ep_len']:.0f} / {env.max_steps}   "
          f"falls {100*real['fall_rate']:.0f}% -> {100*last['fall_rate']:.0f}%"
          f"   (from iter {real['iter']})")


if __name__ == "__main__":
    main()
