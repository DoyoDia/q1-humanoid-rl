"""Deterministic evaluation: how good is the policy, not how noisy is the training loop.

Training reward is the wrong thing to compare algorithms on. It is collected with each
algorithm's own exploration noise -- PPO's Gaussian log_std, SAC's entropy-tuned squashed
Gaussian -- and those shrink at different rates, so a gap in training reward can be a gap
in noise rather than in skill. The first PPO-vs-SAC comparison fell into exactly that.

This runs the mean action on a fresh environment with a fixed seed, one full episode per
robot, no resets: every robot either survives the episode or it doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .envs.motion_tracking import EnvConfig, MotionTrackingEnv


@dataclass
class EvalResult:
    fall_rate: float          # fraction of robots that fell before the episode ended
    joint_rmse: float         # rad, over the steps each robot was still up
    reward: float             # env reward per step, over the same steps
    survived_steps: float     # mean steps before falling (max_steps if never)

    def as_dict(self):
        return dict(eval_fall_rate=self.fall_rate, eval_joint_rmse=self.joint_rmse,
                    eval_reward=self.reward, eval_survived=self.survived_steps)


def deterministic_policy(net):
    """The mean action, for either trainer's network (the same rule export.py uses)."""
    if hasattr(net, "deterministic"):
        return lambda obs: net.deterministic(obs)
    return lambda obs: net.actor(net.norm(obs))


@torch.no_grad()
def evaluate(net, model, motion, contract, num_envs: int = 256, seed: int = 12345,
             backend: str = "mjbatch", cfg: EnvConfig | None = None, device: str = "cpu") -> EvalResult:
    cfg = cfg or EnvConfig()
    cfg = EnvConfig(**{**cfg.__dict__, "num_envs": num_envs, "backend": backend})
    env = MotionTrackingEnv(model, motion, contract, cfg, seed=seed)
    act = deterministic_policy(net)
    alive = np.ones(num_envs, bool)
    fell_at = np.full(num_envs, env.max_steps, np.float64)
    err_sum = np.zeros(num_envs)
    rew_sum = np.zeros(num_envs)
    steps_up = np.zeros(num_envs)
    obs = env.obs()
    for t in range(env.max_steps):
        a = act(torch.as_tensor(obs, device=device)).cpu().numpy()
        obs, rew, done, fell, timeout = env.step(a)
        q_err = np.sqrt(np.mean((env.qpos[:, 7:] - env.ref_qpos[env.phase]) ** 2, axis=1))
        err_sum[alive] += q_err[alive]
        rew_sum[alive] += rew[alive]
        steps_up[alive] += 1
        newly = alive & fell
        fell_at[newly] = t + 1
        alive &= ~fell
        if not alive.any() or (timeout & alive).all():
            break
        # no resets: a fallen robot stays out of the averages from here on
    up = np.maximum(steps_up, 1)
    return EvalResult(
        fall_rate=float(1.0 - alive.mean()),
        joint_rmse=float(np.mean(err_sum / up)),
        reward=float(np.mean(rew_sum / up)),
        survived_steps=float(fell_at.mean()),
    )
