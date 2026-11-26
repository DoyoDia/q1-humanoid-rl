"""PPO, kept small enough to read in one sitting.

Deliberately not a dependency on rsl_rl / skrl / sb3. The algorithm is ~150 lines and we
need to own two things they do not give us for free: an actor that runs on whatever the
developer has (CPU, CUDA or Apple MPS -- the framework's whole premise), and an export
path that emits exactly the ONNX the robot's runtime expects, observation normalisation
folded in. Vendoring a training library would put a second opinion about observation
layout into the deployment path, which is the one thing the contract exists to prevent.

Structure follows mjbatch's go1_joystick.py example, which is a known-good reference for
this class of task.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PPOConfig:
    iters: int = 600
    horizon: int = 24              # policy steps per env per iteration
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 1e-3
    lr_end: float = 3e-4
    lr_to: int = 400
    ent_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    init_log_std: float = np.log(0.5)
    hidden: tuple[int, ...] = (512, 256, 128)
    device: str = "auto"
    actor_device: str = "auto"     # the rollout actor; auto = learner GPU at >= 4096 envs, else CPU
    seed: int = 0


class RunningNorm(nn.Module):
    """Observation normalisation that travels with the policy.

    Kept as module buffers rather than a side-car file so that export folds it into the
    ONNX graph. A normaliser that lives outside the exported model is the classic way to
    get a policy that works in training and does nothing on the robot.
    """

    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def absorb(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).to(self.mean.device)
        n = torch.tensor(float(x.shape[0]), device=self.mean.device)
        m, v = x.mean(0), x.var(0, unbiased=False)
        delta = m - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        self.var.copy_((self.var * self.count + v * n + delta.pow(2) * self.count * n / tot) / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


class ActorCritic(nn.Module):
    """critic_dim: None, the critic reads the policy's observation; else its own input of that
    width (the env's critic_obs(), with its own normaliser) -- an asymmetric actor-critic.
    The actor and self.norm, which is all that is exported, are the same either way."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, critic_dim: int | None = None):
        super().__init__()
        self.obs_dim, self.act_dim, self.critic_dim = obs_dim, act_dim, critic_dim
        self.norm = RunningNorm(obs_dim)
        if critic_dim:
            self.critic_norm = RunningNorm(critic_dim)

        def trunk(out, d_in=obs_dim):
            layers, d = [], d_in
            for h in cfg.hidden:
                layers += [nn.Linear(d, h), nn.ELU()]
                d = h
            layers.append(nn.Linear(d, out))
            return nn.Sequential(*layers)

        self.actor = trunk(act_dim)
        self.critic = trunk(1, critic_dim or obs_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(cfg.init_log_std)))
        for m in self.modules():                       # small last layer: start near zero
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)

    @classmethod
    def from_state_dict(cls, sd: dict, cfg: PPOConfig | None = None) -> "ActorCritic":
        """Rebuild from a saved policy.pt, symmetric or asymmetric, dims read off the weights."""
        net = cls(sd["norm.mean"].shape[0], sd["log_std"].shape[0], cfg or PPOConfig(),
                  sd["critic_norm.mean"].shape[0] if "critic_norm.mean" in sd else None)
        net.load_state_dict(sd)
        return net

    def forward(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None):
        z = self.norm(obs)
        zc = self.critic_norm(critic_obs) if self.critic_dim else z
        return self.actor(z), self.critic(zc).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor | None = None, deterministic: bool = False):
        mean, value = self(obs, critic_obs)
        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device), value
        std = self.log_std.exp()
        action = mean + std * torch.randn_like(mean)
        return action, log_prob(action, mean, self.log_std), value


def log_prob(action, mean, log_std):
    z = (action - mean) / log_std.exp()
    return -0.5 * (z * z).sum(-1) - log_std.sum() - 0.5 * action.shape[-1] * np.log(2 * np.pi)


class Trainer:
    def __init__(self, env, cfg: PPOConfig):
        torch.manual_seed(cfg.seed)
        self.env, self.cfg = env, cfg
        self.device = pick_device(cfg.device)
        # The rollout actor: CPU for small batches (no transfer, and a 512-256-128 MLP is
        # cheap), the learner's GPU for large ones -- at 8192 envs the CPU forward pass
        # took 75 ms per policy step against 12 ms on the GPU, a third of the step.
        if cfg.actor_device != "auto":
            self.actor_device = pick_device(cfg.actor_device)
        else:
            self.actor_device = (self.device if self.device != "cpu" and env.num_envs >= 4096
                                 else "cpu")
        self.cdim = getattr(env, "critic_dim", None)
        self.net = ActorCritic(env.obs_dim, env.act_dim, cfg, self.cdim).to(self.device)
        self.actor = ActorCritic(env.obs_dim, env.act_dim, cfg, self.cdim).to(self.actor_device)
        self.actor.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), cfg.lr)
        self.history: list[dict] = []
        self.start_iter = 0            # > 0 after load(): where the LR schedule picks up

    # -- checkpoints ----------------------------------------------------------

    def checkpoint(self) -> dict:
        """Everything needed to continue: weights (with the obs normaliser and log_std),
        Adam's moments, and the iteration -- which fixes where the LR schedule is."""
        return {"format": 1, "net": self.net.state_dict(), "opt": self.opt.state_dict(),
                "iter": self.start_iter + len(self.history)}

    def load(self, ckpt, iter_hint: int | None = None) -> str:
        """Continue from a checkpoint() dict, or from a bare policy.pt state_dict.

        A bare state_dict carries no iteration, and restarting the LR schedule at its
        peak (1e-3) visibly undid a trained walk policy within 30 iterations. So unless
        told the iteration, assume the schedule had finished and continue at lr_end."""
        if isinstance(ckpt, dict) and "net" in ckpt:
            self.net.load_state_dict(ckpt["net"])
            self.opt.load_state_dict(ckpt["opt"])
            self.start_iter = int(ckpt["iter"])
            how = f"full checkpoint at iteration {self.start_iter} (weights, optimiser, LR schedule)"
        else:
            self.net.load_state_dict(ckpt)
            self.start_iter = int(iter_hint) if iter_hint is not None else self.cfg.lr_to
            how = (f"weights only; LR schedule resumes at iteration {self.start_iter}"
                   + ("" if iter_hint is not None else " (assumed finished: no --resume-iter given)"))
        self.actor.load_state_dict(self.net.state_dict())
        return how

    # -- one iteration ------------------------------------------------------

    def rollout(self):
        cfg, env, n = self.cfg, self.env, self.env.num_envs
        shapes = dict(obs=(env.obs_dim,), act=(env.act_dim,), logp=(), val=(), rew=(), alive=())
        if self.cdim:
            shapes["cobs"] = (self.cdim,)
        buf = {k: np.zeros((cfg.horizon, n, *s), np.float32) for k, s in shapes.items()}
        dev = self.actor_device
        cob = (lambda: env.critic_obs()) if self.cdim else (lambda: None)
        tt = lambda x: None if x is None else torch.as_tensor(x, device=dev)
        obs, cobs = env.obs(), cob()
        term_means, falls, episodes, returns, raw_reward = [], 0, 0, [], []
        for t in range(cfg.horizon):
            o = torch.as_tensor(obs, device=self.actor_device)
            action, logp, value = self.actor.act(o, tt(cobs))
            a = action.cpu().numpy()
            nxt, reward, done, fell, timeout = env.step(a)
            cnxt = cob()                               # terminal state, before any reset
            raw_reward.append(float(reward.mean()))    # report this, NOT the bootstrapped one
            if timeout.any():                          # bootstrap value through a time limit
                with torch.no_grad():
                    vb = self.actor(torch.as_tensor(nxt[timeout], device=self.actor_device),
                                    None if cnxt is None else tt(cnxt[timeout]))[1]
                reward = reward.copy()
                reward[timeout] += cfg.gamma * vb.cpu().numpy()
            for k, v in dict(obs=obs, act=a, logp=logp.cpu().numpy(), val=value.cpu().numpy(),
                             rew=reward, alive=~done).items():
                buf[k][t] = v
            if self.cdim:
                buf["cobs"][t] = cobs
            term_means.append({k: float(v.mean()) for k, v in env.reward_terms.items()})
            ids = np.flatnonzero(done)
            if ids.size:
                episodes += ids.size
                falls += int(fell[ids].sum())
                returns.append(float(env.steps[ids].mean()))
                env.reset(ids)
                nxt = env.obs()
                cnxt = cob()
            obs, cobs = nxt, cnxt

        batch = {k: torch.as_tensor(v).to(self.device) for k, v in buf.items()}
        with torch.no_grad():
            batch["last_val"] = self.actor(torch.as_tensor(obs, device=self.actor_device), tt(cobs))[1].to(self.device)
        stats = {k: float(np.mean([m[k] for m in term_means])) for k in term_means[0]}
        # nan, not 0, when nothing finished: an iteration with no completed episode has
        # no fall rate, and reporting 0 makes "time to fall rate <= 5%" fire on iteration 1.
        stats["fall_rate"] = falls / episodes if episodes else float("nan")
        stats["ep_len"] = float(np.mean(returns)) if returns else float(self.env.max_steps)
        # The raw environment reward, not buf["rew"] -- that one has the timeout bootstrap
        # value folded in, which makes it exceed the sum of the reward weights and makes it
        # incomparable with any algorithm that does not bootstrap the same way.
        stats["reward"] = float(np.mean(raw_reward))
        return batch, stats

    def gae(self, batch):
        cfg = self.cfg
        vals = torch.cat([batch["val"], batch["last_val"][None]])
        adv = torch.zeros_like(batch["rew"])
        carry = 0.0
        for t in reversed(range(cfg.horizon)):
            alive = batch["alive"][t]
            delta = batch["rew"][t] + cfg.gamma * alive * vals[t + 1] - vals[t]
            adv[t] = carry = delta + cfg.gamma * cfg.lam * alive * carry
        return adv, adv + batch["val"]

    def update(self, batch, adv, ret):
        cfg, net = self.cfg, self.net
        obs = batch["obs"].reshape(-1, self.env.obs_dim)
        cobs = batch["cobs"].reshape(-1, self.cdim) if self.cdim else None
        act = batch["act"].reshape(-1, self.env.act_dim)
        logp_old, adv, ret = batch["logp"].reshape(-1), adv.reshape(-1), ret.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(cfg.epochs):
            for idx in torch.randperm(obs.shape[0], device=self.device).chunk(cfg.minibatches):
                mean, value = net(obs[idx], None if cobs is None else cobs[idx])
                logp = log_prob(act[idx], mean, net.log_std)
                ratio = (logp - logp_old[idx]).exp()
                a = adv[idx]
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vf = (value - ret[idx]).pow(2).mean()
                loss = pg + cfg.value_coef * vf - cfg.ent_coef * net.log_std.sum()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                self.opt.step()
        net.norm.absorb(obs)        # after the epochs: the batch was collected under the old stats
        if cobs is not None:
            net.critic_norm.absorb(cobs)

    # -- the loop -----------------------------------------------------------

    def train(self, on_iter=None):
        """Runs iterations start_iter .. cfg.iters: cfg.iters is the TOTAL, so a run
        resumed at 500 with iters=600 does 100 more, on the schedule's 500..600 stretch."""
        cfg, start = self.cfg, time.perf_counter()
        per_iter = cfg.horizon * self.env.num_envs
        for it in range(self.start_iter, cfg.iters):
            self.opt.param_groups[0]["lr"] = float(np.interp(it, (0, cfg.lr_to), (cfg.lr, cfg.lr_end)))
            batch, stats = self.rollout()
            self.update(batch, *self.gae(batch))
            self.actor.load_state_dict(self.net.state_dict())
            dt = time.perf_counter() - start
            done = it + 1 - self.start_iter
            stats.update(iter=it + 1, seconds=dt, lr=self.opt.param_groups[0]["lr"],
                         steps=(it + 1) * per_iter, steps_per_s=done * per_iter / dt)
            self.history.append(stats)
            if on_iter:
                on_iter(stats)
        return self.history
