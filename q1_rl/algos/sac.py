"""SAC for massively parallel environments, in the spirit of FastSAC.

Why bother when PPO already works: PPO throws every batch away after a handful of epochs,
so its cost scales with environment steps. SAC keeps a replay buffer, so each simulated
step is reused many times. The price is gradient compute, and it is a real price: with a
CPU learner those updates cost more than the simulation they save, so SAC here is an
option for developers with a GPU (`--device cuda`), not the portable default. PPO is.

Differences from a textbook SAC, all of which exist because this is a 2048-environment
setting rather than a single-environment gym one:

  * the buffer is indexed by (env step, env), so one insert writes a whole batch;
  * the update-to-data ratio is high on purpose. At 8 gradient steps per iteration this
    implementation plateaued at a 100% fall rate and looked broken; at 48 it learns. That
    is the whole trade restated -- SAC buys sample efficiency with gradient compute, so
    starving the gradient side removes the only reason to use it. On a CPU learner those
    48 updates cost more than the simulation they replace, which is why SAC here wants a
    GPU learner (`--device cuda`) while PPO is perfectly happy CPU-only;
  * timeouts do not bootstrap as terminal -- the environment ends episodes on a clock, and
    treating that as death teaches the policy that standing still eventually kills it.

The squash range is per joint and sized by the reference motion, not hardcoded to [-1, 1]
and not stretched to the joint limits -- both of those were tried and failed for
different reasons; contract_action_scale() records why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .ppo import RunningNorm, pick_device

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


@dataclass
class SACConfig:
    iters: int = 600
    steps_per_iter: int = 8        # env steps collected per iteration (all envs at once)
    buffer_steps: int = 400        # capacity, in env steps (x num_envs transitions)
    warmup_steps: int = 40         # env steps of random action before learning starts
    batch: int = 4096              # transitions per gradient step
    updates_per_iter: int = 48     # gradient steps per iteration -- see note below
    gamma: float = 0.99
    tau: float = 0.005             # target network Polyak rate
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    target_entropy_scale: float = 1.0   # target entropy = -scale * act_dim (standard: 1.0)
    action_margin: float = 1.5          # action range = margin x what the reference needs
    action_floor_rad: float = 0.2       # ...but never narrower than this
    hidden: tuple[int, ...] = (512, 256, 128)
    layernorm: bool = True         # critic LayerNorm: the cheap half of the "fast" recipe
    max_grad_norm: float = 1.0
    device: str = "auto"
    actor_device: str = "auto"
    seed: int = 0


def mlp(inp, hidden, out, layernorm=False):
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ELU())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class SquashedActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, scale: np.ndarray):
        super().__init__()
        self.net = mlp(obs_dim, hidden, 2 * act_dim)
        self.act_dim = act_dim
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, z):
        # Explicit slices rather than chunk(): chunk exports as Split with a `num_outputs`
        # attribute, which onnxruntime rejects below opset 18. Slicing is opset-agnostic
        # and the deployment path must not depend on an exporter detail.
        out = self.net(z)
        mean = out[..., : self.act_dim]
        log_std = out[..., self.act_dim:]
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, z):
        mean, log_std = self(z)
        std = log_std.exp()
        x = mean + std * torch.randn_like(mean)
        y = torch.tanh(x)
        logp = (-0.5 * ((x - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - torch.log(1 - y * y + 1e-6).sum(-1)      # tanh change of variables
        return y * self.scale, logp

    def deterministic(self, z):
        return torch.tanh(self(z)[0]) * self.scale


class SACPolicy(nn.Module):
    """What gets trained, saved and exported. Mirrors ActorCritic's export surface."""

    def __init__(self, obs_dim, act_dim, cfg: SACConfig, scale):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.norm = RunningNorm(obs_dim)
        self.pi = SquashedActor(obs_dim, act_dim, cfg.hidden, scale)
        self.q1 = mlp(obs_dim + act_dim, cfg.hidden, 1, cfg.layernorm)
        self.q2 = mlp(obs_dim + act_dim, cfg.hidden, 1, cfg.layernorm)

    def deterministic(self, obs):
        return self.pi.deterministic(self.norm(obs))

    def q(self, z, a):
        za = torch.cat([z, a], -1)
        return self.q1(za).squeeze(-1), self.q2(za).squeeze(-1)


class ReplayBuffer:
    """Ring buffer over (step, env). One insert stores a whole environment batch."""

    def __init__(self, steps, num_envs, obs_dim, act_dim, device):
        self.n, self.e = steps, num_envs
        z = lambda *s: torch.zeros(*s, device=device)          # noqa: E731
        self.obs, self.next_obs = z(steps, num_envs, obs_dim), z(steps, num_envs, obs_dim)
        self.act = z(steps, num_envs, act_dim)
        self.rew, self.term = z(steps, num_envs), z(steps, num_envs)
        self.ptr, self.full = 0, False

    def add(self, obs, act, rew, next_obs, term):
        i = self.ptr
        for dst, src in ((self.obs, obs), (self.act, act), (self.next_obs, next_obs)):
            dst[i].copy_(torch.as_tensor(src))
        self.rew[i].copy_(torch.as_tensor(rew))
        self.term[i].copy_(torch.as_tensor(term))
        self.ptr = (i + 1) % self.n
        self.full |= self.ptr == 0

    def __len__(self):
        return (self.n if self.full else self.ptr) * self.e

    def sample(self, batch):
        rows = self.n if self.full else self.ptr
        i = torch.randint(0, rows, (batch,), device=self.obs.device)
        j = torch.randint(0, self.e, (batch,), device=self.obs.device)
        return (self.obs[i, j], self.act[i, j], self.rew[i, j],
                self.next_obs[i, j], self.term[i, j])


def contract_action_scale(contract, motion=None, margin: float = 1.5,
                          floor_rad: float = 0.2) -> np.ndarray:
    """Per-joint tanh scale, in action units. Sized by the task, not by the robot.

    Two obvious choices are both wrong, and measurably so on this robot:

      span the whole joint range -- scales reach 12.6, i.e. +-3.1 rad of offset per step.
        A freshly initialised squashed policy explores by throwing every joint to its
        limit, every episode ends on the first step, and the buffer fills with failure.
        (PPO on the same task operates around |a| ~ 0.5, which is +-0.125 rad.)

      a flat cap, say +-0.5 rad -- exploration behaves, but the squat reference needs the
        knee 0.9 rad away from the nominal pose, so the policy physically cannot reach the
        thing it is being rewarded for tracking. Reward plateaus with a 100% fall rate and
        looks like an optimisation failure when it is a reachability failure.

    So the range comes from the reference motion: whatever offset the clip demands, times a
    margin for the correction the policy needs on top, floored so an unused joint still has
    a little authority. A joint the motion never moves gets a narrow range and stops
    flailing; the knee in a squat gets a wide one. Without a motion this falls back to the
    flat cap.
    """
    reach = np.maximum(contract.upper - contract.default_dof_pos,
                       contract.default_dof_pos - contract.lower)
    if motion is not None:
        need = np.abs(np.asarray(motion.joint_pos, np.float64) - contract.default_dof_pos).max(axis=0)
        want = np.maximum(need * margin, floor_rad)
    else:
        want = np.full_like(reach, max(floor_rad, 0.5))
    want = np.minimum(want, reach)
    return np.minimum(want / max(contract.action_scale, 1e-9), contract.action_clip)


class SACTrainer:
    def __init__(self, env, cfg: SACConfig, contract, motion=None):
        torch.manual_seed(cfg.seed)
        self.env, self.cfg = env, cfg
        self.device = pick_device(cfg.device)
        self.actor_device = "cpu" if cfg.actor_device == "auto" else pick_device(cfg.actor_device)
        scale = contract_action_scale(contract, motion if motion is not None else getattr(env, "motion", None),
                                      cfg.action_margin, cfg.action_floor_rad)
        self.net = SACPolicy(env.obs_dim, env.act_dim, cfg, scale).to(self.device)
        self.target = SACPolicy(env.obs_dim, env.act_dim, cfg, scale).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.actor = SACPolicy(env.obs_dim, env.act_dim, cfg, scale).to(self.actor_device)
        self.actor.load_state_dict(self.net.state_dict())

        self.pi_opt = torch.optim.Adam(self.net.pi.parameters(), cfg.lr)
        self.q_opt = torch.optim.Adam(
            list(self.net.q1.parameters()) + list(self.net.q2.parameters()), cfg.lr)
        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], cfg.alpha_lr)
        self.target_entropy = -cfg.target_entropy_scale * env.act_dim

        self.buf = ReplayBuffer(cfg.buffer_steps, env.num_envs, env.obs_dim, env.act_dim, self.device)
        self.scale = torch.as_tensor(scale, dtype=torch.float32, device=self.actor_device)
        self.history: list[dict] = []
        self.env_steps = 0
        self._obs = env.obs()

    # -- collection ---------------------------------------------------------

    @torch.no_grad()
    def collect(self, steps, random=False):
        env, stats = self.env, []
        falls = episodes = 0
        lengths = []
        for _ in range(steps):
            o = torch.as_tensor(self._obs, device=self.actor_device)
            if random:
                a = (torch.rand(env.num_envs, env.act_dim, device=self.actor_device) * 2 - 1) * self.scale
            else:
                a = self.actor.pi.sample(self.actor.norm(o))[0]
            a_np = a.cpu().numpy()
            nxt, rew, done, fell, timeout = env.step(a_np)
            # terminated != done: a timeout is the clock, not death, and must still bootstrap
            self.buf.add(self._obs, a_np, rew, nxt, fell.astype(np.float32))
            stats.append({k: float(v.mean()) for k, v in env.reward_terms.items()} | {"rew": float(rew.mean())})
            ids = np.flatnonzero(done)
            if ids.size:
                episodes += ids.size
                falls += int(fell[ids].sum())
                lengths.append(float(env.steps[ids].mean()))   # completed episodes, as PPO reports
                env.reset(ids)
                nxt = env.obs()
            self._obs = nxt
            self.env_steps += env.num_envs
        agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        # nan, not 0, when nothing finished -- see the note in ppo.py
        agg["fall_rate"] = falls / episodes if episodes else float("nan")
        agg["ep_len"] = float(np.mean(lengths)) if lengths else float(env.max_steps)
        return agg

    # -- learning -----------------------------------------------------------

    def update(self):
        cfg, net, tgt = self.cfg, self.net, self.target
        obs, act, rew, next_obs, term = self.buf.sample(cfg.batch)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            # net.norm for BOTH sides, not tgt.norm. The Polyak update below walks
            # parameters(), and a RunningNorm's statistics are buffers, not parameters --
            # so the target's normaliser would sit at mean 0 / var 1 forever while the
            # online critic saw normalised inputs. The two Qs would then be computed on
            # different input scales and the bootstrap target would be nonsense.
            zn = net.norm(next_obs)
            a2, logp2 = net.pi.sample(zn)
            q1t, q2t = tgt.q(zn, a2)
            target_q = rew + cfg.gamma * (1 - term) * (torch.min(q1t, q2t) - alpha * logp2)

        z = net.norm(obs)
        q1, q2 = net.q(z, act)
        q_loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(net.q1.parameters()) + list(net.q2.parameters()), cfg.max_grad_norm)
        self.q_opt.step()

        a_new, logp = net.pi.sample(z.detach())
        q1p, q2p = net.q(z.detach(), a_new)
        pi_loss = (alpha * logp - torch.min(q1p, q2p)).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        nn.utils.clip_grad_norm_(net.pi.parameters(), cfg.max_grad_norm)
        self.pi_opt.step()

        alpha_loss = -(self.log_alpha.exp() * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(net.parameters(), tgt.parameters()):
                pt.mul_(1 - cfg.tau).add_(p, alpha=cfg.tau)
        return dict(q_loss=float(q_loss), pi_loss=float(pi_loss),
                    alpha=float(self.log_alpha.exp()), entropy=float(-logp.mean()))

    def train(self, on_iter=None):
        cfg, start = self.cfg, time.perf_counter()
        self.collect(cfg.warmup_steps, random=True)
        self.net.norm.absorb(torch.as_tensor(self.buf.obs[: self.buf.ptr or 1].reshape(-1, self.env.obs_dim)))
        self.actor.load_state_dict(self.net.state_dict())

        for it in range(cfg.iters):
            stats = self.collect(cfg.steps_per_iter)
            losses = [self.update() for _ in range(cfg.updates_per_iter)]
            self.net.norm.absorb(self.buf.sample(min(cfg.batch, len(self.buf)))[0])
            self.actor.load_state_dict(self.net.state_dict())
            dt = time.perf_counter() - start
            stats.update({k: float(np.mean([x[k] for x in losses])) for k in losses[0]})
            stats.update(iter=it + 1, seconds=dt, steps=self.env_steps,
                         steps_per_s=self.env_steps / dt, reward=stats.pop("rew"),
                         buffer=len(self.buf))
            self.history.append(stats)
            if on_iter:
                on_iter(stats)
        return self.history
