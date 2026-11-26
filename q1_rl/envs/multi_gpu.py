"""One MotionTrackingEnv spread over several GPUs, one worker process per GPU.

    env = MultiGPUEnv(model_path, motion, contract_name, cfg, seed, gpus=[0, 1])

The warp backend's step is GPU-bound -- mujoco_warp's kernels are small and latency-bound,
so one A6000 runs this humanoid at ~200 W of 300 and adding worlds helps little (8192 ->
16384 worlds: +18% steps/s). A second GPU doubles the hardware instead. Each worker owns
cfg.num_envs / len(gpus) worlds on its GPU (CUDA_VISIBLE_DEVICES set before warp loads, so
no device state is shared) and runs the env's whole step -- physics, reward, observation --
in parallel with the others; the parent only scatters actions and gathers results.

It has the attributes and methods the trainers use: num_envs, obs_dim, act_dim, max_steps,
cfg, obs(), step(a), reset(ids), reward_terms, steps -- and critic_dim / critic_obs() when
cfg.critic_obs is set. Each worker keeps its own adaptive
start sampling (fall statistics over its share of the episodes), seeded differently.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from dataclasses import replace

import numpy as np


def _worker(conn, gpu, model_path, motion, contract_name, cfg, seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)          # before anything touches CUDA
    import mujoco

    from q1_rl.contract import Contract
    from q1_rl.envs.motion_tracking import MotionTrackingEnv
    env = MotionTrackingEnv(mujoco.MjModel.from_xml_path(model_path), motion, Contract(contract_name), cfg, seed=seed)
    conn.send(dict(obs_dim=env.obs_dim, act_dim=env.act_dim, max_steps=env.max_steps, repr=repr(env),
                   critic_dim=env.critic_dim))
    cob = env.critic_obs if env.critic_dim else (lambda: None)
    while True:
        try:
            cmd, arg = conn.recv()
        except (EOFError, OSError):                          # the parent is gone
            return
        if cmd == "step":
            # Episodes that end are reset here, right away, rather than on the parent's next
            # message: that was one more round trip per step (something ends in nearly every
            # step at thousands of worlds), with both GPUs idle while it went back and forth.
            # The terminal observation goes back (the trainer bootstraps time-outs from it),
            # and so do the first observations of the new episodes, for reset() to swap in.
            t0 = time.perf_counter()
            obs, rew, done, fell, timeout = env.step(arg)
            steps = env.steps.copy()
            terms = env.reward_terms
            cobs = cob()                                     # terminal, like obs
            ids = np.flatnonzero(done)
            fresh = cfresh = None
            if ids.size:
                obs = obs.copy()
                env.reset(ids)
                fresh = env.obs()[ids]
                cfresh = None if cobs is None else cob()[ids]
            busy = time.perf_counter() - t0
            conn.send((obs, rew, done, fell, timeout, terms, steps, ids, fresh, busy, cobs, cfresh))
        elif cmd == "reset":
            env.reset(arg)
            conn.send(env.obs())
        elif cmd == "obs":
            conn.send(env.obs())
        elif cmd == "critic_obs":
            conn.send(cob())
        elif cmd == "close":
            conn.close()
            return


class MultiGPUEnv:
    def __init__(self, model_path, motion, contract_name: str, cfg, seed: int = 0, gpus=(0, 1)):
        gpus = list(gpus)
        n = cfg.num_envs
        share = [n // len(gpus) + (1 if i < n % len(gpus) else 0) for i in range(len(gpus))]
        self.cuts = np.cumsum([0] + share)
        self.cfg, self.num_envs, self.gpus = cfg, n, gpus
        ctx = mp.get_context("spawn")
        self._conns, self._procs = [], []
        for i, (g, k) in enumerate(zip(gpus, share)):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(b, g, str(model_path), motion, contract_name, replace(cfg, num_envs=k), seed + 1000 * i))
            p.start()
            self._conns.append(a)
            self._procs.append(p)
        info = [c.recv() for c in self._conns]
        self.obs_dim, self.act_dim, self.max_steps = info[0]["obs_dim"], info[0]["act_dim"], info[0]["max_steps"]
        self._repr = f"MultiGPUEnv(gpus {gpus}, worlds {share}) of {info[0]['repr']}"
        self.critic_dim = info[0]["critic_dim"]
        self._obs = self._gather("obs", [None] * len(gpus))
        self._cobs = self._gather("critic_obs", [None] * len(gpus)) if self.critic_dim else None
        self.reward_terms, self.steps = {}, np.zeros(n, np.int64)
        self._busy, self._nstep, self._t_last = np.zeros(len(gpus)), 0, time.perf_counter()
        import atexit
        atexit.register(self.close)

    def _gather(self, cmd, args):
        for c, x in zip(self._conns, args):
            c.send((cmd, x))
        return np.concatenate([c.recv() for c in self._conns])

    def obs(self):
        return self._obs

    def critic_obs(self):
        return self._cobs

    def step(self, action):
        action = np.asarray(action)
        for i, c in enumerate(self._conns):
            c.send(("step", action[self.cuts[i]:self.cuts[i + 1]]))
        parts = [c.recv() for c in self._conns]
        obs, rew, done, fell, timeout = (np.concatenate([p[k] for p in parts]) for k in range(5))
        self.reward_terms = {k: np.concatenate([p[5][k] for p in parts]) for k in parts[0][5]}
        self.steps = np.concatenate([p[6] for p in parts])
        self._obs = obs
        self._fresh = [(self.cuts[i] + p[7], p[8], p[11]) for i, p in enumerate(parts) if p[8] is not None]
        if self.critic_dim:
            self._cobs = np.concatenate([p[10] for p in parts])
        # where the time goes: each worker's own step (physics, reward, resets) against the
        # wall time between the parent's steps (adds inference, transfer, the PPO update)
        now = time.perf_counter()
        self._busy += np.array([p[9] for p in parts])
        self._nstep += 1
        if self._nstep % 1000 == 0:
            wall = now - self._t_last
            print(f"[multi_gpu] last 1000 steps: wall {wall:.1f} s; worker busy "
                  + ", ".join(f"gpu{g} {b:.1f} s" for g, b in zip(self.gpus, self._busy)), file=sys.stderr, flush=True)
            self._busy[:] = 0
            self._t_last = now
        return obs, rew, done, fell, timeout

    def reset(self, ids):
        ids = np.asarray(ids)
        fresh = getattr(self, "_fresh", None)
        if fresh is not None and np.array_equal(np.sort(ids), np.sort(np.concatenate([f[0] for f in fresh] or [np.zeros(0, int)]))):
            for rows, o, co in fresh:                        # the workers have reset these already
                self._obs[rows] = o
                if co is not None:
                    self._cobs[rows] = co
                self.steps[rows] = 0
            self._fresh = None
            return
        args, which = [], []
        for i in range(len(self._conns)):
            lo, hi = self.cuts[i], self.cuts[i + 1]
            local = ids[(ids >= lo) & (ids < hi)] - lo
            args.append(local)
            which.append(local.size > 0)
        for c, x, w in zip(self._conns, args, which):
            if w:
                c.send(("reset", x))
        for i, (c, w) in enumerate(zip(self._conns, which)):
            if w:
                self._obs[self.cuts[i]:self.cuts[i + 1]] = c.recv()
                self.steps[self.cuts[i] + args[i]] = 0
        if self.critic_dim:
            self._cobs = self._gather("critic_obs", [None] * len(self._conns))

    def close(self):
        if not self._procs:
            return
        for c in self._conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
        self._procs = []

    def __repr__(self):
        return self._repr
