"""Play a SuperMimic policy and/or a q1_rl PPO policy through the same q1_rl environment.

    python tools/eval_supermimic.py --sm logs/run/ckpt_latest.pt --sm-npz data/npz/q1/clip.npz \
        --ppo runs/<run> [--envs 16] [--clean]

SuperMimic (_ref/SuperMimic-Reproduced, SM_ROBOT=q1) trains in mjlab at 2 ms with its own
observation: 10 frames of [q - q0, qd, IMU w, projected gravity, last action] (term-major,
oldest first) plus 10 future reference frames. Here that observation is assembled from
q1_rl's environment -- the 1 ms physics, firmware target rate limit, action and observation
latency, encoder bias and sensor noise the robot policies are trained and judged under --
so both policies are compared on one ruler and the SuperMimic number is also a sim2sim.

Both play the whole clip from frame 0, deterministic, --envs robots (noise and latency
differ per robot unless --clean). Reported: falls (q1_rl's fall rule), joint RMSE against
the reference while up, tracked-body position error relative to the pelvis, and the
pelvis's horizontal drift from the reference at the end.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SM_ROOT = Path(os.environ.get("SM_ROOT", ROOT / "_ref/SuperMimic-Reproduced"))
sys.path.insert(0, str(ROOT))

from q1_rl.algos.ppo import ActorCritic  # noqa: E402
from q1_rl.contract import Contract  # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv  # noqa: E402
from q1_rl.evaluate import deterministic_policy  # noqa: E402
from q1_rl.motion import Motion  # noqa: E402

QUIET = dict(noise_ang_vel=0.0, noise_ori=0.0, imu_bias=0.0, noise_joint_pos=0.0, noise_joint_vel=0.0,
             act_delay_steps=(0, 0), obs_delay_steps=(0, 0), encoder_offset=0.0)
# SuperMimic's own observation noise (paper Table A.1, uniform half-widths): q, qd, w, gravity.
SM_NOISE = (0.01, 0.5, 0.2, 0.05)


def _grav(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # R^T (0, 0, -1)
    return -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1)


class SMAgent:
    """SuperMimic policy fed from a q1_rl env."""

    def __init__(self, ckpt: Path, npz: Path, c: Contract, n: int, noise: bool, seed: int, device: str):
        os.environ["SM_ROBOT"] = "q1"
        sys.path.insert(0, str(SM_ROOT / "src"))
        from supermimic.envs.mdp.motion_library import MotionLibrary
        from supermimic.eval.benchmark import load_policy_from_checkpoint
        from supermimic.robots import TRACKED_BODIES

        self.dev = device
        self.policy, cfg = load_policy_from_checkpoint(ckpt, device)
        assert not cfg["env"].get("anchor_obs", False), "anchor_obs policies not supported here"
        self.k = int(cfg["env"].get("future_frames", 10))
        self.H = int(cfg["policy"].get("history", 10))
        self.q0 = c.default_dof_pos.astype(np.float32)
        self.lib = MotionLibrary([str(npz)], TRACKED_BODIES, torch.as_tensor(self.q0), device=device)
        self.n, self.noise, self.rng = n, noise, np.random.default_rng(seed)
        self.hist = None
        self.last = np.zeros((n, c.n_joints), np.float32)

    def _frame(self, env) -> np.ndarray:
        q = env.obs_qpos[:, 7:] + env.enc_bias - self.q0
        qd = env.obs_qvel[:, 6:].copy()
        w = env.obs_qvel[:, 3:6].copy()
        g = _grav(env.obs_qpos[:, 3:7])
        if self.noise:
            u = lambda x, h: x + self.rng.uniform(-h, h, x.shape)  # noqa: E731
            q, qd, w, g = u(q, SM_NOISE[0]), u(qd, SM_NOISE[1]), u(w, SM_NOISE[2]), u(g, SM_NOISE[3])
        return [q, qd, w, g, self.last.copy()]

    @torch.no_grad()
    def act(self, env) -> np.ndarray:
        f = self._frame(env)
        if self.hist is None:                     # mjlab fills the history with the first frame
            self.hist = [[t.copy() for t in f] for _ in range(self.H)]
        else:
            self.hist = self.hist[1:] + [f]
        terms = [np.stack([h[i] for h in self.hist], axis=1).reshape(self.n, -1) for i in range(5)]
        obs = torch.as_tensor(np.concatenate(terms, axis=1), dtype=torch.float32, device=self.dev)
        frame = torch.as_tensor(env.phase, dtype=torch.long, device=self.dev)
        goal = self.lib.goal_window(frame, self.k)
        a = self.policy(obs, goal).cpu().numpy().astype(np.float32)
        self.last = a
        return a


def run(env, act, motion, c: Contract):
    n = env.num_envs
    up = np.ones(n, bool)
    fell_at = np.full(n, -1)
    jerr, berr = [], []
    obs = env.obs()
    for t in range(env.max_steps):
        a = act(obs)
        obs, _, done, fell, _ = env.step(a)
        newly = fell & up
        fell_at[newly] = t
        up &= ~fell
        ph = env.phase
        if up.any():
            jerr.append(np.sqrt(np.mean((env.qpos[up, 7:] - env.ref_qpos[ph[up]]) ** 2, axis=1)))
            rel = env.xpos[up][:, env.key_ids] - env.xpos[up][:, env.root_id][:, None]
            ref = env.ref_body[ph[up]] - env.ref_root[ph[up]][:, None]
            berr.append(np.linalg.norm(rel - ref, axis=2).mean(1))
        if done.all() or (~up).all():
            break
    drift = np.linalg.norm(env.qpos[up, 0:2] - env.ref_root[env.phase[up], 0:2], axis=1) if up.any() else np.array([np.nan])
    return dict(falls=int((~up).sum()), n=n, fell_at_s=sorted(t / c.policy_hz for t in fell_at if t >= 0),
                joint_rmse=float(np.mean(np.concatenate(jerr))) if jerr else float("nan"),
                body_err_cm=100 * float(np.mean(np.concatenate(berr))) if berr else float("nan"),
                drift_cm=100 * float(np.nanmean(drift)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", type=Path, help="SuperMimic checkpoint")
    ap.add_argument("--sm-npz", type=Path, help="the clip in SuperMimic's npz format (scripts/data/q1_motion_to_npz.py)")
    ap.add_argument("--ppo", type=Path, help="q1_rl run dir (policy.pt + motion.npz)")
    ap.add_argument("--onnx", type=Path, action="append", default=[],
                    help="a deployment ONNX (obs 119 -> action 22), fed the contract observation the env builds; "
                         "repeatable. tools/export_supermimic.py writes one for an obs_mode mimic SuperMimic policy")
    ap.add_argument("--motion", type=Path, help="q1_rl motion npz (default: the PPO run's motion.npz)")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--clean", action="store_true", help="no noise, no latency, no encoder bias")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="mjbatch", help="physics backend of the q1_rl env (mjbatch = CPU)")
    ap.add_argument("--no-dr", action="store_true",
                    help="nominal physics: friction 1.0, PD gains, masses and COM as modelled")
    ap.add_argument("--sim-dt", type=float, default=0.0, help="physics step (0: the model's 1 ms)")
    ap.add_argument("--no-rate-limit", action="store_true", help="drop the firmware's per-frame target rate limit")
    ap.add_argument("--no-latency", action="store_true", help="no action / observation delay (noise kept)")
    ap.add_argument("--push", type=float, default=0.0, help="push the base by up to this many m/s every --push-every s")
    ap.add_argument("--push-every", type=float, default=1.0)
    a = ap.parse_args()

    c = Contract()
    if a.no_rate_limit:
        c.max_step_rad = np.full_like(np.asarray(c.max_step_rad, np.float64), 1e3)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/q1/q1_physics.xml"))
    motion = Motion.load_npz(a.motion or (a.ppo or a.onnx[0].parent) / "motion.npz")
    nodr = dict(friction_range=(1.0, 1.0), kp_scale_range=(1.0, 1.0), kd_scale_range=(1.0, 1.0),
                mass_scale_range=(1.0, 1.0), com_offset=0.0) if a.no_dr else {}
    if a.no_latency:
        nodr.update(act_delay_steps=(0, 0), obs_delay_steps=(0, 0))
    cfg = replace(EnvConfig(), num_envs=a.envs, random_phase=False, episode_seconds=motion.duration, push_vel=a.push,
                  push_interval_s=a.push_every,
                  lift_prob=0.0, init_joint_noise=0.0, init_vel_noise=0.0, backend=a.backend, sim_dt=a.sim_dt,
                  **(QUIET if a.clean else {}), **nodr)
    tag = ("clean" if a.clean else "noise+latency") + (" no-DR" if a.no_dr else "") + \
        (f" dt{a.sim_dt * 1000:g}ms" if a.sim_dt else "") + (" no-rate-limit" if a.no_rate_limit else "") + (" no-latency" if a.no_latency else "") + \
        (f" push {a.push:g} m/s every {a.push_every:g} s" if a.push else "")
    if a.ppo:
        env = MotionTrackingEnv(model, motion, c, cfg, seed=a.seed)
        net = ActorCritic.from_state_dict(torch.load(a.ppo / "policy.pt", map_location="cpu")).eval()
        pol = deterministic_policy(net)
        with torch.no_grad():
            r = run(env, lambda o: pol(torch.as_tensor(o, dtype=torch.float32)).numpy(), motion, c)
        print(f"PPO  {a.ppo.name:24s} [{tag}] {r}", flush=True)
    for path in a.onnx:
        import onnxruntime as ort
        env = MotionTrackingEnv(model, motion, c, cfg, seed=a.seed)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        r = run(env, lambda o: sess.run(None, {iname: c.pad_for_onnx(o).astype(np.float32)})[0], motion, c)
        print(f"ONNX {path.parent.name + '/' + path.name:24s} [{tag}] {r}", flush=True)
    if a.sm:
        env = MotionTrackingEnv(model, motion, c, cfg, seed=a.seed)
        agent = SMAgent(a.sm, a.sm_npz, c, a.envs, noise=not a.clean, seed=a.seed, device=a.device)
        r = run(env, lambda o: agent.act(env), motion, c)
        print(f"SM   {a.sm.parent.name + '/' + a.sm.name:24s} [{tag}] {r}", flush=True)


if __name__ == "__main__":
    main()
