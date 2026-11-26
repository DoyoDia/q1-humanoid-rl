"""Distil a deployment ONNX (obs 119 -> action 22) into a small MLP the robot can run in time.

    python tools/distill_policy.py runs/<teacher_run> -o runs/<student_run> [--hidden 512 256 128]
    python tools/make_deploy_package.py runs/<student_run> --name <name>

Why: the SuperMimic 119-d policy (2048-1024-512-256, a 12 MB ONNX) overran the robot's 20 ms
mimic period on ~70 of ~310 steps per play and made the 5 ms joint_command worker miss its
wakeups; the vendor's motion policies are ~0.9 MB and never overrun (deploy/robot/README.md).

How: DAgger in q1_rl's own environment, with everything the robot policies are judged
under (domain randomisation, sensor noise, action / observation latency, encoder bias,
pushes). Every state visited is labelled with the teacher's action; round 0 is driven by
the teacher, later rounds increasingly by the student (beta = share of robots the teacher
drives), so the student also learns what the teacher does from the states its own mistakes
lead to. The student is trained on the aggregate (capped, oldest dropped) with an MSE loss
on the raw actions. Its input normalisation (mean / std of round 0's observations) is
frozen inside the network, so the export is a plain obs -> action graph with q1_rl's usual
names and metadata (export_onnx, algo "distill").

Written to the output dir: policy.onnx, motion.npz (the teacher run's), student.pt and
distill.json (the rounds' losses and the teacher it came from).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl.contract import Contract  # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv  # noqa: E402
from q1_rl.export import export_onnx  # noqa: E402
from q1_rl.motion import Motion  # noqa: E402


class Student(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        return self.mlp((obs - self.mean) / self.std)


class _Net(nn.Module):
    """What export_onnx expects of a trainer's network: .norm and .actor."""

    def __init__(self, student):
        super().__init__()
        self.norm = nn.Identity()           # the student's normalisation is inside it
        self.actor = student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher", type=Path, help="run dir with policy.onnx (obs 119 -> 22) and motion.npz")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200, help="policy steps per round")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=8, help="passes over the aggregate per round")
    ap.add_argument("--max-samples", type=int, default=3_000_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    c = Contract()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    motion = Motion.load_npz(a.teacher / "motion.npz")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/q1/q1_physics.xml"))
    cfg = replace(EnvConfig(), num_envs=a.envs, lift_prob=0.0)
    env = MotionTrackingEnv(model, motion, c, cfg, seed=a.seed)
    env.reset(np.arange(a.envs))

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(a.teacher / "policy.onnx"), so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    teacher = lambda o: sess.run(None, {iname: c.pad_for_onnx(o).astype(np.float32)})[0]  # noqa: E731

    O, A = [], []
    student, opt, log = None, None, []
    for rnd in range(a.rounds):
        beta = 1.0 if rnd == 0 else max(0.0, 0.5 ** rnd)
        driven = rng.random(a.envs) < beta          # these robots follow the teacher this round
        t0, falls, obs = time.time(), 0, env.obs()
        for _ in range(a.steps):
            ta = teacher(obs)
            O.append(obs.astype(np.float32))
            A.append(ta.astype(np.float32))
            if student is None:
                act = ta
            else:
                with torch.no_grad():
                    sa = student(torch.as_tensor(obs, dtype=torch.float32, device=a.device)).cpu().numpy()
                act = np.where(driven[:, None], ta, sa)
            obs, _, done, fell, _ = env.step(act)
            falls += int(fell[~driven].sum()) if student is not None else int(fell.sum())
            ids = np.flatnonzero(done)
            if ids.size:
                env.reset(ids)
                obs = env.obs()
        t_roll = time.time() - t0
        X = np.concatenate(O)
        Y = np.concatenate(A)
        if len(X) > a.max_samples:                  # keep the newest
            k = len(X) - a.max_samples
            X, Y = X[k:], Y[k:]
        O, A = [X], [Y]
        if student is None:
            std = X.std(0)
            student = Student(c.obs_dim, c.n_joints, a.hidden, X.mean(0), np.where(std < 1e-3, 1.0, std)).to(a.device)
            opt = torch.optim.Adam(student.parameters(), lr=a.lr)
        Xt = torch.as_tensor(X, device=a.device)
        Yt = torch.as_tensor(Y, device=a.device)
        t1 = time.time()
        for _ in range(a.epochs):
            perm = torch.randperm(len(Xt), device=a.device)
            tot = 0.0
            for i in range(0, len(Xt), a.batch):
                j = perm[i:i + a.batch]
                loss = ((student(Xt[j]) - Yt[j]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(j)
        mse = tot / len(Xt)
        student_robots = int((~driven).sum()) if rnd else 0
        rec = dict(round=rnd, beta=round(beta, 4), samples=len(Xt), mse=mse, student_robots=student_robots,
                   student_falls=falls if rnd else None, rollout_s=round(t_roll, 1), train_s=round(time.time() - t1, 1))
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if rnd == a.rounds // 2:
            for g in opt.param_groups:
                g["lr"] = a.lr * 0.3

    student = student.to("cpu").eval()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": student.state_dict(), "hidden": a.hidden}, a.out / "student.pt")
    export_onnx(_Net(student), c, a.out / "policy.onnx", motion_name=motion.name if hasattr(motion, "name") else "",
                algo="distill")
    shutil.copy(a.teacher / "motion.npz", a.out / "motion.npz")
    n_par = sum(p.numel() for p in student.parameters())
    (a.out / "distill.json").write_text(json.dumps({
        "teacher": str((a.teacher / "policy.onnx").resolve()), "hidden": a.hidden, "params": n_par,
        "args": {k: str(v) for k, v in vars(a).items()}, "rounds": log}, indent=2))
    print(f"wrote {a.out}/policy.onnx ({n_par} params, {(a.out / 'policy.onnx').stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
