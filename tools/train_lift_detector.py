"""Train a lift detector for one of our policies, in the vendor's format, and export it.

    python tools/train_lift_detector.py runs/bt_v1 [--envs 512 --steps 3000]

On the robot, a MIMIC_OPENSOURCE_V2 motion can carry a lift_detector (CustomConfigs in
rl/mimic_opensource_v2_config.yaml): an ONNX net that reads the last history_len (10)
observations' slice obs[44:119] -- reference gravity, gravity, gyro, joint pos, joint vel,
last action, 75 numbers -- as one [1, 750] input and outputs one logit. mc smooths the logit
(EMA, logit_ema_alpha), and when sigmoid > lift_threshold for lift_debounce frames it
switches to damping. The vendor ships one detector per policy: the slice holds the policy's
own actions, so a detector is tied to the policy it was trained beside.

This rolls the policy out in the training env with lifts in half the episodes (base raised
and held, EnvConfig.lift_prob), labels each frame lifted (base raised > 3 cm) or not, and
trains a small MLP. Negatives include everything else the policy does -- pushes, and in
broadcast gymnastics the jumping section, whose flight is the obvious false alarm. Which end
of the 750 numbers is the newest frame is not known (mc's code does not say); training sees
both orders at random, so the net does not depend on it.

It then replays held-out rollouts through mc's logic (EMA, threshold, debounce, skipped
first/last frames) and reports: share of lifts caught and how fast, and false alarms per
minute of normal play. Writes <run>/deploy/lift_detector.onnx and a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl.algos.ppo import ActorCritic  # noqa: E402
from q1_rl.contract import Contract  # noqa: E402
from q1_rl.envs.motion_tracking import EnvConfig, MotionTrackingEnv  # noqa: E402
from q1_rl.evaluate import deterministic_policy  # noqa: E402
from q1_rl.motion import Motion  # noqa: E402

START, NOBS, HIST = 44, 75, 10
RAISED = 0.03               # m: lifted and this far up = feet off the floor = positive


def rollout(run: Path, envs: int, steps: int, seed: int, backend: str, lift_prob: float):
    """(T, N, 75) obs slices, (T, N) labels (1 lifted, 0 not, -1 ambiguous), (T, N) frames
    since the episode began, (T, N) frames until the clip ends, (T, N) fell this step,
    (T, N) clip frame."""
    c = Contract()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/q1/q1_physics.xml"))
    motion = Motion.load_npz(run / "motion.npz")
    cfg = replace(EnvConfig(), num_envs=envs, backend=backend, adaptive_sampling=False,
                  episode_seconds=8.0, push_vel=0.5, push_ang_vel=0.5, lift_prob=lift_prob,
                  lift_start_s=(0.6, 6.0), lift_hold_s=(0.5, 2.5))
    env = MotionTrackingEnv(model, motion, c, cfg, seed=seed)
    env.reset(np.arange(envs))
    net = ActorCritic.from_state_dict(torch.load(run / "policy.pt", map_location="cpu"))
    act = deterministic_policy(net.eval())

    X = np.zeros((steps, envs, NOBS), np.float32)
    Y = np.zeros((steps, envs), np.int8)
    age = np.zeros((steps, envs), np.int32)
    left = np.zeros((steps, envs), np.int32)
    fellv = np.zeros((steps, envs), bool)
    phase = np.zeros((steps, envs), np.int32)
    obs = env.obs()
    with torch.no_grad():
        for t in range(steps):
            obs, _, done, fell, _ = env.step(act(torch.as_tensor(obs)).numpy())
            fellv[t] = fell
            phase[t] = env.phase
            X[t] = obs[:, START:START + NOBS]
            raised = env.qpos[:, 2] - env.lift_pose[:, 2]
            Y[t] = np.where(env.lifted, np.where(raised > RAISED, 1, -1), 0)
            age[t] = env.steps
            left[t] = env.T - 1 - env.phase
            if done.any():
                ids = np.flatnonzero(done)
                env.reset(ids)
                obs = env.obs()
    return X, Y, age, left, fellv, phase


def windows(X, Y, age, left, keep_neg=0.35, seed=0):
    """(t, env) with a full history inside one episode -> (M, 10, 75) oldest first. All the
    lifted ones, a keep_neg share of the rest (neighbouring frames say nearly the same)."""
    keep = (age >= HIST) & ((Y == 1) | ((Y == 0) & (np.random.default_rng(seed).random(Y.shape) < keep_neg)))
    t, n = np.nonzero(keep)
    idx = t[:, None] - np.arange(HIST - 1, -1, -1)[None, :]          # oldest .. newest
    return X[idx, n[:, None]], Y[t, n].astype(np.float32)


class Detector(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(HIST * NOBS, 256), nn.LayerNorm(256), nn.ELU(),
                                 nn.Linear(256, 128), nn.LayerNorm(128), nn.ELU(), nn.Linear(128, 1))

    def forward(self, obs):                                  # (B, 750) -> (B,)
        x = obs.view(-1, HIST, NOBS)
        x = ((x - self.mean) / self.std).view(-1, HIST * NOBS)
        return self.net(x).view(-1)


class Export(nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det

    def forward(self, obs):                                  # [1, 750] -> logits [1]
        return self.det(obs)


def train(W, Y, device, epochs=6, batch=4096):
    mean = W.reshape(-1, NOBS).mean(0)
    std = W.reshape(-1, NOBS).std(0) + 1e-3
    det = Detector(mean, std).to(device)
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3, weight_decay=1e-4)
    # no up-weighting of the lifted class: a false alarm (damping a robot mid-routine) is the
    # costlier mistake, and the windows are already ~1/3 lifted
    lossf = nn.BCEWithLogitsLoss()
    Wt, Yt = torch.as_tensor(W), torch.as_tensor(Y)
    n = len(Wt)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * ((n + batch - 1) // batch))
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            x = Wt[b].to(device)
            flip = torch.rand(len(b), device=device) < 0.5           # either history order
            x = torch.where(flip[:, None, None], x.flip(1), x)
            x = x + 0.02 * torch.randn_like(x) * det.std           # a little extra sensor noise
            loss = lossf(det(x.reshape(len(b), -1)), Yt[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        print(f"  epoch {ep + 1}/{epochs}  loss {tot / n:.4f}", flush=True)
    return det.eval()


def logits(det, X, device, reverse):
    T, N, _ = X.shape
    out = np.zeros((T, N), np.float32)
    with torch.no_grad():
        for t in range(HIST - 1, T):
            h = torch.as_tensor(X[t - HIST + 1:t + 1]).permute(1, 0, 2)    # (N, 10, 75) oldest first
            if reverse:
                h = h.flip(1)
            out[t] = det(h.reshape(N, -1).to(device)).cpu().numpy()
    return out


def decide(logit, Y, age, left, fell, phase, alpha, thr, debounce, skip_start, skip_end, fps, before_fall=1.0):
    """mc's decision on every env's stream (EMA of the logit, sigmoid > thr for `debounce`
    frames, inactive in the clip's first/last skip frames) -> lift catch stats and false
    alarms. An alarm within before_fall seconds of a fall in the same episode is not counted
    false: damping a robot that is going down anyway costs nothing."""
    T, N = logit.shape
    # frames from which a fall follows within before_fall s, same episode
    soon = np.zeros((T, N), bool)
    nxt = np.full(N, 10 ** 9)
    for t in range(T - 1, -1, -1):
        nxt = np.where(fell[t], t, nxt)
        soon[t] = nxt - t <= before_fall * fps
        nxt = np.where(age[t] == 1, 10 ** 9, nxt)                 # an episode starts here
    ema = np.zeros(N)
    run = np.zeros(N, np.int64)
    fired = np.zeros(N, bool)
    lift_t0 = np.full(N, -1)
    lifts = caught = alarms = excused = 0
    delays, where = [], []
    normal = 0
    for t in range(T):
        new = age[t] == 1
        ema = np.where(new, logit[t], alpha * logit[t] + (1 - alpha) * ema)
        run[new] = 0
        fired[new] = False
        lift_t0[new] = -1
        active = (age[t] > max(skip_start, HIST)) & (left[t] > skip_end)
        hit = active & (1 / (1 + np.exp(-ema)) > thr)
        run = np.where(hit, run + 1, 0)
        start = (Y[t] == 1) & (lift_t0 < 0)
        lift_t0[start] = t
        lifts += int(start.sum())
        normal += int(((Y[t] == 0) & active).sum())
        f = (run >= debounce) & ~fired
        fired |= f
        c = f & (lift_t0 >= 0)
        caught += int(c.sum())
        delays += list((t - lift_t0[c]) / fps)
        fa = f & (lift_t0 < 0) & (Y[t] == 0)
        excused += int((fa & soon[t]).sum())
        bad = fa & ~soon[t]
        alarms += int(bad.sum())
        where += list(phase[t][bad] / fps)
    mins = normal / fps / 60
    return dict(lifts=lifts, caught=caught, catch_rate=caught / max(lifts, 1),
                delay_median_s=float(np.median(delays)) if delays else None,
                delay_p90_s=float(np.percentile(delays, 90)) if delays else None,
                false_alarms=alarms, excused_before_fall=excused, normal_minutes=mins,
                false_alarms_per_min=alarms / max(mins, 1e-9),
                false_alarm_clip_s=sorted(round(float(w), 1) for w in where))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-steps", type=int, default=1500)
    ap.add_argument("--backend", default="mjbatch")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=6)
    # mc's settings that will go in the config. The vendor uses 0.35 / 0.6 / 1 / 25 / 25; for
    # broadcast gymnastics that fired 0.22 times a minute in normal play (pushes, jumps), and
    # with 3 frames of debounce at 0.9: no false alarm in 108 min, 99.7% of lifts within 0.14 s
    ap.add_argument("--ema", type=float, default=0.35)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--debounce", type=int, default=3)
    ap.add_argument("--skip", type=int, default=25)
    a = ap.parse_args()
    c = Contract()

    print(f"rollout {a.envs} envs x {a.steps} steps (train)", flush=True)
    X, Y, age, left, _, _ = rollout(a.run, a.envs, a.steps, 1, a.backend, 0.5)
    W, L = windows(X, Y, age, left)
    print(f"  {len(L)} windows, {L.mean() * 100:.1f}% lifted", flush=True)
    det = train(W, L, a.device, a.epochs)
    del W, L, X

    print(f"rollout {a.envs // 2} envs x {a.eval_steps} steps (held out)", flush=True)
    Xe, Ye, ae, le, fe, pe = rollout(a.run, a.envs // 2, a.eval_steps, 2, a.backend, 0.3)
    report = {"sweep": []}
    lg = {rev: logits(det, Xe, a.device, rev) for rev in (False, True)}
    for rev in (False, True):
        r = decide(lg[rev], Ye, ae, le, fe, pe, a.ema, a.threshold, a.debounce, a.skip, a.skip, c.policy_hz)
        report["newest_first" if rev else "oldest_first"] = r
        print(f"  {'newest-first' if rev else 'oldest-first'}: caught {r['caught']}/{r['lifts']} lifts "
              f"({r['catch_rate'] * 100:.0f}%), delay median {r['delay_median_s']}s p90 {r['delay_p90_s']}s; "
              f"false alarms {r['false_alarms']} in {r['normal_minutes']:.0f} min "
              f"({r['false_alarms_per_min']:.3f}/min, {r['excused_before_fall']} more just before a fall)", flush=True)
        if r["false_alarm_clip_s"]:
            h = np.histogram(r["false_alarm_clip_s"], bins=np.arange(0, 280, 10))[0]
            print("    false alarms per 10 s of the clip:", " ".join(f"{10*i}:{k}" for i, k in enumerate(h) if k))
    print("  sweep (worse of the two orders): threshold debounce -> catch%, delay p90, false/min")
    for thr in (0.6, 0.8, 0.9, 0.95, 0.98):
        for db in (1, 3, 5, 8):
            rr = [decide(lg[rev], Ye, ae, le, fe, pe, a.ema, thr, db, a.skip, a.skip, c.policy_hz) for rev in (False, True)]
            row = dict(threshold=thr, debounce=db, catch=min(x["catch_rate"] for x in rr),
                       delay_p90=max((x["delay_p90_s"] or 0) for x in rr),
                       false_per_min=max(x["false_alarms_per_min"] for x in rr))
            report["sweep"].append(row)
            print(f"    {thr:.2f} {db}: {row['catch'] * 100:5.1f}%  {row['delay_p90']:.2f}s  {row['false_per_min']:.3f}", flush=True)

    out = a.run / "deploy"
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "lift_detector.onnx"
    exp = Export(det.cpu()).eval()
    torch.onnx.export(exp, torch.zeros(1, HIST * NOBS), str(onnx_path), input_names=["obs"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx_path))
    x = np.random.default_rng(0).normal(size=(1, HIST * NOBS)).astype(np.float32)
    ref = exp(torch.as_tensor(x)).detach().numpy()
    got = s.run(None, {"obs": x})[0]
    assert got.shape == (1,) and np.allclose(got, ref, atol=1e-4), (got, ref)
    report["settings"] = dict(ema=a.ema, threshold=a.threshold, debounce=a.debounce, skip=a.skip)
    (out / "lift_detector_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {onnx_path} (input obs [1, 750], output logits [1]) and lift_detector_report.json")


if __name__ == "__main__":
    main()
