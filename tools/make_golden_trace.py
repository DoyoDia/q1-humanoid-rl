#!/usr/bin/env python3
"""Record a golden trace: raw state in, observation out, for N policy steps.

This is the artefact that keeps five independent observation builders agreeing --
the training env, the ONNX exporter, the browser, the on-robot bridge, and (once the
deployment spike lands) the vendor's own mc_main. Each one replays the recorded inputs
and must reproduce the recorded observations.

The trace is generated from a real MuJoCo rollout of q1_physics.xml so the states are
physically reachable, not synthetic noise: a policy that only ever sees plausible
quaternions will not exercise the gravity projection the way a falling robot does.
The "policy" here is a fixed pseudo-random function -- we are testing plumbing, not skill.

    python3 tools/make_golden_trace.py [--steps 100] [--contract mimic_opensource_v2]
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from q1_rl.contract import Contract  # noqa: E402


def pseudo_policy(obs: np.ndarray, step: int, n: int) -> np.ndarray:
    """Deterministic, obs-dependent, no learning. Bounded so the robot stays reachable."""
    rng = np.random.default_rng(1234 + step)
    return (0.3 * np.tanh(obs[:n] if obs.size >= n else np.resize(obs, n))
            + 0.2 * rng.standard_normal(n)).astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=ROOT / "robots" / "q1" / "q1_physics.xml")
    p.add_argument("--contract", default="mimic_opensource_v2")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default=ROOT / "robots" / "q1" / "contract" / "golden_trace.json")
    a = p.parse_args()

    c = Contract(a.contract)
    model = mujoco.MjModel.from_xml_path(str(a.model))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)

    # A stand-still reference motion: enough to exercise the 44 motion_input dims.
    ref_pos = np.tile(c.default_dof_pos, (a.steps, 1))
    ref_pos += 0.05 * np.sin(np.linspace(0, 4 * np.pi, a.steps))[:, None]
    ref_vel = np.gradient(ref_pos, 1.0 / c.policy_hz, axis=0)
    # a reference pelvis that rocks (roll and pitch up to ~0.15 rad), to exercise ref_gravity
    th = np.linspace(0, 2 * np.pi, a.steps)
    roll, pitch = 0.15 * np.sin(th), 0.1 * np.cos(2 * th)
    ref_quat = np.stack([np.cos(roll / 2) * np.cos(pitch / 2), np.sin(roll / 2) * np.cos(pitch / 2),
                         np.cos(roll / 2) * np.sin(pitch / 2), -np.sin(roll / 2) * np.sin(pitch / 2)], 1)

    records = []
    prev_action = np.zeros(c.n_joints)
    prev_target = c.default_dof_pos.copy()

    for t in range(a.steps):
        state = dict(
            base_quat=data.qpos[3:7].copy(),
            base_ang_vel=data.qvel[3:6].copy(),
            joint_pos=data.qpos[7:].copy(),
            joint_vel=data.qvel[6:].copy(),
            prev_action=prev_action.copy(),
            motion_ref_pos=ref_pos[t],
            motion_ref_vel=ref_vel[t],
            motion_ref_quat=ref_quat[t],
        )
        obs = c.build_obs(state)
        action = pseudo_policy(obs, t, c.n_joints)
        target = c.action_to_target(action, prev_target)

        records.append(dict(
            step=t,
            state={k: np.asarray(v, np.float64).round(12).tolist() for k, v in state.items()},
            obs=obs.astype(np.float32).tolist(),
            action=action.round(12).tolist(),
            target=target.round(12).tolist(),
        ))

        data.ctrl[:] = target
        for _ in range(c.decimation):
            mujoco.mj_step(model, data)
        prev_action, prev_target = action, target

    trace = dict(
        _comment=[
            "Golden trace for the Q1 deployment contract. Replay `state` through your own",
            "observation builder and compare with `obs`; tolerance 1e-6 absolute.",
            "Exact bitwise equality across languages is not expected and not required --",
            "float32 summation order differs. 1e-6 is ~10x float32 epsilon at these magnitudes.",
            "Also replay `action` through your action->target mapping and compare with `target`.",
        ],
        contract=a.contract,
        obs_dim=c.obs_dim,
        onnx_obs_dim=c.onnx_obs_dim,
        n_joints=c.n_joints,
        joint_names=c.joint_names,
        policy_hz=c.policy_hz,
        decimation=c.decimation,
        tolerance=1e-6,
        model=Path(a.model).name,
        fields={k: [f.start, f.size] for k, f in c.fields.items()},
        records=records,
    )
    Path(a.out).write_text(json.dumps(trace), encoding="utf-8")

    o = np.array([r["obs"] for r in records])
    print(f"wrote {a.out}  ({Path(a.out).stat().st_size / 1024:.0f} KB)")
    print(f"  {a.steps} steps of contract {a.contract!r}, obs {c.obs_dim} wide")
    print(f"  obs range [{o.min():.4f}, {o.max():.4f}]   mean |obs| {np.abs(o).mean():.4f}")
    print(f"  base height {data.qpos[2]:.4f} m after {a.steps} steps "
          f"({'still standing' if data.qpos[2] > 0.25 else 'FELL -- fine for a plumbing test'})")
    gv = o[:, c.fields["gvec"].sl]
    print(f"  gvec norm range [{np.linalg.norm(gv, axis=1).min():.6f}, "
          f"{np.linalg.norm(gv, axis=1).max():.6f}]  (should be ~1)")


if __name__ == "__main__":
    main()
