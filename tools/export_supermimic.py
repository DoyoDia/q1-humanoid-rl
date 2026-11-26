"""A SuperMimic checkpoint trained with --env.obs-mode mimic -> a q1_rl run dir the deploy tools take.

    python tools/export_supermimic.py SM_LOG/ckpt_latest.pt --motion runs/motions/clip.npz -o runs/<name>
    python tools/make_deploy_package.py runs/<name> --name <name>        # the robot's three files

The policy (_ref/SuperMimic-Reproduced, SM_ROBOT=q1, obs_mode mimic) is one MLP on the
MIMIC_OPENSOURCE_V2 runtime's 119-d observation with its input scaling inside, so it goes
through q1_rl's own export_onnx unchanged: same input/output names, width and contract
metadata as a PPO export. Written: policy.onnx, motion.npz (the reference the runtime plays,
which must be the clip it was trained on) and supermimic.json (where it came from).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
SM_ROOT = Path(os.environ.get("SM_ROOT", ROOT / "_ref/SuperMimic-Reproduced"))
sys.path.insert(0, str(ROOT))

from q1_rl.contract import Contract  # noqa: E402
from q1_rl.export import export_onnx  # noqa: E402


class _Actor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        return self.policy.forward_mimic(obs)


class _Net(nn.Module):
    """What export_onnx expects of a trainer's network: .norm and .actor."""

    def __init__(self, policy):
        super().__init__()
        self.norm = nn.Identity()           # SuperMimic's fixed input scales are inside forward_mimic
        self.actor = _Actor(policy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--motion", type=Path, required=True, help="q1_rl motion npz the policy was trained on")
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()

    os.environ["SM_ROBOT"] = "q1"
    sys.path.insert(0, str(SM_ROOT / "src"))
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    policy, cfg = load_policy_from_checkpoint(a.ckpt, "cpu")
    if not policy.cfg.mimic:
        raise SystemExit(f"{a.ckpt}: trained with obs_mode {cfg['env'].get('obs_mode', 'paper')!r}; only 'mimic' "
                         "policies see the runtime's 119-d observation")
    c = Contract("mimic_opensource_v2")
    q0 = np.asarray(policy.q0, np.float64)
    if np.abs(q0 - c.default_dof_pos).max() > 1e-6:
        raise SystemExit(f"policy q0 differs from the contract's default_dof_pos by {np.abs(q0 - c.default_dof_pos).max():.2e}")
    policy = policy.float().eval()
    a.out.mkdir(parents=True, exist_ok=True)
    name = a.motion.stem
    export_onnx(_Net(policy), c, a.out / "policy.onnx", motion_name=name, algo="supermimic")
    shutil.copy(a.motion, a.out / "motion.npz")
    (a.out / "supermimic.json").write_text(json.dumps({
        "checkpoint": str(a.ckpt.resolve()), "iteration_config": cfg.get("schedule", {}),
        "env": cfg["env"], "policy": {k: v for k, v in cfg["policy"].items() if k != "mimic_q0"},
    }, indent=2, default=str))
    print(f"wrote {a.out}/policy.onnx (119 -> 22), motion.npz ({name}), supermimic.json")


if __name__ == "__main__":
    main()
