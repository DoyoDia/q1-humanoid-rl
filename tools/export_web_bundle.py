#!/usr/bin/env python3
"""Collect everything the browser needs to run one trained policy, into web/run/<name>/.

The browser gets the self-contained model (q1_web.xml, no external assets), the ONNX, the
contract bundle and the reference motion as JSON -- JavaScript has no npz reader and the
motion is small at policy rate, so JSON is the right shape here rather than a new format.

    python3 tools/export_web_bundle.py runs/squat_mimic_opensource_v2
    python3 tools/export_web_bundle.py runs/bt_v3 --label 广播体操 --audio runs/bt/bt.mp3
    python3 tools/export_web_bundle.py runs/stand_v1 --label 站立 --role idle

Bundles given a --label are listed in web/run/index.json, which the interactive viewer
(sim2sim_view.html) offers as its actions; the --role idle one keeps the robot standing
between them.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from q1_rl.motion import Motion  # noqa: E402


def write_index(root: Path):
    """web/run/index.json: every labelled bundle, in --order."""
    runs = []
    for meta in sorted(root.glob("*/meta.json")):
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("label"):
            runs.append(dict(run=meta.parent.name, label=m["label"], role=m.get("role", "clip"), order=m.get("order", 50)))
    runs.sort(key=lambda r: (r["role"] != "idle", r["role"] != "walk", r["order"], r["label"]))
    (root / "index.json").write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--audio", type=Path, default=None, help="music for the clip (mp3), played by the viewer")
    p.add_argument("--label", default=None, help="name shown in the viewer; labelled bundles go into web/run/index.json")
    p.add_argument("--role", default="clip", choices=["clip", "idle", "walk"],
                   help="idle: the policy the viewer stands the robot with between clips; "
                        "walk: the velocity-command policy WASD drives (train.py --task velocity)")
    p.add_argument("--order", type=int, default=50, help="position among the viewer's actions (small first)")
    a = p.parse_args()

    run = Path(a.run)
    out = a.out or ROOT / "web" / "run" / run.name
    out.mkdir(parents=True, exist_ok=True)

    # The page builds observations from today's contract; a policy exported against an older
    # layout (the 116-wide one before 3917ae4) would run on misplaced inputs without an error.
    contract = json.loads((ROOT / "robots" / "q1" / "contract" / "contract.json").read_text(encoding="utf-8"))
    meta_onnx = {p.key: p.value for p in onnx.load(str(run / "policy.onnx"), load_external_data=False).metadata_props}
    want = contract["contracts"][meta_onnx.get("q1_contract", "mimic_opensource_v2")]["obs_dim"]
    if int(meta_onnx.get("q1_obs_dim", -1)) != want:
        sys.exit(f"{run}: policy was exported with obs_dim {meta_onnx.get('q1_obs_dim')}, "
                 f"the contract now builds {want} -- retrain it")

    shutil.copyfile(ROOT / "robots" / "q1" / "q1_web.xml", out / "model.xml")
    shutil.copyfile(ROOT / "robots" / "q1" / "contract" / "contract.json", out / "contract.json")
    shutil.copyfile(run / "policy.onnx", out / "policy.onnx")

    m = Motion.load_npz(run / "motion.npz")
    (out / "motion.json").write_text(json.dumps({
        "name": m.name, "fps": m.fps, "n_frames": m.n_frames,
        # only a clip whose end runs back into its start may be looped (as in training)
        "cyclic": bool(m.cyclic),
        "joint_pos": np.asarray(m.joint_pos, np.float32).round(6).tolist(),
        "joint_vel": np.asarray(m.joint_vel, np.float32).round(6).tolist(),
        # pelvis orientation (w, x, y, z) per frame: the contract's ref_gravity comes from it
        "root_quat": np.asarray(m.body_quat_w[:, 0], np.float32).round(6).tolist(),
        # pelvis position per frame: the viewer's reference ghost and its starting pose
        "root_pos": np.asarray(m.body_pos_w[:, 0], np.float32).round(5).tolist(),
    }), encoding="utf-8")
    if a.audio:                                  # music the viewer plays in step with the clip
        shutil.copyfile(a.audio, out / ("audio" + Path(a.audio).suffix))

    # PPO runs keep history.json; SuperMimic exports and distilled students have none
    hf = run / "history.json"
    hist = json.loads(hf.read_text(encoding="utf-8")) if hf.is_file() else []
    (out / "meta.json").write_text(json.dumps({
        "run": run.name,
        "motion": m.name,
        "contract": meta_onnx.get("q1_contract", "mimic_opensource_v2"),
        "iters": len(hist) or None,
        "final": {k: hist[-1][k] for k in ("reward", "ep_len", "fall_rate", "steps")} if hist else None,
        "audio": ("audio" + Path(a.audio).suffix) if a.audio else None,
        "label": a.label, "role": a.role, "order": a.order,
        # a walking policy's command ranges and CPG period law (train.py's env.json)
        "walk": json.loads((run / "env.json").read_text(encoding="utf-8")).get("walk") if (run / "env.json").exists() else None,
    }, indent=1), encoding="utf-8")
    write_index(out.parent)

    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"wrote {out}  ({total/1024:.0f} KB total)")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:16s} {f.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    main()
