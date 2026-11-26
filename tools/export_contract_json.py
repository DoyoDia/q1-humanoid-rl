#!/usr/bin/env python3
"""Bundle the contract YAML/JSON files into one contract.json for non-Python consumers.

The browser and the C++ bridge should not each grow a YAML parser, and more importantly
they should not each re-derive anything. This emits exactly what a consumer needs, with
a content hash so a stale bundle is detectable rather than silently wrong.

    python3 tools/export_contract_json.py
"""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "robots" / "q1" / "contract"
SOURCES = ["joints.yaml", "gains.yaml", "rates.yaml", "obs_spec.json"]


def source_hash() -> str:
    h = hashlib.sha256()
    for name in SOURCES:
        h.update((CONTRACT / name).read_bytes())
    return h.hexdigest()[:16]


def main():
    joints = yaml.safe_load((CONTRACT / "joints.yaml").read_text(encoding="utf-8"))
    gains = yaml.safe_load((CONTRACT / "gains.yaml").read_text(encoding="utf-8"))
    rates = yaml.safe_load((CONTRACT / "rates.yaml").read_text(encoding="utf-8"))
    obs = json.loads((CONTRACT / "obs_spec.json").read_text(encoding="utf-8"))

    bundle = {
        "_generated_by": "tools/export_contract_json.py -- do not edit; edit the YAML sources",
        "_source_hash": source_hash(),
        "_sources": SOURCES,
        "n_joints": joints["n_joints"],
        "joint_names": [j["name"] for j in joints["joints"]],
        "lower": [j["lower"] for j in joints["joints"]],
        "upper": [j["upper"] for j in joints["joints"]],
        "effort": [j["effort"] for j in joints["joints"]],
        "velocity": [j["velocity"] for j in joints["joints"]],
        "command": joints["command"],
        "rates": rates,
        "contracts": {
            name: {
                "obs_dim": spec["obs_dim_valid"],
                "onnx_obs_dim": spec.get("obs_dim_onnx_input", spec["obs_dim_valid"]),
                "frame_stack": spec.get("frame_stack", 1),
                "policy_action_dim": spec.get("policy_action_dim", spec.get("action_dim", 22)),
                "fields": [{"name": f["name"], "start": f["start"], "size": f["size"]} for f in spec["fields"]],
                "obs_scales": spec["obs_scales"],
                "obs_clip": spec["clips"]["obs"],
                "action_clip": spec["clips"]["action"],
                "action_scale": gains[name]["action_scale"],
                "kps": gains[name]["kps"],
                "kds": gains[name]["kds"],
                "default_dof_pos": gains[name]["default_dof_pos"],
                "unverified": spec.get("unverified"),
            }
            for name, spec in obs.items()
            if not name.startswith("$") and name in gains
        },
    }
    out = CONTRACT / "contract.json"
    out.write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)  hash {bundle['_source_hash']}")
    for name, c in bundle["contracts"].items():
        flag = f"  [unverified: {c['unverified']}]" if c["unverified"] else ""
        print(f"  {name:22s} obs {c['obs_dim']}->{c['onnx_obs_dim']}  "
              f"frame_stack {c['frame_stack']}  action_scale {c['action_scale']}{flag}")


if __name__ == "__main__":
    main()
