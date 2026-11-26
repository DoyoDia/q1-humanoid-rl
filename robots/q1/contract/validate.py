#!/usr/bin/env python3
"""Check that the contract files agree with each other and with the MJCF.

This is the test that stops the five obs-builders (training env, ONNX exporter, the
browser, the on-robot bridge, and the vendor's own mc_main) from silently drifting apart.
Run it in CI; run it before every deploy.

    python3 validate.py [--mjcf path] [--onnx policy.onnx]

Exit status 0 = contract is self-consistent.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
FAIL, WARN = [], []


def check(cond, msg, unverified=False):
    """unverified=True downgrades a failure to a warning: a known-open question in the
    vendor's own config, recorded rather than pretended away. It does not fail CI."""
    print(("  ok   " if cond else "  WARN " if unverified else "  FAIL ") + msg)
    if not cond:
        (WARN if unverified else FAIL).append(msg)


def load_yaml(path):
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        sys.exit("need pyyaml: pip install pyyaml")


def main():
    p = argparse.ArgumentParser()
    vendor = HERE / "../../../_ref/q1_v3/xml/q1_v3_mc.xml"      # the official model, when unpacked there
    p.add_argument("--mjcf", default=vendor if vendor.exists() else HERE / "../q1_physics.xml",
                   help="default: the official MJCF in _ref/q1_v3, else robots/q1/q1_physics.xml built from it")
    p.add_argument("--onnx", help="also check an exported policy's input/output shapes")
    a = p.parse_args()

    joints = load_yaml(HERE / "joints.yaml")
    gains = load_yaml(HERE / "gains.yaml")
    rates = load_yaml(HERE / "rates.yaml")
    obs = json.loads((HERE / "obs_spec.json").read_text(encoding="utf-8"))

    names = [j["name"] for j in joints["joints"]]
    n = joints["n_joints"]

    print("joints.yaml")
    check(len(names) == n == 22, f"22 joints ({len(names)})")
    check([j["index"] for j in joints["joints"]] == list(range(n)), "indices are 0..21 in order")
    check(len(set(names)) == n, "no duplicate joint names")
    for j in joints["joints"]:
        if not j["lower"] < j["upper"]:
            check(False, f"{j['name']}: lower < upper")
        if not (j["effort"] > 0 and j["velocity"] > 0):
            check(False, f"{j['name']}: positive effort/velocity caps")

    print("gains.yaml")
    for key, g in gains.items():
        for f in ("kps", "kds", "default_dof_pos"):
            check(len(g[f]) == n, f"{key}.{f} has {n} entries (got {len(g[f])})")
        check(all(v >= 0 for v in g["kps"] + g["kds"]), f"{key}: gains are non-negative")

    print("gains vs joints: nominal pose inside the URDF limits")
    for key, g in gains.items():
        bad = [
            (names[i], round(q, 4), joints["joints"][i]["lower"], joints["joints"][i]["upper"])
            for i, q in enumerate(g["default_dof_pos"])
            if not joints["joints"][i]["lower"] <= q <= joints["joints"][i]["upper"]
        ]
        check(not bad, f"{key}.default_dof_pos within limits" + (f" -- outside: {bad}" if bad else ""))

    print("obs_spec.json")
    for key, spec in obs.items():
        if key.startswith("$"):
            continue
        fields = spec["fields"]
        cursor, gaps = 0, []
        for f in fields:
            if f["start"] != cursor:
                gaps.append((f["name"], cursor, f["start"]))
            cursor = f["start"] + f["size"]
        check(not gaps, f"{key}: fields are contiguous from 0" + (f" -- gaps {gaps}" if gaps else ""))
        check(cursor == spec["obs_dim_valid"],
              f"{key}: fields sum to obs_dim_valid ({cursor} vs {spec['obs_dim_valid']})")
        check(spec["action_dim"] == n, f"{key}: action_dim == {n}")
        # A policy may drive fewer joints than the robot has (walk: 13 of 22, the rest
        # position-controlled); its prev_action is then as wide as its own output.
        pol = spec.get("policy_action_dim", n)
        for f in fields:
            if f["name"] == "prev_action" and f["size"] == pol:
                continue
            if f["size"] not in (n, 1) and f["name"].startswith(("joint_", "motion_", "prev_action")):
                check(f["size"] == n,
                      f"{key}.{f['name']}: joint-ordered field should be {n} wide, is {f['size']}",
                      unverified=f.get("unverified", False))

    print("rates.yaml")
    check(abs(rates["sim"]["timestep"] * rates["sim"]["decimation"] - 1.0 / rates["policy_hz"]) < 1e-12,
          f"timestep*decimation == 1/policy_hz ({rates['sim']['timestep']}*{rates['sim']['decimation']} "
          f"vs 1/{rates['policy_hz']})")
    check(rates["robot"]["joint_command_hz"] >= 500, "joint command >= 500 Hz (watchdog floor)")

    print("MJCF")
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(a.mjcf))
        mj_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                     for i in range(m.njnt) if m.jnt_type[i] > 1]
        check(m.nu == n, f"MJCF has {n} actuators (has {m.nu})")
        check(set(mj_joints) == set(names), "MJCF hinge joints == contract joints")
        for j in joints["joints"]:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
            lo, hi = m.jnt_range[i]
            if abs(lo - j["lower"]) > 1e-6 or abs(hi - j["upper"]) > 1e-6:
                check(False, f"{j['name']}: MJCF range {lo:.5g},{hi:.5g} != contract {j['lower']},{j['upper']}")
        check(abs(m.opt.timestep - rates["sim"]["timestep"]) < 1e-12,
              f"MJCF timestep == rates.yaml ({m.opt.timestep} vs {rates['sim']['timestep']})")
    except ImportError:
        print("  skip  mujoco not installed")

    print("contract.json (the bundle non-Python consumers read)")
    bundle_path = HERE / "contract.json"
    if not bundle_path.exists():
        check(False, "contract.json exists -- run tools/export_contract_json.py")
    else:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        h = hashlib.sha256()
        for src in bundle["_sources"]:
            h.update((HERE / src).read_bytes())
        check(h.hexdigest()[:16] == bundle["_source_hash"],
              "contract.json is up to date with the YAML sources "
              "(stale -> rerun tools/export_contract_json.py)")
        check(bundle["joint_names"] == names, "contract.json joint order == joints.yaml")
        for key, spec in obs.items():
            if key.startswith("$") or key not in bundle["contracts"]:
                continue
            check(bundle["contracts"][key]["obs_dim"] == spec["obs_dim_valid"],
                  f"contract.json {key}.obs_dim == obs_spec.json")

    print("golden trace (Python replay)")
    trace_path = HERE / "golden_trace.json"
    if not trace_path.exists():
        check(False, "golden_trace.json exists -- run tools/make_golden_trace.py", unverified=True)
    else:
        sys.path.insert(0, str(ROOT))
        import numpy as np
        from q1_rl.contract import Contract as PyContract
        tr = json.loads(trace_path.read_text(encoding="utf-8"))
        c = PyContract(tr["contract"])
        check(c.obs_dim == tr["obs_dim"], f"trace obs_dim matches the contract ({tr['obs_dim']})")
        worst_obs = worst_tgt = 0.0
        prev_target = None
        for r in tr["records"]:
            state = {k: np.asarray(v, np.float64) for k, v in r["state"].items()}
            worst_obs = max(worst_obs, float(np.abs(c.build_obs(state) - np.asarray(r["obs"], np.float32)).max()))
            tgt = c.action_to_target(np.asarray(r["action"]), prev_target)
            worst_tgt = max(worst_tgt, float(np.abs(tgt - np.asarray(r["target"])).max()))
            prev_target = np.asarray(r["target"])
        tol = tr["tolerance"]
        check(worst_obs <= tol, f"q1_rl/contract.py reproduces the trace obs (max {worst_obs:.3e} <= {tol:g})")
        check(worst_tgt <= tol, f"q1_rl/contract.py reproduces the trace targets (max {worst_tgt:.3e} <= {tol:g})")
        print(f"        (JS side: serve the repo and open web/golden_check.html)")

    if a.onnx:
        print("ONNX")
        import onnx
        mo = onnx.load(a.onnx)
        def dims(vi):
            return [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]
        ins = [(v.name, dims(v)) for v in mo.graph.input]
        outs = [(v.name, dims(v)) for v in mo.graph.output]
        print(f"        inputs {ins}  outputs {outs}")
        meta = {p.key: p.value for p in mo.metadata_props}
        name = meta.get("q1_contract")
        check(name in obs, f"ONNX metadata names a known contract ({name!r})")
        if name in obs:
            # Exactly the width the runtime feeds for THAT contract. An earlier version
            # accepted any contract's valid or padded width, which let a 116-wide policy
            # through for a runtime that feeds 119.
            want = obs[name].get("obs_dim_onnx_input", obs[name]["obs_dim_valid"])
            got = ins[0][1][-1] if ins and ins[0][1] else None
            check(got == want, f"ONNX input width == runtime obs_nums for {name} ({got} vs {want})")
        check(any(d and d[-1] == n for _, d in outs), f"ONNX output width == {n}")

    print()
    for w in WARN:
        print(f"warning (known-open): {w}")
    if FAIL:
        print(f"{len(FAIL)} check(s) failed")
        return 1
    print(f"contract is self-consistent ({len(WARN)} known-open warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
