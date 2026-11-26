"""A trained run -> the three files the robot's MIMIC_OPENSOURCE_V2 runtime loads.

    python tools/make_deploy_package.py runs/<run> --name dance

writes runs/<run>/deploy/:
  q1rl_<name>.onnx   the policy, output renamed `actions` and batch fixed at 1 like the
                     vendor's; checked against the original to 1e-5
  q1rl_<name>.npz    the reference, exactly the arrays and dtypes of the vendor's files
  q1rl_<name>.csv    the reference gravity, one row per frame (frame_id,gx,gy,gz) -- the
                     runtime reads obs[44:47] from this file, and only for a motion whose
                     gravity_guard is enabled (deploy/robot/install.sh does that). It is the
                     reference pelvis gravity, the same function of the same npz the
                     training env feeds (q1_rl.contract.quat_to_gravity of body_quat_w[:, 0]).
  q1rl_<name>.detector.onnx   only if tools/train_lift_detector.py has written
                     deploy/lift_detector.onnx for this run: the same file under the name
                     install.sh looks for, which turns on mc's lift detection for the motion.

Run tools/check_vendor_policies.py first: it is what tells you the observation layout these
files are trained against is the one the runtime builds.

A walk run (train.py --task walk, ONNX metadata q1_contract = walk) packages for the Walk
runtime instead: q1rlw_<name>.onnx only, with the vendor walking policy's exact signature
(observations [1, 71], observation_history [1, 355] -> action [1, 22]), checked against both
the run's policy.onnx and the vendor's file. Gate it with tools/check_vendor_walk.py --onnx
first. Installing it on the robot is not covered by this repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl.contract import Contract, quat_to_gravity  # noqa: E402

NPZ_KEYS = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


VENDOR_WALK = ROOT / "_ref/robot_119/q1_t2d0/rl_models/q1_walkRun_upperPosCtrl_v3.2_20260816.onnx"


def signature(path):
    m = onnx.load(str(path))
    dims = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
    return [(v.name, dims(v)) for v in m.graph.input], [(v.name, dims(v)) for v in m.graph.output]


def package_walk(run: Path, out: Path, stem: Path) -> None:
    m = onnx.load(str(run / "policy.onnx"))
    for v in list(m.graph.input) + list(m.graph.output):
        d = v.type.tensor_type.shape.dim[0]
        d.ClearField("dim_param")
        d.dim_value = 1
    onnx.checker.check_model(m)
    onnx.save(m, str(stem) + ".onnx", save_as_external_data=False)
    ours, theirs = signature(str(stem) + ".onnx"), signature(VENDOR_WALK)
    if ours != theirs:
        sys.exit(f"signature {ours} differs from the vendor walking policy's {theirs}")
    src = ort.InferenceSession(str(run / "policy.onnx"), providers=["CPUExecutionProvider"])
    dst = ort.InferenceSession(str(stem) + ".onnx", providers=["CPUExecutionProvider"])
    h = np.random.default_rng(0).standard_normal((32, 355)).astype(np.float32)
    feed = lambda i: {"observations": h[i:i + 1, -71:], "observation_history": h[i:i + 1]}
    diff = max(float(np.abs(src.run(None, feed(i))[0] - dst.run(None, feed(i))[0]).max()) for i in range(len(h)))
    assert diff < 1e-5, diff
    print(f"{stem}.onnx  {ours[0]} -> {ours[1]}, the vendor's signature; matches policy.onnx to {diff:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--name", required=True, help="file stem: q1rl_<name>.*")
    a = ap.parse_args()

    meta = {p.key: p.value for p in onnx.load(str(a.run / "policy.onnx")).metadata_props}
    if meta.get("q1_contract") == "walk":
        out = a.run / "deploy"
        out.mkdir(exist_ok=True)
        # q1rlw_, not q1rl_: install.sh / rollback.sh clear rl_models/q1rl_*.onnx, and the walk
        # yaml must never point at a file they deleted
        return package_walk(a.run, out, out / f"q1rlw_{a.name}")
    c = Contract()
    out = a.run / "deploy"
    out.mkdir(exist_ok=True)
    stem = out / f"q1rl_{a.name}"

    # policy
    m = onnx.load(str(a.run / "policy.onnx"))
    old = m.graph.output[0].name
    for n in m.graph.node:
        n.output[:] = ["actions" if o == old else o for o in n.output]
    m.graph.output[0].name = "actions"
    for v in list(m.graph.input) + list(m.graph.output):
        d = v.type.tensor_type.shape.dim[0]
        d.ClearField("dim_param")
        d.dim_value = 1
    onnx.checker.check_model(m)
    onnx.save(m, str(stem) + ".onnx", save_as_external_data=False)
    src = ort.InferenceSession(str(a.run / "policy.onnx"), providers=["CPUExecutionProvider"])
    dst = ort.InferenceSession(str(stem) + ".onnx", providers=["CPUExecutionProvider"])
    width = dst.get_inputs()[0].shape[1]
    if width != c.onnx_obs_dim:
        sys.exit(f"policy takes {width} inputs, the contract says {c.onnx_obs_dim}: trained on another layout")
    x = np.random.default_rng(0).standard_normal((32, width)).astype(np.float32)
    ya = src.run(None, {src.get_inputs()[0].name: x})[0]
    yb = np.concatenate([dst.run(None, {"obs": x[i:i + 1]})[0] for i in range(len(x))])
    diff = float(np.abs(ya - yb).max())
    assert diff < 1e-5, diff

    # reference
    z = np.load(a.run / "motion.npz")
    arrays = {k: (np.asarray(z[k], np.float64).reshape(1) if k == "fps" else np.asarray(z[k], np.float32))
              for k in NPZ_KEYS}
    np.savez(str(stem) + ".npz", **arrays)

    # reference gravity, from the npz as written (float32), as the env computes it
    g = quat_to_gravity(arrays["body_quat_w"][:, 0].astype(np.float64))
    with open(str(stem) + ".csv", "w") as f:
        f.write("frame_id,gx,gy,gz\n")
        for i, (gx, gy, gz) in enumerate(g):
            f.write(f"{i},{gx:.8f},{gy:.8f},{gz:.8f}\n")

    T = len(arrays["joint_pos"])
    print(f"{stem}.onnx  obs {width} -> actions 22, matches policy.onnx to {diff:.1e}")
    print(f"{stem}.npz   {T} frames @ {arrays['fps'][0]:g} fps, vendor keys and dtypes")
    print(f"{stem}.csv   {T} rows of reference gravity")
    det = out / "lift_detector.onnx"
    if det.is_file():
        import shutil
        shutil.copyfile(det, str(stem) + ".detector.onnx")
        print(f"{stem}.detector.onnx  lift detector (from {det.name})")


if __name__ == "__main__":
    main()
