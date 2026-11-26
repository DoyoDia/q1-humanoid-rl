"""robot_retargeter output CSV -> a Q1 motion npz the trainer and the robot both read.

    python retarget/to_motion.py OUT.csv --fps 30 -o motions/dance.npz [--start 2 --end 12]

The CSV is one row per frame of the retarget model's qpos: base position (m), base
quaternion (x, y, z, w), then the joints in that model's joint order. This script

  1. maps the joints into contract order by name,
  2. resamples base and joints to the policy rate (50 Hz) -- slerp for the base
     orientation, linear for the rest -- BEFORE forward kinematics, so no body quaternion
     is ever interpolated,
  3. clamps joints to the URDF limits (reporting how much that changed),
  4. pushes the arms clear of the body where the retarget left them inside it (Q1's arms
     are short and its elbows stop 40 deg short of straight, so hanging hands land in the
     hip-pitch links: 34% of frames on LAFAN walk1),
  5. with --balance, moves the centre of mass over the feet that are down (retarget/
     balance.py) -- for clips the robot cannot balance as retargeted, like dance1, which
     stands on one leg with its centre of mass far outside the stance foot,
  6. runs forward kinematics on our physics model (q1_rl.motion.from_root_trajectory),
     which also puts the clip's feet on the floor,

and prints what it had to do, so a clip that needed heavy correction is visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl.contract import Contract  # noqa: E402
from q1_rl.motion import from_root_trajectory  # noqa: E402
from retarget.balance import balance, plant_feet  # noqa: E402


def joint_names_in_qpos_order(xml: Path) -> list[str]:
    m = mujoco.MjModel.from_xml_path(str(xml))
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    return [m.jnt(j).name for j in sorted(hinge, key=lambda j: m.jnt_qposadr[j])]


ARM_BODIES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw")


def repair_arm_collisions(model, root_pos, root_quat, q, c, step=0.02, max_extra=0.6, smooth=7):
    """Abduct each arm (shoulder roll, outward) just enough that it does not touch the
    torso or legs, frame by frame; then smooth the added angle in time so the arm eases
    out and back instead of jumping. Returns the repaired joints and stats."""
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in c.joint_names]
    sides = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        arm = {model.body(f"{side}_{b}_link").id for b in ARM_BODIES}
        sides[side] = (c.joint_names.index(f"{side}_shoulder_roll_joint"), sign, arm)
    robot = set(range(1, model.nbody))

    def touching(t, qt, arm):
        data.qpos[0:3], data.qpos[3:7], data.qpos[qadr] = root_pos[t], root_quat[t], qt
        mujoco.mj_forward(model, data)
        for con in data.contact[:data.ncon]:
            b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
            if (b1 in arm) != (b2 in arm) and {b1, b2} <= robot:
                return True
        return False

    T = len(q)
    extra = {side: np.zeros(T) for side in sides}
    before = 0
    for t in range(T):
        hit_any = False
        for side, (j, sign, arm) in sides.items():
            qt = q[t].copy()
            while touching(t, qt, arm) and extra[side][t] < max_extra:
                hit_any = True
                extra[side][t] += step
                qt[j] = q[t, j] + sign * extra[side][t]
        before += hit_any
    out = q.copy()
    k = np.ones(smooth)
    for side, (j, sign, _) in sides.items():
        e = extra[side]
        # widen each correction over its neighbours (max filter), then average: the arm
        # starts moving out before the contact and comes back after it, smoothly.
        pad = np.pad(e, smooth // 2, mode="edge")
        wide = np.array([pad[i:i + smooth].max() for i in range(T)])
        e = np.convolve(np.pad(wide, smooth // 2, mode="edge"), k / smooth, mode="valid")
        out[:, j] = np.clip(q[:, j] + sign * e, c.lower[j], c.upper[j])
    after = sum(any(touching(t, out[t], arm) for _, _, arm in sides.values()) for t in range(T))
    biggest = max(float(np.max(np.abs(out[:, j] - q[:, j]))) for j, _, _ in sides.values())
    return out, dict(before=before / T, after=after / T, max_added=biggest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--fps", type=float, required=True, help="frame rate of the CSV (the retargeter's source fps)")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--start", type=float, default=0.0, help="seconds")
    p.add_argument("--end", type=float, default=None, help="seconds")
    p.add_argument("--cyclic", action="store_true", help="the clip loops (only if its ends really meet)")
    p.add_argument("--name", default=None)
    p.add_argument("--retarget-model", type=Path, default=ROOT / "_ref/robot_retargeter/asset/robot/q1/q1.xml")
    p.add_argument("--model", type=Path, default=ROOT / "robots/q1/q1_physics.xml")
    p.add_argument("--keep-arm-collisions", action="store_true", help="skip the arm/body collision repair")
    p.add_argument("--balance", action="store_true", help="move the centre of mass over the supporting feet")
    p.add_argument("--plant-feet", action="store_true",
                   help="put feet that are down flat on the floor and keep them still, one foot always "
                        "(for sources that stand flat-footed but whose estimate tilts and floats the feet, e.g. video)")
    p.add_argument("--flight", action="append", default=[], metavar="START-END",
                   help="with --plant-feet: seconds (of the CSV) where both feet may leave the floor together, "
                        "e.g. 207-248 for jumping jacks; repeatable")
    p.add_argument("--ground-window", type=float, default=0.0,
                   help="seconds; put the feet on the floor per sliding window instead of one shift for "
                        "the whole clip (for video sources whose height drifts)")
    a = p.parse_args()

    c = Contract()
    rows = np.loadtxt(a.csv, delimiter=",", ndmin=2)
    names = joint_names_in_qpos_order(a.retarget_model)
    if rows.shape[1] != 7 + len(names):
        sys.exit(f"{a.csv}: {rows.shape[1]} columns, expected 7 + {len(names)} for {a.retarget_model.name}")
    col = {n: 7 + i for i, n in enumerate(names)}
    missing = [n for n in c.joint_names if n not in col]
    if missing:
        sys.exit(f"retarget model lacks contract joints {missing}")

    t_src = np.arange(len(rows)) / a.fps
    end = t_src[-1] if a.end is None else min(a.end, t_src[-1])
    t_dst = np.arange(a.start, end + 1e-9, 1.0 / c.policy_hz)
    if len(t_dst) < 2:
        sys.exit("selected span is shorter than two policy frames")

    root_pos = np.stack([np.interp(t_dst, t_src, rows[:, i]) for i in range(3)], -1)
    rot = Rotation.from_quat(rows[:, 3:7])                       # x, y, z, w
    root_quat = Slerp(t_src, rot)(t_dst).as_quat()[:, [3, 0, 1, 2]]
    q = np.stack([np.interp(t_dst, t_src, rows[:, col[n]]) for n in c.joint_names], -1)

    clamped = np.clip(q, c.lower, c.upper)
    over = np.abs(clamped - q)
    worst = int(np.argmax(over.max(axis=0)))

    model = mujoco.MjModel.from_xml_path(str(a.model))
    arms = None
    if not a.keep_arm_collisions:
        clamped, arms = repair_arm_collisions(model, root_pos, root_quat, clamped, c)
    plant = None
    if a.plant_feet:
        wins = [tuple(float(x) - a.start for x in w.split("-")) for w in a.flight]
        root_pos, root_quat, clamped, plant = plant_feet(model, root_pos, root_quat, clamped, c.policy_hz,
                                                         flight_windows=wins)
    bal = None
    if a.balance:
        root_pos, root_quat, clamped, bal = balance(model, root_pos, root_quat, clamped, c.policy_hz)
    motion, ground = from_root_trajectory(model, root_pos, root_quat, clamped, c.policy_hz,
                                          name=a.name or a.out.stem, cyclic=a.cyclic,
                                          ground_window_s=a.ground_window, ground_align=not a.plant_feet)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    motion.save_npz(a.out)

    print(motion)
    print(f"  source       {len(rows)} frames @ {a.fps:g} fps, used {a.start:.2f}-{end:.2f} s")
    print(f"  joint limits clamped {np.mean(over > 1e-6) * 100:.1f}% of joint-frames, "
          f"worst {over.max():.3f} rad on {c.joint_names[worst]}")
    if arms:
        print(f"  arm/body     touching {arms['before'] * 100:.1f}% of frames -> {arms['after'] * 100:.1f}% "
              f"after abducting the shoulders (at most {arms['max_added']:.2f} rad)")
    if plant:
        print(f"  plant feet   both down {plant['both_down'] * 100:.0f}% of frames, one {plant['any_down'] * 100:.0f}% "
              f"({plant['runs']} contacts); planted sole tilt {plant['tilt_before']:.1f} -> {plant['tilt_after']:.1f} deg "
              f"(p95 {plant['tilt_after_p95']:.1f}); height gap between planted feet {plant['gap_before_mm']:.1f} -> "
              f"{plant['gap_after_mm']:.1f} mm; legs changed {plant['leg_change_rms']:.3f} rad RMS, base up to "
              f"{plant['base_shift_max'] * 100:.1f} cm; IK gave up on {plant['fallback_frames'] * 100:.1f}% of frames")
        print(f"               lower foot planted through {plant['forced'] * 100:.1f}% of frames where neither was down; "
              f"flight (both off, allowed) {plant['flight'] * 100:.1f}%; no flat foot elsewhere {plant['no_flat_foot'] * 100:.1f}%")
    if bal:
        print(f"  balance      ZMP beyond the sole edges by >1 cm on {bal['zmp_out_before'] * 100:.0f}% of frames -> "
              f"{bal['zmp_out_after'] * 100:.0f}% (worst {bal['zmp_worst_before'] * 100:.1f} -> {bal['zmp_worst_after'] * 100:.1f} cm); "
              f"single support: com outside the stance foot by {bal['com_single_p50_before'] * 100:.1f} -> "
              f"{bal['com_single_p50_after'] * 100:.1f} cm (median)")
        print(f"               base moved up to {bal['base_shift_max'] * 100:.1f} cm (mean {bal['base_shift_mean'] * 100:.1f}), "
              f"leg joints changed {bal['leg_change_rms']:.3f} rad RMS, planted feet held to {bal['foot_resid_p99'] * 1000:.1f} mm (p99, max {bal['foot_resid_max'] * 1000:.0f}); "
              f"com short of the plan on {bal['com_short_frames'] * 100:.0f}% of frames")
    print(f"  ground       shifted {ground['shift']:+.3f} m (varying over {ground['shift_range'] * 100:.1f} cm); penetration max {ground['max_penetration'] * 1000:.1f} mm "
          f"({ground['frames_penetrating'] * 100:.1f}% of frames >5 mm), highest lowest-point {ground['max_float']:.3f} m")
    print(f"  joint speed  max {np.abs(motion.joint_vel).max():.1f} rad/s "
          f"(limit min {c.max_vel.min():.1f})")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
