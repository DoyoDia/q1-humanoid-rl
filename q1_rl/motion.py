"""Reference motions in the format the robot already speaks.

The robot's own open-source policies each carry a .npz next to their ONNX
(motion_list.yaml -> custom.trajectory_path). We read one off the robot and decoded the
headers; it is the mjlab / IsaacLab motion convention:

    fps             (1,)        float64
    joint_pos       (T, 22)     float32   contract joint order
    joint_vel       (T, 22)     float32
    body_pos_w      (T, 26, 3)  float32   world frame
    body_quat_w     (T, 26, 4)  float32   world frame, w x y z
    body_lin_vel_w  (T, 26, 3)  float32
    body_ang_vel_w  (T, 26, 3)  float32

Using it as our own format costs nothing and means a motion we train against is a motion
the robot can already be handed. 26 == MJCF nbody minus `world`.

Two ways to get a Motion:
  * load_npz()             -- a vendor file, or one exported from a retarget pipeline
  * from_joint_trajectory  -- joint angles only; forward kinematics fills the rest,
                              with the base height solved per frame so the feet stay down

The body_* arrays are in MJCF body order excluding world (pelvis first). Checked against
every vendor npz on the robot: FK of our MJCF reproduces their body_pos_w to 1e-7 m (see
contract/obs_spec.json). We also record which model produced a file, so a future
mismatch is detectable rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .geometry import lowest_contact_z

ARRAYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


@dataclass
class Motion:
    fps: float
    joint_pos: np.ndarray        # (T, nj)
    joint_vel: np.ndarray        # (T, nj)
    body_pos_w: np.ndarray       # (T, nb, 3)
    body_quat_w: np.ndarray      # (T, nb, 4)
    body_lin_vel_w: np.ndarray   # (T, nb, 3)
    body_ang_vel_w: np.ndarray   # (T, nb, 3)
    name: str = "unnamed"
    body_names: tuple[str, ...] = ()
    # Does the last frame run straight back into the first? Only then may an environment
    # loop the clip. A get-up or a flip must not: looping teleports the reference from the
    # final pose back to the opening one mid-episode.
    cyclic: bool = False

    @property
    def n_frames(self) -> int:
        return self.joint_pos.shape[0]

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    def __repr__(self) -> str:
        return (f"Motion({self.name!r}, {self.n_frames} frames @ {self.fps:g} fps "
                f"= {self.duration:.2f}s, {self.joint_pos.shape[1]} joints, "
                f"{self.body_pos_w.shape[1]} bodies, {'cyclic' if self.cyclic else 'one-shot'})")

    # -- io ----------------------------------------------------------------

    @classmethod
    def load_npz(cls, path, name: str | None = None) -> "Motion":
        z = np.load(path)
        missing = [k for k in (*ARRAYS, "fps") if k not in z]
        if missing:
            raise ValueError(f"{path}: not a Q1 motion npz, missing {missing}")
        stored = str(z["name"]) if "name" in z else None
        arrays = {k: np.asarray(z[k], np.float32) for k in ARRAYS}
        cyclic = bool(z["cyclic"]) if "cyclic" in z else looks_cyclic(arrays)
        return cls(fps=float(np.asarray(z["fps"]).reshape(-1)[0]), **arrays,
                   name=name or stored or Path(path).stem, cyclic=cyclic)

    def save_npz(self, path) -> None:
        # `name` rides along as an extra array: harmless to the vendor's reader, which
        # looks up arrays by name, and it keeps a clip identifiable after a round trip.
        np.savez(path, fps=np.array([self.fps], np.float64),
                 name=np.array(self.name), cyclic=np.array(self.cyclic),
                 **{k: np.asarray(getattr(self, k), np.float32) for k in ARRAYS})

    # -- resampling --------------------------------------------------------

    def resample(self, fps: float) -> "Motion":
        """Linear resample onto a new rate. Quaternions are slerp'd via normalised lerp,
        which is accurate enough between adjacent frames of a 30-120 fps capture."""
        if abs(fps - self.fps) < 1e-9:
            return self
        n = max(2, int(round(self.duration * fps)))
        src = np.arange(self.n_frames) / self.fps
        dst = np.arange(n) / fps

        def lerp(a):
            flat = a.reshape(self.n_frames, -1)
            out = np.stack([np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], -1)
            return out.reshape(n, *a.shape[1:]).astype(np.float32)

        q = lerp(self.body_quat_w)
        q /= np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-9)
        return Motion(fps=fps, joint_pos=lerp(self.joint_pos), joint_vel=lerp(self.joint_vel),
                      body_pos_w=lerp(self.body_pos_w), body_quat_w=q,
                      body_lin_vel_w=lerp(self.body_lin_vel_w),
                      body_ang_vel_w=lerp(self.body_ang_vel_w),
                      name=self.name, body_names=self.body_names, cyclic=self.cyclic)


def looks_cyclic(arrays: dict, joint_tol: float = 0.05, root_tol: float = 0.02) -> bool:
    """Guess whether a clip with no stored flag loops: does the last frame land within a
    step of the first? The vendor's files carry no flag, and they are one-shots (get-ups,
    flips), so when in doubt this says no."""
    j = np.asarray(arrays["joint_pos"])
    r = np.asarray(arrays["body_pos_w"])[:, 0]
    return (float(np.abs(j[-1] - j[0]).max()) < joint_tol
            and float(np.abs(r[-1, 2] - r[0, 2])) < root_tol)


def from_joint_trajectory(model, joint_pos: np.ndarray, fps: float, name: str = "generated",
                          ground_clearance: float = 0.001, free_base: bool = False,
                          cyclic: bool = False) -> Motion:
    """Joint angles -> a full Motion, by running MuJoCo's forward kinematics per frame.

    The base is placed at the origin and dropped until the lowest colliding geom rests on
    the floor, so a generated motion is at least kinematically standing. This is NOT a
    dynamics simulation: it produces a reference to track, not a proof the robot can do it.
    """
    joint_pos = np.asarray(joint_pos, np.float64)
    T, nj = joint_pos.shape
    if nj != model.nu:
        raise ValueError(f"trajectory has {nj} joints, model has {model.nu} actuators")

    data = mujoco.MjData(model)
    nb = model.nbody - 1                      # drop `world`
    body_pos = np.zeros((T, nb, 3), np.float32)
    body_quat = np.zeros((T, nb, 4), np.float32)

    for t in range(T):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0                    # identity quaternion
        data.qpos[7:] = joint_pos[t]
        data.qpos[2] = 1.0
        mujoco.mj_forward(model, data)
        if not free_base:
            # The same lowest-point rule the model builder uses for the stand keyframe;
            # see q1_rl/geometry.py for why a bounding sphere is not good enough.
            data.qpos[2] = 1.0 - lowest_contact_z(model, data) + ground_clearance
            mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos[1:]
        body_quat[t] = data.xquat[1:]

    dt = 1.0 / fps
    if cyclic:      # central differences across the seam, so the loop point has no kink
        joint_vel = ((np.roll(joint_pos, -1, 0) - np.roll(joint_pos, 1, 0)) / (2 * dt)).astype(np.float32)
    else:
        joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    lin_vel = np.gradient(body_pos.astype(np.float64), dt, axis=0).astype(np.float32)
    ang_vel = _quat_angular_velocity(body_quat.astype(np.float64), dt).astype(np.float32)

    names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody))
    return Motion(fps=float(fps), joint_pos=joint_pos.astype(np.float32), joint_vel=joint_vel,
                  body_pos_w=body_pos, body_quat_w=body_quat,
                  body_lin_vel_w=lin_vel, body_ang_vel_w=ang_vel,
                  name=name, body_names=names, cyclic=cyclic)


def from_root_trajectory(model, root_pos: np.ndarray, root_quat: np.ndarray,
                         joint_pos: np.ndarray, fps: float, name: str = "retargeted",
                         cyclic: bool = False, ground_clearance: float = 0.001,
                         ground_percentile: float = 5.0,
                         ground_window_s: float = 0.0, ground_align: bool = True) -> tuple[Motion, dict]:
    """A base trajectory plus joint angles -> a full Motion, by forward kinematics.

    For motions that move through space (a retargeter's output): the base path comes from
    the source instead of being solved per frame like from_joint_trajectory() does, so
    walking, turning and jumping survive. The whole clip is shifted vertically so that
    the lowest contact point sits on the floor at the `ground_percentile`-th percentile
    of frames -- a low percentile rather than the minimum, so one bad frame cannot lift
    the entire clip off the ground. With ground_window_s > 0 the shift varies over time
    instead: the percentile is taken over a sliding window of that length and smoothed,
    for sources whose height drifts (video pose estimation: a standing person floats
    3 cm at the start and 1 cm two minutes later). The window must span a few landings
    so jumps keep their flight. ground_align=False leaves the heights as given. Returns the Motion and the ground statistics
    (penetration and float, in metres) so a caller can reject a clip that needs it.

    root_quat is (w, x, y, z). Joint order is the model's actuator order.
    """
    root_pos = np.asarray(root_pos, np.float64)
    root_quat = np.asarray(root_quat, np.float64)
    joint_pos = np.asarray(joint_pos, np.float64)
    T = joint_pos.shape[0]
    if joint_pos.shape[1] != model.nu:
        raise ValueError(f"trajectory has {joint_pos.shape[1]} joints, model has {model.nu} actuators")

    data = mujoco.MjData(model)
    nb = model.nbody - 1
    body_pos = np.zeros((T, nb, 3))
    body_quat = np.zeros((T, nb, 4))
    lowest = np.zeros(T)
    for t in range(T):
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_quat[t] / np.linalg.norm(root_quat[t])
        data.qpos[7:] = joint_pos[t]
        mujoco.mj_forward(model, data)
        lowest[t] = lowest_contact_z(model, data)
        body_pos[t] = data.xpos[1:]
        body_quat[t] = data.xquat[1:]

    if not ground_align:                    # the caller has put it on the floor already
        shift = np.zeros(T)
    elif ground_window_s > 0:
        from scipy.ndimage import gaussian_filter1d, percentile_filter
        win = max(3, int(round(ground_window_s * fps)) | 1)
        base = percentile_filter(lowest, ground_percentile, size=win, mode="nearest")
        shift = ground_clearance - gaussian_filter1d(base, sigma=win / 4, mode="nearest")
        body_pos[..., 2] += shift[:, None]
    else:
        shift = ground_clearance - np.percentile(lowest, ground_percentile)
        body_pos[..., 2] += shift
    lowest += shift
    # Keep each body's quaternion on one hemisphere through time, so differences and any
    # later interpolation do not see a spurious 2*pi flip.
    for t in range(1, T):
        flip = np.sum(body_quat[t] * body_quat[t - 1], axis=-1) < 0
        body_quat[t, flip] *= -1

    dt = 1.0 / fps
    if cyclic:
        joint_vel = (np.roll(joint_pos, -1, 0) - np.roll(joint_pos, 1, 0)) / (2 * dt)
    else:
        joint_vel = np.gradient(joint_pos, dt, axis=0)
    names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody))
    motion = Motion(fps=float(fps), joint_pos=joint_pos.astype(np.float32),
                    joint_vel=joint_vel.astype(np.float32),
                    body_pos_w=body_pos.astype(np.float32), body_quat_w=body_quat.astype(np.float32),
                    body_lin_vel_w=np.gradient(body_pos, dt, axis=0).astype(np.float32),
                    body_ang_vel_w=_quat_angular_velocity(body_quat, dt).astype(np.float32),
                    name=name, body_names=names, cyclic=cyclic)
    stats = dict(shift=float(np.mean(shift)), shift_range=float(np.ptp(shift)),
                 max_penetration=float(max(0.0, ground_clearance - lowest.min())),
                 frames_penetrating=float(np.mean(lowest < ground_clearance - 0.005)),
                 max_float=float(lowest.max()))
    return motion, stats


def _quat_angular_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference angular velocity in the world frame, from a quaternion track."""
    T = q.shape[0]
    out = np.zeros((*q.shape[:-1], 3))
    nxt = np.minimum(np.arange(T) + 1, T - 1)
    prv = np.maximum(np.arange(T) - 1, 0)
    span = np.where((np.arange(T) > 0) & (np.arange(T) < T - 1), 2.0, 1.0)[:, None, None] * dt
    a, b = q[prv], q[nxt]
    b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)     # shortest arc
    # relative rotation b * conj(a), then the log map's vector part, doubled
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    rw = bw * aw + (bv * av).sum(-1, keepdims=True)
    rv = aw * bv - bw * av - np.cross(bv, av)
    norm = np.linalg.norm(rv, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.abs(rw))
    axis = rv / np.clip(norm, 1e-12, None)
    out = np.sign(rw) * axis * angle / span
    return np.nan_to_num(out)


# -- procedural motions -----------------------------------------------------
# Enough to train and demo against without needing a capture or the vendor's files.
# Each returns joint angles only; from_joint_trajectory() does the rest.

def _base_pose(contract):
    return np.asarray(contract.default_dof_pos, np.float64)


def squat(contract, seconds=4.0, fps=50.0, depth=0.45, cycles=2):
    """Bend the knees and hips in phase, ankles compensating to keep the shins plausible."""
    t = np.linspace(0, 2 * np.pi * cycles, int(seconds * fps), endpoint=False)
    s = 0.5 * (1.0 - np.cos(t))                 # 0 -> 1 -> 0, smooth at the ends
    q = np.tile(_base_pose(contract), (t.size, 1))
    idx = {n: i for i, n in enumerate(contract.joint_names)}
    for side in ("left", "right"):
        q[:, idx[f"{side}_hip_pitch_joint"]] -= depth * s
        q[:, idx[f"{side}_knee_joint"]] += 2.0 * depth * s
        q[:, idx[f"{side}_ankle_pitch_joint"]] -= depth * s
    return q


def wave(contract, seconds=4.0, fps=50.0, cycles=3):
    """Stand still and wave the right arm -- exercises the upper body with a stable base."""
    n = int(seconds * fps)
    t = np.linspace(0, 1, n, endpoint=False)
    q = np.tile(_base_pose(contract), (n, 1))
    idx = {n_: i for i, n_ in enumerate(contract.joint_names)}
    # raise over the first quarter, lower over the last, so the loop seam is at rest
    raise_ = np.minimum(1.0, np.clip(t, 0, 0.25) * 4) * np.minimum(1.0, np.clip(1 - t, 0, 0.25) * 4)
    swing = np.sin(np.linspace(0, 2 * np.pi * cycles, n, endpoint=False))
    q[:, idx["right_shoulder_pitch_joint"]] += raise_ * -1.4
    q[:, idx["right_shoulder_roll_joint"]] += raise_ * -0.9
    q[:, idx["right_elbow_joint"]] += raise_ * (0.3 + 0.35 * swing)
    q[:, idx["right_shoulder_yaw_joint"]] += raise_ * 0.25 * swing
    return q


def step_in_place(contract, seconds=4.0, fps=50.0, lift=0.35, cycles=4):
    """Alternating leg lift. The hardest of the three: the base must actually balance."""
    n = int(seconds * fps)
    ph = np.linspace(0, 2 * np.pi * cycles, n, endpoint=False)
    q = np.tile(_base_pose(contract), (n, 1))
    idx = {n_: i for i, n_ in enumerate(contract.joint_names)}
    for side, off in (("left", 0.0), ("right", np.pi)):
        s = np.clip(np.sin(ph + off), 0, None)          # only the swing half
        q[:, idx[f"{side}_hip_pitch_joint"]] -= lift * s
        q[:, idx[f"{side}_knee_joint"]] += 1.6 * lift * s
        q[:, idx[f"{side}_ankle_pitch_joint"]] -= 0.6 * lift * s
    return q


def stand(contract, seconds=10.0, fps=50.0):
    """The default stance, held: the browser sim2sim's idle policy, trained against pushes and
    lifts so it can be shoved and picked up between clips."""
    return np.tile(_base_pose(contract), (int(seconds * fps), 1))


CATALOG = {"squat": squat, "wave": wave, "step_in_place": step_in_place, "stand": stand}


def generate(name: str, model, contract, **kw) -> Motion:
    if name not in CATALOG:
        raise KeyError(f"no such motion {name!r}; have {sorted(CATALOG)}")
    fps = kw.pop("fps", float(contract.policy_hz))
    q = CATALOG[name](contract, fps=fps, **kw)
    q = np.clip(q, contract.lower, contract.upper)
    return from_joint_trajectory(model, q, fps, name=name, cyclic=True)   # all built periodic
