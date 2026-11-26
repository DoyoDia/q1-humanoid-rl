"""Load robots/q1/contract/ and build observations from it.

This module is the single Python implementation of the deployment contract. The training
env, the ONNX exporter and the on-robot bridge all go through it; the browser has a twin
in web/contract.js, and tools/make_golden_trace.py keeps the two honest.

Nothing here hardcodes a layout. The field order, offsets and widths are read from
obs_spec.json, so when the spike settles which contract we deploy against, only that
file changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CONTRACT_DIR = Path(__file__).resolve().parent.parent / "robots" / "q1" / "contract"


def quat_to_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity direction expressed in the base frame.

    quat is the base orientation in the world (w, x, y, z), matching both MuJoCo's
    qpos[3:7] and the robot's ImuState.orientation. The result is the world's -Z axis
    rotated into the base frame, i.e. R(q)^T @ [0, 0, -1]. Level and upright gives
    [0, 0, -1]; it is a unit vector, not scaled by 9.81.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # third column of R(q), then negated -- R^T @ [0,0,-1] == -R[:, 2] read row-wise
    return np.stack([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ], axis=-1)


def quat_to_roll_pitch(quat_wxyz: np.ndarray) -> np.ndarray:
    """(roll, pitch) of the base, Z-Y-X Euler angles, rad -- the Walk runtime's imu_euler."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.stack([roll, pitch], axis=-1)


@dataclass(frozen=True)
class Field:
    name: str
    start: int
    size: int

    @property
    def sl(self) -> slice:
        return slice(self.start, self.start + self.size)


class Contract:
    """The deployment contract, loaded from disk. Read-only."""

    def __init__(self, name: str = "mimic_opensource_v2", contract_dir: Path = CONTRACT_DIR):
        d = Path(contract_dir)
        self.name = name
        self.dir = d
        self._joints = yaml.safe_load((d / "joints.yaml").read_text(encoding="utf-8"))
        self._gains = yaml.safe_load((d / "gains.yaml").read_text(encoding="utf-8"))
        self.rates = yaml.safe_load((d / "rates.yaml").read_text(encoding="utf-8"))
        spec = json.loads((d / "obs_spec.json").read_text(encoding="utf-8"))
        if name not in spec:
            raise KeyError(f"no such contract {name!r}; have {[k for k in spec if not k.startswith('$')]}")
        self.spec = spec[name]

        self.joint_names: list[str] = [j["name"] for j in self._joints["joints"]]
        self.n_joints: int = self._joints["n_joints"]
        self.lower = np.array([j["lower"] for j in self._joints["joints"]], np.float64)
        self.upper = np.array([j["upper"] for j in self._joints["joints"]], np.float64)
        self.effort = np.array([j["effort"] for j in self._joints["joints"]], np.float64)
        self.max_vel = np.array([j["velocity"] for j in self._joints["joints"]], np.float64)
        self.max_step_rad: float = self._joints["command"]["max_step_rad"]

        g = self._gains[name]
        self.kp = np.array(g["kps"], np.float64)
        self.kd = np.array(g["kds"], np.float64)
        self.default_dof_pos = np.array(g["default_dof_pos"], np.float64)
        self.action_scale = float(g["action_scale"])

        self.fields = {f["name"]: Field(f["name"], f["start"], f["size"]) for f in self.spec["fields"]}
        self.obs_dim: int = self.spec["obs_dim_valid"]
        self.onnx_obs_dim: int = self.spec.get("obs_dim_onnx_input", self.obs_dim)
        self.frame_stack: int = self.spec.get("frame_stack", 1)
        self.obs_scales: dict = self.spec["obs_scales"]
        self.obs_clip: float = self.spec["clips"]["obs"]
        self.action_clip: float = self.spec["clips"]["action"]

        self.policy_hz: int = self.rates["policy_hz"]
        self.sim_dt: float = self.rates["sim"]["timestep"]
        self.decimation: int = self.rates["sim"]["decimation"]

    # -- the two directions -------------------------------------------------

    def build_obs(self, state: dict) -> np.ndarray:
        """Assemble observations. `state` keys, joint fields in joints.yaml order:

            base_quat      (..., 4)   base orientation in world, w x y z
            base_ang_vel   (..., 3)   base angular velocity in the base frame, rad/s
            joint_pos      (..., 22)  rad
            joint_vel      (..., 22)  rad/s
            prev_action    (..., 22)  raw network output of the previous step
            motion_ref_pos (..., 22)  rad      -- only for motion-tracking contracts
            motion_ref_vel (..., 22)  rad/s    -- only for motion-tracking contracts
            motion_ref_quat (..., 4)  reference pelvis orientation in world, w x y z
                                      -- for ref_gravity; the robot reads the same
                                      vector from the motion's gravity_guard csv

        Any leading batch shape works -- one robot in the bridge or the golden trace, 4096 in
        the training environment. The environment calls this rather than keeping its own
        copy: an earlier version did, which made it a sixth observation builder that the
        golden trace never exercised.

        Returns float32 of shape (..., obs_dim), NOT padded to onnx_obs_dim (see pad_for_onnx).
        """
        batch = np.shape(state["joint_pos"])[:-1]
        obs = np.zeros(batch + (self.obs_dim,), np.float64)
        s = self.obs_scales
        for name, f in self.fields.items():
            if name == "gvec":
                v = quat_to_gravity(state["base_quat"])
            elif name == "ref_gravity":
                v = quat_to_gravity(state["motion_ref_quat"])
            elif name in ("base_ang_vel", "imu_ang_vel"):
                v = np.asarray(state["base_ang_vel"], np.float64) * s.get("ang_vel", 1.0)
            elif name == "imu_euler":
                v = quat_to_roll_pitch(state["base_quat"])
            elif name == "command" and "lin_vel" in s:
                # the Walk runtime scales the command like legged_gym: (vx, vy) by lin_vel, yaw rate by ang_vel
                v = np.asarray(state["command"], np.float64) * np.array([s["lin_vel"], s["lin_vel"], s.get("ang_vel", 1.0)])
            elif name == "joint_pos_err":
                v = (np.asarray(state["joint_pos"], np.float64) - self.default_dof_pos) * s.get("dof_pos", 1.0)
            elif name == "joint_pos":
                v = np.asarray(state["joint_pos"], np.float64) * s.get("dof_pos", 1.0)
            elif name == "joint_vel":
                v = np.asarray(state["joint_vel"], np.float64) * s.get("dof_vel", 1.0)
            elif name == "prev_action":
                v = np.asarray(state["prev_action"], np.float64)[..., : f.size] * s.get("actions", 1.0)
            elif name in state:
                v = np.asarray(state[name], np.float64)
            else:
                raise KeyError(f"obs field {name!r} has no value in state and no rule in build_obs()")
            if np.shape(v)[-1:] != (f.size,):
                raise ValueError(f"obs field {name!r}: expected last dim {f.size}, got shape {np.shape(v)}")
            obs[..., f.sl] = v
        np.clip(obs, -self.obs_clip, self.obs_clip, out=obs)
        return obs.astype(np.float32)

    def pad_for_onnx(self, obs: np.ndarray) -> np.ndarray:
        """Zero-pad obs_dim -> the width the exported ONNX actually takes."""
        if self.onnx_obs_dim == self.obs_dim:
            return obs
        out = np.zeros((*obs.shape[:-1], self.onnx_obs_dim), np.float32)
        out[..., : self.obs_dim] = obs
        return out

    def action_to_target(self, action: np.ndarray, prev_target: np.ndarray | None = None,
                         clamp: bool = True) -> np.ndarray:
        """action -> joint position target, with every limit the firmware enforces.

        target = default_dof_pos + action_scale * clip(action)
        then clipped to the joint limits (unless clamp is False: the Walk runtime does not clip
        them, it saturates the torque instead), then rate-limited to max_step_rad per frame.
        The rate limit is NOT cosmetic: the firmware rejects larger jumps, so a policy
        trained without it would behave differently on hardware than in sim.
        """
        a = np.clip(np.asarray(action, np.float64), -self.action_clip, self.action_clip)
        target = self.default_dof_pos + self.action_scale * a
        if clamp:
            target = np.clip(target, self.lower, self.upper)
        if prev_target is not None:
            d = np.clip(target - prev_target, -self.max_step_rad, self.max_step_rad)
            target = prev_target + d
        return target

    def __repr__(self) -> str:
        return (f"Contract({self.name!r}, {self.n_joints} joints, obs {self.obs_dim}"
                f"->{self.onnx_obs_dim}, {self.policy_hz} Hz, decimation {self.decimation})")
