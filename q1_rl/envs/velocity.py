"""Velocity commands: walk where the operator steers (WASD in the browser viewer).

Everything physical is MotionTrackingEnv's -- the backend, domain randomisation, sensor
noise and latency, pushes, being picked up, the 0.2 rad/frame command limit -- on the
"stand" clip, which only serves as the reset pose. What differs is the task: instead of a
reference trajectory the policy sees a command (forward, sideways, yaw rate) and a gait
clock, and is rewarded for the base velocity it achieves and for lifting its feet in time
with the clock.

The observation layout is the `velocity` contract in robots/q1/contract/obs_spec.json. Its
gains, nominal pose and action scale are MIMIC_OPENSOURCE_V2's, so a velocity policy and
the motion-tracking policies drive the same actuators the same way and the viewer can hand
the robot from one to the other mid-stride. It is OUR layout, not one the robot's runtime
knows; deploying it would take a runtime of our own.

Command ranges are the vendor's walking policy's (gains.yaml `walk`; vx in [-0.3, 0.4] m/s,
vy in [-0.4, 0.4], yaw rate +-0.785 rad/s), which were tuned on the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..contract import quat_to_gravity
from .motion_tracking import (EnvConfig, MotionTrackingEnv, _rotate_local_to_world,
                              _rotate_world_to_local, _to_heading, _twist_z, _wrap, _yaw)


@dataclass
class VelocityWeights:
    lin_vel: float = 2.0          # exp(-10 |v_cmd - v|^2), heading frame: 0.1 m/s off -> 0.90, 0.3 -> 0.41
    ang_vel: float = 1.0          # exp(-4 (w_cmd - w)^2)
    # Heading against the commanded yaw rate's integral since the command was drawn,
    # exp(-(err / 0.15 rad)^2). ang_vel misses a steady bias: walk_v5 turned left at 0.1-0.15
    # rad/s on a straight command (60-80 deg in 10 s) for 1-2% of ang_vel. Here it adds up.
    heading: float = 0.0
    # The same for distance: the velocity error (world frame, the command turned with the robot's
    # own heading) integrated with a leak of path_tau_s, exp(-(|err| / 0.25 m)^2). A steady error e
    # settles at e * tau (0.07 m/s -> 0.14 m -> 0.73); the lag of speeding up from a stand leaks
    # away. With heading on, walk_v6 walked 1.2 m/s commands at 1.05 and drifted sideways at
    # 0.04-0.07 m/s, each costing lin_vel a few percent.
    path: float = 0.0
    # Each foot's height against the clock's: a half-sine of swing_height while that foot's
    # half of the cycle is in the air, zero otherwise -- and zero for both when standing.
    feet_swing: float = 1.0
    # The swing leg's joints against a stepping pattern from the clock: hip pitch -a, knee +2a,
    # ankle pitch -a (sums to zero, so the sole stays level) times the swing half-sine, the
    # nominal pose for the stance leg and when standing; exp(-2 |q_leg - ref|). Without it the
    # first policy found standing still safer than stepping and never lifted a foot: 150
    # iterations, every command answered with 0.00 m/s and the feet on the floor. (The
    # pattern humanoid-gym uses; the policy departs from it to walk.)
    ref_joint: float = 2.5
    # Each foot down in its stance half and up in its swing half (both down when standing):
    # credit for any lift at the right time, where feet_swing pays only near the right height.
    contact: float = 0.5
    # The swing foot's load (sole touch sensor, robots/q1/build_model.py): exp(-F / 0.2 mg) while
    # that foot should swing, 1 otherwise; mean over the feet. Standing on both pays 0.08, a foot
    # half unloaded 0.37, lifted 1 -- the gradient toward shifting the weight that contact and
    # feet_swing lack until the foot is already up (Siekmann et al.'s periodic reward for Cassie).
    # Without it four runs bent the swing knee in time and never lifted a foot.
    swing_force: float = 2.0
    # Standing still earns these whatever the command, so they are kept small: with upright 1,
    # height 0.5, posture 0.5, feet_twist 0.5 and alive 0.5 a second policy stood still for 100
    # iterations as well -- a step risked a fall that would lose them all for the rest of the
    # episode, and doing it right paid barely more.
    upright: float = 0.5          # exp(-10 |g_xy|^2): 10 deg of tilt -> 0.74
    height: float = 0.2           # pelvis at its standing height
    # Waist and arms near the nominal pose; hips' roll and yaw loosely (turning and side
    # steps need some).
    posture: float = 0.3
    hip_posture: float = 0.2
    # The torso's sideways lean, exp(-(g_y / 0.1)^2) in the torso frame: 3 deg -> 0.76, 6 -> 0.33.
    # posture alone barely sees the waist (13 deg of waist roll costs it 3%), and walk_v2 rocked
    # the torso +-12 deg every step where the vendor's policy keeps it within +-3.
    torso_roll: float = 0.0
    feet_twist: float = 0.3       # MotionTrackingEnv's: a planted foot not spinning
    alive: float = 0.0
    lin_vel_z: float = -1.0       # (m/s)^2
    ang_vel_xy: float = -0.05     # (rad/s)^2
    feet_slip: float = -0.5       # planted feet's horizontal speed^2, (m/s)^2
    feet_close: float = -10.0     # m the feet are closer sideways than 80% of the stance width
    action_rate: float = -1.0
    action_accel: float = -1.0
    # rad the raw target (default + scale * action) lies beyond the joint limits. The target is
    # clipped to them before the motors, so an action past the limit is a free full-throw
    # command: walk_v2 drove the ankle rolls at +-17 (targets +-8.5 rad against limits of 0.35),
    # banging them limit to limit -- and nothing says the walking runtime clips as we do.
    action_over: float = 0.0
    torque: float = -1e-5
    joint_limit: float = -0.5
    self_contact: float = -0.5
    lift_calm: float = 2.0


@dataclass
class VelocityConfig(EnvConfig):
    episode_seconds: float = 20.0
    adaptive_sampling: bool = False       # nothing to sample: the clip is a pose
    max_tilt_rad: float = 0.8             # from vertical
    push_interval_s: float = 5.0
    push_vel: float = 0.2                 # gentler than tracking's 0.35 while it learns to step
    cmd_vx: tuple[float, float] = (-0.3, 0.4)
    cmd_vy: tuple[float, float] = (-0.4, 0.4)
    cmd_wz: tuple[float, float] = (-0.785, 0.785)
    cmd_resample_s: tuple[float, float] = (2.0, 6.0)
    # Shares of the commands drawn: stand still; move along one axis only (what a key gives).
    cmd_zero_prob: float = 0.1
    cmd_single_prob: float = 0.4
    cmd_deadband: float = 0.05            # a command component smaller than this is zero
    # Step in place under a zero command too (the clock never stops): a curriculum start, so the
    # stepping itself is learned before standing still is an option.
    always_step: bool = False
    gait_period_s: float = 0.6
    swing_height: float = 0.04            # m, ankle above its planted height at mid-swing
    ref_amp: float = 0.45                 # rad, ref_joint's hip-pitch swing (about 4 cm of lift)
    # Fade ref_joint out with the commanded speed: weight x max(0, 1 - |v_cmd| / this); 0 keeps it
    # whole. Its stepping pattern lifts the foot in place, and a stride long enough for 1 m/s
    # swings the hip pitch further than it allows.
    ref_fade_speed: float = 0.0           # m/s
    path_tau_s: float = 2.0               # the path reward's leak (VelocityWeights.path)
    weights: VelocityWeights = field(default_factory=VelocityWeights)


def _clock_targets(clock: np.ndarray, moving: np.ndarray, h: float) -> np.ndarray:
    """(n, 2) wanted ankle heights above planted: left swings in the first half of the
    cycle, right in the second; zero for both when not moving."""
    ph = np.stack([clock, (clock + 0.5) % 1.0], axis=1)
    return np.where(moving[:, None], h * np.maximum(0.0, np.sin(2 * np.pi * ph)), 0.0)


class VelocityEnv(MotionTrackingEnv):
    def __init__(self, model, motion, contract, cfg: VelocityConfig, seed: int = 0):
        n = cfg.num_envs
        self.command = np.zeros((n, 3))
        self.cmd_left = np.zeros(n, np.int64)         # policy steps until the next resample
        self.clock = np.zeros(n)
        self.heading_ref = np.zeros(n)                # yaw the commanded yaw rate has brought it to
        self.path_err = np.zeros((n, 2))              # m, the leaky integral of the velocity error
        self.prev_feet_xy = np.zeros((n, 2, 2))
        # The base class ends by sizing the critic, which needs what is set below: size it here.
        super().__init__(model, motion, contract, replace(cfg, critic_obs=False), seed)
        self.cfg = cfg
        hz = contract.policy_hz
        self.clock_step = 1.0 / (cfg.gait_period_s * hz)
        self.stand_z = float(self.ref_root[0, 2])
        self.ankle_z0 = float(self.ref_body[0, self.feet_k, 2].min())       # planted ankle height
        feet0 = self.ref_body[0, self.feet_k]
        self.stance_w = float(abs(feet0[0, 1] - feet0[1, 1]))
        names = contract.joint_names
        self.upper_j = np.array([i for i, x in enumerate(names) if not x.startswith(("left_hip", "right_hip",
                                 "left_knee", "right_knee", "left_ankle", "right_ankle"))])
        self.hip_j = np.array([i for i, x in enumerate(names) if x.endswith(("hip_roll_joint", "hip_yaw_joint"))])
        self.leg_j = np.arange(12)
        sole = [model.sensor(f"sole_{b}") for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.sole_adr = np.array([sn.adr[0] for sn in sole])
        self.mg = float(model.body_subtreemass[self.root_id]) * 9.81
        self.torso_id = model.body("torso_link").id
        # per side: hip pitch, knee, ankle pitch
        self.pitch_j = np.array([[names.index(f"{s}_{j}_joint") for j in ("hip_pitch", "knee", "ankle_pitch")]
                                 for s in ("left", "right")])
        self.critic_dim = self.critic_obs().shape[1] if cfg.critic_obs else None

    # -- commands ------------------------------------------------------------

    def _sample_commands(self, ids) -> None:
        cfg, rng, k = self.cfg, self.rng, len(ids)
        if k == 0:
            return
        cmd = np.stack([rng.uniform(*cfg.cmd_vx, k), rng.uniform(*cfg.cmd_vy, k),
                        rng.uniform(*cfg.cmd_wz, k)], axis=1)
        u = rng.random(k)
        single = (u >= cfg.cmd_zero_prob) & (u < cfg.cmd_zero_prob + cfg.cmd_single_prob)
        keep = np.eye(3, dtype=bool)[rng.integers(0, 3, k)]
        cmd = np.where(single[:, None] & ~keep, 0.0, cmd)
        cmd[u < cfg.cmd_zero_prob] = 0.0
        cmd[np.abs(cmd) < cfg.cmd_deadband] = 0.0
        self.command[ids] = cmd
        self.heading_ref[ids] = _yaw(self.qpos[ids, 3:7])
        self.path_err[ids] = 0.0
        self.cmd_left[ids] = (rng.uniform(*cfg.cmd_resample_s, k) * self.c.policy_hz).astype(np.int64)

    def moving(self) -> np.ndarray:
        if self.cfg.always_step:
            return np.ones(self.num_envs, bool)
        return np.any(self.command != 0.0, axis=1)

    def clock_targets(self) -> np.ndarray:
        """(n, 2) wanted ankle heights above planted (_clock_targets); WalkEnv's follow its CPG."""
        return _clock_targets(self.clock, self.moving(), self.cfg.swing_height)

    def gait_obs(self) -> np.ndarray:
        """(n, 2): sin/cos of the clock while moving, (0, 0) when told to stand."""
        a = 2 * np.pi * self.clock
        return np.stack([np.sin(a), np.cos(a)], axis=1) * self.moving()[:, None]

    # -- episode -----------------------------------------------------------------

    def reset(self, ids) -> None:
        super().reset(ids)
        ids = np.asarray(ids, np.int64)
        if ids.size == 0:
            return
        self._sample_commands(ids)
        self.clock[ids] = self.rng.random(ids.size)
        self.prev_feet_xy[ids] = self.xpos[ids][:, self.key_ids[self.feet_k], :2]

    def step(self, action: np.ndarray):
        self.clock = (self.clock + self.clock_step) % 1.0
        self.heading_ref += self.command[:, 2] / self.c.policy_hz
        _, reward, done, fell, timeout = super().step(action)
        self.cmd_left -= 1
        self._sample_commands(np.flatnonzero((self.cmd_left <= 0) & ~done))
        return self.obs(), reward, done, fell, timeout

    # -- observation ---------------------------------------------------------------

    def _task_state(self) -> dict:
        """The observation fields that are the task's rather than the robot's sensors'."""
        return dict(command=self.command, gait=self.gait_obs())

    def obs(self) -> np.ndarray:
        return self.c.build_obs({**self._state(), **self._task_state()})

    def critic_obs(self) -> np.ndarray:
        """The true state: the observation without noise or latency, then base velocity and
        height, the feet against the clock, and the episode's randomisation."""
        c, cfg, n = self.c, self.cfg, self.num_envs
        q, qd = self.qpos, self.qvel
        clean = c.build_obs(dict(base_quat=q[:, 3:7], base_ang_vel=qd[:, 3:6], joint_pos=q[:, 7:],
                                 joint_vel=qd[:, 6:], prev_action=self.action, **self._task_state()))
        fz = self.xpos[:, self.key_ids[self.feet_k], 2] - self.ankle_z0
        hz = c.policy_hz
        push_every = max(1, int(cfg.push_interval_s * hz))
        extra = np.concatenate([
            _rotate_world_to_local(q[:, 3:7], qd[:, 0:3]), q[:, 2:3] - self.stand_z,
            fz, self.clock_targets(),
            np.stack([self.friction[:, self.floor_geoms[0], 0], self.kp_s, self.kd_s,
                      self.act_delay / self.dec, self.obs_delay / self.dec], axis=1),
            self.imu_bias, self._dr_critic(),
            np.stack([self.lifted.astype(float), (self.steps % push_every) / push_every,
                      np.clip(self.cmd_left / hz, 0, 10)], axis=1),
        ], axis=1)
        return np.concatenate([clean, extra], axis=1).astype(np.float32)

    # -- reward ------------------------------------------------------------------------

    def _reward(self):
        w, c, cfg = self.cfg.weights, self.c, self.cfg
        q, qd = self.qpos[:, 7:], self.qvel[:, 6:]
        root_q = self.xquat[:, self.root_id]
        yaw = _yaw(root_q)
        v = _to_heading(self.qvel[:, 0:2], yaw)
        wz = _rotate_local_to_world(root_q, self.qvel[:, 3:6])[:, 2]
        g = quat_to_gravity(root_q)
        ang_b = self.qvel[:, 3:6]

        feet = self.xpos[:, self.key_ids[self.feet_k]]
        fz = feet[:, :, 2] - self.ankle_z0
        want = self.clock_targets()
        down = fz < 0.01
        fvel = (feet[:, :, :2] - self.prev_feet_xy) * c.policy_hz
        self.prev_feet_xy = feet[:, :, :2].copy()
        slip = np.sum(np.where(down, np.sum(fvel ** 2, axis=2), 0.0), axis=1)
        d = feet[:, 0, :2] - feet[:, 1, :2]
        lateral = np.abs(-np.sin(yaw) * d[:, 0] + np.cos(yaw) * d[:, 1])

        ftw = _twist_z(self.xquat[:, self.key_ids[self.feet_k]])
        rate = _wrap(ftw - self.prev_feet_tw) * c.policy_hz
        self.feet_rate = rate
        self.prev_feet_tw = ftw
        k = np.exp(-(rate / 0.5) ** 2)
        nd = down.sum(axis=1)
        feet_twist = np.where(nd > 0, np.sum(np.where(down, k, 0.0), axis=1) / np.maximum(nd, 1), 1.0)

        over = np.maximum(q - c.upper, 0.0) + np.maximum(c.lower - q, 0.0)
        raw = c.default_dof_pos + c.action_scale * np.clip(self.action, -c.action_clip, c.action_clip)
        raw_over = np.maximum(raw - c.upper, 0.0) + np.maximum(c.lower - raw, 0.0)
        g_torso = quat_to_gravity(self.xquat[:, self.torso_id])
        dq = q - c.default_dof_pos
        swing = want / max(cfg.swing_height, 1e-9)                    # (n, 2) in [0, 1]
        ref = np.tile(c.default_dof_pos, (self.num_envs, 1))
        for side in range(2):
            ref[:, self.pitch_j[side]] += np.outer(cfg.ref_amp * swing[:, side], [-1.0, 2.0, -1.0])
        e_ref = np.linalg.norm(q[:, self.leg_j] - ref[:, self.leg_j], axis=1)
        stance = want <= 0.0
        cy, sy = np.cos(yaw), np.sin(yaw)
        vc = np.stack([cy * self.command[:, 0] - sy * self.command[:, 1], sy * self.command[:, 0] + cy * self.command[:, 1]], axis=1)
        self.path_err = self.path_err * np.exp(-1.0 / (cfg.path_tau_s * c.policy_hz)) + (self.qvel[:, 0:2] - vc) / c.policy_hz
        fade = (np.clip(1.0 - np.linalg.norm(self.command[:, :2], axis=1) / cfg.ref_fade_speed, 0.0, 1.0)
                if cfg.ref_fade_speed > 0 else 1.0)
        t = {
            "lin_vel":     np.exp(-10.0 * np.sum((self.command[:, :2] - v) ** 2, axis=1)),
            "ang_vel":     np.exp(-4.0 * (self.command[:, 2] - wz) ** 2),
            "heading":     np.exp(-(_wrap(yaw - self.heading_ref) / 0.15) ** 2),
            "path":        np.exp(-np.sum(self.path_err ** 2, axis=1) / 0.25 ** 2),
            "feet_swing":  np.exp(-np.mean((fz - want) ** 2, axis=1) / 0.025 ** 2),
            "ref_joint":   np.exp(-2.0 * e_ref) * fade,
            "contact":     np.mean(down == stance, axis=1).astype(float),
            "swing_force": np.mean(np.where(stance, 1.0, np.exp(-self.sensordata[:, self.sole_adr] / (0.2 * self.mg))), axis=1),
            "upright":     np.exp(-10.0 * np.sum(g[:, :2] ** 2, axis=1)),
            "height":      np.exp(-40.0 * (self.qpos[:, 2] - self.stand_z) ** 2),
            "posture":     np.exp(-2.0 * np.mean(dq[:, self.upper_j] ** 2, axis=1)),
            "hip_posture": np.exp(-3.0 * np.mean(dq[:, self.hip_j] ** 2, axis=1)),
            "torso_roll":  np.exp(-(g_torso[:, 1] / 0.1) ** 2),
            "feet_twist":  feet_twist,
            "alive":       np.ones(self.num_envs),
            "lin_vel_z":   self.qvel[:, 2] ** 2,
            "ang_vel_xy":  np.sum(ang_b[:, :2] ** 2, axis=1),
            "feet_slip":   slip,
            "feet_close":  np.maximum(0.8 * self.stance_w - lateral, 0.0),
            "action_rate": np.sum(self.target_delta ** 2, axis=1),
            "action_accel": np.sum((self.target_delta - self.prev_delta) ** 2, axis=1),
            "action_over": np.sum(raw_over, axis=1),
            "self_contact": (np.sum(self.sensordata[:, self.touch_adr] > 1e-6, axis=1).astype(float)
                             if self.sensordata is not None else np.zeros(self.num_envs)),
            "torque":      np.mean(self.torque ** 2, axis=1),
            "joint_limit": np.sum(over, axis=1),
        }
        L = self.lifted
        self.heading_ref = np.where(L, yaw, self.heading_ref)      # put down facing wherever it was turned
        self.path_err = np.where(L[:, None], 0.0, self.path_err)
        t["lift_calm"] = np.where(L, np.exp(-0.1 * np.mean(qd[:, :12] ** 2, axis=1)), 0.0)
        if L.any():
            for key in ("lin_vel", "ang_vel", "heading", "path", "feet_swing", "ref_joint", "contact", "swing_force", "upright", "height", "hip_posture", "torso_roll",
                        "feet_twist", "lin_vel_z", "ang_vel_xy", "feet_slip", "feet_close"):
                t[key] = np.where(L, 0.0, t[key])
        reward = sum(getattr(w, key) * val for key, val in t.items())
        return reward, t

    def _fell(self) -> np.ndarray:
        g = quat_to_gravity(self.qpos[:, 3:7])
        tilt = np.arccos(np.clip(-g[:, 2], -1.0, 1.0))
        return (((tilt > self.cfg.max_tilt_rad) | (self.qpos[:, 2] < self.cfg.min_root_height)) & ~self.lifted) \
            | ~np.isfinite(self.qpos).all(axis=1)

    def __repr__(self) -> str:
        return (f"VelocityEnv({self.num_envs} envs, obs {self.obs_dim}, act {self.act_dim}, "
                f"{self.max_steps} steps/episode)")
