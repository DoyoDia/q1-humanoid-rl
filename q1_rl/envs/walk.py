"""Velocity-command walking on the Q1's Walk contract.

The robot's motion controller walks with a velocity-command policy (model_type Walk). Its
runtime builds the observation, keeps the 5-frame history, runs a CPG that clocks the gait,
swings the arms in time with it and holds waist yaw; the ONNX only moves the legs and waist
roll. This env reproduces that interface so a policy trained here sees what the robot feeds it.

  * Observation: the `walk` contract in robots/q1/contract/obs_spec.json, 71 per frame, five
    frames oldest first. The fields the config does not spell out were found by running the
    vendor's released policy in this sim (obs_spec.json layout_source, tools/check_vendor_walk.py).
  * Action: 13 (the 12 leg joints and waist roll, joint order), target = default + 0.5 a,
    walk's gains. The ONNX appends 9 zeros to make the 22 the runtime reads.
  * The CPG. cpg_law "runtime" (walk_v8 on): a model of the robot's own gait clock
    (RuntimeCPG below) -- the period law, the pattern field, the freeze while standing, and
    targets that are not clipped to the joint limits (clamp_targets False). cpg_law "random"
    (walk_v2..v7): (sin, cos, -sin, -cos) of one phase, the period drawn per command from
    cycle_time_s; at a zero command half the episodes keep it running and half zero it. Only
    "runtime" policies transfer to the robot (walk_v7 falls 4/4 under the runtime clock).
  * Arms and waist yaw: position-controlled by the robot. Here they hold the nominal pose with
    mimic's arm gains and the shoulders swing at a random amplitude in anti-phase with the legs.

Everything else -- rewards, commands, domain randomisation, pushes, lifts, latency -- is
VelocityEnv's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .velocity import VelocityConfig, VelocityEnv

PATTERN_WALK = (1.0, 0.0)
CPG_STAND = (0.0, 1.0, 0.0, 1.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class RuntimeCPG:
    """A model of the robot's gait clock.

    Two Hopf oscillators, state (xL, yL, xR, yR) -- also the observation's `cpg` field as is.
    Per 20 ms tick, one Euler step x += dt (A(x) x + R x):
      A per leg: [[50 (1 - r_i^2), w_i], [-w_i, 50 (1 - r_i^2)]], r_i^2 = x_i^2 + y_i^2,
      w_i = (pi / P) [1 / (beta (1 + e^(-50 y_i))) + 1 / ((1 - beta) (1 + e^(50 y_i)))],
      R = [[I, -I], [-I, I]] (the legs held in anti-phase, gain 1).
    So each leg's y > 0 half (its stance) lasts beta P and its y < 0 half (swing) (1 - beta) P.
    P and beta follow the command:
      s = 0.7 sqrt(vx^2 + (2 vy)^2), yaw not used; s < 0.7: P 0.8, beta 0.6; else with e = s - 0.7
      P = 0.8 (1 - 0.57143 e), beta = clip(0.6 - 0.42857 e, 0.25, 0.6); e > 0.8: 0.4342848, 0.257144.
    The observation's `pattern` is (P, beta - 0.6).
    Standing, n = |(vx, vy, wz)|:
      frozen at (0, 1, 0, 1) while n < 0.1 and the posture is fine; a command (n >= 0.1) snaps it
      to (1, 0, -1, 0) or (-1, 0, 1, 0) by the sign of vy, else wz, else vx (the leg on that side
      lifts first), as does a tilt beyond the standing limits (by the sign of roll);
      walking, n < 0.1 and the posture and feet fine: it runs on until either leg's y crosses
      from > 0 to <= 0, then snaps to (0, 1, 0, 1). The robot also checks the feet's alignment
      before stopping; that stands in here as a random number of crossings let pass.
    """

    def __init__(self, n: int):
        self.x = np.tile(CPG_STAND, (n, 1))
        self.standing = np.ones(n, bool)
        self.skip = np.zeros(n, np.int64)          # crossings to let pass before stopping

    @staticmethod
    def law(command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = 0.7 * np.sqrt(command[:, 0] ** 2 + (2.0 * command[:, 1]) ** 2)
        e = s - 0.7
        P = np.where(s < 0.7, 0.8, 0.8 * (1.0 - e * 0.57143))
        beta = np.where(s < 0.7, 0.6, np.clip(0.6 - e * 0.42857, 0.25, 0.6))
        far = e > 0.8
        return np.where(far, 0.4342848, P), np.where(far, 0.257144, beta)

    def reset(self, ids) -> None:
        self.x[ids] = CPG_STAND
        self.standing[ids] = True

    def tick(self, command: np.ndarray, stand_ok: np.ndarray, stop_ok: np.ndarray, roll_neg: np.ndarray,
             dt: float = 0.02) -> None:
        """One runtime tick. stand_ok: the standing posture holds (euler_stand_*); stop_ok:
        walking may stop here (euler_need_stand_*); roll_neg: which way a tilt restarts it."""
        n = np.linalg.norm(command, axis=1)
        vx, vy, wz = command.T
        first = np.where(np.abs(vy) > 0.01, vy > 0.01, np.where(np.abs(wz) > 0.01, wz > 0.01, vx > 0.01))
        go = self.standing & (n >= 0.1)
        tip = self.standing & (n < 0.1) & ~stand_ok
        start = go | tip
        lead = np.where(go, first, roll_neg)
        self.x[start] = np.where(lead[start, None], [1.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 1.0, 0.0])
        self.standing[start] = False

        run = ~self.standing & ~start
        if run.any():
            x = self.x[run]
            P, beta = self.law(command[run])
            new = x.copy()
            for i in (0, 1):
                xi, yi = x[:, 2 * i], x[:, 2 * i + 1]
                d = 1.0 - (xi ** 2 + yi ** 2)
                w = (np.pi / P) * (_sigmoid(50.0 * yi) / beta + _sigmoid(-50.0 * yi) / (1.0 - beta))
                xo, yo = x[:, 2 * (1 - i)], x[:, 2 * (1 - i) + 1]
                new[:, 2 * i] = xi + dt * (50.0 * d * xi + w * yi + xi - xo)
                new[:, 2 * i + 1] = yi + dt * (-w * xi + 50.0 * d * yi + yi - yo)
            crossed = np.any((x[:, 1::2] > 0.0) & (new[:, 1::2] <= 0.0), axis=1)
            want_stop = (n[run] < 0.1) & stop_ok[run] & crossed
            idx = np.flatnonzero(run)
            let_pass = want_stop & (self.skip[idx] > 0)
            self.skip[idx[let_pass]] -= 1
            stop = want_stop & ~let_pass
            new[stop] = CPG_STAND
            self.x[idx] = new
            self.standing[idx[stop]] = True

    def pattern(self, command: np.ndarray) -> np.ndarray:
        P, beta = self.law(command)
        return np.stack([P, beta - 0.6], axis=1)

    def swing(self) -> np.ndarray:
        """(n, 2) swing progress in [0, 1] per leg: -y / r while y < 0 (sin of the swing phase)."""
        y = self.x[:, 1::2]
        r = np.sqrt(self.x[:, 0::2] ** 2 + y ** 2)
        return np.maximum(0.0, -y / np.maximum(r, 1e-6))


@dataclass
class WalkConfig(VelocityConfig):
    cmd_deadband: float = 0.1                   # walk_run.yaml cmd_*_threshold_zeroing
    cycle_time_s: tuple[float, float] = (0.45, 0.7)
    # Shorter cycles when fast: the period's range is capped at this stride (m per cycle) over the
    # commanded planar speed, its low end at 80% of the cap -- 1.2 m/s draws 0.40-0.50 s, 1.0 m/s
    # 0.45-0.60 s, slower stays cycle_time_s. 0: the period ignores the command.
    cycle_stride_m: float = 0.0
    cpg_idle_running: float = 0.5               # share of episodes whose CPG keeps running at a zero command
    # "runtime": the robot's own CPG (RuntimeCPG) -- period and duty from the command, frozen
    # while standing, pattern (P, beta - 0.6); cycle_time_s, cycle_stride_m and cpg_idle_running
    # are then unused. "random": the earlier fixed-law clock (sim only, see the module doc).
    cpg_law: str = "random"
    cpg_stop_skip: int = 3                      # runtime: up to this many crossings pass before a stop (feet alignment)
    arm_swing: float = 0.3                      # rad, shoulder pitch amplitude drawn in [0, this]
    arm_gains: tuple[float, float] = (20.0, 1.5)
    # Share of episodes that start part-way to MIMIC_OPENSOURCE_V2's stance (knees 0.22 rad
    # instead of 0.70; the blend uniform in [0, 1]). On the robot the walking state takes over
    # from standing and from the end of every mimic motion (auto_next_action), and in the
    # viewer from the idle policy: the policy has to crouch into its own stance from there.
    start_from_mimic: float = 0.0


class WalkEnv(VelocityEnv):
    def __init__(self, model, motion, contract, cfg: WalkConfig, seed: int = 0):
        if contract.frame_stack < 2 or "cpg" not in contract.fields:
            raise ValueError(f"WalkEnv needs the walk contract, got {contract.name!r}")
        n, names = cfg.num_envs, contract.joint_names
        self.na = int(contract.spec["policy_action_dim"])            # 13: legs + waist roll
        self.pos_j = np.arange(self.na, contract.n_joints)           # the runtime's position-controlled joints
        self.arm_j = np.array([i for i, x in enumerate(names) if "shoulder" in x or "elbow" in x])
        self.shoulder_j = np.array([names.index(f"{s}_shoulder_pitch_joint") for s in ("left", "right")])
        self.frame_dim, self.stack = contract.obs_dim, contract.frame_stack
        self.hist = np.zeros((n, self.stack, self.frame_dim), np.float32)
        self.period = np.full(n, float(np.mean(cfg.cycle_time_s)))
        self.idle_running = np.zeros(n, bool)
        from ..contract import Contract
        self.mimic_pose = Contract("mimic_opensource_v2", contract.dir).default_dof_pos
        self.arm_amp = np.zeros(n)
        if cfg.cpg_law not in ("random", "runtime"):
            raise ValueError(f"cpg_law {cfg.cpg_law!r}")
        self.cpg = RuntimeCPG(n) if cfg.cpg_law == "runtime" else None
        super().__init__(model, motion, contract, cfg, seed)
        # MotionTrackingEnv built the backend with the contract's gains; the arms get theirs on
        # what the backend reads and on what each episode's kp randomisation scales from.
        kp, kd = cfg.arm_gains
        self.gain0[self.arm_j, 0] = kp
        self.bias0[self.arm_j, 1:3] = (-kp, -kd)
        self.gainprm[:, self.arm_j, 0] = kp * self.kp_s[:, None]
        self.biasprm[:, self.arm_j, 1] = -kp * self.kp_s[:, None]
        self.biasprm[:, self.arm_j, 2] = -kd * self.kd_s[:, None]
        self.clock_step = 1.0 / (self.period * contract.policy_hz)    # VelocityEnv set its fixed one
        self.upper_j = np.array([names.index("waist_roll_joint")])   # the rest is the runtime's
        self.obs_dim = self.stack * self.frame_dim
        self.act_dim = self.na
        self.critic_dim = self.critic_obs().shape[1] if cfg.critic_obs else None

    # -- commands and clock --------------------------------------------------------

    def _sample_commands(self, ids) -> None:
        super()._sample_commands(ids)
        k = len(ids)
        if k:
            self.period[ids] = self.rng.uniform(*self.cycle_range(self.command[ids]))
            self.clock_step = 1.0 / (self.period * self.c.policy_hz)
            if self.cpg is not None:
                self.cpg.skip[ids] = self.rng.integers(0, self.cfg.cpg_stop_skip + 1, k)

    def cycle_range(self, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) of the CPG period for each command row (cycle_stride_m)."""
        lo, hi = (np.full(len(command), x) for x in self.cfg.cycle_time_s)
        if self.cfg.cycle_stride_m > 0:
            hi = np.minimum(hi, self.cfg.cycle_stride_m / np.maximum(np.linalg.norm(command[:, :2], axis=1), 1e-6))
            lo = np.minimum(lo, 0.8 * hi)
        return lo, hi

    def _cpg(self) -> np.ndarray:
        if self.cpg is not None:
            return self.cpg.x.copy()
        a = 2 * np.pi * self.clock
        s, c = np.sin(a), np.cos(a)
        on = self.moving() | self.idle_running
        return np.stack([s, c, -s, -c], axis=1) * on[:, None]

    def _task_state(self) -> dict:
        n = self.num_envs
        pattern = self.cpg.pattern(self.command) if self.cpg is not None else np.tile(PATTERN_WALK, (n, 1))
        return dict(command=self.command, gait=self.gait_obs(), cpg=self._cpg(), pattern=pattern)

    def clock_targets(self) -> np.ndarray:
        if self.cpg is None:
            return super().clock_targets()
        return self.cfg.swing_height * self.cpg.swing()

    def _tick_cpg(self) -> None:
        """The runtime's CPG tick, on the posture the IMU reads now (cpg_settings euler_*)."""
        w, x, y, z = self.qpos[:, 3:7].T
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        stand_ok = (np.abs(roll) < 0.06) & (pitch > -0.05) & (pitch < 0.125)
        stop_ok = (np.abs(roll) < 0.04) & (pitch > -0.03) & (pitch < 0.100)
        self.cpg.tick(self.command, stand_ok, stop_ok, roll < 0, 1.0 / self.c.policy_hz)
        self.clock = (np.arctan2(self.cpg.x[:, 0], self.cpg.x[:, 1]) / (2 * np.pi)) % 1.0
        self.period = self.cpg.law(self.command)[0]

    # -- episode -------------------------------------------------------------------

    def reset(self, ids) -> None:
        super().reset(ids)
        ids = np.asarray(ids, np.int64)
        if ids.size == 0:
            return
        self.idle_running[ids] = self.rng.random(ids.size) < self.cfg.cpg_idle_running
        if self.cpg is not None:
            self.cpg.reset(ids)
        self.arm_amp[ids] = self.rng.uniform(0.0, self.cfg.arm_swing, ids.size)
        self.hist[ids] = self._frame()[ids][:, None, :]               # the runtime starts full

    def _start_pose(self, ids, ref: np.ndarray) -> np.ndarray:
        k = len(ids)
        mix = (self.rng.random(k) < self.cfg.start_from_mimic) * self.rng.random(k)
        return ref + mix[:, None] * (self.mimic_pose - ref)

    def _frame(self) -> np.ndarray:
        return self.c.build_obs({**self._state(), **self._task_state()})

    def _runtime_action(self, action: np.ndarray) -> np.ndarray:
        """The policy's 13 -> the 22 the motors get: zeros for the position-controlled joints
        except the shoulders' swing, as an action (target = default + scale * action)."""
        full = np.zeros((self.num_envs, self.c.n_joints))
        full[:, : self.na] = action
        if self.cpg is not None:
            swing = self.arm_amp * np.sin(2 * np.pi * self.clock) * ~self.cpg.standing
        else:
            swing = self.arm_amp * np.sin(2 * np.pi * self.clock) * self.moving()
        full[:, self.shoulder_j[0]] = swing / self.c.action_scale
        full[:, self.shoulder_j[1]] = -swing / self.c.action_scale
        return full

    def step(self, action: np.ndarray):
        _, reward, done, fell, timeout = super().step(self._runtime_action(np.asarray(action)))
        if self.cpg is not None:                                       # the runtime's tick before its next obs
            self._tick_cpg()
        self.hist[:, :-1] = self.hist[:, 1:]
        self.hist[:, -1] = self._frame()
        return self.obs(), reward, done, fell, timeout

    def obs(self) -> np.ndarray:
        """(n, 5 x 71): the frames oldest first, the newest (the runtime's `observations`) last."""
        return self.hist.reshape(self.num_envs, -1).copy()

    def critic_obs(self) -> np.ndarray:
        idle = self.cpg.standing if self.cpg is not None else self.idle_running
        extra = np.stack([self.period, idle.astype(float), self.arm_amp], axis=1)
        return np.concatenate([super().critic_obs(), extra.astype(np.float32)], axis=1)

    def __repr__(self) -> str:
        return (f"WalkEnv({self.num_envs} envs, obs {self.stack} x {self.frame_dim}, act {self.act_dim}, "
                f"{self.max_steps} steps/episode)")
