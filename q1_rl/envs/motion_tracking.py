"""Motion tracking: make the robot reproduce a reference clip.

This is the first task because it lines up with the only deployment contract we have
actually verified. MIMIC_OPENSOURCE_V2's observation already carries 44 dimensions of
reference trajectory, so a policy trained here has exactly the input the robot's own
runtime will hand it -- there is no impedance mismatch to paper over at deploy time.

Rewards are DeepMimic-shaped: a weighted sum of exp(-k * error^2) terms, which behaves
far better than raw negative error because every term stays in [0, 1] and no single one
can dominate the sum early in training.

Everything in here goes through Contract for observations and action mapping, so the
policy this trains is contract-conforming by construction rather than by review.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contract import Contract, quat_to_gravity
from ..motion import Motion

# Bodies whose world position is tracked. A handful of extremities carries almost all the
# information about whether a pose looks right, and costs far less than all 26.
KEY_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
              "left_elbow_link", "right_elbow_link",
              "torso_link", "head_link")

# Model fields this environment randomises per episode.
EXPAND_FIELDS = ("geom_friction", "actuator_gainprm", "actuator_biasprm",
                 "body_mass", "body_inertia", "body_ipos")

# Bodies whose centre of mass is shifted per episode (EnvConfig.com_offset): the two that
# carry the battery, the boards and whatever gets bolted on.
COM_BODIES = ("pelvis", "torso_link")


@dataclass
class RewardWeights:
    joint_pos: float = 1.0        # matching the reference pose -- the main objective
    joint_vel: float = 0.2
    body_pos: float = 1.0         # extremities in the root frame: catches "right angles, wrong shape"
    # Feet on their own, root-relative, with a tight kernel. Averaged in with the hands and
    # head in body_pos, a foot that should be up but stays down barely registered: a dance
    # policy kept both feet planted through the clip's leg lifts and lost little reward.
    feet_pos: float = 1.0
    # How far each foot is above the other one, against the reference. feet_pos is root-
    # relative and tuned for decimetres: a dance foot that should be 2.6 cm up but stays
    # on the floor still scored 0.97, so a policy kept both feet down through every toe tap
    # (lifting when the reference did in 12% of frames). Relative to the lower foot, not
    # the floor, because a retargeted clip floats a centimetre or two.
    feet_height: float = 0.5
    body_ori: float = 0.3
    # The torso's tilt (gravity in the torso frame: roll and pitch, not heading) against the
    # reference's. A walk policy balanced by throwing its upper body side to side -- torso
    # roll 26 deg peak to peak against the reference's 8 -- and body_ori, averaged over six
    # bodies with a loose kernel, barely noticed.
    torso_ori: float = 0.5
    root_ori: float = 0.5         # staying upright
    root_height: float = 0.3
    # Base velocity in the heading frame (forward/sideways) and yaw rate. Without these the
    # env never asked the robot to go anywhere: on a retargeted walk the policy matched the
    # joint angles stepping in place while the reference walked off.
    root_vel: float = 1.0
    yaw_rate: float = 0.3
    # Heading against the reference's, from the episode's start (they start aligned). The
    # observation has no heading -- gravity and the gyro only -- so a policy cannot see that it
    # has turned and cannot correct it; it can only avoid turning. Without this term nothing
    # asked it to: broadcast gymnastics, 270 s from frame 0, ended 225 deg off, in jumps of
    # ~100 deg during side steps (each landing pivoting the same way), while yaw_rate's loose
    # kernel charged 3% for turning 100 deg in 10 s. Linear, 1 - |e| / pi, not a Gaussian:
    # the turning is systematic -- with every randomisation off the policy still turned
    # the same way, -570 to -940 deg, in the same sections -- so it is learnable, but at
    # ~3 deg/s an episode drifts 60 deg, where exp(-(e/0.15)^2) is 0 whether the robot
    # turned 30 or 60 and there was nothing to climb.
    heading: float = 0.5
    # A planted foot not twisting about the vertical. The heading term alone did not stop the
    # turning: the stance foot of the broadcast gymnastics policy jittered in yaw at a median
    # 1.28 rad/s (+-1.5 deg every 20 ms) and ratcheted round -- 180 deg in 25 s of side steps
    # -- with friction 1.2 and with noslip alike. Unlike heading this is seen every step.
    # Per foot within 1 cm of flat on the floor: exp(-(yaw rate / 0.5 rad/s)^2), mean over
    # the feet that are down (1 when none is).
    feet_twist: float = 0.5
    alive: float = 0.5
    # Penalty on how far the joint targets jump per step: sum over joints of (delta rad)^2.
    # Sized so that jitter costs something (random-policy deltas ~0.1 rad -> about -0.2)
    # while tracking itself is nearly free (the squat reference moves at most ~0.03 rad
    # per step on a few joints -> about -0.003). Measured in radians, not action units,
    # so PPO's unbounded actions and SAC's scaled tanh are penalised on the same footing.
    action_rate: float = -1.0
    # Second difference of the targets, sum over joints of rad^2. The first walk policy
    # chattered: targets moved 0.157 rad RMS per step against the reference's 0.018,
    # reversing direction on half of all joint-steps, pinned at the 0.2 rad/frame rate
    # limit. action_rate alone let it; a zig-zag costs (2 delta)^2 here, smooth motion
    # almost nothing.
    action_accel: float = -1.0
    torque: float = -1e-5
    joint_limit: float = -0.5
    # Waist and arms alone against the reference, exp(-mean(e^2) / 0.1^2): 0.05 rad RMS -> 0.78,
    # 0.1 -> 0.37. joint_pos averages all 22 joints, so an arm 0.3 rad off costs it little: the
    # idle policy (stand_v3) stood with both elbows locked straight against their limits, the
    # right arm 0.3 rad out to the side and the waist turned 8 deg.
    upper_pose: float = 0.0
    # rad the raw target (default + scale * action) lies beyond the joint limits -- the clip to
    # the limits makes such an action a constant, noise-free command (stand_v3's elbows).
    action_over: float = 0.0
    # Arm links touching anything (touch sensors on shoulder-yaw, elbow, hand links; see
    # robots/q1/build_model.py): per touching link per step. The same policy had a hand or
    # elbow in its hip 97% of the time.
    self_contact: float = -0.5
    # Lifted off the floor: legs still (exp(-0.1 mean qd^2) over the 12 leg joints).
    lift_calm: float = 2.0


@dataclass
class EnvConfig:
    num_envs: int = 2048
    backend: str = "mjbatch"
    episode_seconds: float = 4.0
    # termination
    # Both relative to the reference, so a clip that bends over (pelvis pitched 90 deg to
    # touch the toes) or squats (pelvis at 0.22 m) is not a fall. For upright clips this is
    # the same as before: tilt is the angle from the reference's gravity direction, which
    # is ~vertical, and the height floor stays min_root_height unless the reference itself
    # goes within root_height_drop of it.
    max_tilt_rad: float = 1.05            # ~60 deg off the reference; the robot's own guard trips at 30
    min_root_height: float = 0.22
    root_height_drop: float = 0.15        # m below the reference pelvis, when that is lower
    max_joint_error: float = 1.2          # rad, RMS over joints -- lost the plot, stop early
    # reset randomisation
    init_joint_noise: float = 0.05
    init_vel_noise: float = 0.10
    init_clearance: float = 0.002         # m between the lowest sole point and the floor
    random_phase: bool = True
    # Where episodes start. Uniform over the clip spends most practice on the easy parts:
    # a dance policy learned to skip its few single-leg moves. Adaptive sampling splits
    # the clip into bins, keeps a decaying count of falls per bin (credited partly to the
    # bins just before, where a recovery has to start), and draws start bins in proportion,
    # mixed with uniform so the easy parts are not forgotten.
    adaptive_sampling: bool = True
    sampling_bin_s: float = 1.0
    sampling_uniform: float = 0.2         # share of starts drawn uniformly
    sampling_decay: float = 0.999         # per policy step: ~1000-step memory
    # domain randomisation, resampled per episode
    friction_range: tuple[float, float] = (0.4, 1.2)
    kp_scale_range: tuple[float, float] = (0.85, 1.15)
    kd_scale_range: tuple[float, float] = (0.85, 1.15)
    # Inertial parameters. The two official models (q1_v2, q1_v3) disagree by 3% in total
    # mass and by 40% on the ankle-pitch links, so the robot's own numbers are uncertain by
    # at least that much. Every link's mass (and inertia, with it) is scaled independently,
    # and the pelvis and torso centres of mass move by up to com_offset per axis.
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    com_offset: float = 0.02              # m, uniform +- per axis
    # Encoder zero offset, per joint per episode: the robot reads q + b and its PD loop
    # drives the reading to the target, so the joint really settles at target - b. Both
    # sides are simulated -- the observation sees q + b, the motors get target - b.
    encoder_offset: float = 0.02          # rad, uniform +-
    # Physics substep. 0: the contract's (1 ms, decimation 20). 0.002 halves the physics
    # per policy step; latencies stay in milliseconds (rounded to the substep).
    sim_dt: float = 0.0
    # Clip position targets to the joint limits before the motors (what the firmware path of
    # the mimic runners does). False: the target goes out as is and the actuator's forcerange
    # saturates the torque -- the robot's Walk runtime does not clamp the position, it clamps
    # kp (q_t - q) - kd dq to a torque limit.
    clamp_targets: bool = True
    push_interval_s: float = 1.5
    push_vel: float = 0.35                # m/s applied to the base, both horizontal axes
    push_ang_vel: float = 0.0             # rad/s added to the base's angular velocity
    # Per body axis (roll, pitch, yaw) share of push_ang_vel. A yaw kick turns the robot and,
    # with no heading in the observation, leaves an error it can neither see nor undo -- it
    # drowned the heading reward's signal. (1, 1, 0) still trains balance against tips.
    push_ang_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Being picked up. In a lift_prob share of episodes, at a random moment the base is
    # raised by lift_height over lift_rise_s and held there (position and attitude, as
    # hands on the waist would) for lift_hold_s; the episode then ends. While lifted only
    # the upper body is tracked and the legs are rewarded for staying still (lift_calm):
    # on the robot a lift detector switches to damping, and this covers the moments
    # before it does -- a tracking policy with its feet in the air otherwise kicks.
    lift_prob: float = 0.0
    lift_height: tuple[float, float] = (0.05, 0.25)   # m
    lift_start_s: tuple[float, float] = (0.3, 3.0)
    lift_hold_s: tuple[float, float] = (0.5, 2.0)
    lift_rise_s: float = 0.3
    # Sensor noise: uniform +-, on what the policy SEES only -- reward and termination use
    # the true state. Joint velocity is the noisiest signal on the robot (differentiated
    # encoder, low-passed at 100 Hz), angular velocity next.
    noise_ang_vel: float = 0.2            # rad/s, per step
    noise_ori: float = 0.02               # rad, per step, small rotation of the IMU quaternion
    imu_bias: float = 0.03                # rad, per episode, roll/pitch mounting offset
    noise_joint_pos: float = 0.01         # rad, per step
    noise_joint_vel: float = 1.0          # rad/s, per step
    # Latency, sampled per episode, in physics substeps (1 ms each). Action delay: the new
    # target reaches the motors this many substeps into the control period, the previous
    # one holds until then (inference + iceoryx + the 1 kHz command thread on the robot).
    # Observation delay: the observation is the state this many substeps before the end of
    # the period (joint state arrives at 500 Hz, so it is up to 2 ms old when read).
    # Values are drawn from a grid so the substep schedule stays a handful of backend calls.
    act_delay_steps: tuple[int, int] = (0, 10)
    obs_delay_steps: tuple[int, int] = (0, 2)
    delay_grid: int = 2
    # What the policy sees. "contract": the Contract's single frame (MIMIC_OPENSOURCE_V2, 119).
    obs_mode: str = "contract"
    # Asymmetric actor-critic: the critic (training only, never exported) gets critic_obs() --
    # the true state the policy cannot sense -- instead of the policy's observation.
    critic_obs: bool = False
    weights: RewardWeights = field(default_factory=RewardWeights)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _twist_z(q: np.ndarray) -> np.ndarray:
    """Rotation about the world vertical of (w, x, y, z) quaternions (the twist of a swing-twist
    split). Unlike Euler yaw it stays defined with the body pitched 90 deg: bending to the toes,
    Euler yaw jumped a full turn and a heading error computed from it with it."""
    return 2.0 * np.arctan2(q[..., 3], q[..., 0])


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _to_heading(v_xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v_xy[:, 0] + s * v_xy[:, 1], -s * v_xy[:, 0] + c * v_xy[:, 1]], axis=-1)


def _rotate_local_to_world(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q) @ v for (w, x, y, z) quaternions."""
    w = q[:, :1]
    u = q[:, 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _rotate_world_to_local(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _rotate_local_to_world(q * np.array([1.0, -1.0, -1.0, -1.0]), v)


def _small_rot(v: np.ndarray) -> np.ndarray:
    """Quaternion of the rotation vector v (..., 3)."""
    ang = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = v / np.maximum(ang, 1e-12)
    return np.concatenate([np.cos(ang / 2), axis * np.sin(ang / 2)], axis=-1)


class MotionTrackingEnv:
    def __init__(self, model, motion: Motion, contract: Contract, cfg: EnvConfig, seed: int = 0):
        from ..backends import make_backend

        if abs(motion.fps - contract.policy_hz) > 1e-6:
            motion = motion.resample(contract.policy_hz)
        self.cfg, self.c, self.motion = cfg, contract, motion
        self.model = model
        if cfg.sim_dt:
            model.opt.timestep = cfg.sim_dt
        dt = float(model.opt.timestep)
        self.dec = int(round(1.0 / (contract.policy_hz * dt)))
        if abs(self.dec * dt * contract.policy_hz - 1.0) > 1e-9:
            raise ValueError(f"sim_dt {dt} does not divide the {contract.policy_hz} Hz policy period")
        self.ms_per_substep = dt * 1000.0
        # The actuators are PD servos (gainprm kp, biasprm -kp -kd); the contract says which
        # gains. q1_physics.xml carries MIMIC_OPENSOURCE_V2's, so for it this changes nothing.
        model.actuator_gainprm[:, 0] = contract.kp
        model.actuator_biasprm[:, 1] = -contract.kp
        model.actuator_biasprm[:, 2] = -contract.kd
        if not cfg.clamp_targets:
            model.actuator_ctrllimited[:] = 0
        # mujoco_warp fixes per-world model-field batching inside put_model(), so the
        # fields this env randomises have to be declared at construction. mjbatch ignores
        # the argument -- it can expand anything on demand.
        self.backend = make_backend(cfg.backend, model, cfg.num_envs,
                                    expand_fields=EXPAND_FIELDS)
        n, nj = cfg.num_envs, contract.n_joints
        self.num_envs, self.nj = n, nj

        b = self.backend
        self.qpos, self.qvel, self.ctrl = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl")
        self.torque = b.bind("actuator_force")
        touch = [i for i in range(model.nsensor) if model.sensor(i).name.startswith("touch_")]
        self.touch_adr = np.array([model.sensor_adr[i] for i in touch], np.int64)
        self.sensordata = b.bind("sensordata") if len(touch) else None
        self.xpos, self.xquat = b.bind("xpos"), b.bind("xquat")
        self.geom_xpos = b.bind("geom_xpos")
        self.friction = b.expand("geom_friction")
        self.gainprm, self.biasprm = b.expand("actuator_gainprm"), b.expand("actuator_biasprm")
        self.gain0 = np.array(model.actuator_gainprm, copy=True)
        self.bias0 = np.array(model.actuator_biasprm, copy=True)
        self.dr_mass = cfg.mass_scale_range != (1.0, 1.0) or cfg.com_offset > 0
        if self.dr_mass:
            self.body_mass, self.body_inertia = b.expand("body_mass"), b.expand("body_inertia")
            self.body_ipos = b.expand("body_ipos")
            self.mass0 = np.array(model.body_mass, copy=True)
            self.inertia0 = np.array(model.body_inertia, copy=True)
            self.ipos0 = np.array(model.body_ipos, copy=True)
            self.com_ids = np.array([model.body(x).id for x in COM_BODIES])

        import mujoco
        self.key_ids = np.array([model.body(x).id for x in KEY_BODIES])
        self.feet_k = np.array([i for i, x in enumerate(KEY_BODIES) if "ankle" in x])
        self.torso_k = KEY_BODIES.index("torso_link")
        self.root_id = model.body(1).id
        self.stand_key = model.key("stand").id
        # Friction is randomised on the FLOOR, not on the robot. The floor carries
        # priority=1, and when two geoms' priorities differ MuJoCo takes the contact
        # parameters from the higher one -- so randomising the robot's geoms changed
        # nothing: contact friction stayed at the floor's 0.6 whatever was written.
        self.floor_geoms = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] == 0])
        if not len(self.floor_geoms):
            raise ValueError("model has no world-body geom to randomise friction on")

        # The sole contact points: the floor-touching spheres. Used to put a freshly reset
        # robot's feet on the ground whatever pose and noise it was reset into.
        self.sole_geoms = np.array([g for g in range(model.ngeom)
                                    if model.geom_bodyid[g] != 0 and model.geom_conaffinity[g]
                                    and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE])
        self.sole_radius = model.geom_size[self.sole_geoms, 0] if len(self.sole_geoms) else None

        self.ref_qpos = np.asarray(motion.joint_pos, np.float64)
        self.ref_qvel = np.asarray(motion.joint_vel, np.float64)
        self.ref_body = np.asarray(motion.body_pos_w, np.float64)[:, self.key_ids - 1]
        self.ref_bquat = np.asarray(motion.body_quat_w, np.float64)[:, self.key_ids - 1]
        self.ref_root = np.asarray(motion.body_pos_w, np.float64)[:, 0]
        self.ref_rquat = np.asarray(motion.body_quat_w, np.float64)[:, 0]
        self.ref_rvel = np.asarray(motion.body_lin_vel_w, np.float64)[:, 0]        # world
        self.ref_rangvel = np.asarray(motion.body_ang_vel_w, np.float64)[:, 0]     # world
        self.T = motion.n_frames
        self.cyclic = motion.cyclic

        self.obs_dim = contract.obs_dim
        self.act_dim = nj
        self.max_steps = int(round(cfg.episode_seconds * contract.policy_hz))

        self.rng = np.random.default_rng(seed)
        self.phase = np.zeros(n, np.int64)
        hi = self.T if self.cyclic else max(1, self.T - 1)
        self.bin_len = max(1, int(round(cfg.sampling_bin_s * contract.policy_hz)))
        self.n_bins = int(np.ceil(hi / self.bin_len))
        self.bin_fails = np.zeros(self.n_bins)
        self.steps = np.zeros(n, np.int64)
        self.lift_at = np.full(n, -1, np.int64)         # step the lift starts, -1: no lift
        self.lift_end = np.zeros(n, np.int64)
        self.lift_h = np.zeros(n)
        self.lift_pose = np.zeros((n, 7))
        self.lifted = np.zeros(n, bool)
        self.prev_feet_tw = np.zeros((n, len(self.feet_k)))
        # a foot is down within 1 cm of the reference's planted ankle height
        self.foot_down_z = float(self.ref_body[:, self.feet_k, 2].min()) + 0.01
        self.action = np.zeros((n, nj))
        self.prev_target = np.tile(contract.default_dof_pos, (n, 1))
        self.target_delta = np.zeros((n, nj))
        self.prev_delta = np.zeros((n, nj))
        self.reward_terms: dict[str, np.ndarray] = {}
        # latency state: per-env delays, and the (possibly stale) state the policy observes
        self.act_delay = np.zeros(n, np.int64)
        self.obs_delay = np.zeros(n, np.int64)
        self.obs_qpos = np.array(self.qpos, copy=True)
        self.obs_qvel = np.array(self.qvel, copy=True)
        self.imu_bias_q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        self.imu_bias = np.zeros((n, 2))
        self.kp_s, self.kd_s = np.ones(n), np.ones(n)
        self.mass_s = np.ones(n)                        # total robot mass / nominal
        self.com_off = np.zeros((n, len(COM_BODIES), 3))
        self.enc_bias = np.zeros((n, nj))
        self.feet_rate = np.zeros((n, len(self.feet_k)))

        if cfg.obs_mode != "contract":
            raise ValueError(f"obs_mode {cfg.obs_mode!r}")
        self.reset(np.arange(n))
        self.critic_dim = self.critic_obs().shape[1] if cfg.critic_obs else None

    # -- episode management -------------------------------------------------

    def reset(self, ids) -> None:
        ids = np.asarray(ids, np.int64)
        if ids.size == 0:
            return
        n, cfg, c = ids.size, self.cfg, self.c
        self.backend.reset(ids, keyframe=self.stand_key)

        # A one-shot clip may start anywhere but its last frame -- there must be at least
        # one step of reference ahead.
        hi = self.T if self.cyclic else max(1, self.T - 1)
        if not cfg.random_phase:
            self.phase[ids] = 0
        elif cfg.adaptive_sampling and self.bin_fails.sum() > 0:
            p = self.sampling_probs()
            b = self.rng.choice(self.n_bins, size=n, p=p)
            self.phase[ids] = np.minimum(b * self.bin_len + self.rng.integers(0, self.bin_len, n), hi - 1)
        else:
            self.phase[ids] = self.rng.integers(0, hi, n)
        ph = self.phase[ids]
        ref = self.ref_qpos[ph]
        # The base comes from the reference too, not from the stand keyframe. With the base
        # left at stand height and the knees bent into a squat phase, the robot started up to
        # 39 mm in the air and every such episode opened with a drop.
        self.qpos[ids, 0:3] = self.ref_root[ph]
        self.qpos[ids, 3:7] = self.ref_rquat[ph]
        self.qpos[ids, 7:] = self._start_pose(ids, ref) + self.rng.uniform(-cfg.init_joint_noise, cfg.init_joint_noise, (n, self.nj))
        self.qvel[ids, :] = 0.0
        self.qvel[ids, 6:] = self.ref_qvel[self.phase[ids]] * 0.5
        # The base starts moving with the reference (plus noise): a walk entered mid-stride at
        # standstill would spend its first steps catching up. Free-joint qvel is world-frame
        # linear, body-frame angular.
        self.qvel[ids, :6] = self.rng.uniform(-cfg.init_vel_noise, cfg.init_vel_noise, (n, 6))
        self.qvel[ids, 0:3] += self.ref_rvel[ph]
        self.qvel[ids, 3:6] += _rotate_world_to_local(self.ref_rquat[ph], self.ref_rangvel[ph])

        # domain randomisation, resampled every episode
        self.friction[ids[:, None], self.floor_geoms, 0] = self.rng.uniform(*cfg.friction_range, (n, 1))
        kp_s = self.rng.uniform(*cfg.kp_scale_range, (n, 1))
        kd_s = self.rng.uniform(*cfg.kd_scale_range, (n, 1))
        self.kp_s[ids], self.kd_s[ids] = kp_s[:, 0], kd_s[:, 0]
        self.gainprm[ids, :, 0] = self.gain0[:, 0] * kp_s
        self.biasprm[ids, :, 1] = self.bias0[:, 1] * kp_s
        self.biasprm[ids, :, 2] = self.bias0[:, 2] * kd_s
        if self.dr_mass:
            nb = self.mass0.shape[0]
            ms = self.rng.uniform(*cfg.mass_scale_range, (n, nb))
            ms[:, 0] = 1.0                                          # the world body
            self.body_mass[ids] = self.mass0 * ms
            self.body_inertia[ids] = self.inertia0 * ms[:, :, None]
            off = self.rng.uniform(-cfg.com_offset, cfg.com_offset, (n, len(self.com_ids), 3))
            ipos = np.broadcast_to(self.ipos0, (n,) + self.ipos0.shape).copy()
            ipos[:, self.com_ids] += off
            self.body_ipos[ids] = ipos
            self.mass_s[ids] = (self.mass0 * ms)[:, 1:].sum(1) / self.mass0[1:].sum()
            self.com_off[ids] = off
            self.backend.set_const(ids)
        self.enc_bias[ids] = self.rng.uniform(-cfg.encoder_offset, cfg.encoder_offset, (n, self.nj)) \
            if cfg.encoder_offset > 0 else 0.0

        self.backend.kinematics(ids)
        # Joint noise moves the feet, so settle each robot onto its own lowest sole point.
        # Only qpos is corrected: nothing reads the derived fields before the next step,
        # which recomputes them, so a second forward() would be pure cost (and on the warp
        # backend a full-batch one).
        if self.sole_radius is not None:
            low = (self.geom_xpos[ids][:, self.sole_geoms, 2] - self.sole_radius).min(axis=1)
            self.qpos[ids, 2] -= low - cfg.init_clearance

        self.prev_feet_tw[ids] = _twist_z(self.xquat[ids][:, self.key_ids[self.feet_k]])
        self.feet_rate[ids] = 0.0
        self.steps[ids] = 0
        hz = c.policy_hz
        lift = self.rng.random(n) < cfg.lift_prob
        self.lift_at[ids] = np.where(lift, (self.rng.uniform(*cfg.lift_start_s, n) * hz).astype(np.int64), -1)
        self.lift_end[ids] = self.lift_at[ids] + int(cfg.lift_rise_s * hz) + (self.rng.uniform(*cfg.lift_hold_s, n) * hz).astype(np.int64)
        self.lift_h[ids] = self.rng.uniform(*cfg.lift_height, n)
        self.lifted[ids] = False
        self.action[ids] = 0.0
        self.target_delta[ids] = 0.0
        self.prev_delta[ids] = 0.0
        # The command already in force is the pose the robot is in. It used to be the stand
        # pose, so an episode started deep in a squat had its first targets dragged toward
        # standing by the 0.2 rad/frame rate limit -- and, with an action delay, the motors
        # would be pulled there before the first new target arrived.
        self.prev_target[ids] = self.qpos[ids, 7:] + self.enc_bias[ids]   # as the robot reads it
        self.ctrl[ids] = self.qpos[ids, 7:]

        def grid(lo_hi):                 # drawn in ms (= 1 ms substeps), applied in substeps
            lo, hi = lo_hi
            ms = self.rng.choice(np.arange(lo, hi + 1, max(1, cfg.delay_grid)), n)
            return np.minimum(np.round(ms / self.ms_per_substep).astype(np.int64), self.dec)
        self.act_delay[ids] = grid(cfg.act_delay_steps)
        self.obs_delay[ids] = grid(cfg.obs_delay_steps)
        bias = np.zeros((n, 3))
        bias[:, :2] = self.rng.uniform(-cfg.imu_bias, cfg.imu_bias, (n, 2))
        self.imu_bias_q[ids] = _small_rot(bias)
        self.imu_bias[ids] = bias[:, :2]
        self.obs_qpos[ids] = self.qpos[ids]
        self.obs_qvel[ids] = self.qvel[ids]

    def _start_pose(self, ids, ref: np.ndarray) -> np.ndarray:
        """The joints an episode starts in: the reference's (subclasses may vary it)."""
        return ref

    # -- observation ---------------------------------------------------------

    def _state(self) -> dict:
        """What the robot's sensors would report: the delayed snapshot, plus noise."""
        cfg, rng, ph, n = self.cfg, self.rng, self.phase, self.num_envs
        u = lambda s, shape: rng.uniform(-s, s, shape) if s > 0 else 0.0
        quat = self.obs_qpos[:, 3:7]
        if cfg.noise_ori > 0 or cfg.imu_bias > 0:
            quat = _quat_mul(_quat_mul(quat, self.imu_bias_q), _small_rot(u(cfg.noise_ori, (n, 3)) * np.ones((n, 3))))
        return dict(
            base_quat=quat,
            base_ang_vel=self.obs_qvel[:, 3:6] + u(cfg.noise_ang_vel, (n, 3)),
            joint_pos=self.obs_qpos[:, 7:] + self.enc_bias + u(cfg.noise_joint_pos, (n, self.nj)),
            joint_vel=self.obs_qvel[:, 6:] + u(cfg.noise_joint_vel, (n, self.nj)),
            prev_action=self.action,
            motion_ref_pos=self.ref_qpos[ph],
            motion_ref_vel=self.ref_qvel[ph],
            motion_ref_quat=self.ref_rquat[ph],
        )

    def obs(self) -> np.ndarray:
        """(num_envs, obs_dim) float32 -- by Contract.build_obs, the same function the golden
        trace checks and the browser mirrors. Not a copy of it."""
        return self.c.build_obs(self._state())

    CRITIC_AHEAD = 10                   # critic_obs: how far ahead (policy steps) it sees the reference

    def critic_obs(self) -> np.ndarray:
        """(num_envs, critic_dim) float32: what the critic sees with EnvConfig.critic_obs.

        The policy's observation has no heading, no base velocity or height, no contacts and
        nothing of the episode's randomisation, so a value function fed only that cannot
        tell a robot about to fall (sliding sideways, turned 40 deg off) from a steady one,
        and the advantage of every action inherits that noise. This is the true state
        instead: the contract frame without noise or latency, then heading error, base
        velocity and height, key bodies against the reference, the feet, the randomised
        physics, lift and push timing, and the reference 10 steps ahead. Training only --
        the exported policy is the actor and still sees exactly the contract."""
        c, cfg, ph, n = self.c, self.cfg, self.phase, self.num_envs
        q, qd = self.qpos, self.qvel
        root_q = q[:, 3:7]
        clean = c.build_obs(dict(
            base_quat=root_q, base_ang_vel=qd[:, 3:6], joint_pos=q[:, 7:], joint_vel=qd[:, 6:],
            prev_action=self.action, motion_ref_pos=self.ref_qpos[ph], motion_ref_vel=self.ref_qvel[ph],
            motion_ref_quat=self.ref_rquat[ph]))
        e_yaw = _wrap(_twist_z(root_q) - _twist_z(self.ref_rquat[ph]))
        v_b = _rotate_world_to_local(root_q, qd[:, 0:3])
        v_ref = _rotate_world_to_local(self.ref_rquat[ph], self.ref_rvel[ph])
        root_p = self.xpos[:, self.root_id]
        yaw = _twist_z(root_q)
        e_body = ((self.xpos[:, self.key_ids] - root_p[:, None, :])
                  - (self.ref_body[ph] - self.ref_root[ph][:, None, :]))
        cy, sy = np.cos(yaw)[:, None], np.sin(yaw)[:, None]           # into the robot's heading frame
        e_body = np.stack([cy * e_body[..., 0] + sy * e_body[..., 1],
                           -sy * e_body[..., 0] + cy * e_body[..., 1], e_body[..., 2]], axis=-1)
        fz = self.xpos[:, self.key_ids[self.feet_k], 2]
        hz, dec = c.policy_hz, self.dec
        push_every = max(1, int(cfg.push_interval_s * hz))
        lift_planned = self.lift_at >= 0
        ahead = np.minimum(ph + self.CRITIC_AHEAD, self.T - 1)
        extra = np.concatenate([
            np.stack([np.sin(e_yaw), np.cos(e_yaw)], axis=1),
            v_b, v_ref,
            np.stack([q[:, 2], self.ref_root[ph, 2]], axis=1),
            e_body.reshape(n, -1),
            self._dr_critic(),
            fz - self.foot_down_z, (fz < self.foot_down_z).astype(float), self.feet_rate,
            np.stack([self.friction[:, self.floor_geoms[0], 0], self.kp_s, self.kd_s,
                      self.act_delay / dec, self.obs_delay / dec], axis=1),
            self.imu_bias,
            np.stack([self.lifted.astype(float),
                      np.where(lift_planned, np.clip((self.lift_at - self.steps) / hz, -5, 5), -5.0),
                      (self.steps % push_every) / push_every,
                      (self.max_steps - self.steps) / max(1, self.max_steps),
                      np.clip((self.T - 1 - ph) / hz, 0, 10)], axis=1),
            self.ref_qpos[ahead] - self.ref_qpos[ph],
            np.stack([self.ref_root[ahead, 2] - self.ref_root[ph, 2],
                      _wrap(_twist_z(self.ref_rquat[ahead]) - _twist_z(self.ref_rquat[ph]))], axis=1),
        ], axis=1)
        return np.concatenate([clean, extra], axis=1).astype(np.float32)

    def _dr_critic(self) -> np.ndarray:
        """The episode's inertial and encoder randomisation, for the critic. Empty when off,
        so a checkpoint trained without it keeps its critic width."""
        parts = []
        if self.dr_mass:
            parts += [self.mass_s[:, None] - 1.0, self.com_off.reshape(self.num_envs, -1)]
        if self.cfg.encoder_offset > 0:
            parts.append(self.enc_bias)
        return np.concatenate(parts, axis=1) if parts else np.zeros((self.num_envs, 0))

    # -- stepping ------------------------------------------------------------

    def step(self, action: np.ndarray):
        c, cfg = self.c, self.cfg
        self.action[:] = action
        target = c.action_to_target(action, self.prev_target, clamp=cfg.clamp_targets)
        # Taken BEFORE prev_target is overwritten. It used to be computed in _reward() as
        # ctrl - prev_target after both had been set to the same target, so the action-rate
        # penalty was identically zero and no smoothness pressure ever existed.
        self.prev_delta = self.target_delta
        self.target_delta = target - self.prev_target
        self.prev_target[:] = target
        self._advance(target)

        self.steps += 1
        if self.cyclic:
            self.phase = (self.phase + 1) % self.T
        else:
            self.phase = np.minimum(self.phase + 1, self.T - 1)

        if cfg.push_vel > 0:
            due = (self.steps % max(1, int(cfg.push_interval_s * c.policy_hz))) == 0
            if due.any():
                k = int(due.sum())
                self.qvel[due, :2] += self.rng.uniform(-cfg.push_vel, cfg.push_vel, (k, 2))
                if cfg.push_ang_vel > 0:
                    self.qvel[due, 3:6] += (self.rng.uniform(-cfg.push_ang_vel, cfg.push_ang_vel, (k, 3))
                                            * np.asarray(cfg.push_ang_axes))

        if cfg.lift_prob > 0:
            self._lift()

        reward, terms = self._reward()
        fell = self._fell()
        # Running off the end of a one-shot clip is a timeout, not a death: the task is
        # over, not failed, so the learner should still bootstrap from it.
        timeout = self.steps >= self.max_steps
        timeout |= (self.lift_at >= 0) & (self.steps >= self.lift_end)
        if not self.cyclic:
            timeout |= self.phase >= self.T - 1
        done = fell | timeout
        if self.cfg.adaptive_sampling:
            self.bin_fails *= self.cfg.sampling_decay
            if fell.any():
                b = np.minimum(self.phase[fell] // self.bin_len, self.n_bins - 1)
                for back, w in ((0, 1.0), (1, 0.5), (2, 0.25)):   # credit the run-up too
                    np.add.at(self.bin_fails, np.maximum(b - back, 0), w)
        self.reward_terms = terms
        return self.obs(), reward, done, fell, timeout

    def _lift(self) -> None:
        """Hold lifted envs' bases up in the air (see EnvConfig.lift_prob)."""
        on = (self.lift_at >= 0) & (self.steps >= self.lift_at)
        start = on & (self.steps == self.lift_at)
        if start.any():
            self.lift_pose[start] = self.qpos[start, 0:7]
        self.lifted = on
        if not on.any():
            return
        rise = max(1, int(self.cfg.lift_rise_s * self.c.policy_hz))
        a = np.clip((self.steps[on] - self.lift_at[on]) / rise, 0.0, 1.0)
        self.qpos[on, 0:2] = self.lift_pose[on, 0:2]
        self.qpos[on, 2] = self.lift_pose[on, 2] + a * self.lift_h[on]
        self.qpos[on, 3:7] = self.lift_pose[on, 3:7]
        self.qvel[on, 0:6] = 0.0

    def sampling_probs(self) -> np.ndarray:
        """Start-bin distribution: falls (decayed) mixed with uniform."""
        u = 1.0 / self.n_bins
        if self.bin_fails.sum() <= 0:
            return np.full(self.n_bins, u)
        f = self.bin_fails / self.bin_fails.sum()
        p = (1.0 - self.cfg.sampling_uniform) * f + self.cfg.sampling_uniform * u
        return p / p.sum()

    def _advance(self, target: np.ndarray) -> None:
        """One control period: each env's new target reaches the motors after its action
        delay, and the observation snapshot is taken obs_delay substeps before the end.
        The backend does the scheduling -- a few step() calls on mjbatch, one captured
        graph on warp."""
        dec = self.dec
        # target is in the robot's encoder reading; the joint goes to target - offset
        self.backend.step_scheduled(target - self.enc_bias, self.act_delay, dec - self.obs_delay, dec,
                                    self.obs_qpos, self.obs_qvel)

    # -- reward ---------------------------------------------------------------

    def _reward(self):
        w, c, ph = self.cfg.weights, self.c, self.phase
        q, qd = self.qpos[:, 7:], self.qvel[:, 6:]

        e_q = q - self.ref_qpos[ph]
        e_qd = qd - self.ref_qvel[ph]

        root_p, root_q = self.xpos[:, self.root_id], self.xquat[:, self.root_id]
        body_rel = self.xpos[:, self.key_ids] - root_p[:, None, :]
        ref_rel = self.ref_body[ph] - self.ref_root[ph][:, None, :]
        e_body = body_rel - ref_rel

        e_bori = 1.0 - np.abs(np.einsum("nkj,nkj->nk", self.xquat[:, self.key_ids], self.ref_bquat[ph]))
        gvec = quat_to_gravity(root_q)
        tilt = np.arccos(np.clip(-gvec[:, 2], -1.0, 1.0))
        e_h = root_p[:, 2] - self.ref_root[ph][:, 2]

        fz = self.xpos[:, self.key_ids[self.feet_k], 2]
        rz = self.ref_body[ph][:, self.feet_k, 2]
        e_lift = (fz - fz.min(1, keepdims=True)) - (rz - rz.min(1, keepdims=True))

        over = np.maximum(q - c.upper, 0.0) + np.maximum(c.lower - q, 0.0)
        raw = c.default_dof_pos + c.action_scale * np.clip(self.action, -c.action_clip, c.action_clip)
        raw_over = np.maximum(raw - c.upper, 0.0) + np.maximum(c.lower - raw, 0.0)

        ftw = _twist_z(self.xquat[:, self.key_ids[self.feet_k]])
        rate = _wrap(ftw - self.prev_feet_tw) * c.policy_hz
        self.feet_rate = rate
        self.prev_feet_tw = ftw
        down = fz < self.foot_down_z
        k = np.exp(-(rate / 0.5) ** 2)
        nd = down.sum(axis=1)
        feet_twist = np.where(nd > 0, np.sum(np.where(down, k, 0.0), axis=1) / np.maximum(nd, 1), 1.0)

        # Horizontal base velocity, each in its own heading frame, so a robot that has
        # drifted in yaw is still judged on walking forward rather than on compass direction.
        v_rob = _to_heading(self.qvel[:, 0:2], _yaw(root_q))
        v_ref = _to_heading(self.ref_rvel[ph, 0:2], _yaw(self.ref_rquat[ph]))
        yaw_rate = _rotate_local_to_world(root_q, self.qvel[:, 3:6])[:, 2]   # qvel[3:6] is body-frame
        e_v = np.sum((v_rob - v_ref) ** 2, axis=1)
        e_w = (yaw_rate - self.ref_rangvel[ph, 2]) ** 2

        t = {
            # exp(-6 e^2), was exp(-2 e^2): at 0.31 rad RMS (a dance policy skipping its leg
            # lifts) the old kernel still paid 0.83 against 0.95 for tracking well, so
            # skipping hard moves was nearly free. Now 0.56 against 0.85.
            "joint_pos":   np.exp(-6.0 * np.mean(e_q ** 2, axis=1)),
            "joint_vel":   np.exp(-0.1 * np.mean(e_qd ** 2, axis=1)),
            "body_pos":    np.exp(-40.0 * np.mean(np.sum(e_body ** 2, axis=2), axis=1)),
            "torso_ori":   np.exp(-30.0 * np.sum((quat_to_gravity(self.xquat[:, self.key_ids[self.torso_k]])
                                                  - quat_to_gravity(self.ref_bquat[ph, self.torso_k])) ** 2, axis=1)),
            "feet_pos":    np.exp(-100.0 * np.mean(np.sum(e_body[:, self.feet_k] ** 2, axis=2), axis=1)),
            # 1 cm off -> 0.78, 2.6 cm (a skipped toe tap) -> 0.18
            "feet_height": np.exp(-2500.0 * np.sum(e_lift ** 2, axis=1)),
            "body_ori":    np.exp(-10.0 * np.mean(e_bori ** 2, axis=1)),
            "root_ori":    np.exp(-5.0 * tilt ** 2),
            "root_height": np.exp(-40.0 * e_h ** 2),
            # Tight kernels on purpose: the walks here go ~0.3 m/s, and at exp(-4 e) standing
            # still earned 0.66 of this term -- too little reason to risk a step. At
            # exp(-16 e) standing earns ~0.2.
            "root_vel":    np.exp(-16.0 * e_v),
            "yaw_rate":    np.exp(-4.0 * e_w),
            "heading":     1.0 - np.abs(_wrap(_twist_z(root_q) - _twist_z(self.ref_rquat[ph]))) / np.pi,
            "feet_twist":  feet_twist,
            "alive":       np.ones(self.num_envs),
            "action_rate": np.sum(self.target_delta ** 2, axis=1),
            "action_accel": np.sum((self.target_delta - self.prev_delta) ** 2, axis=1),
            "self_contact": (np.sum(self.sensordata[:, self.touch_adr] > 1e-6, axis=1).astype(float)
                             if self.sensordata is not None else np.zeros(self.num_envs)),
            "torque":      np.mean(self.torque ** 2, axis=1),
            "joint_limit": np.sum(over, axis=1),
            "upper_pose":  np.exp(-np.mean(e_q[:, 12:] ** 2, axis=1) / 0.1 ** 2),
            "action_over": np.sum(raw_over, axis=1),
        }
        L = self.lifted
        t["lift_calm"] = np.where(L, np.exp(-0.1 * np.mean(qd[:, :12] ** 2, axis=1)), 0.0)
        if L.any():
            # in the air nothing about the base, the feet or the legs can be tracked; the arms can
            t["joint_pos"] = np.where(L, np.exp(-6.0 * np.mean(e_q[:, 12:] ** 2, axis=1)), t["joint_pos"])
            for k in ("joint_vel", "body_pos", "torso_ori", "feet_pos", "feet_height", "body_ori",
                      "root_ori", "root_height", "root_vel", "yaw_rate", "heading", "feet_twist"):
                t[k] = np.where(L, 0.0, t[k])
        reward = sum(getattr(w, k) * v for k, v in t.items())
        return reward, t

    def _fell(self) -> np.ndarray:
        cfg = self.cfg
        gvec = quat_to_gravity(self.qpos[:, 3:7])
        gref = quat_to_gravity(self.ref_rquat[self.phase])
        tilt = np.arccos(np.clip(np.sum(gvec * gref, axis=1), -1.0, 1.0))
        floor = np.minimum(cfg.min_root_height, self.ref_root[self.phase, 2] - cfg.root_height_drop)
        rms = np.sqrt(np.mean((self.qpos[:, 7:] - self.ref_qpos[self.phase]) ** 2, axis=1))
        return (((tilt > cfg.max_tilt_rad)
                 | (self.qpos[:, 2] < floor)
                 | (rms > cfg.max_joint_error)) & ~self.lifted) | ~np.isfinite(self.qpos).all(axis=1)

    def __repr__(self) -> str:
        return (f"MotionTrackingEnv({self.motion.name!r}, {self.num_envs} envs, "
                f"obs {self.obs_dim}, act {self.act_dim}, {self.max_steps} steps/episode)")
