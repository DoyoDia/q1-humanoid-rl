// The browser twin of q1_rl/contract.py. Same contract, same arithmetic.
//
// These two implementations exist because a policy has to behave identically in the
// training loop and in the sim2sim page, and "we were careful" is not a guarantee.
// robots/q1/contract/golden_trace.json is the guarantee: both replay it and must agree
// to 1e-6. If you change one of these files, change the other and re-run the check.

/** Gravity direction in the base frame. quat is (w, x, y, z), world orientation of the base.
 *  Returns the world -Z axis rotated into the base frame: level+upright -> [0, 0, -1]. */
export function quatToGravity(q) {
  const [w, x, y, z] = q;
  return [
    -2.0 * (x * z - w * y),
    -2.0 * (y * z + w * x),
    -(1.0 - 2.0 * (x * x + y * y)),
  ];
}

/** (roll, pitch) of the base, Z-Y-X Euler angles -- the Walk runtime's imu_euler. */
export function quatToRollPitch(q) {
  const [w, x, y, z] = q;
  return [Math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
          Math.asin(Math.min(1.0, Math.max(-1.0, 2.0 * (w * y - z * x))))];
}

export class Contract {
  /** @param bundle parsed robots/q1/contract/contract.json
   *  @param name   which contract, e.g. "mimic_opensource_v2" */
  constructor(bundle, name = 'mimic_opensource_v2') {
    const c = bundle.contracts[name];
    if (!c) throw new Error(`no such contract ${name}; have ${Object.keys(bundle.contracts)}`);
    this.name = name;
    this.sourceHash = bundle._source_hash;
    this.jointNames = bundle.joint_names;
    this.nJoints = bundle.n_joints;
    this.lower = bundle.lower;
    this.upper = bundle.upper;
    this.maxStepRad = bundle.command.max_step_rad;
    this.policyHz = bundle.rates.policy_hz;
    this.decimation = bundle.rates.sim.decimation;
    this.simTimestep = bundle.rates.sim.timestep;

    this.obsDim = c.obs_dim;
    this.onnxObsDim = c.onnx_obs_dim;
    this.frameStack = c.frame_stack;
    this.fields = c.fields;
    this.scales = c.obs_scales;
    this.obsClip = c.obs_clip;
    this.actionClip = c.action_clip;
    this.actionScale = c.action_scale;
    this.kp = c.kps;
    this.kd = c.kds;
    this.defaultDofPos = c.default_dof_pos;
    this.policyActionDim = c.policy_action_dim ?? this.nJoints;
  }

  /** state: {base_quat, base_ang_vel, joint_pos, joint_vel, prev_action,
   *          motion_ref_pos?, motion_ref_vel?, motion_ref_quat?} -- all arrays, joint fields
   *          in contract order; motion_ref_quat is the reference pelvis orientation (w x y z).
   *  Returns Float32Array of width obsDim. */
  buildObs(state) {
    const obs = new Float64Array(this.obsDim);
    const s = this.scales;
    for (const f of this.fields) {
      let v;
      switch (f.name) {
        case 'gvec':
          v = quatToGravity(state.base_quat);
          break;
        case 'ref_gravity':
          v = quatToGravity(state.motion_ref_quat);
          break;
        case 'base_ang_vel':
        case 'imu_ang_vel': {
          const k = s.ang_vel ?? 1.0;
          v = Array.from(state.base_ang_vel, (x) => x * k);
          break;
        }
        case 'imu_euler':
          v = quatToRollPitch(state.base_quat);
          break;
        case 'command': {
          // the Walk runtime scales the command like legged_gym (contract.py)
          const k = 'lin_vel' in s ? [s.lin_vel, s.lin_vel, s.ang_vel ?? 1.0] : [1, 1, 1];
          v = Array.from(state.command, (x, i) => x * k[i]);
          break;
        }
        case 'joint_pos_err': {
          const k = s.dof_pos ?? 1.0;
          v = Array.from(state.joint_pos, (x, i) => (x - this.defaultDofPos[i]) * k);
          break;
        }
        case 'joint_pos': {
          const k = s.dof_pos ?? 1.0;
          v = Array.from(state.joint_pos, (x) => x * k);
          break;
        }
        case 'joint_vel': {
          const k = s.dof_vel ?? 1.0;
          v = Array.from(state.joint_vel, (x) => x * k);
          break;
        }
        case 'prev_action': {
          const k = s.actions ?? 1.0;
          v = Array.from({ length: f.size }, (_, i) => state.prev_action[i] * k);
          break;
        }
        default:
          if (!(f.name in state)) {
            throw new Error(`obs field ${f.name} has no value in state and no rule in buildObs()`);
          }
          v = state[f.name];
      }
      if (v.length !== f.size) {
        throw new Error(`obs field ${f.name}: expected ${f.size} values, got ${v.length}`);
      }
      for (let i = 0; i < f.size; i++) {
        obs[f.start + i] = Math.min(this.obsClip, Math.max(-this.obsClip, v[i]));
      }
    }
    return Float32Array.from(obs);
  }

  /** Zero-pad obsDim -> the width the exported ONNX takes. */
  padForOnnx(obs) {
    if (this.onnxObsDim === this.obsDim) return obs;
    const out = new Float32Array(this.onnxObsDim);
    out.set(obs);
    return out;
  }

  /** action -> joint position target, with the joint limits and the firmware's
   *  0.2 rad/frame step limit applied. Pass prevTarget to enable the rate limit. */
  actionToTarget(action, prevTarget = null, clamp = true) {
    const out = new Float64Array(this.nJoints);
    for (let i = 0; i < this.nJoints; i++) {
      const a = Math.min(this.actionClip, Math.max(-this.actionClip, action[i]));
      let t = this.defaultDofPos[i] + this.actionScale * a;
      if (clamp) t = Math.min(this.upper[i], Math.max(this.lower[i], t));
      if (prevTarget) {
        const d = Math.min(this.maxStepRad, Math.max(-this.maxStepRad, t - prevTarget[i]));
        t = prevTarget[i] + d;
      }
      out[i] = t;
    }
    return out;
  }
}

/** Replay robots/q1/contract/golden_trace.json. Returns {ok, steps, maxObsDiff, maxTargetDiff}. */
export function checkGoldenTrace(bundle, trace) {
  const c = new Contract(bundle, trace.contract);
  if (c.obsDim !== trace.obs_dim) {
    throw new Error(`contract.json says obs ${c.obsDim}, trace says ${trace.obs_dim}`);
  }
  let maxObs = 0, maxTgt = 0, worstObs = null;
  let prevTarget = null;
  for (const r of trace.records) {
    const obs = c.buildObs(r.state);
    for (let i = 0; i < obs.length; i++) {
      const d = Math.abs(obs[i] - r.obs[i]);
      if (d > maxObs) { maxObs = d; worstObs = { step: r.step, index: i, js: obs[i], py: r.obs[i] }; }
    }
    const tgt = c.actionToTarget(r.action, prevTarget);
    for (let i = 0; i < tgt.length; i++) maxTgt = Math.max(maxTgt, Math.abs(tgt[i] - r.target[i]));
    prevTarget = r.target;   // follow Python's state, so one bad step cannot mask the next
  }
  return {
    ok: maxObs <= trace.tolerance && maxTgt <= trace.tolerance,
    contract: trace.contract,
    steps: trace.records.length,
    tolerance: trace.tolerance,
    maxObsDiff: maxObs,
    maxTargetDiff: maxTgt,
    worstObs,
  };
}
