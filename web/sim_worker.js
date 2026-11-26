// The live simulation of web/sim2sim_view.html, off the main thread: MuJoCo WASM, the
// policies in onnxruntime-web, sensor noise and latency, the hands (lift, mouse spring),
// clips, walking on commands, fall detection. It keeps its own clock; the page only draws.
//
// Every 20 ms: the observation (web/contract.js, the same builder the golden trace checks
// against Python) from the robot's state, the active policy's ONNX, its action to joint
// targets, 20 x 1 ms MuJoCo steps with the PD inside the model's position actuators.
// With noise on, the observation carries the training env's sensor noise and the targets
// arrive late, as on the robot (q1_rl/envs/motion_tracking.py EnvConfig).
//
// Policies (web/run/index.json, tools/export_web_bundle.py):
//   idle  -- holds the default stance against pushes and lifts (train.py --motion stand)
//   walk  -- walks on (vx, vy, yaw rate) commands, WASD: train.py --task velocity (our layout),
//            or --task walk (the vendor's Walk contract: a 5-frame history in, 13 joints out,
//            its own gains -- the robot's runtime can run it)
//   clip  -- tracks a reference clip
// A fall switches to damping, as the robot's own protection does, until reset.
//
// Messages in:  {t: 'init'|'scene'|'spawn'|'reset'|'clip'|'idle'|'push'|'lift'|'ball'|'noise'|
//                'pause'|'speed'|'seek'|'drag'|'dragTo'|'dragEnd'|'cmd', ...}
// Messages out: {t: 'ready'|'scene'|'state'|'status'|'toast'|'ack'|'error', ...}
import { SCENES, sceneXml } from './scenes.js';

const NOISE = { ang_vel: 0.2, ori: 0.02, imu_bias: 0.03, joint_pos: 0.01, joint_vel: 1.0, act_delay: [0, 10], obs_delay: [0, 2], grid: 2 };
const LIFT_H = 0.25, LIFT_RISE = 0.4;            // m, s
// Walking (q1_rl/envs/velocity.py VelocityConfig): the gait clock's period, the deadband the
// policy was trained with, and how fast a key press ramps the command (per second).
const GAIT_PERIOD = 0.6, DEADBAND = 0.05, CMD_RATE = [1.0, 1.0, 2.0], WALK_SETTLE_S = 1.0;
// Walk contract (q1_rl/envs/walk.py WalkConfig): its deadband (the runtime's), the arms' gains
// in simulation (the runtime holds them with kp 5 plus gravity compensation) and their swing.
const WALK_DEADBAND = 0.1, ARM_GAINS = [20.0, 1.5], ARM_SWING = 0.15;
const uni = (a) => (Math.random() * 2 - 1) * a;
const post = (m, transfer) => postMessage(m, transfer || []);

let mj, ort, Contract, quatToGravity, get;
let index, idleEntry, walkEntry, clipEntries, xmlBase;
const bundles = {};
let W = null;
let HZ = 50, DEC = 20, nj = 22, c = null;   // c: a contract, for the shared limits and the default pose
let idle = null, walk = null;

async function bundle(run) {                      // lazily: the clip files are megabytes
  if (!bundles[run]) bundles[run] = (async () => {
    const base = `./run/${run}/`;
    const [meta, motion, cj] = await Promise.all([
      get(base + 'meta.json').then(r => r.json()),
      get(base + 'motion.json').then(r => r.json()),
      get(base + 'contract.json').then(r => r.json())]);
    const sess = await ort.InferenceSession.create(new Uint8Array(await get(base + 'policy.onnx').then(r => r.arrayBuffer())), { executionProviders: ['wasm'] });
    const contract = new Contract(cj, meta.contract || 'mimic_opensource_v2');
    const tensor = new ort.Tensor('float32', new Float32Array(contract.onnxObsDim), [1, contract.onnxObsDim]);
    const b = { run, base, meta, motion, contract, sess, tensor, velocity: contract.name === 'velocity' || contract.frameStack > 1 };
    if (contract.frameStack > 1) {                  // the Walk runtime's two inputs, the history oldest first
      const fd = contract.obsDim, k = contract.frameStack;
      Object.assign(b, { stacked: true, fresh: true, cur: new ort.Tensor('float32', new Float32Array(fd), [1, fd]),
                         hist: new ort.Tensor('float32', new Float32Array(k * fd), [1, k * fd]) });
    }
    return b;
  })();
  return bundles[run];
}

// -- simulation state ---------------------------------------------------------------------
// mode: 'idle' (idle policy, or PD hold without one), 'walk', 'clip', 'fallen' (damping)
let mode = 'idle', clip = null, frame = 0, ground = 0, spawnYaw = 0, refYaw0 = 0, simTime = 0;
let prevAction, prevTarget, errSum = 0, errN = 0;
let paused = false, speed = 1, noisy = true;
let delays = { act: 0, obs: 0 }, imuBias = [0, 0];
let lift = null, drag = null, snap = null;
// cmdWant: the keys; ramp: moving toward it at CMD_RATE; cmd: what the policy sees (deadband applied)
let cmdWant = [0, 0, 0], ramp = [0, 0, 0], cmd = [0, 0, 0], clock = 0, stillFor = 0;
let velGot = [0, 0];                               // the base's velocity in its heading frame, smoothed
const twist = (q) => 2 * Math.atan2(q[3], q[0]);
const robotQuat = () => [W.data.qpos[3], W.data.qpos[4], W.data.qpos[5], W.data.qpos[6]];
function quatMul(a, b) {
  return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
}
const yawQuat = (y) => [Math.cos(y / 2), 0, 0, Math.sin(y / 2)];
function smallRot(rx, ry, rz) {                   // small rotation vector -> quaternion
  const a = Math.hypot(rx, ry, rz); if (a < 1e-12) return [1, 0, 0, 0];
  const s = Math.sin(a / 2) / a; return [Math.cos(a / 2), rx * s, ry * s, rz * s];
}
const status = (text, cls = '') => post({ t: 'status', text, cls });
const toast = (text) => post({ t: 'toast', text });

function sampleRealism() {
  const g = NOISE.grid, pick = ([a, b]) => a + g * Math.floor(Math.random() * ((b - a) / g + 1));
  delays = noisy ? { act: pick(NOISE.act_delay), obs: pick(NOISE.obs_delay) } : { act: 0, obs: 0 };
  imuBias = noisy ? [uni(NOISE.imu_bias), uni(NOISE.imu_bias)] : [0, 0];
}

function buildWorld(scene) {
  const model = mj.from_xml_string(sceneXml(xmlBase, scene));
  const data = new mj.MjData(model);
  const balls = [0, 1, 2].map((i) => {
    const b = model.body(`ball_${i}`).id, j = model.body_jntadr[b];
    return { body: b, q: model.jnt_qposadr[j], v: model.jnt_dofadr[j] };
  });
  const feet = ['left_ankle_roll_link', 'right_ankle_roll_link'].map((n) => model.body(n).id);
  return { scene, model, data, balls, feet, key: model.key('stand').id };
}

/** The robot at (x, y) on ground height z, facing yaw: the stand pose, or clip frame f's pose. */
function placeRobot(x, y, z, yaw, f = null, resetObjects = false) {
  const { model, data } = W;
  // The stand keyframe covers the robot only: the scene's loose bodies either keep their
  // state or go back to where the scene put them (qpos0) -- never to the keyframe's zeros.
  const NQ = 7 + nj, NV = 6 + nj;
  const keepQ = resetObjects ? Array.from(model.qpos0).slice(NQ) : Array.from(data.qpos).slice(NQ);
  const keepV = resetObjects ? null : Array.from(data.qvel).slice(NV);
  mj.mj_resetDataKeyframe(model, data, W.key);
  const q = data.qpos;
  keepQ.forEach((x, i) => q[NQ + i] = x);
  if (f !== null) {
    const m = clip.motion, rq = m.root_quat[f];
    const qq = quatMul(yawQuat(yaw - twist(rq)), rq);
    for (let i = 0; i < 4; i++) q[3 + i] = qq[i];
    for (let i = 0; i < nj; i++) q[7 + i] = m.joint_pos[f][i];
    q[2] = m.root_pos[f][2] + z + 0.003;
  } else {
    const qq = yawQuat(yaw);
    for (let i = 0; i < 4; i++) q[3 + i] = qq[i];
    q[2] += z + 0.003;
  }
  q[0] = x; q[1] = y;
  for (let i = 0; i < data.qvel.length; i++) data.qvel[i] = i >= NV && keepV ? keepV[i - NV] : 0;
  for (let i = 0; i < 6 * model.nbody; i++) data.xfrc_applied[i] = 0;
  mj.mj_forward(model, data);
  for (let i = 0; i < nj; i++) { prevTarget[i] = q[7 + i]; data.ctrl[i] = q[7 + i]; }
  prevAction = new Float64Array(nj);
  errSum = 0; errN = 0; lift = null; drag = null;
  sampleRealism();
  snapshot();
}
function spawn(i) {                               // the robot at the chosen spot, the scene as built
  const s = W.scene.spawns[i] || W.scene.spawns[0];
  W.spawn = W.scene.spawns.indexOf(s);
  ground = s.z; spawnYaw = s.yaw;
  placeRobot(s.x, s.y, s.z, s.yaw, null, true);
}

// the observation's raw state, taken obs-delay substeps before the end of a control period
function snapshot() {
  const q = W.data.qpos, v = W.data.qvel;
  snap = { quat: [q[3], q[4], q[5], q[6]], w: [v[3], v[4], v[5]],
           qj: Float64Array.from({ length: nj }, (_, i) => q[7 + i]), vj: Float64Array.from({ length: nj }, (_, i) => v[6 + i]) };
}
function observedState() {
  const s = snap;
  if (!noisy) return s;
  const quat = quatMul(s.quat, quatMul(smallRot(imuBias[0], imuBias[1], 0), smallRot(uni(NOISE.ori), uni(NOISE.ori), uni(NOISE.ori))));
  return { quat, w: s.w.map((x) => x + uni(NOISE.ang_vel)),
           qj: s.qj.map((x) => x + uni(NOISE.joint_pos)), vj: s.vj.map((x) => x + uni(NOISE.joint_vel)) };
}

// the active policy, and what it tracks this step
function active() {
  if (mode === 'fallen') return null;
  if (mode === 'clip') {
    const m = clip.motion;
    return { b: clip, pos: m.joint_pos[frame], vel: m.joint_vel[frame], quat: m.root_quat[frame] };
  }
  if (mode === 'walk') return { b: walk, quat: [1, 0, 0, 0] };
  if (!idle) return null;
  const m = idle.motion;
  return { b: idle, pos: m.joint_pos[0], vel: m.joint_vel[0].map(() => 0), quat: m.root_quat[0] };
}
const moving = () => cmd.some((x) => x !== 0);

// The robot's gait clock (q1_rl/envs/walk.py RuntimeCPG): two coupled Hopf
// oscillators, period and duty from the command, frozen at (0,1,0,1) while standing. Used when
// the walking bundle was trained on it (meta.walk.cpg === 'runtime').
const CPG_STAND = [0, 1, 0, 1];
let cpgX = CPG_STAND.slice(), cpgStanding = true, cpgSkip = 0;
const runtimeCpg = () => !!(walk && walk.meta.walk && walk.meta.walk.cpg === 'runtime');
function cpgLaw(c) {
  const s = 0.7 * Math.hypot(c[0], 2 * c[1]), e = s - 0.7;
  if (e > 0.8) return [0.4342848, 0.257144];
  if (s < 0.7) return [0.8, 0.6];
  return [0.8 * (1 - e * 0.57143), Math.min(0.6, Math.max(0.25, 0.6 - e * 0.42857))];
}
function cpgReset() { cpgX = CPG_STAND.slice(); cpgStanding = true; }
function cpgTick(c, roll, pitch) {
  const n = Math.hypot(c[0], c[1], c[2]), dt = 1 / HZ;
  const standOk = Math.abs(roll) < 0.06 && pitch > -0.05 && pitch < 0.125;
  const stopOk = Math.abs(roll) < 0.04 && pitch > -0.03 && pitch < 0.100;
  if (cpgStanding) {
    let lead = null;
    if (n >= 0.1) lead = Math.abs(c[1]) > 0.01 ? c[1] > 0.01 : Math.abs(c[2]) > 0.01 ? c[2] > 0.01 : c[0] > 0.01;
    else if (!standOk) lead = roll < 0;
    if (lead !== null) { cpgX = lead ? [1, 0, -1, 0] : [-1, 0, 1, 0]; cpgStanding = false; cpgSkip = Math.floor(Math.random() * 4); }
    return;
  }
  const [P, beta] = cpgLaw(c), x = cpgX, nx = x.slice();
  for (let i = 0; i < 2; i++) {
    const xi = x[2 * i], yi = x[2 * i + 1], xo = x[2 - 2 * i], yo = x[3 - 2 * i], d = 1 - (xi * xi + yi * yi);
    const sg = 1 / (1 + Math.exp(-50 * yi));
    const w = (Math.PI / P) * (sg / beta + (1 - sg) / (1 - beta));
    nx[2 * i] = xi + dt * (50 * d * xi + w * yi + xi - xo);
    nx[2 * i + 1] = yi + dt * (-w * xi + 50 * d * yi + yi - yo);
  }
  const crossed = (x[1] > 0 && nx[1] <= 0) || (x[3] > 0 && nx[3] <= 0);
  if (n < 0.1 && stopOk && crossed) {
    if (cpgSkip > 0) cpgSkip--;
    else { cpgX = CPG_STAND.slice(); cpgStanding = true; return; }
  }
  cpgX = nx;
}

// The actuators' PD gains follow the active policy's contract (the model is built with
// MIMIC_OPENSOURCE_V2's); the Walk contract's arms get what training gives them.
let gainsOf = null;
function setGains(contract) {
  if (gainsOf && gainsOf.name === contract.name) return;
  gainsOf = contract;
  const m = W.model, gp = m.actuator_gainprm, bp = m.actuator_biasprm, G = gp.length / m.nu, B = bp.length / m.nu;
  for (let i = 0; i < nj; i++) {
    const arm = contract.frameStack > 1 && /shoulder|elbow/.test(contract.jointNames[i]);
    const kp = arm ? ARM_GAINS[0] : contract.kp[i], kd = arm ? ARM_GAINS[1] : contract.kd[i];
    gp[G * i] = kp; bp[B * i + 1] = -kp; bp[B * i + 2] = -kd;
  }
}
async function runStacked(b, st) {
  let a = 2 * Math.PI * clock, on = moving() ? 1 : 0;
  if (runtimeCpg()) {
    const [P, beta] = cpgLaw(cmd);
    a = Math.atan2(cpgX[0], cpgX[1]); on = cpgStanding ? 0 : 1;
    Object.assign(st, { command: cmd, pattern: [P, beta - 0.6], cpg: cpgX.slice() });
  } else {
    const s = Math.sin(a) * on, co = Math.cos(a) * on;
    Object.assign(st, { command: cmd, pattern: [1, 0], cpg: [s, co, -s, -co] });
  }
  const f = b.contract.buildObs(st), fd = f.length, h = b.hist.data;
  if (b.fresh) { for (let i = 0; i < b.contract.frameStack; i++) h.set(f, i * fd); b.fresh = false; }
  else { h.copyWithin(0, fd); h.set(f, h.length - fd); }
  b.cur.data.set(f);
  const out = (await b.sess.run({ observations: b.cur, observation_history: b.hist })).action.data;
  const action = Float64Array.from(out);
  // the runtime's: zeros past the policy's 13, the shoulders swinging against the legs
  const names = b.contract.jointNames, sw = ARM_SWING * Math.sin(a) * on / b.contract.actionScale;
  action[names.indexOf('left_shoulder_pitch_joint')] = sw;
  action[names.indexOf('right_shoulder_pitch_joint')] = -sw;
  return action;
}

async function controlStep() {
  const { model, data } = W, q = data.qpos;
  let target;
  const pol = active();
  if (mode === 'walk') rampCommand();
  // a policy of another contract does not read this one's actions as its previous action
  if (pol && gainsOf && pol.b.contract.name !== gainsOf.name) prevAction = new Float64Array(nj);
  setGains(pol ? pol.b.contract : c);
  setCtrlLimited(!(pol && pol.b.stacked && runtimeCpg()));
  if (pol && pol.b.stacked) {
    const s = observedState(), b = pol.b;
    const action = await runStacked(b, { base_quat: s.quat, base_ang_vel: s.w, joint_pos: s.qj, joint_vel: s.vj, prev_action: prevAction });
    target = b.contract.actionToTarget(action, prevTarget, !runtimeCpg());   // the Walk runtime clamps torque, not position
    prevAction = action;
  } else if (pol) {
    const s = observedState(), b = pol.b;
    const st = { base_quat: s.quat, base_ang_vel: s.w, joint_pos: s.qj, joint_vel: s.vj, prev_action: prevAction };
    if (b.velocity) {
      const a = 2 * Math.PI * clock, on = moving() ? 1 : 0;
      Object.assign(st, { command: cmd, gait: [Math.sin(a) * on, Math.cos(a) * on] });
    } else Object.assign(st, { motion_ref_pos: pol.pos, motion_ref_vel: pol.vel, motion_ref_quat: pol.quat });
    b.tensor.data.set(b.contract.padForOnnx(b.contract.buildObs(st)));
    const action = (await b.sess.run({ obs: b.tensor })).action.data;
    target = b.contract.actionToTarget(action, prevTarget);
    prevAction = Float64Array.from(action);
  } else {
    // no policy: the stance held by PD (no idle bundle), or damping (fallen: targets follow the joints)
    target = Float64Array.from({ length: nj }, (_, i) => mode === 'fallen' ? q[7 + i] : c.defaultDofPos[i]);
    prevAction = new Float64Array(nj);
  }
  for (let k = 0; k < DEC; k++) {
    const tgt = k < delays.act ? prevTarget : target;
    for (let i = 0; i < nj; i++) data.ctrl[i] = mode === 'fallen' ? q[7 + i] : tgt[i];
    applyHands(k / DEC);
    mj.mj_step(model, data);
    if (k === DEC - 1 - delays.obs) snapshot();
  }
  prevTarget = target;
  simTime += 1 / HZ;
  {
    const q = data.qpos, v = data.qvel, yaw = Math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
    const vx = Math.cos(yaw) * v[0] + Math.sin(yaw) * v[1], vy = -Math.sin(yaw) * v[0] + Math.cos(yaw) * v[1];
    const k = 1 / HZ / 0.5;                         // 0.5 s low-pass: one gait cycle's sway averaged out
    velGot = [velGot[0] + k * (vx - velGot[0]), velGot[1] + k * (vy - velGot[1])];
  }
  if (mode === 'walk') {
    clock = (clock + 1 / (gaitPeriod() * HZ)) % 1;
    if (runtimeCpg()) {
      const q = data.qpos, w = q[3], x = q[4], y = q[5], z = q[6];
      cpgTick(cmd, Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x)))));
    }
    // keys released and the robot has come to a stand: hand it back to the idle policy
    const still = !moving() && !cmdWant.some((x) => x) && Math.hypot(data.qvel[0], data.qvel[1]) < 0.1
      && (!runtimeCpg() || cpgStanding);
    stillFor = still ? stillFor + 1 / HZ : 0;
    if (stillFor >= WALK_SETTLE_S && idle) mode = 'idle';
  }
  if (mode === 'clip') {
    let e = 0;
    for (let i = 0; i < nj; i++) { const d = q[7 + i] - clip.motion.joint_pos[frame][i]; e += d * d; }
    errSum += Math.sqrt(e / nj); errN++;
    frame++;
    if (frame >= clip.motion.n_frames) {
      status(`${clip.label}：做完了，没摔（关节误差 ${(errSum / errN).toFixed(3)} rad）`, 'ok');
      mode = 'idle'; clip = null;
    }
  }
  if (mode !== 'fallen' && !lift && fellOver()) {
    const where = clip ? `（${clip.label} ${fmt(frame / HZ)}）` : mode === 'walk' ? '（走路时）' : '';
    mode = 'fallen'; drag = null; stopWalk();
    status(`摔倒了${where} → 阻尼保护，按 R 复位`, 'bad');
  }
}
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
function fellOver() {
  const q = W.data.qpos, xp = W.data.xpos, pol = active();
  const g = quatToGravity(robotQuat()), gr = pol && mode === 'clip' ? quatToGravity(pol.quat) : [0, 0, -1];
  const tilt = Math.acos(Math.max(-1, Math.min(1, g[0] * gr[0] + g[1] * gr[1] + g[2] * gr[2]))) * 180 / Math.PI;
  const footZ = Math.min(xp[3 * W.feet[0] + 2], xp[3 * W.feet[1] + 2]);
  return tilt > 60 || (tilt > 35 && q[2] - footZ < 0.2);
}
// The CPG period: the walking policy's own law when its bundle has one (q1_rl/envs/walk.py
// cycle_stride_m: the middle of the range it was trained on at this speed), GAIT_PERIOD otherwise.
// MuJoCo clips ctrl to ctrlrange (the joint limits here). The runtime-trained walk sends its targets
// unclipped (q1_rl/envs/motion_tracking.py clamp_targets): widen the ranges while it drives.
// (actuator_ctrllimited is a bool array the wasm binding cannot hand out.)
let ctrlRange0 = null, ctrlWide = false;
function setCtrlLimited(on) {
  if (ctrlWide === !on) return;
  const cr = W.model.actuator_ctrlrange;
  if (!ctrlRange0) ctrlRange0 = Float64Array.from(cr);
  for (let i = 0; i < cr.length; i++) cr[i] = on ? ctrlRange0[i] : (i % 2 ? 1e3 : -1e3);
  ctrlWide = !on;
}
function gaitPeriod() {
  const g = walk && walk.meta.walk;
  if (g && g.cpg === 'runtime') return cpgLaw(cmd)[0];
  if (!g || !g.cycle) return GAIT_PERIOD;
  let [lo, hi] = g.cycle;
  if (g.stride > 0) { hi = Math.min(hi, g.stride / Math.max(Math.hypot(cmd[0], cmd[1]), 1e-6)); lo = Math.min(lo, 0.8 * hi); }
  return (lo + hi) / 2;
}
function rampCommand() {                          // the keys' command, reached at CMD_RATE per second
  for (let i = 0; i < 3; i++) {
    const m = CMD_RATE[i] / HZ;
    ramp[i] += Math.max(-m, Math.min(m, cmdWant[i] - ramp[i]));
  }
  // the policy never saw a command component under the deadband (q1_rl/envs/velocity.py)
  const db = walk && walk.stacked ? WALK_DEADBAND : DEADBAND;
  cmd = ramp.map((x) => Math.abs(x) < db ? 0 : x);
}
function stopWalk() { cmdWant = [0, 0, 0]; ramp = [0, 0, 0]; cmd = [0, 0, 0]; }

// hands on the robot: the kinematic lift (training holds a lifted base the same way) and the mouse spring
function applyHands(phase) {
  const { model, data } = W, q = data.qpos, v = data.qvel;
  for (let i = 0; i < 6 * model.nbody; i++) data.xfrc_applied[i] = 0;
  if (lift) {
    const now = simTime + phase / HZ;
    let a = lift.down === null ? Math.min(1, (now - lift.t) / LIFT_RISE) : Math.max(0, 1 - (now - lift.down) / LIFT_RISE);
    a = a * a * (3 - 2 * a);
    q[0] = lift.xy[0]; q[1] = lift.xy[1]; q[2] = lift.z0 + LIFT_H * a;
    for (let i = 0; i < 4; i++) q[3 + i] = lift.quat[i];
    for (let i = 0; i < 6; i++) v[i] = 0;
    if (lift.down !== null && a <= 0) lift = null;
  }
  if (drag) {
    const b = drag.body, R = data.xmat, xp = data.xpos, xi = data.xipos, L = drag.local;
    const p = [0, 1, 2].map((r) => xp[3 * b + r] + R[9 * b + 3 * r] * L[0] + R[9 * b + 3 * r + 1] * L[1] + R[9 * b + 3 * r + 2] * L[2]);
    const dt = 1 / (HZ * DEC);
    const vel = drag.prev ? p.map((x, i) => (x - drag.prev[i]) / dt) : [0, 0, 0];
    drag.prev = p; drag.p = p;
    // stiffness as for at least 3 kg, so light things move briskly; damped near critical
    const M = drag.mass, Ms = Math.max(3, M), K = 20 * Ms, D = 2 * 0.6 * Math.sqrt(K * M);
    const F = p.map((x, i) => K * (drag.target[i] - x) - D * vel[i]);
    const n = Math.hypot(...F), cap = 15 * Ms + 9.81 * M;
    if (n > cap) for (let i = 0; i < 3; i++) F[i] *= cap / n;
    const r = [p[0] - xi[3 * b], p[1] - xi[3 * b + 1], p[2] - xi[3 * b + 2]];
    const tq = [r[1] * F[2] - r[2] * F[1], r[2] * F[0] - r[0] * F[2], r[0] * F[1] - r[1] * F[0]];
    for (let i = 0; i < 3; i++) { data.xfrc_applied[6 * b + i] = F[i]; data.xfrc_applied[6 * b + 3 + i] = tq[i]; }
    drag.force = Math.min(n, cap); drag.capped = n > cap;     // for the label by the cursor
  }
}

// -- actions ----------------------------------------------------------------------------------
function startClip(b, label, t0) {
  const wasFallen = mode === 'fallen';
  clip = { ...b, label };
  frame = Math.min(clip.motion.n_frames - 1, Math.round(t0 * HZ));
  // from standing the clip just starts (its first frame is a stance); after a fall, a walk or
  // for a later start, the robot is put in that frame's pose where it is
  if (wasFallen || frame > 0) placeRobot(W.data.qpos[0], W.data.qpos[1], ground, twist(robotQuat()), frame);
  mode = 'clip'; errSum = 0; errN = 0; stopWalk();
  refYaw0 = twist(clip.motion.root_quat[frame]);
  spawnYaw = twist(robotQuat());
  status('', '');
}
function push(dir, s) {
  if (mode === 'fallen' || lift) return;
  const a = dir ?? Math.random() * 2 * Math.PI;
  W.data.qvel[0] += s * Math.cos(a); W.data.qvel[1] += s * Math.sin(a);
  toast(`推了一下：${s.toFixed(1)} m/s`);
}
function toggleLift() {
  if (mode === 'fallen') return;
  const q = W.data.qpos;
  if (!lift) {
    // only the idle and clip policies were trained being picked up
    if (mode === 'walk') { mode = 'idle'; stopWalk(); }
    lift = { t: simTime, z0: q[2], xy: [q[0], q[1]], quat: [q[3], q[4], q[5], q[6]], down: null };
  } else if (lift.down === null) lift.down = simTime;
}
let nextBall = 0;
function throwBall(cam) {
  const { data } = W, ball = W.balls[nextBall++ % 3];
  const to = [data.qpos[0], data.qpos[1], data.qpos[2] + 0.05];
  const dir = to.map((x, i) => x - cam[i]), len = Math.hypot(...dir);
  const start = cam.map((x, i) => x + dir[i] / len * Math.min(1.0, len * 0.3));
  start[2] = Math.max(0.15 + ground, start[2]);
  const T = Math.max(0.25, Math.hypot(...to.map((x, i) => x - start[i])) / 7);
  data.qpos[ball.q] = start[0]; data.qpos[ball.q + 1] = start[1]; data.qpos[ball.q + 2] = start[2];
  data.qpos[ball.q + 3] = 1; data.qpos[ball.q + 4] = data.qpos[ball.q + 5] = data.qpos[ball.q + 6] = 0;
  data.qvel[ball.v] = (to[0] - start[0]) / T; data.qvel[ball.v + 1] = (to[1] - start[1]) / T;
  data.qvel[ball.v + 2] = (to[2] - start[2]) / T + 0.5 * 9.81 * T;
  for (let i = 3; i < 6; i++) data.qvel[ball.v + i] = 0;
}
function setCommand(v) {
  cmdWant = v.slice();
  if (!cmdWant.some((x) => x)) return;
  if (mode === 'clip') { toast('正在做动作：先点「站立」再用 WASD 走'); cmdWant = [0, 0, 0]; return; }
  if (mode === 'fallen' || lift) { cmdWant = [0, 0, 0]; return; }
  if (!walk) { toast('没有走路策略（make view 打包 velocity 训练结果）'); cmdWant = [0, 0, 0]; return; }
  if (mode !== 'walk') { mode = 'walk'; clock = 0; stillFor = 0; cpgReset(); ramp = [0, 0, 0]; prevAction = new Float64Array(nj); walk.fresh = true; }
}
function loadScene(id) {
  const scene = SCENES.find((s) => s.id === id) || SCENES[0];
  mode = 'idle'; clip = null; stopWalk();
  if (W) { W.data.delete(); W.model.delete(); }
  W = buildWorld(scene); gainsOf = null; ctrlRange0 = null; ctrlWide = false;
  spawn(0);
  post({ t: 'scene', id: scene.id });
}
function resetRobot(i = W.spawn) {
  mode = 'idle'; clip = null; stopWalk();
  spawn(i);
  status('', '');
}

// what the page draws: the state, the reference ghost's pose, and what the panel shows
function ghostQpos() {
  if (mode !== 'clip') return null;
  const data = W.data, m = clip.motion, f = Math.min(frame, m.n_frames - 1), rq = m.root_quat[f];
  const ry = twist(robotQuat());
  const out = new Float64Array(7 + nj);
  out[0] = data.qpos[0] + 0.85 * Math.sin(ry); out[1] = data.qpos[1] - 0.85 * Math.cos(ry); out[2] = m.root_pos[f][2] + ground;
  // turned to the robot's heading (align), and also the clip's own heading relative to its start
  const qa = quatMul(yawQuat(ry - twist(rq)), rq), qb = quatMul(yawQuat(spawnYaw - refYaw0), rq);
  for (let i = 0; i < nj; i++) out[7 + i] = m.joint_pos[f][i];
  return { q: out, align: qa, free: qb };
}
let rtShown = 1;
function sendState() {
  if (!W) return;
  const qpos = Float64Array.from(W.data.qpos), g = ghostQpos();
  let dyaw = null;
  if (mode === 'clip') {
    const f = Math.min(frame, clip.motion.n_frames - 1);
    dyaw = (twist(robotQuat()) - spawnYaw - (twist(clip.motion.root_quat[f]) - refYaw0)) * 180 / Math.PI;
    dyaw = ((dyaw % 360) + 540) % 360 - 180;
  }
  post({ t: 'state', qpos, ghost: g, simTime, mode, clip: clip && clip.run, clipLabel: clip && clip.label,
         frame, nFrames: clip ? clip.motion.n_frames : 0, err: errN ? errSum / errN : null, dyaw, ground,
         lift: lift ? (lift.down === null ? 1 : 2) : 0, drag: drag && drag.p ? { p: drag.p, force: drag.force, capped: drag.capped } : null,
         rt: rtShown, delays, noisy, paused, speed, cmd, velGot, scene: W.scene.id, spawn: W.spawn }, [qpos.buffer]);
}

// -- messages: queued, and applied between control steps -----------------------------------------
const queue = [];
onmessage = (e) => {
  const m = e.data;
  if (m.t === 'init') { init(m).catch(fail); return; }
  if (m.t === 'clip') {                            // load it while the simulation runs on
    const entry = clipEntries.find((x) => x.run === m.run);
    if (!entry) { post({ t: 'ack', id: m.id, ok: false }); return; }
    status(`加载 ${entry.label}…`);
    bundle(entry.run).then((b) => queue.push({ t: 'clipReady', b, label: entry.label, t0: m.t0 || 0, id: m.id }), fail);
    return;
  }
  queue.push(m);
};
function apply(m) {
  switch (m.t) {
    case 'scene': loadScene(m.id); break;
    case 'spawn': resetRobot(m.i); break;
    case 'reset': resetRobot(); break;
    case 'clipReady':
      startClip(m.b, m.label, m.t0);
      post({ t: 'ack', id: m.id, ok: true, run: m.b.run, audio: m.b.meta.audio ? m.b.base + m.b.meta.audio : null });
      break;
    case 'idle': if (mode === 'clip' || mode === 'walk') { mode = 'idle'; clip = null; stopWalk(); status(''); } break;
    case 'push': push(m.dir, m.speed); break;
    case 'lift': toggleLift(); break;
    case 'ball': throwBall(m.cam); break;
    case 'noise': noisy = m.on; sampleRealism(); break;
    case 'pause': paused = m.on; break;
    case 'speed': speed = m.v; break;
    case 'seek':
      if (mode !== 'clip') break;
      frame = m.frame;
      placeRobot(W.data.qpos[0], W.data.qpos[1], ground, twist(robotQuat()), frame);
      mode = 'clip';
      refYaw0 = twist(clip.motion.root_quat[frame]); spawnYaw = twist(robotQuat());
      break;
    case 'drag': drag = { body: m.body, local: m.local, target: m.target, mass: m.mass, prev: null, p: null, force: 0 }; break;
    case 'dragTo': if (drag) drag.target = m.target; break;
    case 'dragEnd': drag = null; break;
    case 'cmd': setCommand(m.v); break;
  }
}

// fixed-rate control steps against the wall clock
let last = 0, due = 0, rtWall = 0, rtSim = 0;
async function loop() {
  try {
    let dirty = queue.length > 0;
    while (queue.length) apply(queue.shift());
    const now = performance.now(), wall = Math.min(0.25, (now - last) / 1000); last = now;
    if (!paused) {
      due += wall * speed * HZ;
      const t0 = simTime;
      let n = 0;
      while (due >= 1 && n < 14) { await controlStep(); due -= 1; n++; }
      if (due > 14) due = 0;                                   // too slow: let sim time lag rather than spiral
      rtWall += wall; rtSim += simTime - t0;
      if (rtWall > 1) { rtShown = rtSim / rtWall / speed; rtWall = 0; rtSim = 0; }
      dirty ||= n > 0;
    }
    if (dirty || paused) sendState();
  } catch (e) { fail(e); return; }
  setTimeout(loop, 4);
}

function fail(e) { post({ t: 'error', message: String(e && e.message || e), stack: e && e.stack }); }

async function init(m) {
  const loadMujoco = (await import('./node_modules/@mujoco/mujoco/mujoco.js')).default;
  ort = await import('./node_modules/onnxruntime-web/dist/ort.mjs');
  ort.env.wasm.wasmPaths = new URL('./node_modules/onnxruntime-web/dist/', location).href;
  ort.env.wasm.numThreads = 1;
  ({ Contract, quatToGravity } = await import('./contract.js'));
  // Bundles are re-exported in place, and http.server lets the browser cache them by heuristic:
  // revalidate every data file so a fresh export is never shadowed by a stale copy.
  get = (u) => fetch(u, { cache: 'no-cache' });

  index = await get('./run/index.json').then(r => r.ok ? r.json() : null).catch(() => null);
  if (!index) index = { runs: [{ run: m.run || 'bt_v3', label: m.run || 'bt_v3' }] };
  idleEntry = index.runs.find(r => r.role === 'idle');
  // ?walk=<run> picks one of the walking policies; the first in index.json otherwise
  const walks = index.runs.filter(r => r.role === 'walk');
  walkEntry = walks.find(r => r.run === m.walk) || walks[0];
  clipEntries = index.runs.filter(r => r.role !== 'idle' && r.role !== 'walk');
  const first = idleEntry || clipEntries[0];
  xmlBase = await get(`./run/${first.run}/model.xml`).then(r => r.text());
  [idle, walk] = await Promise.all([idleEntry ? bundle(idleEntry.run) : null, walkEntry ? bundle(walkEntry.run) : null]);
  c = (idle || await bundle(first.run)).contract;
  nj = c.nJoints; HZ = c.policyHz; DEC = c.decimation;
  prevAction = new Float64Array(nj); prevTarget = new Float64Array(nj);
  mj = await loadMujoco();
  post({ t: 'ready', xmlBase, idle: idleEntry ? { run: idleEntry.run } : null, walk: walkEntry ? { run: walkEntry.run, label: walkEntry.label, vx: walk.meta.walk && walk.meta.walk.vx } : null,
         clips: clipEntries.map((e) => ({ run: e.run, label: e.label })), hz: HZ });
  loadScene(m.scene);
  last = performance.now();
  loop();
}
