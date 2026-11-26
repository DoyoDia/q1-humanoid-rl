"""Make a retargeted clip balanceable: move the robot's centre of mass so its ZMP stays over
the feet that are on the ground, without touching the feet or the upper body.

A human dancer's balance does not survive retargeting onto a body with other proportions
and much smaller feet (Q1's soles are 14.5 x 4.6 cm). On LAFAN dance1, the retargeted
clip stands on one leg with its centre of mass up to 20 cm outside the stance foot; a
policy that tracks it cannot lift that foot without falling, so it learns to keep both
feet down -- and every toe tap and ankle flick that happens with the foot in the air is
lost. This module fixes the reference instead:

  1. contacts   which sole points (the four sole spheres of each foot) touch the floor,
                frame by frame, from the clip's own foot heights;
  2. trajectory a new centre-of-mass path, as one quadratic program over the whole clip:
                as close to the original as possible and changing smoothly, subject to its
                ZMP (cart-table model: com - h/g * com_acc) lying inside the support polygon
                of the touching sole points, shrunk by a margin. Violations are slack
                variables with a large cost, so an impossible frame degrades instead of
                failing the clip;
  3. inverse kinematics, per frame (mink, in parallel processes): both feet keep their
                pose in the world, the pelvis keeps its orientation (softly), the arms and
                waist keep their joint angles, and the base position and the leg joints
                move to put the centre of mass on the new path. Where the legs cannot
                reach it, the com is bounded to what they reached and step 2 runs again,
                so the final path is smooth and reachable.

balance() returns the new base trajectory and joints, and what it did.
"""

from __future__ import annotations

import mujoco
import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp
from scipy.spatial import ConvexHull

G = 9.81
SIDES = ("left", "right")


def _sole_spheres(model):
    """Per foot: the sole contact spheres (geom ids) on the ankle roll link."""
    out = {}
    for s in SIDES:
        b = model.body(f"{s}_ankle_roll_link").id
        out[s] = [g for g in range(model.ngeom)
                  if model.geom_bodyid[g] == b and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    return out


def _fk(model, data, root_pos, root_quat, q):
    data.qpos[0:3], data.qpos[3:7], data.qpos[7:] = root_pos, root_quat, q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def _contacts(points_z, fps, height=0.012, min_frames=3):
    """points_z (T, n): sole point heights above the floor -> boolean (T, n), with
    contact or air runs shorter than min_frames merged into their surroundings."""
    on = points_z < height
    for k in range(on.shape[1]):
        col = on[:, k]
        for val in (False, True):           # drop short air gaps, then short touches
            t = 0
            while t < len(col):
                if col[t] == val:
                    e = t
                    while e < len(col) and col[e] == val:
                        e += 1
                    if e - t < min_frames and t > 0 and e < len(col):
                        col[t:e] = not val
                    t = e
                else:
                    t += 1
    return on


def _halfplanes(pts, margin, pad=0.005):
    """Support polygon of the touching points as rows (n, b): n.p <= b, shrunk by margin
    (less, if the polygon is too thin for it). Fewer than three non-collinear points --
    a foot on its edge -- become a thin polygon padded by `pad`."""
    pts = np.asarray(pts)
    hull = None
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            if hull.volume < 1e-5:          # m^2: collinear in practice
                hull = None
        except Exception:
            hull = None
    if hull is None:
        sq = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * pad
        pts = (pts[:, None, :] + sq[None]).reshape(-1, 2)
        hull, margin = ConvexHull(pts), 0.0
    eq = hull.equations                     # n.x + d <= 0 inside, n unit
    n, b = eq[:, :2], -eq[:, 2]
    centre = pts[hull.vertices].mean(0)
    inr = float(np.min(b - n @ centre))     # distance centre -> nearest edge
    return n, b - min(margin, 0.5 * inr)


def _zmp(com, h, fps):
    acc = np.zeros_like(com)
    acc[1:-1] = (com[2:] - 2 * com[1:-1] + com[:-2]) * fps * fps
    return com - (h / G)[:, None] * acc




# -- inverse kinematics, one frame at a time, in worker processes -------------------------
# Frames are independent but for a warm start, so contiguous runs of them go to a pool of
# forked workers. _IK holds what the workers need; it is set before the pool forks.
_IK: dict = {}


def _ik_setup(model, root_pos, root_quat, q, ik_iters, hold):
    """hold (T, 2): per frame and foot, whether the foot is held in place exactly (it is
    down, or about to be or just was) or only softly (in the air)."""
    import mink
    feet = [model.body(f"{s}_ankle_roll_link").id for s in SIDES]
    legs = [model.jnt_dofadr[model.joint(f"{s}_{j}_joint").id]
            for s in SIDES for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    _IK.update(model=model, root_pos=root_pos, root_quat=root_quat, q=q, ik_iters=ik_iters,
               feet=feet, legs=legs, hold=hold, waist=model.jnt_dofadr[model.joint("waist_roll_joint").id],
               foot_target=None, foot_tol=0.005)


def _ik_run(job):
    """job: (frames, targets (n,2)) -> list of (t, qpos, com, foot_error). A foot on the
    floor is a hard constraint (a mink equality task) holding its pose in the world; a foot
    in the air is a soft task, so it can travel a little with the pelvis -- held exactly,
    its leg ran its ankle into a limit before the pelvis got over the other foot. The
    centre of mass follows the target horizontally; the base keeps its orientation softly;
    the arms keep their angles and the waist roll nearly (stiff posture costs); the legs
    are free; joint limits are constraints of mink's QP. foot_error is that of the held
    feet."""
    import mink
    frames, targets = job[0], job[1]
    warms = job[2] if len(job) > 2 else None              # explicit warm starts, per frame
    keep = job[3] if len(job) > 3 else None               # frames to report (others: run-up)
    W = _IK
    model, feet, legs = W["model"], W["feet"], W["legs"]
    data = mujoco.MjData(model)
    cfg = mink.Configuration(model)
    foot_tasks = [mink.FrameTask(model.body(b).name, "body", position_cost=1.0, orientation_cost=1.0) for b in feet]
    com_task = mink.ComTask(cost=[30.0, 30.0, 0.0])
    base_task = mink.FrameTask(model.body(1).name, "body", position_cost=0.0, orientation_cost=3.0)
    pcost = np.full(model.nv, 100.0)                        # arms, waist: held
    pcost[:6] = 0.0                                         # base: free
    pcost[legs] = 0.5                                       # legs: drawn to the last frame's correction
    pcost[W["waist"]] = 2.0                                 # waist roll: may lean a little
    posture = mink.PostureTask(model, cost=pcost)
    tasks, limits = [com_task, base_task, posture], [mink.ConfigurationLimit(model)]

    out, prev = [], None
    for i, (t, goal) in enumerate(zip(frames, targets)):
        orig = np.concatenate([W["root_pos"][t], W["root_quat"][t], W["q"][t]])
        data.qpos[:] = orig
        mujoco.mj_kinematics(model, data)
        foot_was = data.xpos[feet].copy()
        cfg.update(orig)
        for task, b in zip(foot_tasks, feet):
            task.set_target(cfg.get_transform_frame_to_world(model.body(b).name, "body"))
        if W.get("foot_target") is not None:                 # plant_feet: held feet go to given poses
            for k, (task, hd) in enumerate(zip(foot_tasks, W["hold"][t])):
                if hd:
                    T4 = W["foot_target"][t, k]
                    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(T4[:3, :3]), T4[:3, 3]))
                    foot_was[k] = T4[:3, 3]
        base_task.set_target(cfg.get_transform_frame_to_world(model.body(1).name, "body"))
        # The legs are drawn not to their source pose but to it plus the previous frame's
        # correction: solved independently, neighbouring frames landed on different
        # solutions wherever an ankle sat at its limit (a knee straightening from 0.9 to
        # 0.03 rad in two frames).
        ptarget = orig.copy()
        if prev is not None and prev[0] == t - 1:
            li = np.array(legs) - 6
            ptarget[7:][li] += prev[1][7:][li] - W["q"][t - 1][li]
        posture.set_target(ptarget)
        com_task.set_target(np.r_[goal, 0.0])

        # Start from the frame's own pose with the base where the previous frame put it
        # (only the base: carrying the legs' correction over let one bad solve poison
        # every frame after it).
        warm = orig.copy()
        if warms is not None:
            warm = warms[i].copy()
        elif prev is not None and prev[0] == t - 1:
            warm[0:3] += prev[1][0:3] - W["root_pos"][t - 1]
        cfg.update(warm)
        hold = W["hold"][t]
        held = [ft for ft, hd in zip(foot_tasks, hold) if hd]
        for ft, hd in zip(foot_tasks, hold):
            # in the air: free to slide sideways and fore-aft, much less to sink or rise
            # (the cost is in the foot's frame, whose z is near vertical in the air too)
            ft.set_position_cost(1.0 if hd else [10.0, 10.0, 300.0])
            ft.set_orientation_cost(1.0 if hd else 10.0)
        free_feet = [ft for ft, hd in zip(foot_tasks, hold) if not hd]
        for _ in range(W["ik_iters"]):
            try:
                v = mink.solve_ik(cfg, tasks + free_feet, 1.0, "daqp", damping=1e-3, limits=limits, constraints=held)
            except Exception:                               # constraints infeasible from here
                break
            # small steps: with the com target out of reach a full step overshoots and the
            # linearised foot constraints then chase it around
            cfg.integrate_inplace(v * min(1.0, 0.1 / max(1e-9, np.abs(v).max())), 1.0)
            if np.abs(v).max() < 1e-5:
                break
        data.qpos[:] = cfg.q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        err = float(max([np.linalg.norm(data.xpos[b] - fw) for b, fw, hd in zip(feet, foot_was, hold) if hd], default=0.0))
        if err > W.get("foot_tol", 0.005):                  # lost a foot: keep the frame as it was
            data.qpos[:] = orig
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            err = -1.0                                      # (counted as a fallback)
        prev = (t, data.qpos.copy())
        if keep is None or int(t) in keep:
            out.append((t, data.qpos.copy(), data.subtree_com[1].copy(), err))
    return out


def _ik_frames(frames, targets, workers, lead=10):
    """IK for `frames` (sorted) in parallel contiguous chunks. Each chunk is solved from
    `lead` frames before its start, whose results are dropped: the warm start and the
    legs' pull toward the previous frame's correction need a run-up, or every chunk
    boundary is a jump."""
    frames = np.asarray(frames)
    if not len(frames):
        return []
    T = len(_IK["q"])
    goal = {int(t): g for t, g in zip(frames, targets)}
    n = max(1, min(workers, len(frames) // 16))
    jobs = []
    for part in np.array_split(frames, n):
        start = max(0, int(part[0]) - lead)
        run = np.arange(start, int(part[-1]) + 1)
        jobs.append((run, np.array([goal.get(int(t), _IK["target"][t]) for t in run]), None, set(int(t) for t in part)))
    if n == 1:
        res = [_ik_run(jobs[0])]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n) as pool:
            res = pool.map(_ik_run, jobs)
    return [r for chunk in res for r in chunk]


def balance(model, root_pos, root_quat, q, fps, margin=0.01, accel=1e5, contact_height=0.01,
            flat_tilt=10.0, flight_height=0.05, max_shift=0.10, reach_tol=0.005, rounds=8,
            ik_iters=60, max_joint_step=0.25, workers=None, solver="clarabel", verbose=False):
    """root_pos (T,3), root_quat (T,4 wxyz), q (T,nu) in qpos order -> (root_pos, root_quat, q, stats)."""
    import os
    T = len(q)
    workers = workers or os.cpu_count() or 1
    data = mujoco.MjData(model)
    spheres = _sole_spheres(model)
    allg = [g for s in SIDES for g in spheres[s]]
    rad = model.geom_size[allg, 0]

    # -- 1. contacts, from the clip's own sole heights (floor = 5th percentile of the lowest
    #       point, as q1_rl.motion.from_root_trajectory grounds the clip)
    P = np.zeros((T, len(allg), 3))
    com = np.zeros((T, 3))
    tilt = np.zeros((T, 2))
    for t in range(T):
        _fk(model, data, root_pos[t], root_quat[t], q[t])
        P[t] = data.geom_xpos[allg]
        com[t] = data.subtree_com[1]
        for i, s in enumerate(SIDES):
            zz = data.xmat[model.body(f"{s}_ankle_roll_link").id][8]
            tilt[t, i] = np.degrees(np.arccos(np.clip(zz, -1, 1)))
    z = P[..., 2] - rad
    floor = np.percentile(z.min(1), 5)
    z -= floor
    # Which feet are down. Not by absolute height: a retargeted clip floats and sinks by a
    # centimetre or two (LAFAN dance1 hovers 1.2-1.8 cm up for most of its length), which
    # swamps any threshold. The lower foot is down unless the whole robot is clearly in the
    # air; the other foot is down when it is within contact_height of the lower one.
    # A foot that is down and nearly flat (sole tilted less than flat_tilt) supports with
    # its whole sole: the clip's feet are rarely exactly flat -- 5 deg already lifts the
    # heel 1.3 cm -- and the robot puts them down flat. Tilted more (a toe or heel stance),
    # only the sole points near its lowest one count.
    feet_z = np.stack([z[:, :4].min(1), z[:, 4:].min(1)], 1)
    lowest = feet_z.min(1, keepdims=True)
    down = _contacts(feet_z - lowest, fps, contact_height) & (lowest < flight_height)
    on = np.zeros_like(z, dtype=bool)
    for t in range(T):
        for i in range(2):
            sl = slice(4 * i, 4 * i + 4)
            if down[t, i]:
                on[t, sl] = True if tilt[t, i] < flat_tilt else z[t, sl] < z[t, sl].min() + contact_height
    hold = down.copy()                                      # feet held exactly: down, +-2 frames
    for sh in (1, 2):
        hold[sh:] |= down[:-sh]
        hold[:-sh] |= down[sh:]
    feet_ids = [model.body(f"{s}_ankle_roll_link").id for s in SIDES]
    li_all = np.array([model.jnt_dofadr[model.joint(f"{s}_{j}_joint").id] - 6 for s in SIDES
                       for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")])
    foot_src = np.zeros((T, 2, 3))
    for t in range(T):
        _fk(model, data, root_pos[t], root_quat[t], q[t])
        foot_src[t] = data.xpos[feet_ids]
    planes = [(_halfplanes(P[t, on[t], :2], margin) if on[t].any() else None) for t in range(T)]
    edges = [(_halfplanes(P[t, on[t], :2], 0.0) if on[t].any() else None) for t in range(T)]
    h = com[:, 2] - floor                                   # com height over the floor

    def violation(c):
        # Judged on the path low-passed over 0.1 s, before and after alike: the ZMP takes
        # the second difference of the centre of mass, times h/g*fps^2 (~115 here), so a
        # millimetre of frame-to-frame jitter reads as a ZMP 20 cm off -- in the source
        # clip as much as in the IK's output -- while a policy at 50 Hz never sees it.
        w = max(1, int(round(0.1 * fps)))
        cs = np.stack([np.convolve(np.pad(c[:, i], w // 2, mode="edge"), np.ones(w) / w, mode="valid")[:T]
                       for i in range(2)], 1)
        zmp = _zmp(cs, h, fps)
        vz, vc = np.zeros(T), np.zeros(T)
        for t, pl in enumerate(edges):
            if pl is not None:
                n, b = pl
                vz[t] = max(0.0, float(np.max(n @ zmp[t] - b)))
                vc[t] = max(0.0, float(np.max(n @ cs[t] - b)))
        return vz, vc

    vz0, vc0 = violation(com)

    # -- 2. the new centre-of-mass path: min |c - c0_lp|^2 + accel |D2 c|^2 + w |s|^2
    #       s.t. n.zmp(c)_t <= b_t + s_t, zmp(c) = c - (h/g) D2 c (linear in c).
    #       The path itself must be smooth, not just its change from the source: a
    #       retargeted com path jitters, and a cart-table ZMP constraint on a jittery path
    #       is met by accelerations that cancel the jitter, leaving the com where it was --
    #       outside the stance foot. With the acceleration cost, the only way to put the
    #       ZMP over the foot is to put the com there. c0_lp is the source path low-passed
    #       over 0.2 s.
    nv = 3 * T
    k = (h / G) * fps * fps
    lw = max(1, int(round(0.2 * fps)))
    c0 = np.stack([np.convolve(np.pad(com[:, i], lw // 2, mode="edge"), np.ones(lw) / lw, mode="valid")[:T]
                   for i in range(2)], 1)
    D2 = sp.diags([np.ones(T - 2), -2 * np.ones(T - 2), np.ones(T - 2)], [0, 1, 2], shape=(T - 2, T))
    Hc = sp.eye(T) + accel * (D2.T @ D2)
    H = sp.block_diag([Hc, Hc, 1e4 * sp.eye(T)]).tocsc()
    f = np.concatenate([-c0[:, 0], -c0[:, 1], np.zeros(T)])
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for t, pl in enumerate(planes):
        # zmp_t = c_t - k_t (c_{t+1} - 2 c_t + c_{t-1}). The clip's first and last frames
        # have no acceleration and are left free: a clip cut mid-move cannot be made to
        # stand still at its cut.
        if pl is None or t in (0, T - 1):
            continue
        n, b = pl
        taps = [(t, 1 + 2 * k[t]), (t - 1, -k[t]), (t + 1, -k[t])]
        for e in range(len(n)):
            for tt, w in taps:
                rows += [r, r]; cols += [tt, T + tt]; vals += [w * n[e, 0], w * n[e, 1]]
            rows.append(r); cols.append(2 * T + t); vals.append(-1.0)
            rhs.append(b[e]); r += 1
    A = sp.csc_matrix((vals, (rows, cols)), shape=(r, nv))
    rhs = np.array(rhs)

    # The legs can carry the com only so far with the feet where they are. First ask for
    # at most max_shift; then, wherever the IK fell short, bound the com to what it reached
    # and plan again, so the path stays smooth -- a plan the legs follow in some frames
    # and not in others has a jagged com, and a jagged com has a ZMP all over the floor.
    lo, hi = c0 - max_shift, c0 + max_shift
    sols = np.concatenate([root_pos, root_quat, q], 1)
    com1 = com.copy()
    resid_feet = np.zeros(T)
    todo, target = np.arange(T), None
    for rnd in range(rounds):
        lb = np.concatenate([lo[:, 0], lo[:, 1], np.zeros(T)])
        ub = np.concatenate([hi[:, 0], hi[:, 1], np.full(T, np.inf)])
        sol = solve_qp(H, f, A, rhs, lb=lb, ub=ub, solver=solver)
        if sol is None:
            raise RuntimeError("balance QP failed")
        new = np.stack([sol[:T], sol[T:2 * T]], 1)
        slack = sol[2 * T:]
        if target is not None:
            todo = np.where(np.linalg.norm(new - target, axis=1) > 1e-3)[0]
        target = new
        _ik_setup(model, root_pos, root_quat, q, ik_iters, hold)
        _IK["target"] = target
        for t, qp_, c_, e_ in _ik_frames(todo, target[todo], workers):
            sols[t], com1[t], resid_feet[t] = qp_, c_, e_
        resid_com = np.linalg.norm(com1[:, :2] - target, axis=1)
        short = np.where(resid_com > reach_tol)[0]
        if verbose:
            print(f"  round {rnd}: solved {len(todo)} frames; com short of the plan by >{reach_tol * 1000:.0f} mm "
                  f"on {len(short)} (worst {resid_com.max() * 100:.1f} cm); qp slack max {slack.max() * 100:.2f} cm", flush=True)
        if not len(short) or rnd == rounds - 1:
            break
        # Those frames may move the com no further from the source than the legs took it.
        # A box, not a point: pinning the com to exactly what was reached also forbade the
        # moves the legs could make, and left the ZMP constraints around it unsatisfiable.
        reach = np.abs(com1[short, :2] - c0[short]).max(1, keepdims=True) + reach_tol
        lo[short] = np.maximum(lo[short], c0[short] - reach)
        hi[short] = np.minimum(hi[short], c0[short] + reach)

    # Continuity. Frames are solved one by one, and where an ankle sits at its limit two
    # neighbours can still land on different solutions -- a knee 0.3 rad apart, 15 rad/s
    # for one 20 ms step. The correction (solution minus source, for the base position and
    # the legs) is smoothed in time: a 5-frame median drops one-frame excursions, then a
    # 3-frame mean. The source's own motion is untouched; the feet move by what the
    # smoothing costs, which the stats report.
    from scipy.ndimage import median_filter, uniform_filter1d
    src = np.concatenate([root_pos, root_quat, q], 1)
    cols = np.r_[0:3, 7 + li_all]
    corr = sols[:, cols] - src[:, cols]
    corr = uniform_filter1d(median_filter(corr, size=(5, 1), mode="nearest"), 3, axis=0, mode="nearest")
    sols[:, cols] = src[:, cols] + corr
    for t in range(T):
        data.qpos[:] = sols[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com1[t] = data.subtree_com[1]
        fw = foot_src[t]
        held_err = [np.linalg.norm(data.xpos[b] - fw[i]) for i, b in enumerate(feet_ids) if hold[t, i]]
        if resid_feet[t] >= 0:
            resid_feet[t] = max(held_err, default=0.0)
    steps = np.abs(np.diff(sols[:, 7:], axis=0)).max(1)
    jump = max_joint_step
    fallbacks = int(np.sum(resid_feet < 0))
    resid_feet = np.maximum(resid_feet, 0.0)
    resid_com = np.linalg.norm(com1[:, :2] - target, axis=1)

    out_pos, out_q = sols[:, 0:3], sols[:, 7:]
    out_quat = sols[:, 3:7] / np.linalg.norm(sols[:, 3:7], axis=1, keepdims=True)
    vz1, vc1 = violation(com1)
    supported = np.array([pl is not None for pl in planes])
    single = down.sum(1) == 1
    li = np.array(_IK["legs"]) - 6
    shift = np.linalg.norm(out_pos[:, :2] - root_pos[:, :2], axis=1)
    stats = dict(
        frames=T, single_support=float(single.mean()),
        zmp_out_before=float(np.mean(vz0[supported] > 0.01)), zmp_out_after=float(np.mean(vz1[supported] > 0.01)),
        zmp_p95_before=float(np.percentile(vz0[supported], 95)), zmp_p95_after=float(np.percentile(vz1[supported], 95)),
        zmp_worst_before=float(vz0.max()), zmp_worst_after=float(vz1.max()),
        com_single_p50_before=float(np.median(vc0[single])) if single.any() else 0.0,
        com_single_p50_after=float(np.median(vc1[single])) if single.any() else 0.0,
        base_shift_max=float(shift.max()), base_shift_mean=float(shift.mean()),
        leg_change_rms=float(np.sqrt(np.mean((out_q - q)[:, li] ** 2))),
        qp_slack_max=float(slack.max()),
        foot_resid_max=float(resid_feet.max()), foot_resid_p99=float(np.percentile(resid_feet, 99)),
        com_short_frames=float(np.mean(resid_com > reach_tol)), com_resid_max=float(resid_com.max()),
        fallback_frames=fallbacks, joint_jumps=int(np.sum(steps > jump)))
    return out_pos, out_quat, out_q, stats


def plant_feet(model, root_pos, root_quat, q, fps, window_s=2.0, max_height=0.08, max_speed=0.2,
               min_frames=5, blend=6, flight_windows=(), flight_low=0.012, flight_high=0.07,
               clearance=0.001, ik_iters=60, workers=None, verbose=False):
    """Put the feet that are down flat on the floor and keep them still while down.

    For sources that stand with flat feet but whose estimate does not: video pose
    estimation gives broadcast gymnastics, done with both feet flat nearly throughout,
    soles tilted a median 13 deg and one foot 1-3 cm above the other -- one edge touches,
    and balance() (which reads contacts from relative heights) sees one foot down.

      1. down: a foot within max_height of the floor (the 5th percentile of the lowest sole
         point in a sliding window_s window -- video height drifts) whose ankle moves slower
         than max_speed (smoothed); runs shorter than min_frames merged away. Speed does most
         of the work: standing feet move < 0.1 m/s, swinging ones 0.3-0.7. The height bound is
         loose because the estimate is: standing on both feet, one of them came out 5.5 cm up;
         it only has to keep out a raised knee held still (foot 10 cm and more up).
      1b. at least one foot down, always: where neither is, the lower one (on average over
         the gap) is. The exception is flight -- both feet off together, flight_low to
         flight_high up (a jumping jack: 1-4 cm in the estimate, where a hop with a raised
         knee has one foot 10-19 cm up and the other on the floor) -- and only inside
         flight_windows ((start_s, end_s), ...): elsewhere both feet off is estimation noise.
      2. target per contact run: the ankle's median horizontal position and heading over the
         run (a planted foot does not slide), on the floor, sole level; over the first and
         last `blend` frames of a run the target moves between the source pose and that one,
         but only while the other foot is fully planted -- one foot is always flat and still.
      3. IK per frame (mink, the machinery balance() uses): the planted feet are hard
         constraints at their targets, a foot in the air follows its source pose softly, the
         centre of mass stays over its source point, the pelvis keeps its orientation softly
         and the arms and waist their angles.
    The result is put on the floor by the same floor estimate (planted soles at `clearance`),
    so do not ground it again by percentile. Returns (root_pos, root_quat, q, stats). Run it
    before balance().
    """
    import os
    from scipy.ndimage import gaussian_filter1d, percentile_filter
    T = len(q)
    workers = workers or os.cpu_count() or 1
    data = mujoco.MjData(model)
    spheres = _sole_spheres(model)
    feet_ids = [model.body(f"{s}_ankle_roll_link").id for s in SIDES]
    # ankle height above the floor with the sole level: from the stand keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    g0 = spheres[SIDES[0]]
    ankle_h = float(data.xpos[feet_ids[0]][2] - np.min(data.geom_xpos[g0, 2] - model.geom_size[g0, 0]))

    low = np.zeros((T, 2))
    pos = np.zeros((T, 2, 3))
    yaw = np.zeros((T, 2))
    tilt = np.zeros((T, 2))
    com = np.zeros((T, 3))
    for t in range(T):
        _fk(model, data, root_pos[t], root_quat[t], q[t])
        com[t] = data.subtree_com[1]
        for i, s in enumerate(SIDES):
            g = spheres[s]
            low[t, i] = np.min(data.geom_xpos[g, 2] - model.geom_size[g, 0])
            pos[t, i] = data.xpos[feet_ids[i]]
            R = data.xmat[feet_ids[i]].reshape(3, 3)
            yaw[t, i] = np.arctan2(R[1, 0], R[0, 0])
            tilt[t, i] = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    win = max(3, int(round(window_s * fps)) | 1)
    floor = gaussian_filter1d(percentile_filter(low.min(1), 5, size=win, mode="nearest"), win / 4, mode="nearest")
    speed = np.linalg.norm(np.gradient(gaussian_filter1d(pos, 2.0, axis=0, mode="nearest"), axis=0), axis=-1) * fps
    height = low - floor[:, None]
    down = _contacts(np.where((height < max_height) & (speed < max_speed), 0.0, 1.0), fps, 0.5, min_frames)
    hs = gaussian_filter1d(height, 1.5, axis=0, mode="nearest")
    tsec = np.arange(T) / fps
    inwin = np.zeros(T, bool)
    for a, b in flight_windows:
        inwin |= (tsec >= a) & (tsec < b)
    flight = inwin & (hs > flight_low).all(1) & (hs < flight_high).all(1)
    flight = ~_contacts(np.where(flight, 1.0, 0.0)[:, None], fps, 0.5, 3)[:, 0]      # runs >= 3 frames
    flight &= inwin
    gap = ~down.any(1) & ~flight
    t = 0
    forced = 0
    while t < T:                                             # give each gap the lower foot
        if not gap[t]:
            t += 1
            continue
        e = t
        while e < T and gap[e]:
            e += 1
        down[t:e, int(np.argmin(hs[t:e].mean(0)))] = True
        forced += e - t
        t = e

    # source pose of each ankle, to blend from at the ends of a contact: switching the target
    # from where the source put the foot to the floor in one frame (up to 5 cm and 15 deg)
    # jerks the knee by the same in one 20 ms step
    src_R = np.zeros((T, 2, 3, 3))
    for t in range(T):
        _fk(model, data, root_pos[t], root_quat[t], q[t])
        for i in range(2):
            src_R[t, i] = data.xmat[feet_ids[i]].reshape(3, 3)
    from scipy.spatial.transform import Rotation, Slerp

    # blend weights: 1 = planted pose; ramps at the ends of each run
    W = np.zeros((T, 2))
    for i in range(2):
        t = 0
        while t < T:
            if not down[t, i]:
                t += 1
                continue
            e = t
            while e < T and down[e, i]:
                e += 1
            k = np.arange(t, e)
            W[t:e, i] = np.minimum.reduce([np.ones(e - t), (k - t + 1) / blend if t > 0 else np.ones(e - t),
                                           (e - k) / blend if e < T else np.ones(e - t)])
            t = e
    # one foot fully planted wherever a foot is down: a ramp only while the other is planted
    for t in range(T):
        if down[t].any() and not (W[t] >= 1.0).any():
            W[t, int(np.argmax(np.where(down[t], W[t], -1.0)))] = 1.0

    target = np.zeros((T, 2, 4, 4))
    runs = 0
    for i in range(2):
        t = 0
        while t < T:
            if not down[t, i]:
                t += 1
                continue
            e = t
            while e < T and down[e, i]:
                e += 1
            xy = np.median(pos[t:e, i, :2], axis=0)
            hd = np.angle(np.mean(np.exp(1j * yaw[t:e, i])))
            c, s_ = np.cos(hd), np.sin(hd)
            flat = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            for k in range(t, e):
                w = W[k, i]
                p_flat = np.array([xy[0], xy[1], floor[k] + ankle_h])
                target[k, i] = np.eye(4)
                if w >= 1.0:
                    target[k, i, :3, :3] = flat
                    target[k, i, :3, 3] = p_flat
                else:
                    rots = Rotation.from_matrix(np.stack([src_R[k, i], flat]))
                    target[k, i, :3, :3] = Slerp([0.0, 1.0], rots)(w).as_matrix()
                    target[k, i, :3, 3] = (1 - w) * pos[k, i] + w * p_flat
            runs += 1
            t = e

    _ik_setup(model, root_pos, root_quat, q, ik_iters, down)
    _IK.update(foot_target=target, foot_tol=0.01, target=com[:, :2])
    res = _ik_frames(np.arange(T), com[:, :2], workers)
    rp, rq, qq = root_pos.copy(), root_quat.copy(), q.copy()
    fallback = 0
    for t, qpos, _, err in res:
        if err < 0:
            fallback += 1
            continue
        rp[t], rq[t], qq[t] = qpos[0:3], qpos[3:7], qpos[7:]
    _IK.update(foot_target=None, foot_tol=0.005)
    rp[:, 2] -= floor - clearance                            # planted soles at `clearance` over z = 0

    # what it did: tilt and height of the planted feet before and after
    tilt1 = np.zeros((T, 2))
    low1 = np.zeros((T, 2))
    for t in range(T):
        _fk(model, data, rp[t], rq[t], qq[t])
        for i, s in enumerate(SIDES):
            g = spheres[s]
            low1[t, i] = np.min(data.geom_xpos[g, 2] - model.geom_size[g, 0])
            tilt1[t, i] = np.degrees(np.arccos(np.clip(data.xmat[feet_ids[i]][8], -1, 1)))
    flat1 = (low1 < clearance + 0.004) & (tilt1 < 2.0)
    gap0 = np.abs(low[:, 0] - low[:, 1])[down.all(1)]
    gap1 = np.abs(low1[:, 0] - low1[:, 1])[down.all(1)]
    li = [model.jnt_dofadr[model.joint(f"{s}_{j}_joint").id] - 6 for s in SIDES
          for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    stats = dict(both_down=float(down.all(1).mean()), any_down=float(down.any(1).mean()), runs=runs,
                 tilt_before=float(np.median(tilt[down])), tilt_after=float(np.median(tilt1[down])),
                 tilt_after_p95=float(np.percentile(tilt1[down], 95)),
                 gap_before_mm=float(np.median(gap0) * 1000) if len(gap0) else 0.0,
                 gap_after_mm=float(np.median(gap1) * 1000) if len(gap1) else 0.0,
                 leg_change_rms=float(np.sqrt(np.mean((qq[:, li] - q[:, li]) ** 2))),
                 base_shift_max=float(np.linalg.norm(rp - root_pos, axis=1).max()),
                 fallback_frames=float(fallback / T), forced=float(forced / T), flight=float(flight.mean()),
                 no_flat_foot=float((~flat1.any(1) & ~flight).mean()))
    return rp, rq, qq, stats
