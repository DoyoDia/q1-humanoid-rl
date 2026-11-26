#!/usr/bin/env python3
"""Build the physics model that training, the browser and deployment all share.

Everything here is derived from the vendor MJCF plus contract/. Nothing is typed twice:
gains come from gains.yaml, force limits from joints.yaml, timestep from rates.yaml.

    python3 build_model.py                      # mimic_opensource_v2 gains (default)
    python3 build_model.py --contract walk
    python3 build_model.py --keep-meshes        # visual build, for a desktop viewer

Three deliberate departures from q1_v3_mc.xml:

1. Visual mesh geoms are deleted; the three COLLIDING meshes (both feet and the head,
   contype=1) are kept and copied into robots/q1/meshes/. Deleting them too looks
   harmless -- a standing rollout is identical, because the robot rests on four 2 mm
   contact spheres per foot and the foot mesh never reaches the floor -- but a tumbling
   rollout diverges (tools/check_mesh_substitution.py measures it). The browser build
   (--self-contained) cannot ship STL files, so there the colliding meshes become boxes
   and the divergence is the price; the training/deployment build pays nothing.

2. Torque actuators become position actuators carrying the contract's kp/kv. The robot's
   low level runs PD at 1 kHz from the stiffness/damping fields of JointCommand, so
   putting PD inside MuJoCo is the faithful model -- and it lets us call step(nstep=20)
   for one policy step instead of 20 Python round-trips.

3. A `stand` keyframe is added at the contract's default_dof_pos, with the base height
   solved so the lowest foot contact point rests on the floor.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import sys

import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from q1_rl.geometry import geom_lowest_z, solve_stand_height  # noqa: E402
CONTRACT = HERE / "contract"


def load_contract():
    joints = yaml.safe_load((CONTRACT / "joints.yaml").read_text(encoding="utf-8"))
    gains = yaml.safe_load((CONTRACT / "gains.yaml").read_text(encoding="utf-8"))
    rates = yaml.safe_load((CONTRACT / "rates.yaml").read_text(encoding="utf-8"))
    obs = json.loads((CONTRACT / "obs_spec.json").read_text(encoding="utf-8"))
    return joints, gains, rates, obs


def mesh_aabb(model, mesh_id):
    """Axis-aligned bounding box of a compiled mesh, in the mesh's own frame."""
    adr, num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    v = model.mesh_vert[adr: adr + num]
    lo, hi = v.min(axis=0), v.max(axis=0)
    return (lo + hi) * 0.5, (hi - lo) * 0.5


def handle_meshes(spec, self_contained: bool, meshdir: str):
    """Deal with the vendor's mesh geoms. Returns (deleted, boxed, kept).

    Not all mesh geoms are visual. Each foot and the head carry a *colliding* mesh
    (contype=1) alongside their primitives. Deleting those silently changes the physics --
    a standing rollout cannot tell, because the feet rest on four 2 mm contact spheres and
    the foot mesh never reaches the floor, which is exactly why an earlier 500-step check
    passed while the model was wrong.

    So the two builds diverge here, on purpose:

      training / deployment (default)
          Visual meshes and their assets go; the three colliding meshes stay, copied into
          robots/q1/meshes/ (5 MB). Fidelity is what matters and that is cheap.

      --self-contained (the browser build)
          Colliding meshes become their bounding box so the XML needs no external assets.
          A box is bigger than the foot, so the box would steal ground contact from the
          vendor's sole spheres; demote_undercutting_boxes() prevents that. This build is
          an approximation -- tools/check_mesh_substitution.py measures by how much.
    """
    deleted = boxed = kept = 0
    compiled = spec.compile() if self_contained else None
    keep_names = set()
    for geom in list(spec.geoms):
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH and not geom.meshname:
            continue
        if geom.contype == 0 and geom.conaffinity == 0:
            spec.delete(geom)
            deleted += 1
            continue
        if not self_contained:
            keep_names.add(geom.meshname)
            kept += 1
            continue
        mid = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_MESH, geom.meshname)
        center, half = mesh_aabb(compiled, mid)
        gq = np.asarray(geom.quat, float)
        mq = np.asarray(compiled.mesh_quat[mid], float)
        mpos = np.asarray(compiled.mesh_pos[mid], float)

        def mat(q):
            m = np.zeros(9)
            mujoco.mju_quat2Mat(m, q)
            return m.reshape(3, 3)

        total = np.zeros(4)
        mujoco.mju_mulQuat(total, gq, mq)
        geom.pos = np.asarray(geom.pos, float) + mat(gq) @ (mpos + mat(mq) @ center)
        geom.quat = total
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = half
        geom.meshname = ""
        boxed += 1

    for mesh in list(spec.meshes):
        if mesh.name not in keep_names:
            spec.delete(mesh)
    if keep_names:
        # Copy the few colliding meshes in next to the model. 4 MB, and the model then
        # stands on its own instead of reaching back into the vendor archive.
        dst = Path(meshdir)
        dst.mkdir(parents=True, exist_ok=True)
        src = Path(spec.modelfiledir)
        for name in sorted(keep_names):
            for mesh in spec.meshes:
                if mesh.name == name and mesh.file:
                    f = (src / spec.meshdir / mesh.file).resolve()
                    if f.exists():
                        shutil.copyfile(f, dst / f.name)
        spec.meshdir = str(dst.resolve())
    return deleted, boxed, kept


def demote_undercutting_boxes(spec, tol=1e-4):
    """Stop a bounding box from stealing ground contact from the vendor's contact points.

    Each foot carries four 2 mm spheres at the sole corners: those are the ground contact
    the vendor's own policies were trained against. A mesh's AABB is conservative, and the
    foot box reaches ~5 mm BELOW those spheres -- after boxing, the robot would stand on
    box corners instead, 2 cm higher and rocking.

    The floor is `contype=1 conaffinity=0`, so a geom touches it iff its own conaffinity is
    non-zero. Clearing conaffinity on the box removes exactly the floor contact and leaves
    robot-vs-robot collision intact.
    """
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)

    lowest_sphere = {}
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_conaffinity[g]:
            b = model.geom_bodyid[g]
            lowest_sphere[b] = min(lowest_sphere.get(b, np.inf), geom_lowest_z(model, data, g))

    demoted = set()
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or b not in lowest_sphere:
            continue
        if geom_lowest_z(model, data, g) < lowest_sphere[b] - tol:
            demoted.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b))

    for geom in spec.geoms:
        parent = geom.parent.name if geom.parent else ""
        if parent in demoted and geom.type == mujoco.mjtGeom.mjGEOM_BOX:
            geom.conaffinity = 0
    return sorted(demoted)


def to_position_actuators(spec, joints, gains_for):
    """Replace torque actuators with position actuators carrying the contract gains."""
    by_joint = {j["name"]: j for j in joints["joints"]}
    order = [j["name"] for j in joints["joints"]]
    seen = {}
    for act in spec.actuators:
        jname = act.target
        if jname not in by_joint:
            raise SystemExit(f"actuator {act.name!r} drives {jname!r}, which is not in the contract")
        i = order.index(jname)
        kp, kv = gains_for["kps"][i], gains_for["kds"][i]
        act.set_to_position(kp=kp, kv=kv)
        # The policy commands a joint angle, so ctrl is bounded by the joint, and the
        # torque the PD is allowed to ask for is bounded by the contract's effort cap.
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [by_joint[jname]["lower"], by_joint[jname]["upper"]]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-by_joint[jname]["effort"], by_joint[jname]["effort"]]
        seen[jname] = act.name
    missing = [j for j in order if j not in seen]
    if missing:
        raise SystemExit(f"contract joints with no actuator: {missing}")
    return len(seen)


def style_floor(spec) -> None:
    """Black floor tiles, 1 m x 1 m, with thin white grout lines -- the look of every view
    of this model (MuJoCo's viewer, tools/render_motion.py, tools/rollout_video.py), so a
    clip's scale and drift can be read straight off the floor. Visual only: the floor
    geom's contact parameters are untouched.

    One texture tile is one metre: texuniform + texrepeat 1 maps texture space to world
    metres. 'edge' marks draw a 1-pixel border, so at 256 px per tile each grout line is
    two pixels, about 8 mm.
    """
    tex = spec.texture("groundplane")
    mat = spec.material("groundplane")
    if tex is None or mat is None:
        raise ValueError("source model has no 'groundplane' texture/material to restyle")
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    tex.rgb1 = tex.rgb2 = [0.06, 0.06, 0.07]
    tex.mark = mujoco.mjtMark.mjMARK_EDGE
    tex.markrgb = [0.92, 0.92, 0.92]
    tex.width = tex.height = 256
    mat.texrepeat = [1.0, 1.0]
    mat.texuniform = True
    mat.reflectance = 0.15          # a glazed tile, not a mirror
    mat.specular = 0.3
    mat.shininess = 0.6


TOUCH_BODIES = [f"{side}_{b}_link" for side in ("left", "right")
                for b in ("shoulder_yaw", "elbow", "wrist_yaw")]


def add_touch_sensors(spec, model, margin=0.01) -> int:
    """A box site around each arm link's collision geoms, with a touch sensor on it.

    A trained walk policy kept its hands and elbows in its hip-pitch links 97% of the
    time and nothing in the reward could see it. Touch sensors report the contact normal
    force inside the site, from MuJoCo's own contacts, on both backends; the env reads
    them as sensordata and penalises them. They add no physics."""
    n = 0
    for name in TOUCH_BODIES:
        bid = model.body(name).id
        corners = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[g])
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corners.append(model.geom_pos[g] + R.reshape(3, 3) @ (c + h * [sx, sy, sz]))
        if not corners:
            continue
        lo, hi = np.min(corners, 0) - margin, np.max(corners, 0) + margin
        site = spec.body(name).add_site(name=f"touch_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                        pos=list((lo + hi) / 2), size=list((hi - lo) / 2),
                                        rgba=[1, 0.3, 0.3, 0.0], group=5)
        sensor = spec.add_sensor(name=f"touch_{name}", type=mujoco.mjtSensor.mjSENS_TOUCH,
                                 objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site.name)
        n += 1
    return n


SOLE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def add_sole_sensors(spec, model, margin=0.005) -> int:
    """A box site around each foot's floor-contact spheres, with a touch sensor on it: the load
    on each sole. A walking policy has to unload a foot before it can lift it, and a reward
    that only sees the foot once it is off the floor gives nothing to climb until then -- three
    velocity-command runs bent the swing knee in time with the gait clock and never lifted a
    foot. Named sole_*, not touch_*: the env's self-contact penalty counts the touch_* ones."""
    n = 0
    for name in SOLE_BODIES:
        bid = model.body(name).id
        pts = [model.geom_pos[g] for g in range(model.ngeom)
               if model.geom_bodyid[g] == bid and model.geom_conaffinity[g]
               and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        if not pts:
            continue
        r = max(model.geom_size[g, 0] for g in range(model.ngeom)
                if model.geom_bodyid[g] == bid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
        lo, hi = np.min(pts, 0) - r - margin, np.max(pts, 0) + r + margin
        site = spec.body(name).add_site(name=f"sole_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                        pos=list((lo + hi) / 2), size=list((hi - lo) / 2),
                                        rgba=[0.3, 0.3, 1, 0.0], group=5)
        spec.add_sensor(name=f"sole_{name}", type=mujoco.mjtSensor.mjSENS_TOUCH,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site.name)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=HERE / "../../_ref/q1_v3/xml/q1_v3_mc.xml")
    p.add_argument("--contract", default="mimic_opensource_v2", choices=["mimic_opensource_v2", "walk"])
    p.add_argument("--keep-meshes", action="store_true")
    p.add_argument("--self-contained", action="store_true",
                   help="box the colliding meshes so the XML needs no external assets "
                        "(the browser build; an approximation -- see check_mesh_substitution.py)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    joints, gains, rates, _ = load_contract()
    g = gains[a.contract]
    out = Path(a.out) if a.out else HERE / ("q1_visual.xml" if a.keep_meshes
                                            else "q1_web.xml" if a.self_contained
                                            else "q1_physics.xml")

    spec = mujoco.MjSpec.from_file(str(a.src))
    spec.option.timestep = rates["sim"]["timestep"]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    style_floor(spec)

    meshdir = str(out.parent / "meshes")
    src_meshdir = None
    if a.keep_meshes:
        # The visual build references every vendor STL where it already lives. The spec's
        # meshdir is relative to the SOURCE xml, so pin it absolute for compiling here and
        # rewrite it relative to the OUTPUT xml when writing -- otherwise the written file
        # looks for ../meshes/ next to itself and does not load.
        src_meshdir = (Path(spec.modelfiledir) / spec.meshdir).resolve()
        spec.meshdir = str(src_meshdir)
    deleted, boxed, kept = ((0, 0, 0) if a.keep_meshes
                            else handle_meshes(spec, a.self_contained, meshdir))
    demoted = demote_undercutting_boxes(spec) if boxed else []
    n_act = to_position_actuators(spec, joints, g)

    model = spec.compile()
    n_touch = add_touch_sensors(spec, model)
    n_sole = add_sole_sensors(spec, model)
    model = spec.compile()
    stand_q = np.array(g["default_dof_pos"], dtype=float)
    height = solve_stand_height(model, stand_q)

    key = spec.add_key()
    key.name = "stand"
    key.qpos = [0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0, *stand_q]
    key.ctrl = list(stand_q)

    model = spec.compile()
    xml = spec.to_xml()
    if kept:                                   # write the meshdir as a relative path
        xml = xml.replace(str(Path(meshdir).resolve()), "meshes")
    if src_meshdir is not None:
        xml = xml.replace(str(src_meshdir), os.path.relpath(src_meshdir, out.parent.resolve()))
    out.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(out))     # prove the written file loads

    # --- report, and assert the thing actually stands ---
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    z0 = data.qpos[2]
    for _ in range(int(1.0 / model.opt.timestep)):     # 1 s holding the nominal pose
        mujoco.mj_step(model, data)
    drop = z0 - data.qpos[2]

    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
    print(f"  contract   {a.contract}")
    coll = sum(1 for g in range(model.ngeom)
               if model.geom_bodyid[g] != 0 and (model.geom_contype[g] or model.geom_conaffinity[g]))
    print(f"  geoms      {model.ngeom} ({coll} colliding)   "
          f"deleted {deleted} visual meshes, kept {kept} colliding meshes, boxed {boxed}")
    if demoted:
        print(f"  contact    {', '.join(sorted(set(demoted)))}: box kept for volume, "
              f"floor contact left to the vendor's sole spheres")
    print(f"  actuators  {n_act} position, kp/kv from gains.yaml, forcerange from joints.yaml")
    print(f"  sensors    {n_touch} touch sensors on the arm links (self-collision penalty), "
          f"{n_sole} on the soles (foot load)")
    print(f"  timestep   {model.opt.timestep}  integrator {mujoco.mjtIntegrator(model.opt.integrator).name}")
    print(f"  mass       {model.body_mass.sum():.4f} kg")
    print(f"  keyframe   'stand' at base z={height:.4f} m")
    print(f"  1 s hold   base dropped {drop * 1000:+.1f} mm, |qvel|max={np.abs(data.qvel).max():.3f}")
    if drop > 0.05:
        print("  WARNING: it sank more than 5 cm -- gains or keyframe height are wrong")


if __name__ == "__main__":
    main()
