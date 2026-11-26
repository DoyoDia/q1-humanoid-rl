"""Where does a geom actually touch the ground? One implementation, used everywhere.

Both the model builder (to place the `stand` keyframe) and the reference-motion generator
(to put each frame's root at the right height) need the lowest point of the robot. They
used to compute it separately: the builder exactly, the motion generator with
geom_rbound, the bounding-sphere radius. Once the training model kept the vendor's foot
collision meshes, that bounding sphere put every generated reference root 81 mm above
where the standing robot's root physically sits -- so the root-height reward chased a
height the robot could only reach by leaving the ground.
"""

from __future__ import annotations

import mujoco
import numpy as np


def touches_floor(model, g: int) -> bool:
    """Can geom g ever contact the floor?

    The Q1 floor is contype=1 conaffinity=0, so a pair collides iff the other geom's
    conaffinity is non-zero. `contype or conaffinity` would also admit geoms that exist
    only for robot-vs-robot volume.
    """
    return model.geom_bodyid[g] != 0 and bool(model.geom_conaffinity[g])


def geom_lowest_z(model, data, g: int) -> float:
    """Exact lowest world-z of a geom, per shape.

    geom_rbound would be one line, but for a box it is the half-diagonal and for a mesh the
    radius of a sphere around the whole mesh, both of which lift the robot centimetres off
    its real contact points.
    """
    t = model.geom_type[g]
    z = data.geom_xpos[g][2]
    R = data.geom_xmat[g].reshape(3, 3)
    s = model.geom_size[g]
    down = R[2]                                   # world z row: how each local axis projects up
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return z - s[0]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return z - float(np.abs(down) @ s[:3])
    if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        radial = float(np.hypot(down[0], down[1]))         # local x/y extent -> radius
        axial = float(abs(down[2]))                        # local z extent -> half length
        cap = s[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
        return z - (radial * s[0] + axial * s[1] + cap)
    if t == mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[g] >= 0:
        mid = model.geom_dataid[g]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        v = model.mesh_vert[adr: adr + num]
        return float((data.geom_xpos[g] + v @ R.T)[:, 2].min())
    return z - float(model.geom_rbound[g])                 # conservative fallback


def lowest_contact_z(model, data) -> float:
    """Lowest point of anything on the robot that can touch the floor, at data's pose."""
    return min(geom_lowest_z(model, data, g) for g in range(model.ngeom) if touches_floor(model, g))


def solve_stand_height(model, qpos_joints, quat=(1.0, 0.0, 0.0, 0.0), clearance=0.001) -> float:
    """Base height that puts the lowest floor-touching point `clearance` above z=0."""
    data = mujoco.MjData(model)
    data.qpos[3:7] = quat
    data.qpos[7:] = qpos_joints
    data.qpos[2] = 1.0                      # start high, no contact
    mujoco.mj_forward(model, data)
    return 1.0 - lowest_contact_z(model, data) + clearance
