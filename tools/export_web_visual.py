#!/usr/bin/env python3
"""Export the robot's visual meshes for the browser viewer (web/sim2sim_view.html).

    python3 tools/export_web_visual.py          -> web/assets/q1_visual.json + q1_visual.bin

The physics model the browser simulates (q1_web.xml) has only collision boxes and cylinders.
The viewer draws the visual model's meshes instead, each attached to its body: body poses come
from the simulation, and these files give each mesh geom its body, its pose in that body and
its colour. Vertices and faces are taken from the COMPILED visual model, so they are already in
the frame MuJoCo uses for the geom (it re-centres meshes when it loads them) -- reading the STL
files directly would put every part a few centimetres off.

The .bin holds, mesh after mesh, float32 vertices (x y z) then uint32 faces (a b c); the json
holds the offsets.
"""

import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    m = mujoco.MjModel.from_xml_path(str(ROOT / "robots/q1/q1_visual.xml"))
    out = ROOT / "web" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    blob, meshes = bytearray(), []
    for k in range(m.nmesh):
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(np.float32)
        f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]].astype(np.uint32)
        meshes.append(dict(name=m.mesh(k).name, vert_offset=len(blob), n_vert=len(v)))
        blob += v.tobytes()
        meshes[-1].update(face_offset=len(blob), n_face=len(f))
        blob += f.tobytes()
    geoms = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        geoms.append(dict(body=m.body(m.geom_bodyid[g]).name, mesh=int(m.geom_dataid[g]),
                          pos=m.geom_pos[g].round(6).tolist(), quat=m.geom_quat[g].round(6).tolist(),
                          rgba=m.geom_rgba[g].round(3).tolist()))
    (out / "q1_visual.bin").write_bytes(bytes(blob))
    (out / "q1_visual.json").write_text(json.dumps(dict(meshes=meshes, geoms=geoms)))
    print(f"wrote {out}/q1_visual.json ({len(geoms)} geoms, {len(meshes)} meshes) and "
          f"q1_visual.bin ({len(blob) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
