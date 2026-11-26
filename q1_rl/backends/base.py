"""The five verbs an environment is allowed to use."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """A batch of `num_envs` identical simulations, stepped together.

    Arrays returned by bind()/expand() are LIVE views of shape (num_envs, ...): write to
    them and the simulations see it, step and they update in place. No copies, no
    functional threading of state. Environments must not assume they are NumPy arrays
    beyond basic indexing and arithmetic -- that is what keeps a GPU backend possible.
    """

    model: object          # the template MjModel: names, ids, sizes
    num_envs: int

    # -- state -------------------------------------------------------------

    @abstractmethod
    def bind(self, field: str) -> np.ndarray:
        """A live (num_envs, ...) view of an MjData field: qpos, qvel, ctrl, ..."""

    @abstractmethod
    def expand(self, field: str) -> np.ndarray:
        """A live (num_envs, ...) view of a per-env MjModel field, for domain
        randomisation. Call set_const() after writing fields that have derived constants."""

    def set_const(self, ids=None) -> None:
        """Recompute constants derived from expand()ed model fields."""

    # -- time --------------------------------------------------------------

    @abstractmethod
    def step(self, nstep: int = 1) -> None:
        """Advance every simulation by nstep physics substeps."""

    @abstractmethod
    def forward(self, ids=None) -> None:
        """Recompute derived quantities (sensors, xpos, ...) without advancing time."""

    @abstractmethod
    def reset(self, ids, keyframe: int = -1) -> None:
        """Reset the given env indices, optionally to a keyframe. Call before writing
        the state you actually want, since a reset discards writes made before it."""

    def kinematics(self, ids=None) -> None:
        """Body and geom poses only (xpos, xquat, geom_xpos) -- enough to place a freshly
        reset robot. Backends where a full forward() is expensive override this."""
        self.forward(ids)

    def step_scheduled(self, target: np.ndarray, switch_at: np.ndarray, snapshot_at: np.ndarray,
                       nstep: int, snap_qpos: np.ndarray, snap_qvel: np.ndarray) -> None:
        """nstep substeps in which world w's ctrl becomes target[w] after switch_at[w]
        substeps (the previous ctrl holds until then), and its qpos/qvel are copied into
        snap_qpos[w]/snap_qvel[w] after snapshot_at[w] substeps. Actuation and sensing
        latency, per world.

        This default splits the period at the distinct event times: a few step() calls.
        A GPU backend overrides it to run the whole period as one captured graph."""
        ctrl, qpos, qvel = self.bind("ctrl"), self.bind("qpos"), self.bind("qvel")
        events = sorted(set(switch_at.tolist()) | set(snapshot_at.tolist()) | {nstep})
        s = 0
        for e in events:
            if e > s:
                self.step(nstep=e - s)
                s = e
            m = switch_at == e
            if m.any():
                ctrl[m] = target[m]
            m = snapshot_at == e
            if m.any():
                snap_qpos[m] = qpos[m]
                snap_qvel[m] = qvel[m]

    # -- named accessors ---------------------------------------------------

    def sensor(self, name: str) -> np.ndarray:
        s = self.model.sensor(name)
        return self.bind("sensordata")[:, s.adr[0]: s.adr[0] + s.dim[0]]

    def body_xpos(self, name_or_id) -> np.ndarray:
        i = self.model.body(name_or_id).id
        return self.bind("xpos")[:, i]

    def body_xquat(self, name_or_id) -> np.ndarray:
        i = self.model.body(name_or_id).id
        return self.bind("xquat")[:, i]

    def site_xpos(self, name: str) -> np.ndarray:
        return self.bind("site_xpos")[:, self.model.site(name).id]

    # -- introspection -----------------------------------------------------

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_envs={self.num_envs}, nq={self.model.nq}, nu={self.model.nu})"
