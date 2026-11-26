"""The NVIDIA accelerator: mujoco_warp behind the same five verbs.

This backend exists for the developer who has a 4090, not as the default. mjbatch stays
the default because it runs everywhere; warp's raw physics is ~2.2x the 32-thread
mjbatch figure on this robot on a workstation GPU, and the gap is small enough that requiring a GPU was never justified.

One structural difference the interface has to absorb: mjbatch's bind() hands back a live
NumPy view of the simulation, whereas warp's state lives in device memory and NumPy cannot
view it. So bind() here returns a HOST MIRROR, and the backend moves data across the bus
around step()/forward()/reset(). Environments see the same semantics; they just pay a
copy. For the Q1 at 4096 worlds that is ~1.3 MB each way per policy step, which is far
below what PCIe cares about, and the alternative -- rewriting every environment against a
device array namespace -- would cost the portability this whole layer is for.

What goes up, and when. NumPy cannot tell us that a mirror was written, so:

  * every bound *writable* Data field (qpos, qvel, ctrl, ...) is uploaded before every
    step/forward. They are small, and an environment may write them at any moment --
    a command, a push, a reset.
  * expand()ed Model fields are large (~10 MB at 4096 worlds) and change only at resets,
    so each keeps a shadow copy of what was last uploaded and goes up only when it differs.

An earlier version marked a field dirty once, in bind(), and cleared the mark on the first
upload. Every write after that was silently dropped unless a reset happened to re-mark the
field in the same step -- a commanded +0.6 rad knee moved 0.012 rad, and randomised
friction never reached the device. Frequent resets in large batches hid it.

Per-world model parameters (domain randomisation) must be declared up front, because
mujoco_warp decides field batching inside put_model():

    make_backend("warp", model, n, expand_fields=["geom_friction", "actuator_gainprm"])
"""

from __future__ import annotations

import numpy as np

from . import register
from .base import Backend

_KERNELS = None


def _kernels(wp):
    """The two per-world kernels the scheduled control period needs, built on first use
    (warp is imported lazily so that a machine without it can still load this package)."""
    global _KERNELS
    if _KERNELS is None:
        @wp.kernel
        def copy_rows_at(src: wp.array2d(dtype=wp.float32), dst: wp.array2d(dtype=wp.float32),
                         at: wp.array(dtype=wp.int32), k: int):
            w, j = wp.tid()
            if at[w] == k:
                dst[w, j] = src[w, j]
        _KERNELS = copy_rows_at
    return _KERNELS

# Fields a caller may write; everything else is treated as read-only output.
_WRITABLE = frozenset({"qpos", "qvel", "ctrl", "act", "qacc_warmstart", "mocap_pos", "mocap_quat"})


@register("warp")
class WarpBackend(Backend):
    def __init__(self, model, num_envs: int, expand_fields=(), device: str | None = None,
                 capture: bool = True, njmax: int | None = None, nconmax: int | None = None,
                 **kw):
        try:
            import mujoco
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as e:
            raise ImportError(
                "the warp backend needs mujoco_warp:\n"
                "  pip install 'mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp.git'"
            ) from e
        self._mjw, self._wp, self._mujoco = mjw, wp, mujoco

        self.device = device or ("cuda:0" if wp.get_cuda_device_count() else "cpu")
        if self.device == "cpu":
            raise RuntimeError(
                "mujoco_warp on the CPU device runs ~116x slower than mjbatch on this "
                "robot (measured, see bench/results.md). Use the mjbatch backend instead.")

        self.model = model
        self.num_envs = num_envs
        self._expand_fields = tuple(expand_fields)

        host = mujoco.MjData(model)
        mujoco.mj_forward(model, host)
        # put_data defaults to njmax=64 constraint rows per world. A 22-DoF humanoid with
        # two feet on the ground blows past that the moment it lands on more than a couple
        # of contact points, and mujoco_warp then DROPS constraints -- it prints
        # "nefc overflow" once and carries on with wrong physics. Size it from the model.
        self._njmax = njmax or max(256, 12 * model.ngeom)
        self._nconmax = nconmax or max(128, 4 * model.ngeom)
        with wp.ScopedDevice(self.device):
            self._m = mjw.put_model(model, batch_sizes={f: num_envs for f in self._expand_fields})
            self._d = mjw.put_data(model, host, nworld=num_envs,
                                   njmax=self._njmax, nconmax=self._nconmax)

        self._mirror: dict[str, np.ndarray] = {}     # field -> host array
        self._src: dict[str, object] = {}            # field -> the warp array it mirrors
        self._data_fields: set[str] = set()          # mirrors the sim writes back
        self._writable: set[str] = set()             # Data mirrors uploaded every time
        self._shadow: dict[str, np.ndarray] = {}     # Model field -> last uploaded copy
        self._graphs: dict[int, object] = {}     # nstep -> captured CUDA graph
        self._forward_graph = None
        self._kinematics_graph = None
        self._reset_graph = None
        self._reset_mask = None
        self._sched = None                       # device buffers + graphs for step_scheduled
        self._capture = capture

    # -- host mirrors ------------------------------------------------------

    def _mirror_of(self, owner, field: str, is_data: bool) -> np.ndarray:
        if field not in self._mirror:
            arr = getattr(owner, field, None)
            if arr is None:
                raise KeyError(f"{type(owner).__name__} has no field {field!r}")
            self._src[field] = arr
            self._mirror[field] = np.ascontiguousarray(arr.numpy())
            if is_data:
                self._data_fields.add(field)
        return self._mirror[field]

    def _upload(self) -> None:
        for f in self._writable:
            self._src[f].assign(np.ascontiguousarray(self._mirror[f]))
        for f, last in self._shadow.items():
            cur = self._mirror[f]
            if not np.array_equal(cur, last):
                self._src[f].assign(np.ascontiguousarray(cur))
                np.copyto(last, cur)

    def _download(self) -> None:
        # Only Data fields come back. Model fields (the expand() ones) are parameters the
        # simulation never writes; pulling them every step moved ~10 MB per policy step on
        # this robot and made the GPU backend slower than the CPU one.
        for f in self._data_fields:
            np.copyto(self._mirror[f], self._src[f].numpy())

    def bind(self, field: str) -> np.ndarray:
        m = self._mirror_of(self._d, field, is_data=True)
        if field in _WRITABLE:
            self._writable.add(field)
        return m

    def expand(self, field: str) -> np.ndarray:
        if field not in self._expand_fields:
            raise KeyError(
                f"{field!r} was not declared per-world. mujoco_warp fixes field batching "
                f"inside put_model(), so pass it at construction:\n"
                f"  make_backend('warp', model, n, expand_fields=[..., {field!r}])")
        m = self._mirror_of(self._m, field, is_data=False)
        if field not in self._shadow:
            # Seed the shadow with something that cannot match, so the first upload after
            # the caller's first write always happens.
            self._shadow[field] = np.full_like(m, np.nan)
        return m

    def set_const(self, ids=None) -> None:
        with self._wp.ScopedDevice(self.device):
            self._upload()              # warp recomputes what it needs inside step()

    # -- time ---------------------------------------------------------------

    def _run(self, nstep: int) -> None:
        mjw, wp = self._mjw, self._wp
        if not self._capture or self.device == "cpu":
            for _ in range(nstep):
                mjw.step(self._m, self._d)
            return
        # One graph per substep count: an env with latency splits each control period into
        # a few calls of different lengths, and recapturing on every change cost more than
        # the physics. Kernels must be compiled before the first capture, so the very first
        # call runs uncaptured -- those ARE this call's substeps. (The warm-up used to be
        # extra substeps on top, so the first step after a new nstep advanced time twice.)
        graph = self._graphs.get(nstep)
        if graph is None:
            warm = not self._graphs
            if warm:
                for _ in range(nstep):
                    mjw.step(self._m, self._d)
            with wp.ScopedCapture() as cap:     # records, does not execute
                for _ in range(nstep):
                    mjw.step(self._m, self._d)
            graph = self._graphs[nstep] = cap.graph
            if warm:
                return
        wp.capture_launch(graph)

    def step(self, nstep: int = 1) -> None:
        with self._wp.ScopedDevice(self.device):
            self._upload()
            self._run(nstep)
            self._wp.synchronize()
            self._download()

    def forward(self, ids=None) -> None:
        # Captured too. Run eagerly, mujoco_warp's collision pass allocates its scratch
        # (~1.9 GB at 8192 worlds) on every call, so one forward took 420 ms -- and every
        # env reset does one. forward() does not advance time, so the eager warm-up is free.
        mjw, wp = self._mjw, self._wp
        with wp.ScopedDevice(self.device):
            self._upload()
            if not self._capture or self.device == "cpu":
                mjw.forward(self._m, self._d)
            else:
                if self._forward_graph is None:
                    mjw.forward(self._m, self._d)
                    with wp.ScopedCapture() as cap:
                        mjw.forward(self._m, self._d)
                    self._forward_graph = cap.graph
                wp.capture_launch(self._forward_graph)
            wp.synchronize()
            self._download()

    def kinematics(self, ids=None) -> None:
        # A reset only needs poses to put the robot on the floor; skipping collision and
        # the constraint solve makes it several times cheaper than forward(). Captured for
        # the same reason forward() is.
        mjw, wp = self._mjw, self._wp
        with wp.ScopedDevice(self.device):
            self._upload()
            if not self._capture:
                mjw.kinematics(self._m, self._d)
            else:
                if self._kinematics_graph is None:
                    mjw.kinematics(self._m, self._d)
                    with wp.ScopedCapture() as cap:
                        mjw.kinematics(self._m, self._d)
                    self._kinematics_graph = cap.graph
                wp.capture_launch(self._kinematics_graph)
            wp.synchronize()
            self._download()

    def step_scheduled(self, target, switch_at, snapshot_at, nstep, snap_qpos, snap_qvel) -> None:
        """The whole control period as one graph: per world, a kernel swaps ctrl to the new
        target after its delay and another copies qpos/qvel at its snapshot time, between
        the substeps. One upload, one launch, one download -- splitting the period into
        separate step() calls instead cost ~45 ms per period at 8192 worlds."""
        wp, mjw = self._wp, self._mjw
        if not self._capture:
            return super().step_scheduled(target, switch_at, snapshot_at, nstep, snap_qpos, snap_qvel)
        copy_rows_at = _kernels(wp)
        n = self.num_envs
        with wp.ScopedDevice(self.device):
            if self._sched is None:
                f32 = wp.float32
                self._sched = dict(
                    target=wp.zeros((n, self.model.nu), dtype=f32),
                    switch=wp.zeros(n, dtype=wp.int32), snap=wp.zeros(n, dtype=wp.int32),
                    qpos=wp.zeros((n, self.model.nq), dtype=f32),
                    qvel=wp.zeros((n, self.model.nv), dtype=f32), graphs={})
            S = self._sched
            for f in ("qpos", "qvel", "ctrl"):          # ensure the mirrors exist and go up
                self.bind(f)
            S["target"].assign(np.ascontiguousarray(target, dtype=np.float32))
            S["switch"].assign(np.ascontiguousarray(switch_at, dtype=np.int32))
            S["snap"].assign(np.ascontiguousarray(snapshot_at, dtype=np.int32))
            self._upload()

            def period():
                d = self._d
                for k in range(nstep + 1):
                    wp.launch(copy_rows_at, dim=(n, self.model.nu), inputs=[S["target"], d.ctrl, S["switch"], k])
                    wp.launch(copy_rows_at, dim=(n, self.model.nq), inputs=[d.qpos, S["qpos"], S["snap"], k])
                    wp.launch(copy_rows_at, dim=(n, self.model.nv), inputs=[d.qvel, S["qvel"], S["snap"], k])
                    if k < nstep:
                        mjw.step(self._m, d)

            graph = S["graphs"].get(nstep)
            if graph is None:
                # Modules must be loaded before a capture: launch the copy kernel once as a
                # no-op (k=-1 matches no world), and run the very first period eagerly unless
                # a step() graph already proved mjw.step's kernels are compiled.
                wp.launch(copy_rows_at, dim=(n, 1), inputs=[S["qpos"], S["qpos"], S["snap"], -1])
                compiled = bool(self._graphs) or bool(S["graphs"])
                if not compiled:
                    period()                  # first call: compiles kernels; these ARE the substeps
                with wp.ScopedCapture() as cap:
                    period()
                graph = S["graphs"][nstep] = cap.graph
                if not compiled:
                    wp.synchronize()
                    self._download()
                    np.copyto(snap_qpos, S["qpos"].numpy())
                    np.copyto(snap_qvel, S["qvel"].numpy())
                    return
            wp.capture_launch(graph)
            wp.synchronize()
            self._download()
            np.copyto(snap_qpos, S["qpos"].numpy())
            np.copyto(snap_qvel, S["qvel"].numpy())

    def reset(self, ids, keyframe: int = -1) -> None:
        wp, mjw, model = self._wp, self._mjw, self.model
        ids = np.asarray(ids, np.int64)
        if ids.size == 0:
            return
        with wp.ScopedDevice(self.device):
            self._upload()
            mask = np.zeros(self.num_envs, np.bool_)
            mask[ids] = True
            # Captured, with a mask that lives on the device: reset_data is a dozen small
            # kernels, and launched eagerly with a fresh mask array each time it cost
            # ~10 ms -- paid on almost every policy step at 8192 worlds.
            if not self._capture:
                mjw.reset_data(self._m, self._d, reset=wp.array(mask, dtype=wp.bool))
            else:
                if self._reset_graph is None:
                    self._reset_mask = wp.array(mask, dtype=wp.bool)
                    mjw.reset_data(self._m, self._d, reset=self._reset_mask)
                    with wp.ScopedCapture() as cap:
                        mjw.reset_data(self._m, self._d, reset=self._reset_mask)
                    self._reset_graph = cap.graph
                else:
                    self._reset_mask.assign(mask)
                    wp.capture_launch(self._reset_graph)
            wp.synchronize()
            self._download()
        if keyframe >= 0:
            for f, src in (("qpos", model.key_qpos), ("qvel", model.key_qvel), ("ctrl", model.key_ctrl)):
                if src is None or (f == "ctrl" and not model.nu):
                    continue
                self._mirror_of(self._d, f, is_data=True)[ids] = src[keyframe]
                self._writable.add(f)

    def sensor(self, name: str) -> np.ndarray:
        s = self.model.sensor(name)
        return self.bind("sensordata")[:, s.adr[0]: s.adr[0] + s.dim[0]]

    def __repr__(self) -> str:
        return (f"WarpBackend(num_envs={self.num_envs}, device={self.device}, "
                f"nq={self.model.nq}, nu={self.model.nu}, "
                f"njmax={self._njmax}, nconmax={self._nconmax})")
