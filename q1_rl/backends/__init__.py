"""Batched-physics backends.

The whole point of this layer is that an environment written against it runs on a
MacBook and on an A6000 without changing. It is deliberately tiny -- five verbs plus
named accessors -- because that is the intersection of what mjbatch and mujoco_warp
both offer imperatively. MJX is not behind this interface and cannot be: its functional
`data = step(model, data)` style would force every environment to be written twice.

    from q1_rl.backends import make_backend
    b = make_backend("mjbatch", model, num_envs=4096)

Backends, measured on the same 22-DoF Q1 model (bench/results.md):

    mjbatch   CPU, any machine        1.0 M substeps/s on 32 threads, 0.5 M on 8
    warp      NVIDIA GPU              2.5 M substeps/s on an A6000, +83 s JIT

mujoco_warp's CPU device exists but runs 116x slower than mjbatch, so it is an
accelerator, never the portable default.
"""

from .base import Backend

_REGISTRY = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def available() -> dict[str, bool]:
    """Which backends can actually run here. Cheap import probes, no device allocation."""
    out = {}
    try:
        import mjbatch  # noqa: F401
        out["mjbatch"] = True
    except ImportError:
        out["mjbatch"] = False
    try:
        import warp
        out["warp"] = warp.get_cuda_device_count() > 0
    except Exception:
        out["warp"] = False
    return out


def make_backend(name: str, model, num_envs: int, **kw) -> Backend:
    if name == "auto":
        av = available()
        name = "warp" if av.get("warp") else "mjbatch"
    if name == "mjbatch":
        from . import mjbatch_backend  # noqa: F401
    elif name == "warp":
        from . import warp_backend  # noqa: F401
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](model, num_envs, **kw)


__all__ = ["Backend", "make_backend", "available", "register"]
