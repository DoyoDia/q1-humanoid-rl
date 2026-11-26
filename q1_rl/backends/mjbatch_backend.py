"""The portable default: mjbatch, a C++ thread pool over mj_step with the GIL released.

This is the backend that makes the framework's central claim true -- that a developer
can train a humanoid on whatever machine they have. On the same Q1 model it reaches
1.0 M physics substeps/s on 32 threads and 0.5 M on 8. Those are physics-only ceilings:
a training run also pays for observations, rewards and the learner, and measured
end-to-end throughput is well under half of them -- see bench/results.md.
"""

from __future__ import annotations

import numpy as np

from . import register
from .base import Backend


@register("mjbatch")
class MjBatchBackend(Backend):
    def __init__(self, model, num_envs: int, num_threads: int = 0, expand_fields=(), **kw):
        # expand_fields is accepted and ignored: mjbatch can expand any model field on
        # demand. Only the warp backend needs them declared up front.
        try:
            from mjbatch import Batch
        except ImportError as e:
            raise ImportError("pip install mjbatch  (the default backend)") from e
        self._batch = Batch(model, num_envs, num_threads)
        self.model = model
        self.num_envs = num_envs
        self.num_threads = num_threads or 0

    def bind(self, field: str) -> np.ndarray:
        return self._batch.bind(field)

    def expand(self, field: str) -> np.ndarray:
        return self._batch.expand(field)

    def set_const(self, ids=None) -> None:
        self._batch.set_const() if ids is None else self._batch.set_const(ids)

    def step(self, nstep: int = 1) -> None:
        self._batch.step(nstep=nstep)

    def forward(self, ids=None) -> None:
        self._batch.forward() if ids is None else self._batch.forward(np.asarray(ids))

    def reset(self, ids, keyframe: int = -1) -> None:
        self._batch.reset(np.asarray(ids), keyframe=keyframe)

    def sensor(self, name: str) -> np.ndarray:
        return self._batch.sensor(name)
