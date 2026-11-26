"""Optional experiment tracking (Weights & Biases).

Off unless asked for: training must keep working on a machine with no wandb installed and
no network. One place decides how a run is named and what goes into its config, so live
logging (train.py --wandb) and after-the-fact upload (tools/wandb_upload.py) produce runs
that line up in the same project.
"""

from __future__ import annotations

import dataclasses
import math

DEFAULT_PROJECT = "q1_rl"


def _plain(x):
    if dataclasses.is_dataclass(x):
        return {k: _plain(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def start(name: str, config: dict, project: str = DEFAULT_PROJECT, entity: str | None = None,
          tags: list[str] | None = None, notes: str | None = None):
    import wandb
    return wandb.init(project=project, entity=entity, name=name, config=_plain(config),
                      tags=tags, notes=notes)


def log(run, stats: dict) -> None:
    """One training iteration. NaN (an iteration with no finished episode) is dropped
    rather than plotted as a gap-filling zero."""
    row = {}
    for k, v in stats.items():
        if k == "iter" or not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
            continue
        group = "train" if k in ("reward", "fall_rate", "ep_len") else \
                "perf" if k in ("seconds", "steps", "steps_per_s") else "reward_terms"
        row[f"{group}/{k}"] = v
    run.log(row, step=int(stats["iter"]))
