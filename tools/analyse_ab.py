#!/usr/bin/env python3
"""Score an ab_algos.py result on deterministic evaluation, across seeds.

Reads the {algo/seedN: {train, eval}} file ab_algos.py writes. Everything reported here
comes from evaluate() -- the mean action on a fixed-seed environment, no exploration noise --
because training reward carries each algorithm's own noise level. An earlier version of
this comparison scored training reward from one seed; see bench/results.md for what it
concluded and how that held up.

    python3 tools/analyse_ab.py runs/ab_algos.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def first_time(evals, key, at_most):
    return next((e["t"] for e in evals if e[key] == e[key] and e[key] <= at_most), None)


def spread(xs, fmt):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "not reached"
    if len(xs) == 1:
        return fmt.format(xs[0])
    return f"{fmt.format(np.median(xs))} (min {fmt.format(min(xs))}, max {fmt.format(max(xs))})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, nargs="?", default=Path("runs/ab_algos.json"))
    a = p.parse_args()
    data = json.loads(a.path.read_text(encoding="utf-8"))

    by_algo = defaultdict(list)
    for key, run in data.items():
        algo, seed = key.split("/")
        by_algo[algo].append((seed, run))

    print(f"{a.path}\n")
    for algo, runs in by_algo.items():
        seeds = [s for s, _ in runs]
        final = [r["eval"][-1] for _, r in runs]
        print(f"{algo.upper()}  seeds {', '.join(seeds)}  "
              f"({runs[0][1]['train'][-1]['t']:.0f}s training each, evaluation time excluded)")
        print(f"  final eval   fall rate     {spread([100*f['eval_fall_rate'] for f in final], '{:.1f}%')}")
        print(f"               joint RMSE    {spread([f['eval_joint_rmse'] for f in final], '{:.4f}')} rad")
        print(f"               reward/step   {spread([f['eval_reward'] for f in final], '{:.3f}')}")
        best = [min(r["eval"], key=lambda e: e["eval_fall_rate"]) for _, r in runs]
        print(f"  best eval    fall rate     {spread([100*b['eval_fall_rate'] for b in best], '{:.1f}%')}")
        for lim in (0.50, 0.20, 0.05):
            ts = [first_time(r["eval"], "eval_fall_rate", lim) for _, r in runs]
            reached = sum(t is not None for t in ts)
            print(f"  eval fall <= {100*lim:>3.0f}%  {spread(ts, '{:.0f}s')}   [{reached}/{len(ts)} seeds]")
        steps = [r["train"][-1]["steps"] for _, r in runs]
        print(f"  env steps    {spread(steps, '{:,.0f}')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
