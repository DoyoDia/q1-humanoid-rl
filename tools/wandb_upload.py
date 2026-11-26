"""Upload a finished (or running) train.py run's history.json to Weights & Biases.

    python tools/wandb_upload.py runs/walk1_30s_ppo_gpu [--name ...] [--note "..."]

For runs started without --wandb. Logs every recorded iteration under the same keys
train.py --wandb uses, so live and uploaded runs plot together in the q1_rl project.
history.json is written every --save-every iterations and at the end, so a running job
uploads up to its last checkpoint -- and with --follow keeps uploading each new checkpoint
until the run writes its policy.onnx (train.py's last step).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q1_rl import tracking  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--name", default=None, help="default: the run folder's name")
    p.add_argument("--project", default=tracking.DEFAULT_PROJECT)
    p.add_argument("--entity", default=None)
    p.add_argument("--note", default=None)
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--follow", action="store_true", help="keep uploading new checkpoints until the run finishes")
    p.add_argument("--poll", type=float, default=30.0, help="seconds between checks with --follow")
    a = p.parse_args()

    path = a.run_dir / "history.json"
    started = time.time()
    run = tracking.start(a.name or a.run_dir.name, dict(run_dir=str(a.run_dir), uploaded_from="history.json"),
                         project=a.project, entity=a.entity, tags=["uploaded", *a.tag], notes=a.note)
    print(f"-> {run.url}")
    sent = 0
    while True:
        hist = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        for s in hist[sent:]:
            tracking.log(run, s)
        if len(hist) > sent:
            sent = len(hist)
            run.summary["iters"] = hist[-1]["iter"]
            print(f"uploaded through iteration {hist[-1]['iter']}", flush=True)
        onnx = a.run_dir / "policy.onnx"
        finished = onnx.exists() and onnx.stat().st_mtime >= started - 1 and path.stat().st_mtime >= onnx.stat().st_mtime - 60
        if not a.follow or finished:
            break
        time.sleep(a.poll)
    run.finish()


if __name__ == "__main__":
    main()
