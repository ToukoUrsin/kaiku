"""Demonstrations in, a tested policy out: load, train, then sort scenes nobody has seen.

    python -m kaiku.policy.fit DATA --out DIR [--which kaiku,plain] [--steps 30000] [--scenes 48]

``kaiku`` trains on the operator's demonstrations and their echoes, ``plain`` on the operator's
own alone. Each is saved as ``DIR/<which>.pt`` with its test result inside. Progress goes to
stdout as lines of ``KAIKU {json}``, which the live demo reads (``kaiku.app.trainer``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from kaiku.policy import rollout
from kaiku.policy.dataset import Loader
from kaiku.policy.model import Policy
from kaiku.policy.train import train

# How much of the bar each stage gets.
LOAD, TRAIN = 0.1, 0.7
# Pairs kept in memory for the training: 2.5 million take 6 GB of a GPU.
MOST_PAIRS = 2_500_000


def say(**status) -> None:
    print("KAIKU " + json.dumps(status), flush=True)


def fit(data: Path | str, which: str, out: Path | str, steps: int = 30000, scenes: int = 48, humans: int | None = None,
        seed: int = 0, threads: int | None = None, report=say, eval_seed: int = 100, rounds: int = 1,
        last: int | None = None) -> Policy:
    """Train one policy and test it on ``scenes * rounds`` fresh scenes. ``report(**status)`` hears about it.
    ``humans`` uses the operator's first K episodes, ``last`` the latest K."""
    tell = lambda stage, p, **kw: report(which=which, stage=stage, progress=round(p, 4), **kw)      # noqa: E731
    tell("loading", 0.0)
    loader = Loader()
    data = loader.load(data, humans, echoes=which != "plain", progress=lambda i, n: tell("loading", LOAD * i / n),
                       last=last).sample(MOST_PAIRS, seed)
    policy = train(data, loader.feat, steps, seed=seed, threads=threads or 8, log=None,
                   on_step=lambda i, n, loss: tell("training", LOAD + TRAIN * i / n, loss=round(loss, 5)))
    policy.info.update(which=which, humans=int(data.human_demos), demos=int(data.demos))
    res = rollout.run(policy, scenes, rounds, eval_seed, threads=threads,
                      progress=lambda f: tell("testing", LOAD + TRAIN + (1 - LOAD - TRAIN) * f))
    policy.info.update(success=res["clean_rate"], parts=res["parts_rate"], scenes=res["scenes"],
                       clipped=res["clipped"], aborted=res["aborted"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    policy.save(out)
    tell("done", 1.0, **{k: policy.info[k] for k in ("success", "parts", "scenes", "humans", "demos", "loss")})
    return policy


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="recorder folder")
    ap.add_argument("--out", default="policies", help="folder for <which>.pt")
    ap.add_argument("--which", default="kaiku,plain")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--scenes", type=int, default=48)
    ap.add_argument("--humans", type=int, default=None, help="use the operator's first K episodes")
    ap.add_argument("--last", type=int, default=None, help="use the operator's latest K episodes")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    which = a.which.split(",")
    if len(which) == 1:
        fit(a.data, which[0], Path(a.out) / f"{which[0]}.pt", a.steps, a.scenes, a.humans, a.seed, a.threads,
            last=a.last)
        return
    # Side by side, each in a process of its own.
    procs = [subprocess.Popen([sys.executable, "-u", "-m", "kaiku.policy.fit", a.data, "--out", a.out, "--which", w,
                               "--steps", str(a.steps), "--scenes", str(a.scenes), "--seed", str(a.seed),
                               *(["--humans", str(a.humans)] if a.humans else []),
                               *(["--last", str(a.last)] if a.last else []),
                               *(["--threads", str(a.threads)] if a.threads else [])]) for w in which]
    codes = [p.wait() for p in procs]
    if any(codes):
        say(stage="error", text="Training stopped early. The log above says why.")
        sys.exit(1)


if __name__ == "__main__":
    main()
