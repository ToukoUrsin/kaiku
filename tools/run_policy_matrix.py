"""Train and evaluate one matched policy arm on a held-out layout manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kaiku.policy.dataset import Loader
from kaiku.policy.features import Featurizer
from kaiku.policy.model import Policy
from kaiku.policy.rollout import run
from kaiku.policy.train import train
from kaiku.sim.scene import Layout, Placement, Scene


def layouts_from(path: Path) -> list[Layout]:
    out = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        out.append(Layout([Placement(p["name"], np.asarray(p["pos"], float), float(p["yaw"])) for p in row["layout"]],
                          dict(row.get("mapping", {"red": "left", "yellow": "center", "blue": "right"}))))
    return out


def make_layouts(path: Path, n: int, seed: int) -> None:
    sc, rng = Scene(), np.random.default_rng(seed)
    with path.open("w") as f:
        for _ in range(n):
            lay = sc.sample_layout(rng)
            f.write(json.dumps({"layout": [{"name": p.name, "pos": [float(x) for x in p.pos], "yaw": float(p.yaw)}
                                for p in lay.placements], "mapping": lay.mapping}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path); ap.add_argument("out", type=Path); ap.add_argument("--arm", required=True)
    ap.add_argument("--heldout", type=Path, required=True); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    loader = Loader(); pairs = loader.load(a.data, max_episodes=None, echoes=a.arm != "plain", workers=1, context="fork", orphans=True)
    if a.arm == "plain":
        pairs = pairs.pick(np.unique(pairs.episode), echoes=False)
    policy = train(pairs, loader.feat, steps=a.steps, seed=a.seed, threads=a.threads, log=None, every=a.steps)
    held = layouts_from(a.heldout)
    res = run(policy, n=len(held), episodes=1, seed=9000, threads=a.threads, layouts=held)
    row = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "scenes": res["scenes"], "clean": res["clean"],
           "yield": res["clean_rate"], "parts": res["parts"], "parts_yield": res["parts_rate"],
           "clipped": res["clipped"], "aborted": res["aborted"], "pairs": len(pairs.x), "loss": policy.info["loss"],
           "seconds": policy.info["seconds"]}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(row, sort_keys=True) + "\n"); print(row)


if __name__ == "__main__":
    main()
