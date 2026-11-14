"""The matched baseline policy matrix on Modal: every arm, every seed, the full training budget.

    modal run tools/modal_matrix.py --datasets /bulk3/runs/robotteleop/baseline/policy_datasets \
        --heldout /bulk3/runs/robotteleop/baseline/measurement/heldout96.jsonl --steps 30000 --seeds 0,1,2 \
        --out /bulk3/runs/robotteleop/baseline/policy_runs/final_30000/runs.jsonl

Each job is what ``tools/run_policy_matrix.py`` does for one arm and one seed: train on the arm's
assembled demonstrations, then sort the held-out tables once with nobody driving. One T4 per job.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/kaiku_repo"
G1 = "third_party/mujoco_menagerie/unitree_g1"
ARMS = ("plain", "copy", "mimicgen", "kaiku_matched", "kaiku")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "mujoco==4.0.1", "pyyaml", "torch", "scipy", "mink==1.4.0", "qpsolvers[daqp]",
                 "aiohttp", "websockets", "msgpack")
    .add_local_dir(ROOT / "kaiku", f"{REMOTE}/kaiku", ignore=["**/__pycache__", "web/static/**"])
    .add_local_dir(ROOT / G1, f"{REMOTE}/{G1}")
)
app = modal.App("kaiku-matrix", image=image)
vol = modal.Volume.from_name("kaiku-demos", create_if_missing=True)


@app.function(gpu="T4", cpu=16, memory=32768, timeout=3600, volumes={"/data": vol})
def job(cfg: dict) -> dict:
    import sys
    sys.path.insert(0, REMOTE)
    import numpy as np

    from kaiku.policy.dataset import Loader
    from kaiku.policy.rollout import run
    from kaiku.policy.train import train
    from kaiku.sim.scene import Layout, Placement

    t0 = time.time()
    log = lambda m: print(f"[{cfg['arm']} s{cfg['seed']}] {m}", flush=True)      # noqa: E731
    loader = Loader()
    pairs = loader.load(f"/data/baseline/{cfg['tag']}/{cfg['arm']}", max_episodes=None, echoes=cfg["arm"] != "plain")
    if cfg["arm"] == "plain":
        pairs = pairs.pick(np.unique(pairs.episode), echoes=False)
    policy = train(pairs, loader.feat, steps=cfg["steps"], seed=cfg["seed"], threads=16, log=log)
    held = []
    for line in Path(f"/data/baseline/{cfg['tag']}/heldout.jsonl").read_text().splitlines():
        row = json.loads(line)
        held.append(Layout([Placement(p["name"], np.asarray(p["pos"], float), float(p["yaw"])) for p in row["layout"]],
                           dict(row.get("mapping", {"red": "left", "yellow": "center", "blue": "right"}))))
    res = run(policy, n=len(held), episodes=1, seed=9000, threads=16, layouts=held, log=log)
    out = {"arm": cfg["arm"], "seed": cfg["seed"], "steps": cfg["steps"], "scenes": res["scenes"], "clean": res["clean"],
           "yield": res["clean_rate"], "parts": res["parts"], "parts_yield": res["parts_rate"], "clipped": res["clipped"],
           "aborted": res["aborted"], "pairs": len(pairs.x), "loss": policy.info["loss"],
           "seconds": policy.info["seconds"], "minutes": round((time.time() - t0) / 60, 1), "device": "T4"}
    log(json.dumps(out))
    return out


@app.local_entrypoint()
def main(datasets: str, heldout: str, out: str, steps: int = 30000, seeds: str = "0,1,2", arms: str = ",".join(ARMS),
         tag: str = "m1"):
    src = Path(datasets)
    with vol.batch_upload(force=True) as up:
        for arm in arms.split(","):
            up.put_directory(src / arm, f"/baseline/{tag}/{arm}")
        up.put_file(Path(heldout), f"/baseline/{tag}/heldout.jsonl")
    cfgs = [{"arm": arm, "seed": int(seed), "steps": steps, "tag": tag} for seed in seeds.split(",") for arm in arms.split(",")]
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:                          # written as they come in: a cut run keeps what it has
        for r in job.map(cfgs, order_outputs=False, return_exceptions=True):
            if isinstance(r, Exception):
                print("FAILED", repr(r)[:300], flush=True)
                continue
            print({k: r[k] for k in ("arm", "seed", "clean", "scenes", "yield", "parts_yield", "minutes")}, flush=True)
            f.write(json.dumps(r, sort_keys=True) + "\n")
            f.flush()
    print("->", out_path)
