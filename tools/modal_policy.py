"""Train and test policies on Modal (a GPU for training, CPU workers for the robots).

    modal volume put kaiku-demos DEMOS_DIR /demos            # once: a recorder folder
    modal run tools/modal_policy.py --humans 1,2,4,8,16,32   # the curve: with and without echoes
    modal run tools/modal_policy.py --humans 8 --only kaiku --name try

Each job trains one policy on the operator's first K episodes (with their echoes, or without)
and rolls it out on fresh scenes. Results come back as JSON lines in ``--out``; the policies
stay on the volume under /policies.

The live demo uses ``fit_remote_stream``: one T4 container per policy, in parallel, with
status events returned to the live trainer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/kaiku_repo"
G1 = "third_party/mujoco_menagerie/unitree_g1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "mujoco==4.0.1", "pyyaml", "torch", "scipy", "mink==1.4.0", "qpsolvers[daqp]",
                 "aiohttp", "websockets", "msgpack")
    .add_local_dir(ROOT / "kaiku", f"{REMOTE}/kaiku", ignore=["**/__pycache__", "web/static/**"])
    .add_local_dir(ROOT / G1, f"{REMOTE}/{G1}")
)
app = modal.App("kaiku-policy", image=image)
vol = modal.Volume.from_name("kaiku-demos", create_if_missing=True)


@app.function(gpu="T4", cpu=16, memory=49152, timeout=3 * 3600, volumes={"/data": vol})
def job(cfg: dict) -> dict:
    import sys
    sys.path.insert(0, REMOTE)
    import numpy as np

    from kaiku.policy import rollout
    from kaiku.policy.dataset import Loader
    from kaiku.policy.train import train

    t0 = time.time()
    log = lambda m: print(f"[{cfg['name']}] {m}", flush=True)      # noqa: E731
    loader = Loader()
    data = loader.load(cfg.get("data") or "/data/demos", cfg["humans"], cfg.get("stride", 1), skip=cfg.get("skip", 0))
    demos, humans = data.demos, data.human_demos
    if cfg["plain"]:
        data = data.pick(np.unique(data.episode), echoes=False)
        demos = humans
    data = data.sample(cfg.get("most", 3_500_000), cfg.get("seed", 0))          # what one T4 holds
    log(f"{len(data.x)} pairs, {demos} demonstrations ({humans} by the operator), loaded in {time.time() - t0:.0f} s")
    policy = train(data, loader.feat, cfg.get("steps", 30000), cfg.get("batch", 512), width=cfg.get("width", 512),
                   depth=cfg.get("depth", 4), sigma=cfg.get("sigma", 0.004), sigma_asked=cfg.get("sigma_asked", 0.002),
                   seed=cfg.get("seed", 0), threads=16, log=log)
    policy.info.update(humans=int(humans), demos=int(demos), plain=bool(cfg["plain"]))
    Path("/data/policies").mkdir(exist_ok=True)
    policy.save(f"/data/policies/{cfg['name']}.pt")
    vol.commit()
    out = rollout.run(policy, cfg.get("n", 48), cfg.get("rounds", 2), cfg.get("eval_seed", 100), cfg.get("parts"),
                      log=log, every=cfg.get("every", 12), blend=cfg.get("blend", 4))
    out.update(name=cfg["name"], seed=cfg.get("seed", 0), every=cfg.get("every", 12), humans=int(humans),
               demos=int(demos), plain=bool(cfg["plain"]),
               loss=policy.info["loss"], pairs=policy.info["pairs"], minutes=round((time.time() - t0) / 60, 1))
    log(json.dumps(out))
    return out


@app.function(cpu=16, memory=16384, timeout=3600, volumes={"/data": vol})
def test(cfg: dict) -> dict:
    """Roll an already trained policy out again (other scenes, other settings)."""
    import sys
    sys.path.insert(0, REMOTE)
    from kaiku.policy import rollout
    from kaiku.policy.model import Policy

    policy = Policy.load(f"/data/policies/{cfg['name']}.pt")
    out = rollout.run(policy, cfg.get("n", 48), cfg.get("rounds", 2), cfg.get("eval_seed", 100), cfg.get("parts"),
                      log=lambda m: print(f"[{cfg['name']}] {m}", flush=True), every=cfg.get("every", 12),
                      blend=cfg.get("blend", 4))
    out.update({k: v for k, v in policy.info.items() if k != "seconds"}, name=cfg["name"],
               every=cfg.get("every", 12), blend=cfg.get("blend", 4))
    return out


@app.function(gpu="T4", cpu=16, memory=32768, timeout=3600, volumes={"/data": vol})
def fit_remote(cfg: dict) -> dict:
    """What ``python -m kaiku.policy.fit`` does, for the live demo's Train button."""
    import sys
    sys.path.insert(0, REMOTE)
    from kaiku.policy.fit import fit

    policy = fit(cfg["data"], cfg["which"], f"/data/policies/{cfg['name']}.pt", cfg["steps"], cfg["scenes"],
                 last=cfg.get("last"))
    vol.commit()
    return policy.info


@app.function(gpu="T4", cpu=16, memory=32768, timeout=1800, volumes={"/data": vol})
def fit_remote_stream(cfg: dict):
    """Train one policy and stream fit stages while the live app remains responsive.

    A generator is used instead of Modal's job logs: the live container can consume both
    generators concurrently and turn their events into the existing ``KAIKU`` HUD protocol.
    ``reload`` is intentionally before loading any rows so a Train click sees the live
    container's preceding volume commit.
    """
    import queue
    import threading

    import sys
    sys.path.insert(0, REMOTE)
    from kaiku.policy.fit import fit

    vol.reload()
    events: queue.Queue[dict] = queue.Queue()
    state: dict = {"which": cfg["which"], "stage": "loading", "progress": 0.0}
    result: dict = {}

    def report(**status):
        state.update(status)
        events.put({"kind": "status", **status})

    def work():
        try:
            policy = fit(cfg["data"], cfg["which"], f"/data/policies/{cfg['name']}.pt",
                         cfg.get("steps", 30000), cfg.get("scenes", 48),
                         threads=cfg.get("threads", 16), train_threads=cfg.get("train_threads", 8),
                         rollout_threads=cfg.get("rollout_threads", 16), last=cfg.get("last"), report=report,
                         loader_workers=cfg.get("loader_workers", 16), loader_context="spawn",
                         compile=cfg.get("compile", False))      # a container that is used again trains very slowly compiled
            result["policy"] = policy
        except Exception as exc:
            result["error"] = repr(exc)[:500]

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive() or not events.empty():
        try:
            yield events.get(timeout=1.0)
        except queue.Empty:
            # Loading and GPU work can both spend a second without a fit callback.
            yield {"kind": "heartbeat", **state}
    worker.join()
    if result.get("error"):
        yield {"kind": "error", "text": result["error"]}
        return
    policy = result["policy"]
    policy_path = Path(f"/data/policies/{cfg['name']}.pt")
    vol.commit()
    yield {"kind": "result", "which": cfg["which"], "info": policy.info,
           "policy": policy_path.read_bytes()}


@app.local_entrypoint()
def live(data: str, out: str, steps: int = 30000, scenes: int = 48, last: int = 48):
    """The demonstrations of a running session -> ``out/kaiku.pt`` and ``out/plain.pt``."""
    say = lambda **m: print("KAIKU " + json.dumps(m), flush=True)       # noqa: E731
    src, dst, name = Path(data), Path(out), f"live_{int(time.time())}"
    rows = [line for line in (src / "episodes.jsonl").read_text().splitlines() if line.strip()]
    files = sorted({json.loads(line)["file"] for line in rows})
    try:
        have = {e.path.rsplit("/", 1)[-1] for e in vol.listdir("/live")}
    except Exception:
        have = set()
    side = dst / "episodes.jsonl"
    dst.mkdir(parents=True, exist_ok=True)
    side.write_text("\n".join(rows) + "\n")
    with vol.batch_upload(force=True) as up:
        for f in files:
            if f not in have:
                up.put_file(src / f, f"/live/{f}")
        up.put_file(side, "/live/episodes.jsonl")
    cfgs = [{"which": w, "name": f"{name}_{w}", "data": "/data/live", "steps": steps, "scenes": scenes, "last": last}
            for w in ("kaiku", "plain")]
    try:
        infos = list(fit_remote.map(cfgs))
        for cfg in cfgs:
            with open(dst / f"{cfg['which']}.pt.part", "wb") as f:
                for chunk in vol.read_file(f"policies/{cfg['name']}.pt"):
                    f.write(chunk)
            (dst / f"{cfg['which']}.pt.part").replace(dst / f"{cfg['which']}.pt")
        print(infos)
    except Exception as e:
        say(stage="error", text=f"Training failed: {repr(e)[:200]}")
        raise


@app.local_entrypoint()
def main(humans: str = "1,2,4,8,16,32", only: str = "", name: str = "curve", steps: int = 30000, rounds: int = 2,
         stride: int = 1, out: str = "", retest: str = "", every: str = "12", parts: int = 0, seeds: str = "0", data: str = "",
         skip: int = 0):
    out_path = Path(out or f"/bulk2/runs/robotteleop/policy/{name}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if retest:
        cfgs = [{"name": n, "rounds": rounds, "every": int(e), "parts": parts or None}
                for n in retest.split(",") for e in every.split(",")]
        fn = test
    else:
        cfgs = []
        for seed in [int(v) for v in seeds.split(",")]:
            for k in [int(v) for v in humans.split(",")]:
                for plain in (True, False):
                    if only and only != ("plain" if plain else "kaiku"):
                        continue
                    cfgs.append({"name": f"{name}_{'plain' if plain else 'kaiku'}_{k}_s{seed}", "humans": k,
                                 "plain": plain, "steps": steps, "rounds": rounds,
                                 # every second tick of the largest sets, so they fit in memory while loading
                                 "stride": stride if plain else max(stride, -(-k // 64)),
                                 "every": int(every.split(",")[0]), "parts": parts or None, "seed": seed,
                                 "data": data, "skip": skip})
        fn = job
    # Written as they come in, so a run that is cut short keeps what it has.
    with open(out_path, "a") as f:
        for r in fn.map(cfgs, order_outputs=False, return_exceptions=True):
            if isinstance(r, Exception):
                print("FAILED", repr(r)[:300], flush=True)
                continue
            print({k: r[k] for k in ("name", "demos", "success_rate", "clean_rate", "parts_rate", "clipped") if k in r},
                  flush=True)
            f.write(json.dumps(r) + "\n")
            f.flush()
    print("->", out_path)
