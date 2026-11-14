"""Measure the policy from the head camera on Modal: the 96 held-out scenes, four shards an arm.

    modal run tools/modal_measure.py --out /bulk3/runs/robotteleop/perception/NAME
    modal run tools/modal_measure.py --arms camera-clean --out DIR

Each shard is ``tools/measure_perception_policy.py`` on a T4 (the cameras are rendered with EGL),
24 scenes of 80 s. The shards' rows come back and are joined here into one JSONL and one summary
per arm, the same files the checked-in results are.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/kaiku_repo"
SHARDS = 4

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libegl1")
    .pip_install("numpy==2.5.4", "mujoco==4.0.1", "pyyaml", "torch", "scipy", "mink==1.4.0", "qpsolvers[daqp]",
                 "aiohttp", "websockets", "msgpack", "pillow")
    .add_local_dir(ROOT / "kaiku", f"{REMOTE}/kaiku", ignore=["**/__pycache__", "web/static/**"])
    .add_local_dir(ROOT / "third_party/mujoco_menagerie/unitree_g1", f"{REMOTE}/third_party/mujoco_menagerie/unitree_g1")
    .add_local_dir(ROOT / "third_party/aaltonomous_safety", f"{REMOTE}/third_party/aaltonomous_safety")
    .add_local_file(ROOT / "tools/measure_perception_policy.py", f"{REMOTE}/tools/measure_perception_policy.py")
)
app = modal.App("kaiku-measure", image=image)


@app.function(gpu="T4", cpu=8, memory=16384, timeout=2 * 3600)
def shard(arm: str, index: int, seed: int, max_s: float, policy: str | None) -> dict:
    import os
    import subprocess
    import sys
    import time
    out = Path(f"/tmp/{arm}-{seed}-s{index}.jsonl")
    cmd = [sys.executable, "tools/measure_perception_policy.py", "--arm", arm, "--seed", str(seed), "--out", str(out),
           "--shard", str(index), "--shards", str(SHARDS), "--max-s", str(max_s)]
    if policy:
        cmd += ["--policy", policy]
    began = time.time()
    done = subprocess.run(cmd, cwd=REMOTE, env={**os.environ, "MUJOCO_GL": "egl"}, capture_output=True, text=True)
    return {"arm": arm, "shard": index, "code": done.returncode, "seconds": round(time.time() - began),
            "rows": out.read_text() if out.exists() else "",
            "summary": json.loads(out.with_suffix(".summary.json").read_text()) if out.with_suffix(".summary.json").exists() else None,
            "tail": (done.stdout[-1500:] + "\n" + done.stderr[-3000:])}


@app.local_entrypoint()
def main(out: str, arms: str = "camera-clean,camera-noisy", seed: int = 100, max_s: float = 80.0, policy: str = ""):
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [(arm, i, seed, max_s, policy or None) for arm in arms.split(",") for i in range(SHARDS)]
    got: dict[str, list[dict]] = {}
    for r in shard.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:400])
            continue
        print(f"{r['arm']} shard {r['shard']}: exit {r['code']}, {r['seconds']} s,",
              (r["summary"] or {}).get("success"), "scenes,", (r["summary"] or {}).get("sorted"), "parts")
        if r["code"] != 0 or not r["summary"]:
            print(r["tail"])
            continue
        (folder / f"{r['arm']}-{seed}-s{r['shard']}.jsonl").write_text(r["rows"])
        (folder / f"{r['arm']}-{seed}-s{r['shard']}.summary.json").write_text(json.dumps(r["summary"], indent=2, sort_keys=True) + "\n")
        got.setdefault(r["arm"], []).append(r)
    for arm, parts in got.items():
        parts.sort(key=lambda r: r["shard"])
        total = {k: sum(int(r["summary"][k]) for r in parts) for k in ("scenes", "parts", "success", "sorted", "clipped_commands",
                                                                       "clipped_scenes", "aborted_scenes")}
        total.update(arm=arm, seed=seed, sim_seconds_cap=max_s, shards=[r["shard"] for r in parts], policy=policy or "default",
                     success_rate=round(total["success"] / max(total["scenes"], 1), 6),
                     parts_rate=round(total["sorted"] / max(total["parts"], 1), 6))
        (folder / f"{arm}-{seed}.jsonl").write_text("".join(r["rows"] for r in parts))
        (folder / f"{arm}-{seed}.summary.json").write_text(json.dumps(total, indent=2, sort_keys=True) + "\n")
        print("TOTAL", json.dumps(total, sort_keys=True))
