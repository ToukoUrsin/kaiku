"""Measure one policy arm and write one JSON object per evaluation scene.

Example (the checked-in 96-scene comparison):
    MUJOCO_GL=egl .venv/bin/python tools/measure_perception_policy.py \
      --arm camera-noisy --seed 101 --out /bulk3/runs/robotteleop/perception/camera-noisy-101.jsonl

The JSONL is deliberately separate from the summary JSON so a timed-out run can
still be inspected scene by scene.  ``rollout.run`` owns the simulator and policy
loop; this file only records its result in a reproducible format.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

# Running a file under ``tools/`` puts that directory first on sys.path.  Force
# the checkout containing this driver ahead of any separately installed Kaiku.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaiku.policy.model import Policy
from kaiku.policy.rollout import run


def interval(success: int, total: int) -> list[float]:
    """Exact two-sided 95% Clopper-Pearson interval."""
    if total <= 0:
        return [0.0, 0.0]
    alpha = 0.05
    lo = 0.0 if success == 0 else float(beta.ppf(alpha / 2, success, total - success + 1))
    hi = 1.0 if success == total else float(beta.ppf(1 - alpha / 2, success + 1, total - success))
    return [round(lo, 6), round(hi, 6)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("simulator", "camera-clean", "camera-noisy"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--max-s", type=float, default=80.0)
    a = ap.parse_args(argv)
    perception = None
    depth_noise = edge_noise = 0.0
    if a.arm.startswith("camera"):
        perception = "camera"
    if a.arm == "camera-noisy":
        depth_noise, edge_noise = 3.0, 1.0

    if a.shards > 1:
        a.n = 16 // a.shards
        a.episodes = 6
    scene_offset = a.shard * a.n * a.episodes
    records: list[dict] = []
    result = run(Policy.load("kaiku/policy/pretrained/kaiku.pt"), n=a.n, episodes=a.episodes,
                 seed=a.seed, max_s=a.max_s, threads=None, perception=perception,
                 depth_noise_mm=depth_noise, edge_noise_px=edge_noise, scene_records=records,
                 log=print, scene_offset=scene_offset)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for row in records:
            f.write(json.dumps({"arm": a.arm, **row}, sort_keys=True) + "\n")
    scenes = int(result["scenes"])
    parts = int(result["parts"])
    summary = {
        "arm": a.arm, "seed": a.seed, "scene_offset": scene_offset, "scenes": scenes, "parts": parts,
        "success": int(result["success"]), "sorted": int(result["sorted"]),
        "clipped_commands": int(result["clipped"]),
        "clipped_scenes": int(sum(r["clipped"] > 0 for r in records)),
        "aborted_scenes": int(result["aborted"]),
        "success_rate": round(result["success"] / max(scenes, 1), 6),
        "success_rate_ci95": interval(int(result["success"]), scenes),
        "parts_rate": round(result["sorted"] / max(parts, 1), 6),
        "parts_rate_ci95": interval(int(result["sorted"]), parts),
        "result": result,
    }
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("MEASUREMENT_SUMMARY " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
