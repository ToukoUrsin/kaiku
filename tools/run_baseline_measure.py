"""Make a shared target manifest and run the three offline generation arms.

Examples (the input folder is the recorder snapshot):

    python tools/run_baseline_measure.py manifest /bulk2/.../demos_snap /bulk3/.../targets.jsonl --scenes 16 --tables 47
    python tools/run_baseline_measure.py run /bulk2/.../demos_snap /bulk3/.../targets.jsonl /bulk3/.../mimicgen --method mimicgen

The manifest is written once and passed to every method, so the denominators and target
tables are matched.  ``run`` resumes at completed source episodes and writes one attempt
JSONL plus one summary JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from kaiku.policy.baseline import generate_batch, load_source, target_layout
from kaiku.sim.scene import Layout, Placement, Scene


def human_rows(root: Path) -> list[dict]:
    rows = [json.loads(x) for x in (root / "episodes.jsonl").read_text().splitlines() if x.strip()]
    out, seen = [], set()
    for row in rows:
        if row.get("source") == "human" and row["episode"] not in seen:
            out.append(row)
            seen.add(row["episode"])
    return out


def serial_layout(layout: Layout) -> list[dict]:
    return [{"name": p.name, "pos": [float(x) for x in p.pos], "yaw": float(p.yaw)} for p in layout.placements]


def parse_layout(row: dict) -> Layout:
    return Layout([Placement(p["name"], np.asarray(p["pos"], float), float(p["yaw"])) for p in row["layout"]],
                  dict(row["mapping"]))


def make_manifest(root: Path, out: Path, scenes: int, tables: int, seed: int) -> None:
    sc, rng = Scene(), np.random.default_rng(seed)
    rows = human_rows(root)[:scenes]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            src = load_source(root, row)
            targets = [target_layout(src, sc, rng, same_shape=None, spread=0.08) for _ in range(tables)]
            f.write(json.dumps({"episode": row["episode"], "row": row["row"], "layout": [serial_layout(x) for x in targets],
                                "mapping": row["mapping"]}, sort_keys=True) + "\n")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def bucket(attempts: list[dict], key) -> dict[str, dict[str, float | int]]:
    out = {}
    for name in sorted({str(key(a)) for a in attempts}):
        xs = [a for a in attempts if str(key(a)) == name]
        k = sum(bool(a["clean"]) for a in xs)
        out[name] = {"attempts": len(xs), "clean": k, "yield": k / len(xs), "interval95": wilson(k, len(xs))}
    return out


def run(root: Path, manifest: Path, out: Path, method: str, seed: int) -> None:
    rows = {r["episode"]: r for r in human_rows(root)}
    specs = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    out.mkdir(parents=True, exist_ok=True)
    attempts_path = out / f"{method}_attempts.jsonl"
    done = {json.loads(x)["source_episode"] for x in attempts_path.read_text().splitlines() if x.strip()} if attempts_path.exists() else set()
    started = time.perf_counter()
    with attempts_path.open("a") as f:
        for i, spec in enumerate(specs):
            if spec["episode"] in done:
                continue
            src = load_source(root, rows[spec["episode"]])
            targets = [parse_layout({"layout": x, "mapping": spec["mapping"]}) for x in spec["layout"]]
            summary = generate_batch(src, targets, seed=seed + i * 1000, out_root=out / "dataset", method=method)
            for a in summary.attempts:
                f.write(json.dumps(a.__dict__, sort_keys=True) + "\n")
            f.flush()
    attempts = [json.loads(x) for x in attempts_path.read_text().splitlines() if x.strip()]
    k = sum(bool(a["clean"]) for a in attempts)
    by_distance = bucket(attempts, lambda a: "near" if a["distance"] < .05 else "mid" if a["distance"] < .10 else "far")
    by_parts = bucket(attempts, lambda a: a["parts"])
    by_shape = bucket(attempts, lambda a: "same_shape" if a["same_shape"] else "other_shape")
    summary = {"method": "mimicgen_reimplementation" if method == "mimicgen" else method,
               "source_scenes": len(specs), "tables_per_scene": len(specs[0]["layout"]) if specs else 0,
               "attempts": len(attempts), "clean": k, "yield": k / len(attempts) if attempts else 0.0,
               "interval95": wilson(k, len(attempts)),
               "seconds_per_clean_demonstration": sum(a["seconds"] for a in attempts) / k if k else None,
               "by_distance": by_distance, "by_parts": by_parts, "by_shape": by_shape,
               "wall_seconds": round(time.perf_counter() - started, 2), "offline": True}
    (out / f"{method}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("root", type=Path); m.add_argument("out", type=Path); m.add_argument("--scenes", type=int, default=16); m.add_argument("--tables", type=int, default=47); m.add_argument("--seed", type=int, default=20261114)
    r = sub.add_parser("run")
    r.add_argument("root", type=Path); r.add_argument("manifest", type=Path); r.add_argument("out", type=Path); r.add_argument("--method", choices=("echo", "mimicgen", "copy"), required=True); r.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()
    if a.command == "manifest":
        make_manifest(a.root, a.out, a.scenes, a.tables, a.seed)
    else:
        run(a.root, a.manifest, a.out, a.method, a.seed)


if __name__ == "__main__":
    main()
