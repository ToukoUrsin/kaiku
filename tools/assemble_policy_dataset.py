"""Assemble the same 16 human rows with one generator's clean survivors."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def rows(root: Path) -> list[dict]:
    return [json.loads(x) for x in (root / "episodes.jsonl").read_text().splitlines() if x.strip()]


def assemble(human: Path, generated: Path, out: Path, method: str, limit: int | None, seed: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    out_rows = []
    next_ep = 0
    for r in rows(human):
        name = f"human_{next_ep:06d}.npz"
        d = np.load(human / r["file"])
        np.savez_compressed(out / name, state=d["state"][int(r["row"]):int(r["row"]) + 1],
                            action=d["action"][int(r["row"]):int(r["row"]) + 1],
                            parts=d["parts"][int(r["row"]):int(r["row"]) + 1],
                            span=d["span"][int(r["row"]):int(r["row"]) + 1])
        q = dict(r); q.update(file=name, row=0, episode=next_ep, demo=len(out_rows), source="human", env=0)
        out_rows.append(q); next_ep += 1
    candidates = []
    gen_rows = rows(generated)
    for r in gen_rows:
        if method == "echo" and int(r.get("env", 0)) == 0:
            continue  # the source lead is already present among the 16 human rows
        candidates.append(r)
    if limit is not None and len(candidates) > limit:
        rng = np.random.default_rng(seed)
        candidates = [candidates[i] for i in np.sort(rng.choice(len(candidates), limit, replace=False))]
    for file in sorted({r["file"] for r in candidates}):
        picked = [r for r in candidates if r["file"] == file]
        d = np.load(generated / file)
        idx = [int(r["row"]) for r in picked]
        name = f"generated_{next_ep:06d}.npz"
        np.savez_compressed(out / name, state=d["state"][idx], action=d["action"][idx], parts=d["parts"][idx], span=d["span"][idx])
        for j, r in enumerate(picked):
            q = dict(r); q.update(file=name, row=j, episode=next_ep, demo=len(out_rows), source=method)
            out_rows.append(q)
        next_ep += 1
    (out / "episodes.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out_rows))
    return len(candidates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("human", type=Path); ap.add_argument("generated", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--method", required=True, choices=("echo", "mimicgen", "copy")); ap.add_argument("--limit", type=int); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); print(assemble(a.human, a.generated, a.out, a.method, a.limit, a.seed))


if __name__ == "__main__":
    main()
