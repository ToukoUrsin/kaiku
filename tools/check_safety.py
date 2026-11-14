#!/usr/bin/env python3
"""Replay recorded Kaiku demonstrations through the challenge safety runtime.

The output is deliberately one JSON object, so it can be attached to a run or
consumed by a judge without parsing log text.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "third_party" / "aaltonomous_safety"
FPS = 30.0


def _official():
    """Load the byte-for-byte vendored module without shadowing Kaiku modules."""
    path = VENDORED / "safety.py"
    spec = importlib.util.spec_from_file_location("aaltonomous_official_safety", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official safety runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Keep safety.py byte-for-byte unchanged while supplying the vendored FK-only URDF.
    module.WorkspaceChecker.__init__.__defaults__ = (VENDORED / "g1_body29_hand14.urdf",)
    return module


def _rows(root: Path) -> list[dict]:
    side = root / "episodes.jsonl"
    if not side.exists():
        raise FileNotFoundError(f"{side} is missing")
    return [json.loads(line) for line in side.read_text().splitlines() if line.strip()]


def check(root: Path, limit: int | None = None, seed: int = 0) -> dict:
    official = _official()
    official.log.setLevel(logging.ERROR)
    rows = _rows(root)
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        rng = np.random.default_rng(seed)
        if limit < len(rows):
            rows = [rows[i] for i in np.sort(rng.choice(len(rows), limit, replace=False))]

    by_file: dict[str, list[dict]] = {}
    for row in rows:
        by_file.setdefault(row["file"], []).append(row)
    arrays = {name: np.load(root / name) for name in by_file}
    counts = Counter()
    source_ticks = Counter()
    reason_ticks = Counter()
    clipped_ticks = 0
    largest = 0.0
    aborted: list[dict] = []

    for row in rows:
        data = arrays[row["file"]]
        k = int(row["row"])
        state, action = data["state"][k], data["action"][k]
        start, end = map(int, data["span"][k])
        filt = official.SafetyFilter.from_files(VENDORED / "joint_spec.yaml")
        if end <= start:
            continue
        filt.reset(state[start, :14], state[start, 14:], 0.0)
        source = "operator" if row["source"] == "human" else "echo"
        counts[source] += 1
        did_abort = False
        for tick, (measured, target) in enumerate(zip(state[start:end], action[start:end])):
            result = filt.filter(
                {"arm_q_target": target[:14], "hand_q_target": target[14:]},
                tick / FPS,
                arm_q=measured[:14],
                hand_q=measured[14:],
            )
            returned = np.concatenate([result.arm_q_cmd, result.hand_q_cmd])
            largest = max(largest, float(np.max(np.abs(target - returned))))
            source_ticks[source] += 1
            if result.clipped:
                clipped_ticks += 1
                reason_ticks.update(result.reasons)
            if result.mode == "aborted" and not did_abort:
                did_abort = True
                aborted.append({"demo": row.get("demo"), "source": source,
                                "file": row["file"], "row": k})

    return {
        "demonstrations_checked": {"operator": counts["operator"], "echoes": counts["echo"],
                                    "total": sum(counts.values())},
        "ticks": {"operator": source_ticks["operator"], "echoes": source_ticks["echo"],
                   "total": sum(source_ticks.values())},
        "official_filter_clipped_ticks": clipped_ticks,
        "official_filter_clipped_ticks_by_reason": dict(sorted(reason_ticks.items())),
        "largest_command_difference_rad": largest,
        "demonstrations_it_would_have_aborted": aborted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="a recorder folder containing episodes.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    try:
        result = check(args.data, args.limit, args.seed)
    except ImportError as exc:
        parser.error(f"official runtime needs optional dependency 'pin': {exc}")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
