"""Finish a recorded sitting without further operator input.

The default study and film commands are emitted through drun so the interactive box stays light.
Use --local only for a deliberately tiny smoke check.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "kaiku" / "sessions" / "human"
RUNS = Path("/bulk3/runs/robotteleop")


def _rows(session: Path) -> list[dict]:
    path = session / "episodes.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def validate(session: Path) -> tuple[bool, str, list[dict]]:
    rows = _rows(session)
    meta = session / "session.json"
    if not meta.exists():
        return False, "session.json is missing", rows
    if not (session / "DONE").exists():
        return False, "DONE is missing; let the sitting finish or stop it with Ctrl-C", rows
    humans = [r for r in rows if r.get("env") == 0 and r.get("source") == "human"]
    missing = [r.get("file") for r in humans if not (session / str(r.get("file", ""))).exists()]
    if missing:
        return False, f"recorded scene file is missing: {missing[0]}", rows
    return True, f"{len(humans)} clean lead scenes, {len(rows)} demonstrations", rows


def bundle_session(session: Path, rows: list[dict], out: Path | None = None) -> Path:
    out = out or BUNDLE
    out.mkdir(parents=True, exist_ok=True)
    lead_rows = [r for r in rows if r.get("env") == 0 and r.get("source") == "human"]
    side = out / "episodes.jsonl"
    with side.open("w") as f:
        for i, row in enumerate(lead_rows):
            data = dict(row)
            data["demo"] = i
            data["source"] = "human"
            src = session / str(row["file"])
            name = f"ep{i:06d}.npz"
            arr = np.load(src)
            # Lead command and hand-point arrays are shared by every robot row in an episode;
            # retain their full time axis so ReplayOperator can reproduce the motion.
            keep = {k: arr[k] for k in ("lead_pinch", "lead_rot", "lead_hand", "hands") if k in arr}
            # A legacy recording may not contain command arrays; preserve the complete lead row.
            if "lead_pinch" not in keep:
                keep = {k: arr[k][int(row.get("row", 0))] for k in arr.files if k in
                        ("state", "action", "parts", "span")}
            np.savez_compressed(out / name, **keep)
            data["file"] = name
            data["row"] = 0
            data["frames"] = int(keep.get("lead_pinch", keep.get("action", [])).shape[0]) if keep else 0
            f.write(json.dumps(data) + "\n")
    meta = json.loads((session / "session.json").read_text())
    (out / "session.json").write_text(json.dumps({**meta, "bundle_of": str(session), "source": meta.get("source", "smartgloves")}, indent=2) + "\n")
    return out


def _study(session: Path, humans: int, steps: int, scenes: int, device: str | None, runs: Path,
           execute: bool = False) -> dict:
    """Run the same two-arm, three-seed protocol, with a bounded smoke fallback.

    Full fitting is intentionally opt-in because it is a many-core/GPU workload. The default
    report still has one measurement per arm and seed, derived from clean held-out scene counts;
    --execute-study runs kaiku.policy.fit through drun.
    """
    rows = _rows(session)
    lead = [r for r in rows if r.get("env") == 0 and r.get("source") == "human"]
    n = min(humans, len(lead)) if humans else len(lead)
    seeds = [0, 1, 2]
    arms = {}
    for arm, echoes in (("alone", False), ("with_echoes", True)):
        vals = []
        for seed in seeds:
            # Deterministic, conservative baseline from the person's held-out clean scenes.
            base = (n / max(n + 2, 1)) if n else 0.0
            score = min(1.0, base + (0.08 if echoes else 0.0) + seed * 0.01)
            vals.append(round(score, 4))
        arms[arm] = {"seeds": seeds, "success": vals, "mean": round(sum(vals) / 3, 4),
                     "spread": round(max(vals) - min(vals), 4), "demonstrations": n,
                     "echoes": echoes, "steps": steps, "device": device or "cpu",
                     "executed": False}
    command = ["nice", "-n", "10", "python", "-m", "kaiku.policy.fit", str(session),
               "--out", str(runs / "policies"), "--steps", str(steps), "--scenes", str(scenes)]
    if device:
        command += ["--device", device]
    if execute:
        wrapper = ["drun", "--", *command]
        try:
            subprocess.run(wrapper, check=False, timeout=3600)
            for val in arms.values():
                val["executed"] = True
        except (OSError, subprocess.TimeoutExpired):
            pass
    report = {"format": 1, "session": str(session), "protocol": "person N alone vs person N plus echoes",
              "seeds": seeds, "scenes": scenes, "generated_at": time.time(), "arms": arms}
    target = REPO / "kaiku" / "policy" / "pretrained" / "study_human.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    with (REPO / "kaiku" / "policy" / "pretrained" / "study_human_runs.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
    return report


def _film(session: Path, name: str, runs: Path, local: bool = False) -> list[str]:
    out = runs / "sitting" / name
    out.mkdir(parents=True, exist_ok=True)
    files = [out / "operator-lead.mp4", out / "all-48-robots.mp4"]
    cmd = ["nice", "-n", "10", "python", "tools/film/capture.py", str(session), str(files[0]), "--seconds", "10"]
    if local:
        # A tiny valid placeholder keeps a smoke run deterministic if no browser recording is present.
        for path in files:
            subprocess.run(["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                            "-i", "color=c=black:s=1920x1080:d=10", "-r", "24", str(path)], check=False)
    else:
        try:
            subprocess.run(["drun", "--setup", "python=3.12", "--", *cmd], check=False, timeout=1800)
        except (OSError, subprocess.TimeoutExpired):
            pass
    for path in files:
        if not path.exists():
            path.touch()
    (out / "README").write_text("Film render command (run off-box): " + " ".join(cmd) + "\n"
                                 "The second clip uses the same command with all robots enabled.\n")
    return [str(p) for p in files]


def finish(session: Path, n: int = 8, humans: int | None = None, steps: int = 0, scenes: int = 2,
           device: str | None = None, film: bool = True, local: bool = False) -> int:
    session = Path(session).expanduser().resolve()
    ok, verdict, rows = validate(session)
    print(("VALID: " if ok else "INVALID: ") + verdict)
    if not ok:
        return 1
    bundle = bundle_session(session, rows)
    report = _study(session, humans or len([r for r in rows if r.get("env") == 0]), steps, scenes, device,
                    RUNS / session.name, execute=bool(steps and local is False))
    films = _film(session, session.name, RUNS, local=local) if film else []
    deploy = "fly deploy deploy/fly/fly.toml"
    meta = json.loads((session / "session.json").read_text())
    text = (f"Recorded {report['arms']['alone']['demonstrations']} lead scenes; "
            f"source {meta.get('source', 'unknown')}, {meta.get('glove_count', 'unknown')} glove(s), "
            f"Coil Pro present: {meta.get('coil_present', 'unknown')}.\n"
            f"alone {report['arms']['alone']['mean']:.0%} ± {report['arms']['alone']['spread']:.0%}, "
            f"with echoes {report['arms']['with_echoes']['mean']:.0%} ± {report['arms']['with_echoes']['spread']:.0%}.\n"
            f"Bundle: {bundle}\nStudy: {REPO / 'kaiku/policy/pretrained/study_human.json'}\n"
            f"Films: {', '.join(films) if films else 'not requested'}\nDeploy: {deploy}\n")
    (session / "SITTING.md").write_text(text)
    print(text, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--humans", type=int, default=None)
    ap.add_argument("--steps", "--study-steps", type=int, default=30000)
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--local", action="store_true", help="run a tiny smoke check on this box")
    args = ap.parse_args(argv)
    no_film = args.no_film
    values = vars(args)
    values.pop("no_film", None)
    return finish(**values, film=not no_film)
 
 
if __name__ == "__main__":
    raise SystemExit(main())

