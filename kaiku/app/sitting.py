"""What follows from a recorded sitting, without anybody doing anything more.

    python -m kaiku.app.sitting SESSION_DIR

checks the session, makes the replay the hosted demo plays (a real glove session only), runs the
study on the person's own demonstrations in sandboxes (``drun``; nothing heavy on this machine) and
writes ``SITTING.md`` into the session: what was recorded, what the study said, what to deploy.
Every number in it is measured; a step that did not run says so.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "kaiku" / "sessions" / "human"
RUNS = Path("/bulk3/runs/robotteleop")


def worn(row: dict) -> bool:
    """Did a gloved hand drive this lead scene: the operator in gloves, or a visitor wearing one.
    The stand-in, a pointer and a camera's hand are not gloves, whatever else was plugged in."""
    who = str(row.get("operator", ""))
    return row.get("env") == 0 and row.get("source") == "human" and (who == "smartgloves" or "Smartglove" in who)


def glove_rows(rows: list[dict]) -> list[dict]:
    """The scenes a gloved hand drove, with their echoes."""
    keep = {r.get("episode") for r in rows if worn(r)}
    return [r for r in rows if r.get("episode") in keep]


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
    if json.loads(meta.read_text()).get("source") == "smartgloves":
        rows = glove_rows(rows)                                # only what a gloved hand drove counts
        if not rows:
            return False, "no scene was sorted by a gloved hand", rows
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


SEEDS = (0, 1, 2)
DEPLOY = ("hsec exec --only FLY_API_TOKEN -- fly deploy --config deploy/fly/fly.toml "
          "--dockerfile deploy/fly/Dockerfile --ha=false --yes")


def study_command(data: str, out: str, steps: int, scenes: int, seed: int, device: str | None = None) -> list[str]:
    """One seed of the study: the person's demonstrations alone (``plain``) and with their echoes
    (``kaiku``), trained side by side and each tested on ``scenes`` scenes neither has seen."""
    return ["python", "-m", "kaiku.policy.fit", data, "--out", out, "--which", "kaiku,plain", "--steps", str(steps),
            "--scenes", str(scenes), "--seed", str(seed), "--threads", "2", *(["--device", device] if device else [])]


def read_fit(text: str) -> dict[str, dict]:
    """What a fit said when it finished, per arm: the ``KAIKU {...}`` line with stage ``done``."""
    out = {}
    for line in text.splitlines():
        if line.startswith("KAIKU "):
            try:
                row = json.loads(line[6:])
            except ValueError:
                continue
            if row.get("stage") == "done" and "success" in row:
                out[row["which"]] = row
    return out


def _study(session: Path, rows: list[dict], steps: int, scenes: int, device: str | None, runs: Path,
           execute: bool) -> dict:
    """The person's demonstrations alone against the same ones with their echoes, three seeds of each.

    Every number in the report is one a finished run said itself. A run that did not finish is left
    out and named; nothing is estimated. The fits are many cores for the better part of an hour, so
    they run in sandboxes (``drun``), one per seed, and only when ``execute`` says so."""
    runs.mkdir(parents=True, exist_ok=True)
    stage = REPO / ".sitting" / session.name                   # inside the repo, so the sandbox gets it
    data, out = f".sitting/{session.name}/session", f".sitting/{session.name}/policies"
    commands = {seed: study_command(data, f"{out}/seed{seed}", steps, scenes, seed, device) for seed in SEEDS}
    report = {"format": 2, "session": str(session), "protocol": "the person's scenes alone against the same scenes with their echoes",
              "steps": steps, "scenes": scenes, "seeds": list(SEEDS), "executed": False, "arms": {}, "missing": [],
              "commands": [" ".join(c) for c in commands.values()]}
    if not execute:
        return report
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(session, stage / "session", ignore=shutil.ignore_patterns("*.rgmp", "*.record", "*.log", "*.pid"))
    (stage / "session" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    procs = {}
    for seed, command in commands.items():
        log = (runs / f"study-seed{seed}.log").open("w")
        procs[seed] = (subprocess.Popen(["drun", "--no-fetch", "--setup", "pip install -q -e '.[policy]'", "--", *command],
                                        cwd=REPO, stdout=log, stderr=subprocess.STDOUT), log)
    said = {}
    for seed, (proc, log) in procs.items():
        try:
            proc.wait(timeout=3 * 3600)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        said[seed] = read_fit((runs / f"study-seed{seed}.log").read_text())
    shutil.rmtree(stage, ignore_errors=True)
    for which, name in (("plain", "alone"), ("kaiku", "with_echoes")):
        got = {seed: said[seed][which] for seed in SEEDS if which in said[seed]}
        report["missing"] += [f"{name} seed {seed}" for seed in SEEDS if seed not in got]
        if got:
            success = [float(row["success"]) for row in got.values()]
            report["arms"][name] = {"seeds": list(got), "success": success, "mean": sum(success) / len(success),
                                    "low": min(success), "high": max(success),
                                    "humans": [row.get("humans") for row in got.values()],
                                    "demonstrations": [row.get("demos") for row in got.values()]}
    report["executed"] = bool(report["arms"])
    if report["executed"]:
        report["generated_at"] = time.time()
        target = REPO / "kaiku" / "policy" / "pretrained"
        target.mkdir(parents=True, exist_ok=True)
        (target / "study_human.json").write_text(json.dumps(report, indent=2) + "\n")
        with (target / "study_human_runs.jsonl").open("a") as f:
            f.write(json.dumps(report) + "\n")
    return report


def finish(session: Path, steps: int = 30000, scenes: int = 96, device: str | None = None, study: bool = True) -> int:
    session = Path(session).expanduser().resolve()
    ok, verdict, rows = validate(session)
    print(("VALID: " if ok else "INVALID: ") + verdict)
    if not ok:
        return 1
    meta = json.loads((session / "session.json").read_text())
    lead = [r for r in rows if r.get("env") == 0 and r.get("source") == "human"]
    real = meta.get("source") == "smartgloves"
    # Only a real glove session becomes what the hosted demo replays; an emulated one stays where it is.
    bundle = bundle_session(session, rows) if real else None
    report = _study(session, rows, steps, scenes, device, RUNS / "sitting" / session.name, execute=study)
    lines = [f"{len(lead)} scenes sorted by the lead, {len(rows)} demonstrations with their echoes.",
             f"Source: {'real Smartgloves' if real else meta.get('source', 'unknown')}, {meta.get('glove_count', '?')} glove(s), "
             f"Coil Pro: {meta.get('coil_present', '?')}, driver {meta.get('driver_version') or '?'}."]
    if report["executed"]:
        for name, words in (("alone", "The person's scenes alone"), ("with_echoes", "With their echoes")):
            arm = report["arms"].get(name)
            if arm:
                lines.append(f"{words}: {arm['mean']:.0%} of {scenes} new scenes sorted "
                             f"({', '.join(f'{v:.0%}' for v in arm['success'])}).")
        if report["missing"]:
            lines.append("Did not finish, left out: " + ", ".join(report["missing"]) + ".")
        lines.append("Study file: kaiku/policy/pretrained/study_human.json")
    else:
        lines.append("Study: not run. Each of these is one seed, to run off the box with drun: " + " | ".join(report["commands"]))
    if real:
        lines += [f"Replay for the hosted demo: {bundle} (the demo replays it when nobody drives, after the next deploy).",
                  f"Deploy: {DEPLOY}",
                  "Words: a real glove session exists, so SUBMISSION.gloves.md and deck/kaiku-gloves.pdf "
                  "(deck/kaiku-one-glove.pdf for one glove) are the ones to use.",
                  "Film: film/README.md says how the cut with the glove footage is made."]
    else:
        lines.append("This was not a real glove session: the words, the deck and the hosted demo stay as they are.")
    text = "\n".join(lines) + "\n"
    (session / "SITTING.md").write_text(text)
    print(text, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--steps", "--study-steps", type=int, default=30000)
    ap.add_argument("--scenes", type=int, default=96, help="new scenes each policy is tested on")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-study", action="store_true", help="validate and bundle only; say how the study is run")
    args = ap.parse_args(argv)
    return finish(args.session, args.steps, args.scenes, args.device, study=not args.no_study)


if __name__ == "__main__":
    raise SystemExit(main())
