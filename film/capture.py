#!/usr/bin/python
"""Film every take in film.json: clips in drun sandboxes (many at once), panel stills here.

    /usr/bin/python film/capture.py /bulk3/runs/robotteleop/film/work [NAME ...] [--par 24] [--chunk 90]

Each clip is cut into chunks of --chunk frames; each chunk is one sandbox (4 vCPU, Mesa llvmpipe).
Writes WORK/clips/NAME.mp4 and NAME.says.jsonl (what the page said, twice a second). Skips what exists.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FILM = Path(__file__).resolve().parent
ROOT = FILM.parent


def opt(args: list[str], name: str, default: int) -> int:
    if name in args:
        i = args.index(name)
        v = int(args[i + 1])
        del args[i:i + 2]
        return v
    return default


def shoot_args(takes: dict, c: dict) -> list[str]:
    a = ["--query", c["query"], "--size", c.get("size", "1920x1080"), "--speed", str(c.get("speed", 1))]
    if c.get("css"):
        a += ["--css", takes["css"][c["css"]]]
    if c.get("init"):
        a += ["--init", takes["init"][c["init"]]]
    return a


def run_chunk(job: dict) -> bool:
    d = job["dir"]
    if (d / "out" / "c.mp4").exists() and (d / "out" / "c.mp4.stats.json").exists():
        return True
    if d.exists():
        shutil.rmtree(d)
    (d / "film").mkdir(parents=True)
    (d / "tools" / "film").mkdir(parents=True)
    (d / "out").mkdir()
    for src, dst in ((FILM / "shoot.py", d / "film" / "shoot.py"), (FILM / "sandbox.sh", d / "film" / "sandbox.sh"),
                     (ROOT / "tools" / "film" / "capture.py", d / "tools" / "film" / "capture.py")):
        shutil.copy(src, dst)
    subprocess.run(["cp", "-al", str(job["rec"]), str(d / "rec")], check=True)
    cmd = ["bash", "film/sandbox.sh", "rec", "--out", "out/c.mp4", "--start", repr(job["start"]), "--end", repr(job["end"]),
           "--jobs", str(job["browsers"]), *job["args"]]
    (d / "run.sh").write_text(" ".join(shlex.quote(x) for x in cmd) + "\n")
    for attempt in range(3):
        with (d / f"drun{attempt}.log").open("w") as log:
            rc = subprocess.run(["drun", "--", "bash", "run.sh"], cwd=d, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0 and (d / "out" / "c.mp4.stats.json").exists():
            print(f"  ok {job['name']} [{job['i']}]", flush=True)
            return True
        print(f"  retry {job['name']} [{job['i']}] (exit {rc})", flush=True)
    return False


def main() -> None:
    args = sys.argv[1:]
    par, chunk = opt(args, "--par", 24), opt(args, "--chunk", 90)
    work = Path(args[0]).resolve()
    only = set(args[1:])
    takes = json.loads((FILM / "film.json").read_text())["takes"]
    clips = work / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    recs = work.parent / "rec"

    for name, s in takes["stills"].items():
        out = clips / f"{name}.png"
        if (only and name not in only) or out.exists():
            continue
        subprocess.run(["nice", "-n", "10", "/usr/bin/python", str(FILM / "shoot.py"), str(recs / s["rec"]), "--still", "--out", str(out),
                        "--start", str(s["at"]), "--scale", str(s.get("scale", 1)), "--clip", s.get("clip", ""),
                        *shoot_args(takes, s)], check=True, stdout=subprocess.DEVNULL)
        print(f"still {name}", flush=True)

    jobs, todo = [], {}
    for name, c in takes["clips"].items():
        if (only and name not in only) or (clips / f"{name}.mp4").exists():
            continue
        speed, fps = c.get("speed", 1), 30
        total = int(round((c["end"] - c["start"]) * fps / speed))
        n = 1 if c.get("whole") else max(1, -(-total // chunk))      # whole: one browser films it start to end
        todo[name] = []
        for i in range(n):
            a, b = total * i // n, total * (i + 1) // n
            job = {"name": name, "i": i, "dir": work / "jobs" / f"{name}-{i:02d}", "rec": recs / c["rec"],
                   "start": c["start"] + a * speed / fps, "end": c["start"] + b * speed / fps, "args": shoot_args(takes, c),
                   "browsers": 1 if c.get("whole") else 2}
            jobs.append(job)
            todo[name].append(job)
    print(f"{len(jobs)} chunks, {par} sandboxes at a time", flush=True)
    with ThreadPoolExecutor(par) as pool:
        done = list(pool.map(run_chunk, jobs))
    if not all(done):
        raise SystemExit("some chunks failed; run again to retry only those")
    for name, parts in todo.items():
        lst = work / "jobs" / f"{name}.txt"
        lst.write_text("".join(f"file '{p['dir'] / 'out' / 'c.mp4'}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-movflags", "+faststart", str(clips / f"{name}.mp4")], check=True)
        (clips / f"{name}.says.jsonl").write_text("".join((p["dir"] / "out" / "c.mp4.says.jsonl").read_text() for p in parts))
        print(f"clip {name}", flush=True)


if __name__ == "__main__":
    main()
