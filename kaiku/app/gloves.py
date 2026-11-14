"""Start the Rokoko glove path and Kaiku with a small terminal checklist.

This command owns only the processes it starts. It is deliberately text only: print the URL and
let the operator choose when to open it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import signal
from datetime import datetime
from pathlib import Path

from kaiku.rgmp.glove import GloveStream

SDK_BIN = Path.home() / ".local" / "share" / "rokoko-device-sdk" / "bin"


def _exe(name: str) -> str | None:
    return shutil.which(name) or (str(SDK_BIN / name) if (SDK_BIN / name).exists() else None)


def _open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class Checklist:
    """Probe the driver and solved stream without making a browser connection."""

    def __init__(self, host: str = "127.0.0.1", port: int = 12276, solver_port: int = 12277):
        self.host, self.port, self.solver_port = host, int(port), int(solver_port)
        self.stream: GloveStream | None = None

    def rows(self) -> list[tuple[str, bool, str]]:
        sdk = _exe("rokoko-sdk")
        driver = _open(self.host, self.port)
        solver = _open(self.host, self.solver_port)
        rates = {side: 0.0 for side in ("left", "right")}
        seen = {side: False for side in rates}
        position = {side: False for side in rates}
        if self.stream is not None:
            for side in rates:
                rates[side] = self.stream.rate_hz(side)
                frame = self.stream.latest(side)
                seen[side] = frame is not None
                position[side] = bool(frame is not None and frame.coil_present)
        return [
            ("SDK found and licensed", sdk is not None, "Install Rokoko Device SDK and sign in"),
            ("driver running", driver, "Run rkk-hand-solver --emit world"),
            ("hand solver answering", solver, "Run rkk-hand-solver --emit world"),
            ("left glove streaming", seen["left"] and rates["left"] > 1, "Wake the left glove and check USB"),
            ("right glove streaming", seen["right"] and rates["right"] > 1, "Wake the right glove and check USB"),
            ("position from Coil Pro on both hands", all(position.values()), "Plug in the Coil Pro and move it near the gloves"),
        ]

    @property
    def ready(self) -> bool:
        return all(ok for _, ok, _ in self.rows())

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream = None

    def wait(self, seconds: float, live: bool = False) -> bool:
        self.stream = GloveStream(self.host, self.port).start()
        end = time.monotonic() + seconds
        previous = None
        while time.monotonic() < end:
            current = self.rows()
            if live and current != previous:
                _print(current)
                previous = current
            if self.ready:
                return True
            time.sleep(0.1)
        return self.ready


def _print(rows: list[tuple[str, bool, str]], *, url: str | None = None) -> None:
    for name, ok, fix in rows:
        print(f"[{'x' if ok else ' '}] {name}" + ("" if ok else f" — {fix}"), flush=True)
    if url:
        print(f"Kaiku running: {url}", flush=True)


def _session_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    return Path.home() / "kaiku-sessions" / datetime.now().strftime("%Y%m%d-%H%M")


def _fallback_session() -> Path | None:
    root = Path.home() / "kaiku-sessions"
    candidates = [p for p in root.iterdir()] if root.exists() else []
    def has_human(p: Path) -> bool:
        try:
            return any(json.loads(line).get("env") == 0 and json.loads(line).get("source") == "human"
                       for line in (p / "episodes.jsonl").read_text().splitlines() if line.strip())
        except (OSError, ValueError):
            return False
    candidates = [p for p in candidates if p.is_dir() and (p / "episodes.jsonl").exists() and has_human(p)]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Connect Smartgloves II and start Kaiku.")
    ap.add_argument("--check", action="store_true", help="print the checklist and exit")
    ap.add_argument("--host", default="127.0.0.1", help="Kaiku's web host")
    ap.add_argument("--driver-host", default=None, help="Rokoko driver host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=12276)
    ap.add_argument("--solver-port", type=int, default=12277)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--session-dir", default=None, help="recording folder (default: ~/kaiku-sessions/YYYYMMDD-HHMM)")
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for the checklist")
    ap.add_argument("--kaiku-port", type=int, default=8651)
    args = ap.parse_args(argv)
    driver_host = args.driver_host or "127.0.0.1"
    children: list[subprocess.Popen] = []
    driver_started = False
    check = Checklist(driver_host, args.port, args.solver_port)
    try:
        driver_live = _open(driver_host, args.port)
        if not _open(driver_host, args.solver_port):
            solver = _exe("rkk-hand-solver")
            if solver:
                command = [solver, "--emit", "world"]
                if driver_live:
                    command.append("--no-driver")
                children.append(subprocess.Popen(command, start_new_session=True))
                driver_started = not driver_live
        ready = check.wait(args.timeout, live=True)
        if args.check:
            return 0 if ready else 1
        if not ready:
            fallback = _fallback_session()
            if fallback is None:
                print("Gloves are not ready. Fix the unchecked lines, then run this command again.", file=sys.stderr)
                return 1
            print(f"Gloves are not ready. Replaying the last clean session: {fallback}", flush=True)
            run = subprocess.Popen([sys.executable, "-m", "kaiku.app.run", "--n", str(args.n), "--host", args.host,
                                    "--port", str(args.kaiku_port), "--session", str(fallback)],
                                   start_new_session=True)
            children.append(run)
            print(f"[x] Kaiku running: http://127.0.0.1:{args.kaiku_port}/", flush=True)
            return run.wait()
        session = _session_path(args.session_dir)
        session.mkdir(parents=True, exist_ok=True)
        raw = session / "kaiku.rgmp.record"
        rec = subprocess.Popen([sys.executable, "-m", "kaiku.rgmp.record", "--host", driver_host,
                                "--port", str(args.port), "--out", str(raw)], start_new_session=True)
        children.append(rec)
        run = subprocess.Popen([sys.executable, "-m", "kaiku.app.run", "--n", str(args.n), "--host", args.host,
                                "--port", str(args.kaiku_port), "--gloves", f"{driver_host}:{args.port}",
                                "--record", str(session)], start_new_session=True)
        children.append(run)
        print(f"[x] Kaiku running: http://127.0.0.1:{args.kaiku_port}/", flush=True)
        return run.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        check.close()
        for child in reversed(children):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except OSError:
                    child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except OSError:
                        child.kill()
        if driver_started:
            # The solver starts the driver in its own process tree; terminate its process only.
            pass


if __name__ == "__main__":
    raise SystemExit(main())
