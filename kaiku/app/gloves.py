"""The one command for a Smartglove sitting.

The checklist is intentionally plain text: it is readable on a phone and keeps waiting while the
operator plugs in hardware. The command owns only processes it starts.
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
import re
from datetime import datetime, timezone
from pathlib import Path

from kaiku.rgmp.glove import GloveStream

SDK_BIN = Path.home() / ".local" / "share" / "rokoko-device-sdk" / "bin"
TARGET_SCENES = 30  # enough points for the three-seed study while fitting a 25-minute sitting


def _exe(name: str) -> str | None:
    return shutil.which(name) or (str(SDK_BIN / name) if (SDK_BIN / name).exists() else None)


def _open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _doctor(timeout: float = 4.0) -> dict[str, str | bool | None]:
    """Read only useful, non-secret parts of `rokoko doctor`."""
    exe = _exe("rokoko")
    out: dict[str, str | bool | None] = {"sdk": exe is not None, "signed_in": None, "activated": None,
                                         "licence": None, "version": None, "text": ""}
    if not exe:
        return out
    try:
        p = subprocess.run([exe, "doctor"], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return out
    text = p.stdout + "\n" + p.stderr
    out["text"] = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", "[masked]", text)
    match = re.search(r"driver[^0-9]*([0-9]+(?:\.[0-9]+)+)", text, re.I) or \
        re.search(r"(?:sdk)[^0-9]*([0-9]+(?:\.[0-9]+)+)", text, re.I)
    out["version"] = match.group(1) if match else None
    low = text.lower()
    out["signed_in"] = True if "signed in as" in low else None
    if "activated" in low:
        out["activated"] = any(x in low for x in ("activated ..... yes", "activated: yes", "activated yes"))
    if "licence" in low or "license" in low:
        out["licence"] = not bool(re.search(r"\bexpired\b|no seats?|0\s*/\s*\d+\s+seats", low))
    return out


class Checklist:
    """Probe the driver and solved stream without making a browser connection."""

    def __init__(self, host: str = "127.0.0.1", port: int = 12276, solver_port: int = 12277):
        self.host, self.port, self.solver_port = host, int(port), int(solver_port)
        self.stream: GloveStream | None = None
        self.doctor = _doctor(timeout=1.0)

    def rows(self) -> list[tuple[str, bool, str]]:
        sdk = _exe("rokoko-sdk")
        doctor = self.doctor
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
        sdk_ok = sdk is not None
        worn = [side for side in rates if seen[side]]
        slow = [side for side in worn if rates[side] < 40]
        if doctor.get("signed_in") is False:
            sdk_fix = "Sign in to Rokoko on this Mac, then run this command again."
        elif doctor.get("activated") is False:
            sdk_fix = "Activate the Rokoko Device SDK on this Mac, then run this command again."
        elif doctor.get("licence") is False:
            sdk_fix = "Renew the Rokoko licence or free a seat, then run this command again."
        elif not sdk_ok:
            sdk_fix = "Install Rokoko Device SDK on this Mac, then run this command again."
        else:
            sdk_fix = "The SDK is installed; if it still fails, sign in and activate it."
        if not worn:
            gloves, gloves_fix = "a glove streaming", ("Plug the glove into this Mac by USB, switch it on and put it on. "
                                                      "One glove is enough.")
        else:
            gloves = "gloves streaming: " + ", ".join(f"{side} {round(rates[side] / 10) * 10:.0f} Hz" for side in worn)
            gloves += "" if len(worn) == 2 else " (one glove is enough)"
            gloves_fix = f"The {slow[0] if slow else ''} glove's rate is too low; move closer to its receiver or use the cable."
        rows = [
            ("Rokoko SDK installed and licensed", sdk_ok and doctor.get("signed_in") is not False and
             doctor.get("activated") is not False and doctor.get("licence") is not False, sdk_fix),
            ("Rokoko's driver running", driver, "Start the Rokoko driver and leave it running."),
            ("hand solver answering", solver, "Start rkk-hand-solver with --emit world."),
            (gloves, bool(worn) and not slow, gloves_fix),
        ]
        if worn:
            coil = [side for side in worn if position[side]]
            where = ("the Coil Pro" if len(coil) == len(worn) else
                     "the trackpad or the laptop camera (no Coil Pro)" if not coil else
                     f"the Coil Pro for the {coil[0]} hand only, so one hand drives")
            rows.append(("hand position from " + where, True, ""))
        return rows

    def messages(self) -> list[str]:
        out = [fix for _, ok, fix in self.rows() if not ok]
        if self.stream is not None and self.stream.connected and not any(self.stream.latest(s) for s in ("left", "right")):
            out.append("The driver is open but no glove frames arrived for 3 seconds; start streaming in Rokoko.")
        return out

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
        silent_since = time.monotonic()
        warned_silent = False
        while time.monotonic() < end:
            current = self.rows()
            if live and current != previous:
                _print(current)
                previous = current
            if self.stream.connected and not any(self.stream.latest(s) for s in ("left", "right")):
                if not warned_silent and time.monotonic() - silent_since >= 3.0:
                    print("The driver is open but no glove frames arrived for 3 seconds; start streaming in Rokoko.", flush=True)
                    warned_silent = True
            else:
                silent_since = time.monotonic()
                warned_silent = False
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


def _free_port(preferred: int) -> tuple[int, bool]:
    if not _open("127.0.0.1", preferred):
        return preferred, False
    port = preferred + 1
    while _open("127.0.0.1", port):
        port += 1
    return port, True


def _write_metadata(session: Path, source: str, stream: GloveStream | None = None, driver_version: str | None = None) -> None:
    data = {"format": 1, "source": source, "start_time": time.time(),
            "start_iso": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
            "glove_count": 2, "coil_present": None}
    if stream is not None:
        frames = [stream.latest(side) for side in ("left", "right")]
        data.update({"driver_version": driver_version, "firmware": stream.firmware,
                     "glove_count": sum(frame is not None for frame in frames),
                     "coil_present": any(bool(frame and frame.coil_present) for frame in frames)})
    (session / "session.json").write_text(json.dumps(data, indent=2) + "\n")


def _done(session: Path, started: float, source: str, target: int | None = None) -> None:
    episodes = 0
    try:
        episodes = sum(1 for line in (session / "episodes.jsonl").read_text().splitlines()
                       if line.strip() and json.loads(line).get("env") == 0)
    except (OSError, ValueError):
        pass
    summary = {"scenes_kept": episodes, "scenes_dropped": 0, "seconds": round(time.time() - started, 1),
               "source": source, "target": target}
    (session / "DONE").write_text(json.dumps(summary, indent=2) + "\n")


def _emulated_driver() -> bool:
    """Is the Rokoko driver on this machine running its emulated gloves? They stream exactly what
    real ones do, so the stream cannot say; the driver's own command line can."""
    try:
        out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any("smartglove-emulated" in line for line in out.splitlines())


def _stop(*_) -> None:
    raise KeyboardInterrupt


def drive_args(stream: GloveStream | None, driver: str) -> tuple[list[str], str]:
    """How Kaiku is started for the gloves that stream, and one line that says what drives what.

    Two gloves that both have a place on a Coil Pro drive both arms. Anything less (one glove, or no
    Coil Pro) is one hand: its fingers and pinch drive the robot's, and its place comes from the Coil
    Pro when the glove has one, from the pointer or the laptop camera when it has none."""
    frames = [stream.latest(side) for side in ("left", "right")] if stream is not None else []
    if len(frames) == 2 and all(f is not None and f.coil_present for f in frames):
        return ["--gloves", driver], "Two gloves on a Coil Pro: both hands drive. Hold them flat and still to calibrate."
    worn = [f for f in frames if f is not None]
    if any(f.coil_present for f in worn):
        words = "One hand drives: fingers and position from the glove. Hold it flat and still for 2 seconds to start."
    else:
        words = ("One hand drives: fingers and pinch from the glove, position from the trackpad "
                 "(or press Use my hand on the page for the laptop camera).")
    return ["--mode", "visitor", "--one-glove", driver], words


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Connect Smartgloves II and start Kaiku.")
    ap.add_argument("--check", action="store_true", help="print the checklist and exit")
    ap.add_argument("--host", default="127.0.0.1", help="Kaiku's web host")
    ap.add_argument("--driver-host", default=None, help="Rokoko driver host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=12276)
    ap.add_argument("--solver-port", type=int, default=12277)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--session-dir", default=None, help="recording folder (default: ~/kaiku-sessions/YYYYMMDD-HHMM)")
    ap.add_argument("--wait", type=float, default=300.0, help="seconds to wait for gloves (default: 300)")
    ap.add_argument("--timeout", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--kaiku-port", type=int, default=8651)
    ap.add_argument("--target", type=int, default=TARGET_SCENES, help="clean scenes before suggesting you stop")
    ap.add_argument("--emulator", action="store_true", help="use Kaiku's moving stand-in, without Rokoko hardware")
    args = ap.parse_args(argv)
    if args.timeout is not None:
        args.wait = args.timeout
    driver_host = args.driver_host or "127.0.0.1"
    children: list[subprocess.Popen] = []
    started = time.time()
    session = _session_path(args.session_dir)
    source = "emulated" if args.emulator else "smartgloves"
    check = Checklist(driver_host, args.port, args.solver_port)
    # Ctrl-C, a closed terminal window and a plain kill all end the sitting the same way: everything
    # this started is stopped and the session is marked done.
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _stop)
    try:
        if args.emulator:
            ready = True
        else:
            driver_live = _open(driver_host, args.port)
            if not args.check and not _open(driver_host, args.solver_port):
                solver = _exe("rkk-hand-solver")
                if solver:
                    children.append(subprocess.Popen([solver, "--emit", "world", "--driver-arg=--auto-stream-usb"], start_new_session=True))
            ready = check.wait(args.wait if not args.check else min(args.wait, 1.0), live=True)
        if args.check:
            return 0 if ready else 1
        if not ready:
            print("The gloves are not ready yet. Fix the unchecked line above and run this command again.", file=sys.stderr)
            return 1
        if (check.stream is not None and check.stream.emulated) or _emulated_driver():
            source = "emulated"                                # never a real session by mistake
        session.mkdir(parents=True, exist_ok=True)
        _write_metadata(session, source, check.stream, str(check.doctor.get("version") or "") or None)
        kaiku_port, changed = _free_port(args.kaiku_port)
        if changed:
            print(f"Port {args.kaiku_port} is already in use; using port {kaiku_port} instead.", flush=True)
        command = [sys.executable, "-m", "kaiku.app.run", "--n", str(args.n), "--host", args.host,
                   "--port", str(kaiku_port), "--record", str(session), "--target", str(args.target)]
        if not args.emulator:
            raw = session / "kaiku.rgmp.record"
            children.append(subprocess.Popen([sys.executable, "-m", "kaiku.rgmp.record", "--host", driver_host,
                                               "--port", str(args.port), "--out", str(raw)], start_new_session=True))
            how, words = drive_args(check.stream, f"{driver_host}:{args.port}")
            command += how
            print(words, flush=True)
        run = subprocess.Popen(command, start_new_session=True)
        children.append(run)
        print(f"Kaiku running: http://127.0.0.1:{kaiku_port}/", flush=True)
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
        if session.exists() and not args.check:
            _done(session, started, source, args.target)


if __name__ == "__main__":
    raise SystemExit(main())
