#!/usr/bin/env python3
"""Replay a Kaiku recording on a deterministic virtual clock and encode an H.264 clip."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

try:
    from aiohttp import WSMsgType, web
except ImportError:
    WSMsgType = web = None  # type: ignore[assignment]


# The venv used by Kaiku has aiohttp but not Playwright; the box's system Python has both.
def ensure_browser_python() -> None:
    try:
        import playwright  # noqa: F401
    except ImportError:
        system_python = "/usr/bin/python"
        if os.path.exists(system_python) and os.path.realpath(sys.executable) != os.path.realpath(system_python):
            os.execv(system_python, [system_python, *sys.argv])
        raise SystemExit("Playwright is unavailable; install it on the box or use /usr/bin/python")


class ReplayServer:
    def __init__(self, recording: Path):
        self.recording = recording
        self.manifest = json.loads((recording / "recording.json").read_text())
        self.messages: list[dict[str, Any]] = []
        with (recording / self.manifest["messages"]).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.messages.append(json.loads(line))
        self.messages.sort(key=lambda m: float(m["t"]))
        self.index = 0
        self.ws = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.started = threading.Event()
        self.port = 0
        self.error: BaseException | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="kaiku-replay", daemon=True)
        self.thread.start()
        self.started.wait(30)
        if self.error:
            raise self.error
        if not self.started.is_set():
            raise RuntimeError("replay server did not start")

    def stop(self) -> None:
        if self.loop and self.thread:
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self._stop()))
            self.thread.join(10)
            self.thread = None

    async def _stop(self) -> None:
        if self.ws is not None and not self.ws.closed:
            await self.ws.close()
        if self._stop_event is not None:
            self._stop_event.set()

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
            self.started.set()
        finally:
            self.loop.close()
            self.loop = None

    async def _serve(self) -> None:
        app = web.Application()
        app.router.add_get("/api/ws", self._ws_handler)
        app.router.add_route("GET", "/{path:.*}", self._asset_handler)
        runner = web.AppRunner(app, handle_signals=False, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.port = runner.addresses[0][1]
        self._stop_event = asyncio.Event()
        self.started.set()
        await self._stop_event.wait()
        await runner.cleanup()

    def _asset(self, raw_path: str) -> dict[str, Any] | None:
        key = raw_path or "/"
        assets = self.manifest.get("assets", {})
        item = assets.get(key)
        if item is None and key == "/":
            item = assets.get("/") or assets.get("")
        return item

    async def _asset_handler(self, request: web.Request) -> web.StreamResponse:
        item = self._asset(request.raw_path.split("?", 1)[0])
        if item is None:
            # The recorder keys query-bearing paths exactly; try the full path as a fallback.
            item = self._asset(request.raw_path)
        if item is None:
            raise web.HTTPNotFound(text=f"replay asset not recorded: {request.raw_path}")
        path = self.recording / item["file"]
        if not path.is_file():
            raise web.HTTPNotFound(text="replay asset file missing")
        headers = {"Content-Type": item.get("content_type", "application/octet-stream"), "Cache-Control": "no-cache"}
        return web.Response(body=path.read_bytes(), headers=headers, status=int(item.get("status", 200)))

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        if self.ws is not None and not self.ws.closed:
            await self.ws.close()
        ws = web.WebSocketResponse(heartbeat=20, compress=False, max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        self.ws = ws
        try:
            async for msg in ws:
                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
                # Browser ACKs and UI commands are intentionally ignored; the recording is authoritative.
        finally:
            if self.ws is ws:
                self.ws = None
        return ws

    async def _tick_async(self, t: float) -> None:
        if self.ws is None or self.ws.closed:
            return
        sent = 0
        while self.index < len(self.messages) and float(self.messages[self.index]["t"]) <= t + 1e-8:
            msg = self.messages[self.index]
            self.index += 1
            if msg["kind"] == "text":
                await self.ws.send_str(msg["data"])
            else:
                await self.ws.send_bytes(base64.b64decode(msg["data"]))
            sent += 1

    def tick(self, t: float) -> None:
        if self.loop is None:
            return
        done = threading.Event()
        def go() -> None:
            async def run() -> None:
                try:
                    await self._tick_async(float(t))
                finally:
                    done.set()
            asyncio.create_task(run())
        self.loop.call_soon_threadsafe(go)
        done.wait(10)


def read_actions(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise SystemExit("--script must contain a JSON list")
    return sorted(data, key=lambda a: float(a.get("t", 0)))


def add_quality(query: str) -> str:
    query = query.lstrip("?")
    pairs = [p for p in query.split("&") if p]
    if not any(p.split("=", 1)[0] == "quality" for p in pairs):
        pairs.append("quality=high")
    return "&".join(pairs)


def advance_virtual(cdp, milliseconds: float) -> None:
    if milliseconds <= 0:
        return
    cdp.send("Emulation.setVirtualTimePolicy", {
        "policy": "advance",
        "budget": float(milliseconds),
        "maxVirtualTimeTaskStarvationCount": 10000,
    })
    # Keep the browser stopped between explicit video frames. This also prevents wall-clock
    # SwiftShader rendering from sneaking in while ffmpeg is consuming the previous frame.
    cdp.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})


def run_ffmpeg(out: Path, width: int, height: int, fps: int):
    cmd = ["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_single(args: argparse.Namespace, out: Path, start: float, end: float) -> dict[str, Any]:
    ensure_browser_python()
    from PIL import Image
    from io import BytesIO
    from playwright.sync_api import sync_playwright

    recording = Path(args.recording).expanduser().resolve()
    replay = ReplayServer(recording)
    replay.start()
    width, height = (int(x) for x in args.size.lower().split("x", 1))
    scale = float(args.scale)
    fps = int(args.fps)
    nframes = max(0, int(round((end - start) * fps)))
    out.parent.mkdir(parents=True, exist_ok=True)
    actions = read_actions(args.script)
    action_index = 0
    first_binary_t = next((float(m["t"]) for m in replay.messages if m["kind"] == "binary"), 0.25)
    boot_t = max(0.5, first_binary_t + 0.05)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=[
                "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--enable-webgl",
                "--run-all-compositor-stages-before-draw",
            ])
            context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=scale,
                                          ignore_https_errors=True)
            page = context.new_page()
            browser_logs: list[str] = []
            page.on("console", lambda message: browser_logs.append(f"console {message.type}: {message.text}"))
            page.on("pageerror", lambda error: browser_logs.append(f"pageerror: {error}"))
            page.on("requestfailed", lambda request: browser_logs.append(f"requestfailed {request.url}: {request.failure}"))
            page.on("response", lambda response: browser_logs.append(f"response {response.status} {response.url}"))
            page.add_init_script("""
                (() => { const r = window.requestAnimationFrame.bind(window); let n = 0;
                  window.__filmRafCount = 0; window.__filmLastRaf = 0;
                  window.requestAnimationFrame = (cb) => r((t) => { window.__filmRafCount = ++n; window.__filmLastRaf = t; cb(t); });
                })();
            """)
            query = add_quality(args.query)
            url = f"http://127.0.0.1:{replay.port}/" + ("?" + query if query else "")
            page.goto(url, wait_until="domcontentloaded", timeout=120000)
            # Let the page load its static scene and consume the first keyframe. No frames are
            # captured during boot, so this warm-up does not enter the output clip.
            for _ in range(100):
                replay.tick(boot_t)
                try:
                    if page.evaluate("Boolean(window.kaiku && document.querySelector('[data-ref=veil]')?.dataset.done)"):
                        break
                except Exception:
                    pass
                page.wait_for_timeout(20)
            else:
                raise RuntimeError("page did not connect to replay websocket\n" + "\n".join(browser_logs[-80:]) +
                                   f"\nurl={page.url} ready={page.evaluate('document.readyState')}\ntext={page.locator('body').inner_text()[:1000]}")
            cdp = context.new_cdp_session(page)
            cdp.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
            # Fast-forward the page's state to the requested window start without writing frames.
            replay.tick(max(boot_t, start))
            advance_virtual(cdp, max(0.0, start - boot_t) * 1000.0)
            # Ensure the first due message has been delivered before the first screenshot.
            page.wait_for_timeout(5)
            ff = run_ffmpeg(out, int(round(width * scale)), int(round(height * scale)), fps)
            if ff.stdin is None:
                raise RuntimeError("ffmpeg stdin unavailable")
            cursor = max(boot_t, start)
            raf_counts: list[int] = []
            capture_wall_start = time.monotonic()
            for frame_no in range(nframes):
                target = start + frame_no / fps
                target_for_replay = max(target, boot_t)
                # Apply scripted UI actions at the same virtual time as the video frame.
                while action_index < len(actions) and float(actions[action_index].get("t", 0)) <= target + 1e-9:
                    action = actions[action_index]
                    if "click" in action:
                        page.locator(str(action["click"])).click(timeout=30000)
                    elif "key" in action:
                        page.keyboard.press(str(action["key"]))
                    elif "eval" in action:
                        page.evaluate(str(action["eval"]))
                    action_index += 1
                replay.tick(target_for_replay)
                if target > cursor:
                    advance_virtual(cdp, (target - cursor) * 1000.0)
                cursor = max(cursor, target)
                # A screenshot forces the compositor to finish the canvas draw at the current
                # virtual timestamp. The RAF counter is retained as proof in the stats file.
                # JPEG avoids spending most of the SwiftShader-bound wall time deflating
                # a lossless screenshot; the decoded RGB is encoded losslessly enough by
                # the final CRF-14 H.264 pass.
                screenshot = page.screenshot(type="jpeg", quality=98, timeout=240000)
                image = Image.open(BytesIO(screenshot)).convert("RGB")
                if image.size != (int(round(width * scale)), int(round(height * scale))):
                    image = image.resize((int(round(width * scale)), int(round(height * scale))), Image.Resampling.LANCZOS)
                ff.stdin.write(image.tobytes())
                raf_counts.append(int(page.evaluate("window.__filmRafCount || 0")))
            ff.stdin.close()
            rc = ff.wait(timeout=600)
            if rc:
                raise RuntimeError(f"ffmpeg exited {rc}")
            stats = {"frames": nframes, "fps": fps, "size": [int(round(width * scale)), int(round(height * scale))],
                     "start": start, "end": end, "wall_seconds": time.monotonic() - capture_wall_start,
                     "raf_first": raf_counts[0] if raf_counts else 0, "raf_last": raf_counts[-1] if raf_counts else 0,
                     "raf_nonduplicate_steps": sum(b > a for a, b in zip(raf_counts, raf_counts[1:])),
                     "query": query}
            Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
            browser.close()
    finally:
        replay.stop()
    return stats


def split_ranges(start: float, end: float, fps: int, jobs: int) -> list[tuple[float, float]]:
    total = int(round((end - start) * fps))
    jobs = min(max(1, jobs), max(1, total))
    ranges = []
    for j in range(jobs):
        a = (total * j) // jobs
        b = (total * (j + 1)) // jobs
        ranges.append((start + a / fps, start + b / fps))
    return [r for r in ranges if r[1] > r[0]]


def render_parallel(args: argparse.Namespace, start: float, end: float) -> dict[str, Any]:
    out = Path(args.out).expanduser().resolve()
    ranges = split_ranges(start, end, args.fps, args.jobs)
    with tempfile.TemporaryDirectory(prefix="kaiku-film-", dir="/bulk2/runs/robotteleop/film") as td:
        children = []
        for i, (a, b) in enumerate(ranges):
            seg = Path(td) / f"seg-{i:03d}.mp4"
            cmd = ["nice", "-n", "10", sys.executable, __file__, args.recording, "--out", str(seg),
                   "--start", str(a), "--end", str(b), "--fps", str(args.fps), "--size", args.size,
                   "--scale", str(args.scale), "--query", args.query, "--jobs", "1"]
            if args.script:
                cmd += ["--script", args.script]
            children.append((seg, subprocess.Popen(cmd)))
        for seg, proc in children:
            rc = proc.wait()
            if rc:
                raise SystemExit(f"segment {seg.name} failed with exit {rc}")
        concat = Path(td) / "concat.txt"
        concat.write_text("\n".join(f"file {shlex.quote(str(seg))}" for seg, _ in children) + "\n")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(out)], check=True)
    return {"jobs": len(ranges), "segments": len(ranges), "start": start, "end": end, "fps": args.fps}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording")
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--query", default="clean=1&shot=overview&quality=high")
    p.add_argument("--script")
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()
    manifest = json.loads((Path(args.recording) / "recording.json").read_text())
    duration = float(manifest.get("seconds_requested", manifest.get("seconds_wall", 0)))
    end = duration if args.end is None else args.end
    if args.start < 0 or end <= args.start:
        raise SystemExit("capture window must have end > start >= 0")
    if args.fps < 1 or args.fps > 120:
        raise SystemExit("fps must be between 1 and 120")
    if args.jobs > 1:
        result = render_parallel(args, args.start, end)
    else:
        result = render_single(args, Path(args.out).expanduser().resolve(), args.start, end)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
