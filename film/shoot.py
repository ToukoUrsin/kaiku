#!/usr/bin/python
"""Render a recorded take to a clip, frame-exact, and write down what the page said while it ran.

Same replay as tools/film/capture.py (its ReplayServer and virtual clock), plus:
  * CLIP.says.jsonl: twice a second, the words and numbers the page itself showed (source line,
    banner, toasts, totals, the policy tab's finding). The edit reads its numbers from this file.
  * --jobs N splits the range over N browsers (use inside a drun sandbox, not on the box).
  * --css adds film-only styling that hides or enlarges nothing but what is named (never changes a word).

    python film/shoot.py REC --out clip.mp4 --start 5 --end 20 --query 'clean=1&shot=overview&quality=high' --jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "film"))
from capture import ReplayServer, add_quality, advance_virtual, read_actions  # noqa: E402

CHROME = os.environ.get("CHROME") or next((p for p in ("/usr/bin/google-chrome", "/usr/bin/google-chrome-stable") if os.path.exists(p)), None)
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

# The page's clock, handed to the film. Headless Chrome runs animation frames on real time even while
# everything else is on the paused virtual clock, so the camera, the hands and the rings would move at
# render speed and the page would think its stream had stalled. With this, one film frame is exactly
# one page frame, at the recording's time, and every browser that films a part of a take agrees.
CLOCK = """(() => {
  const real = window.requestAnimationFrame.bind(window);
  let q = [], film = false, now = 0;
  window.requestAnimationFrame = (cb) => { if (!film) return real(() => cb(performance.now())); q.push(cb); return 0; };
  const idle = () => real(idle);                             // keeps the compositor ticking for screenshots
  window.__filmTake = () => { film = true; now = performance.now(); performance.now = () => now; idle(); };
  window.__filmWaiting = () => q.length;
  window.__filmNow = (t) => { now = t; };
  window.__filmFrame = (t) => { now = t; const run = q; q = []; for (const cb of run) cb(t); return run.length; };
})();"""
CLOCK_AT = 100000.0      # page milliseconds at recording time 0, the same in every browser
HOLD = """(() => { const s = window.kaiku.stage; s.__own = Object.hasOwn(s, 'render'); s.__render = s.render; s.render = () => {}; })()"""
DRAW = """(() => { const s = window.kaiku.stage; if (s.__own) s.render = s.__render; else delete s.render; })()"""

# What the page says, read from the page. Only elements that are showing.
SAYS = """() => {
  const shown = (el) => el && !el.closest('[hidden]') && el.getClientRects().length > 0;
  const one = (ref) => { const el = document.querySelector(`[data-ref="${ref}"]`);
    return shown(el) ? el.textContent.replace(/\\s+/g, ' ').trim() : null; };
  const out = {};
  for (const ref of ['source-text', 'banner-text', 'caption', 'opening', 'chain-count', 'chain-who', 'robots', 'demos',
    'demos-label', 'stat1-label', 'amplification', 'stat2-label', 'yield', 'stat3-label', 'rate', 'clipped', 'finding',
    'study-note', 'policy-kaiku', 'policy-plain', 'policy-data', 'policy-test', 'policy-alone', 'policy-here', 'tag-text',
    'headline-top', 'headline-main', 'ds-episodes', 'ds-frames', 'ds-time', 'drive-say', 'scene-index', 'sorted-text',
    'fleet-note', 'history-note', 'conn-text']) {
    const v = one(ref); if (v) out[ref] = v; }
  const toasts = [...document.querySelectorAll('[data-ref="toasts"] > *')].filter(shown).map((el) => el.textContent.replace(/\\s+/g, ' ').trim());
  if (toasts.length) out.toasts = toasts;
  const said = document.querySelector('.said, [data-ref="said"]');
  if (shown(said)) out.said = said.textContent.replace(/\\s+/g, ' ').trim();
  return out; }"""


def encoder(out: Path, width: int, height: int, fps: int):
    return subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                             "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)


def render(a: argparse.Namespace, out: Path, start: float, end: float) -> dict:
    from PIL import Image
    from playwright.sync_api import sync_playwright

    replay = ReplayServer(Path(a.recording).expanduser().resolve())
    replay.start()
    width, height = (int(x) for x in a.size.lower().split("x", 1))
    fps = a.fps
    nframes = max(0, int(round((end - start) * fps / a.speed)))
    if a.still:
        nframes = 1
    actions, ai = read_actions(a.script), 0
    first = next((float(m["t"]) for m in replay.messages if m["kind"] == "binary"), 0.25)
    boot_t = max(0.5, first + 0.05)
    says_path = Path(str(out) + ".says.jsonl")
    try:
        with sync_playwright() as p:
            # Mesa's llvmpipe draws a frame three to four times faster than SwiftShader, with the same picture.
            gl = ["--use-angle=gl-egl"] if a.gl == "egl" else ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
            launch = dict(headless=True, args=[*gl, "--ignore-gpu-blocklist", "--enable-webgl", "--run-all-compositor-stages-before-draw"])
            if CHROME:
                launch["executable_path"] = CHROME
            browser = p.chromium.launch(**launch)
            context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=a.scale)
            page = context.new_page()
            logs: list[str] = []
            page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
            page.add_init_script(CLOCK)
            page.goto(f"http://127.0.0.1:{replay.port}/?{add_quality(a.query)}", wait_until="domcontentloaded", timeout=120000)
            for _ in range(3000):
                replay.tick(boot_t)
                try:
                    if page.evaluate("Boolean(window.kaiku && document.querySelector('[data-ref=veil]')?.hasAttribute('data-done'))"):
                        break
                except Exception:
                    pass
                page.wait_for_timeout(20)
            else:
                raise RuntimeError("the page did not start\n" + "\n".join(logs[-40:]))
            for _ in range(60):                            # the loading veil fades out in real time
                replay.tick(boot_t)
                page.wait_for_timeout(25)
            if a.css:
                page.add_style_tag(content=a.css)
            if a.init:                                     # film-only camera framing; never the page's words
                page.evaluate(a.init)
            binary = [m["kind"] == "binary" for m in replay.messages]
            sent = lambda: sum(binary[:replay.index])      # stream frames the replay has sent so far
            base = page.evaluate("window.kaiku.stream.frames") - sent()
            page.evaluate("window.__filmTake()")
            for _ in range(200):                           # the page's own frame loop is now waiting for the film
                if page.evaluate("window.__filmWaiting()"):
                    break
                page.wait_for_timeout(10)
            cdp = context.new_cdp_session(page)
            cdp.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
            step_ms = 1000.0 * a.speed / fps

            def frame(t: float, before: float) -> None:
                """Bring the page to recording time t and run one page frame there."""
                # Stream frames are stamped one film frame back, so the page shows the newest one whole.
                page.evaluate(f"window.__filmNow({CLOCK_AT + t * 1000.0 - step_ms})")
                replay.tick(max(t, boot_t))
                if t > before:
                    advance_virtual(cdp, (t - before) * 1000.0)
                want = base + sent()
                for _ in range(4000):                      # every frame sent has reached the page, however slow the machine
                    if page.evaluate("window.kaiku.stream.frames") >= want:
                        break
                    page.wait_for_timeout(5)
                else:
                    logs.append(f"stream frames late at {t:.3f}")
                page.evaluate(f"window.__filmFrame({CLOCK_AT + t * 1000.0})")

            # Run the page up to the start without drawing: coarse steps, then the last seconds frame by
            # frame, so the camera, the hands and the counters have settled exactly as in a single run.
            page.evaluate(HOLD)
            cursor = boot_t
            fine = max(boot_t, start - a.settle * a.speed)
            while cursor < start - 1e-9:
                step = min(a.warm_step, fine - cursor) if cursor < fine - 1e-9 else min(a.speed / fps, start - cursor)
                frame(cursor + step, cursor)
                cursor += step
            page.evaluate(DRAW)
            page.wait_for_timeout(5)
            wall = time.monotonic()
            if a.still:                                    # one PNG, optionally only a region, at --scale
                clip = None
                if a.clip:
                    x, y, w, h = (float(v) for v in a.clip.split(","))
                    clip = {"x": x, "y": y, "width": w, "height": h}
                frame(start, cursor)
                page.screenshot(path=str(out), type="png", clip=clip, timeout=240000)
                says_path.write_text(json.dumps({"t": round(start, 3), **page.evaluate(SAYS)}) + "\n")
                browser.close()
                return {"frames": 1, "start": start, "end": start, "wall_seconds": round(time.monotonic() - wall, 1), "errors": logs[-5:]}
            ff = encoder(out, int(round(width * a.scale)), int(round(height * a.scale)), fps)
            with says_path.open("w") as says:
                for n in range(nframes):
                    target = start + n * a.speed / fps
                    while ai < len(actions) and float(actions[ai].get("t", 0)) <= target + 1e-9:
                        act = actions[ai]
                        ai += 1
                        if "click" in act:
                            page.locator(str(act["click"])).click(timeout=30000)
                        elif "key" in act:
                            page.keyboard.press(str(act["key"]))
                        elif "eval" in act:
                            page.evaluate(str(act["eval"]))
                    frame(target, cursor)
                    cursor = max(cursor, target)
                    for attempt in range(6):               # a screenshot now and then waits for a frame that never comes
                        try:
                            shot = page.screenshot(type="jpeg", quality=98, timeout=30000)
                            break
                        except Exception:
                            if attempt == 5:
                                raise
                            logs.append(f"screenshot retried at {target:.3f}")
                    image = Image.open(BytesIO(shot)).convert("RGB")
                    ff.stdin.write(image.tobytes())
                    if n % max(1, fps // 2) == 0:
                        says.write(json.dumps({"t": round(target, 3), **page.evaluate(SAYS)}) + "\n")
            ff.stdin.close()
            if ff.wait(timeout=900):
                raise RuntimeError("ffmpeg failed")
            browser.close()
    finally:
        replay.stop()
    return {"frames": nframes, "start": start, "end": end, "wall_seconds": round(time.monotonic() - wall, 1), "errors": logs[-5:]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--query", default="clean=1&shot=overview&quality=high")
    p.add_argument("--script")
    p.add_argument("--css", default="")
    p.add_argument("--init", default="", help="JavaScript run once after the page has loaded (camera framing)")
    p.add_argument("--speed", type=float, default=1.0, help="recorded seconds per film second")
    p.add_argument("--still", action="store_true", help="write one PNG at --start instead of a clip")
    p.add_argument("--clip", default="", help="with --still: x,y,w,h in page pixels")
    p.add_argument("--warm-step", type=float, default=0.5)
    p.add_argument("--settle", type=float, default=3.0, help="film seconds run frame by frame, undrawn, before the first frame")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--gl", choices=["egl", "swiftshader"], default=os.environ.get("KAIKU_GL", "egl"))
    a = p.parse_args()
    out = Path(a.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.still:
        a.end, a.jobs = a.start, 1
    total = int(round((a.end - a.start) * a.fps / a.speed))
    jobs = max(1, min(a.jobs, total)) if not a.still else 1
    if jobs == 1:
        stats = render(a, out, a.start, a.end)
    else:
        with tempfile.TemporaryDirectory(prefix="shoot-", dir=str(out.parent)) as td:
            parts, procs = [], []
            for j in range(jobs):
                s, e = a.start + (total * j // jobs) * a.speed / a.fps, a.start + (total * (j + 1) // jobs) * a.speed / a.fps
                part = Path(td) / f"p{j:02d}.mp4"
                parts.append(part)
                cmd = [sys.executable, __file__, a.recording, "--out", str(part), "--start", repr(s), "--end", repr(e),
                       "--fps", str(a.fps), "--size", a.size, "--scale", str(a.scale), "--query", a.query,
                       "--css", a.css, "--warm-step", str(a.warm_step), "--settle", str(a.settle), "--gl", a.gl, "--init", a.init, "--speed", str(a.speed)] + (["--script", a.script] if a.script else [])
                procs.append(subprocess.Popen(["nice", "-n", "10", *cmd]))
            if any(pr.wait() for pr in procs):
                raise SystemExit("a part failed")
            (Path(td) / "list.txt").write_text("".join(f"file '{x}'\n" for x in parts))
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(Path(td) / "list.txt"),
                            "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
            with Path(str(out) + ".says.jsonl").open("w") as says:
                for part in parts:
                    says.write(Path(str(part) + ".says.jsonl").read_text())
            stats = {"frames": total, "start": a.start, "end": a.end, "jobs": jobs,
                     "parts": [json.loads(Path(str(x) + ".stats.json").read_text()) for x in parts]}
    stats["query"] = a.query
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
