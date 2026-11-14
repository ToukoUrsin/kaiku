#!/usr/bin/python
"""Shot 1: film of a real hand, played into the page as its camera, drives the lead robot's fingers.

    nice -n 10 /usr/bin/python film/hand.py https://kaiku-echo.fly.dev CAM.y4m OUT --seconds 45

Headless Chromium gets CAM.y4m as its camera (it loops), opens the page and presses what a visitor
presses: Drive it yourself, Use my hand, Grasp it myself. While the hand drives, three things are
written, all on this machine's clock, so the edit can put them side by side without guessing:

  OUT/rec/            the page's stream, recorded by film/rec.py (it only listens); recording.json has t0_epoch
  OUT/inset/N.jpg     the page's own camera picture with the hand it tracked drawn over it, as this browser
                      showed it (Chrome's screencast), and OUT/inset.jsonl with the time of every picture
  OUT/hand.jsonl      what the page read off every camera frame, and what the runner said was driving

Film-only styling in this browser: the camera picture fills the window and keeps its colour (the page
shows it small and grey), and the 3D view is not drawn here (the film draws it from the recorded stream).
Nothing the page sends to the runner is touched. The turn is handed back at the end.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
W, H = 960, 720
CSS = (".hand-cam{position:fixed!important;left:0!important;top:0!important;right:auto!important;bottom:auto!important;"
       f"width:{W}px!important;border-radius:0!important;box-shadow:none!important;z-index:99999!important}}"
       ".hand-cam video{filter:none!important}.hand-cam p{display:none!important}")
HOOK = """() => { const d = window.kaiku.drive, draw = d.draw.bind(d); window.__hand = [];
  d.draw = (camera, hand) => { const o = window.kaiku.hud.hud.operator || {};
    window.__hand.push({e: Date.now(), v: +camera.video.currentTime.toFixed(3), seen: Boolean(hand), grip: d.grip, own: d.own,
      at: d.at && [+d.at.x.toFixed(3), +d.at.y.toFixed(3)], depth: +camera.depth.toFixed(3), right: camera.right,
      drive: o.drive || o.driver || null, fingers: o.fingers || null, gap: o.gap_mm ?? null, closure: o.closure ?? null});
    return draw(camera, hand); }; }"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("camera")
    p.add_argument("out")
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--cpu", action="store_true", help="software WebGL reads the hand four times a second; the page's CPU path is faster here")
    p.add_argument("--assisted", action="store_true", help="do not press Grasp it myself")
    a = p.parse_args()
    out = Path(a.out).resolve()
    (out / "inset").mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[
            "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
            "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
            f"--use-file-for-fake-video-capture={Path(a.camera).resolve()}"])
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        logs: list[str] = []
        page.on("pageerror", lambda e: logs.append("PAGEERROR " + str(e)[:300]))
        page.on("console", lambda m: logs.append(m.type + ": " + m.text[:200]) if m.type in ("error", "warning") else None)
        for attempt in range(6):                             # this machine's Chromium drops loads now and then
            try:
                page.goto(a.url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_function("window.kaiku && window.kaiku.stream.frames > 0", timeout=120000)
                break
            except Exception as e:  # noqa: BLE001
                print(f"load failed ({str(e)[:80]}), again", flush=True)
                time.sleep(3)
        else:
            raise SystemExit("the page did not load")
        page.evaluate("window.kaiku.stage.render = () => {}")
        if a.cpu:                                            # the page tries the GPU first and falls back to the CPU itself
            page.evaluate("""async () => { const main = [...document.querySelectorAll('script[type=module]')].map((x) => x.src).find((x) => /main\\.js/.test(x));
              const m = await import(new URL('../vendor/mediapipe/vision_bundle.mjs', main).href);
              const make = m.HandLandmarker.createFromOptions;
              m.HandLandmarker.createFromOptions = function (files, options) {
                if (options?.baseOptions?.delegate === 'GPU') throw new Error('film: no GPU here');
                return make.call(this, files, options); }; }""")
        page.wait_for_timeout(1500)
        ref = "drive-stage" if page.locator("[data-ref=drive-stage]").is_visible() else "drive-panel"
        page.click(f"[data-ref={ref}]", timeout=60000)
        page.wait_for_function("window.kaiku.drive.active", timeout=300000)
        print("driving", flush=True)
        rec = None
        try:
            page.click("[data-ref=drive-camera]", timeout=60000)
            page.wait_for_function("window.kaiku.drive.camera", timeout=60000)
            page.add_style_tag(content=CSS)
            page.evaluate(HOOK)
            if not a.assisted:
                page.wait_for_function("document.querySelector('[data-ref=drive-own]') && !document.querySelector('[data-ref=drive-own]').hidden", timeout=60000)
                page.evaluate("document.querySelector('[data-ref=drive-own]').click()")
                page.wait_for_function("window.kaiku.drive.own", timeout=20000)
            print("camera", page.evaluate("(() => { const v = window.kaiku.drive.camera.video; return [v.videoWidth, v.videoHeight]; })()"), flush=True)
            cdp = page.context.new_cdp_session(page)

            def on_frame(params: dict) -> None:
                n = len(frames)
                (out / "inset" / f"{n:06d}.jpg").write_bytes(base64.b64decode(params["data"]))
                frames.append({"n": n, "e": params["metadata"]["timestamp"]})
                cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

            cdp.on("Page.screencastFrame", on_frame)
            cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "everyNthFrame": 1})
            page.wait_for_timeout(3000)                      # the hand's range settles before the take
            rec = subprocess.Popen(["/usr/bin/python", str(HERE / "rec.py"), a.url.split("?")[0], str(out / "rec"),
                                    "--seconds", str(a.seconds), "--force"], stdout=(out / "rec.log").open("w"), stderr=subprocess.STDOUT)
            while rec.poll() is None:
                page.wait_for_timeout(40)
            cdp.send("Page.stopScreencast")
            rows = page.evaluate("window.__hand")
            (out / "hand.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (out / "inset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in frames))
            seen = sum(r["seen"] for r in rows)
            span = (rows[-1]["e"] - rows[0]["e"]) / 1000 if len(rows) > 1 else 0
            print(f"{len(frames)} pictures, {len(rows)} camera frames read ({len(rows) / max(span, 1e-9):.1f} a second), hand seen in {seen}; "
                  f"runner said {sorted({str(r['drive']) for r in rows})}, fingers {sorted({str(r['fingers']) for r in rows})}", flush=True)
        finally:
            if rec and rec.poll() is None:
                rec.terminate()
            try:
                page.evaluate("document.querySelector('[data-ref=drive-back]').click()")       # hand the turn back
                page.wait_for_timeout(1500)
                print("handed back", flush=True)
            except Exception as e:  # noqa: BLE001
                print("could not hand back:", str(e)[:200], flush=True)
            if logs:
                print("\n".join(logs[-6:]))
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
