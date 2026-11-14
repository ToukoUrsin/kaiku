#!/usr/bin/env python3
"""Record the hosted stream for one shot. Same files as tools/film/record.py, so shoot.py can replay them.

It only listens, unless --at asks for a press. It first waits (on a separate connection, nothing sent)
until the page is in the state the shot needs, so a visitor who is driving is never interrupted:

    --wait standin    Operator mode, the scripted stand-in driving, nobody's turn
    --wait copyfree   the same, and "See it without Kaiku" can be pressed
    --wait policy     a policy is loaded (Policy mode can be switched on)

Presses: --at 'T:copy_scene:null'  --at 'T:mode:"policy"'. If this take switched to Policy it switches
back to Operator at the end; otherwise it sends nothing at all.

    .venv/bin/python film/rec.py https://kaiku-echo.fly.dev OUT --seconds 40 --wait standin
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "film"))
from record import fetch_asset, parse_at  # noqa: E402


def brief(d: dict) -> str:
    op, copy, ep, echo = d.get("operator") or {}, d.get("copy") or {}, d.get("episode") or {}, d.get("echo") or {}
    flags = "".join(k[0] for k in ("on", "next", "again", "over", "free") if copy.get(k)) or "-"
    return (f"mode={d.get('mode')} src={op.get('source')} standin={op.get('stand_in')} copy={flags} "
            f"scene={ep.get('index')} parts={ep.get('sorted')}/{ep.get('parts')} echo_ok={echo.get('ok')} "
            f"failed={echo.get('failed')} status={d.get('status')} speed={(d.get('perf') or {}).get('realtime')}")


def ready(d: dict, want: str) -> bool:
    op, copy = d.get("operator") or {}, d.get("copy") or {}
    if want == "policy":
        return bool((d.get("policy") or {}).get("ready", True)) and d.get("status") == "running"
    ok = d.get("mode") == "gloves" and bool(op.get("stand_in")) and d.get("status") == "running"
    if want == "copyfree":
        ok = ok and bool(copy.get("free")) and not any(copy.get(k) for k in ("on", "next", "again"))
    if want == "scenestart":                               # a fresh scene: the take holds one whole sort
        ok = ok and float((d.get("episode") or {}).get("elapsed_s") or 99) < 1.5
    return ok


async def wait_for(session: aiohttp.ClientSession, ws_url: str, want: str, limit: float, operator: bool = False) -> None:
    t0, last, asked = time.monotonic(), "", False
    async with session.ws_connect(ws_url, heartbeat=20, max_msg_size=16 * 1024 * 1024) as ws:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                if len(msg.data) >= 8:
                    await ws.send_bytes(bytes((1,)) + bytes(msg.data[4:8]))
                continue
            if msg.type != aiohttp.WSMsgType.TEXT:
                break
            m = json.loads(msg.data)
            if m.get("_") != "hud":
                continue
            line = brief(m["d"])
            if operator and not asked and m["d"].get("mode") == "policy":   # never while a visitor drives
                asked = True
                await ws.send_str(json.dumps({"type": "cmd", "name": "mode", "value": "gloves"}, separators=(",", ":")))
                print("  pressed Operator", flush=True)
            if line.split(" scene=")[0] != last:
                last = line.split(" scene=")[0]
                print(f"  waiting for {want}: {line}", flush=True)
            if ready(m["d"], want):
                return
            if time.monotonic() - t0 > limit:
                raise SystemExit(f"the page did not reach '{want}' in {limit:g} s: {line}")


async def record(a: argparse.Namespace) -> None:
    out = Path(a.out).expanduser().resolve()
    if (out / "recording.json").exists() and not a.force:
        raise SystemExit(f"{out} exists; --force replaces it")
    (out / "assets").mkdir(parents=True, exist_ok=True)
    u = urlparse(a.url)
    base = f"{u.scheme}://{u.netloc}/"
    ws_url = ("wss" if u.scheme == "https" else "ws") + "://" + u.netloc + "/api/ws"
    commands = sorted(a.at or [], key=lambda c: c["t"])
    if any(c["name"] in ("train", "reset", "drive_start", "pause") for c in commands):
        raise SystemExit("this recorder does not press Train, reset, pause or Drive it yourself")
    assets: dict = {}
    counts = {"text": 0, "binary": 0}
    went_policy = False
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)) as session:
        for first in ("/", "/api/scene", "/api/pack/lod0.bin", "/api/pack/lod1.bin", "/api/pack/lod2.bin"):
            await fetch_asset(session, base, first, out, assets, set())
        meta = assets.get("/api/scene")
        if meta:
            for tex in json.loads((out / meta["file"]).read_text()).get("textures", []):
                if isinstance(tex, dict) and tex.get("file"):
                    await fetch_asset(session, base, "/api/tex/" + str(tex["file"]), out, assets, set())
        if a.wait:
            await wait_for(session, ws_url, a.wait, a.wait_limit, a.operator)
        t0 = time.monotonic()
        print(f"recording {a.seconds:g} s", flush=True)
        async with session.ws_connect(ws_url, heartbeat=20, max_msg_size=16 * 1024 * 1024) as ws:
            async def presses() -> None:
                nonlocal went_policy
                for c in commands:
                    await asyncio.sleep(max(0.0, c["t"] - (time.monotonic() - t0)))
                    await ws.send_str(json.dumps({"type": "cmd", "name": c["name"], "value": c["value"]}, separators=(",", ":")))
                    went_policy = went_policy or (c["name"] == "mode" and c["value"] == "policy")
                    print(f"  pressed @{time.monotonic() - t0:.2f}s {c['name']}={json.dumps(c['value'])}", flush=True)

            task = asyncio.create_task(presses())
            last = ""
            try:
                with (out / "messages.jsonl").open("w", encoding="utf-8") as log:
                    while (left := t0 + a.seconds - time.monotonic()) > 0:
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=left)
                        except asyncio.TimeoutError:
                            break
                        now = time.monotonic() - t0
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            log.write(json.dumps({"t": now, "kind": "text", "data": msg.data}, separators=(",", ":")) + "\n")
                            counts["text"] += 1
                            m = json.loads(msg.data)
                            if m.get("_") == "hud":
                                line = brief(m["d"])
                                key = line.split(" parts=")[0]
                                if key != last:
                                    last = key
                                    print(f"  {now:6.1f}s {line}", flush=True)
                            elif m.get("_") != "hello":
                                print(f"  {now:6.1f}s {json.dumps(m)[:200]}", flush=True)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            data = bytes(msg.data)
                            log.write(json.dumps({"t": now, "kind": "binary", "data": base64.b64encode(data).decode()}, separators=(",", ":")) + "\n")
                            counts["binary"] += 1
                            if len(data) >= 8:
                                await ws.send_bytes(bytes((1,)) + data[4:8])
                        else:
                            print(f"stream closed: {msg.type}", file=sys.stderr)
                            break
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                if went_policy:                            # leave the shared page the way it was found
                    await ws.send_str(json.dumps({"type": "cmd", "name": "mode", "value": "gloves"}, separators=(",", ":")))
                    print("  switched back to Operator", flush=True)
    (out / "recording.json").write_text(json.dumps({
        "format": 1, "source_url": a.url, "ws_url": ws_url, "seconds_requested": a.seconds,
        "seconds_wall": time.monotonic() - t0, "recorded_at": time.time(),
        "commands": [{k: c[k] for k in ("t", "name", "value")} for c in commands],
        "messages": "messages.jsonl", "message_counts": counts, "assets": assets}, indent=2) + "\n")
    print(f"saved {out}: {counts['binary']} binary, {counts['text']} text, {len(assets)} files", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--at", action="append", type=parse_at, default=[])
    p.add_argument("--wait", choices=["standin", "copyfree", "scenestart", "policy"])
    p.add_argument("--wait-limit", type=float, default=600.0)
    p.add_argument("--operator", action="store_true", help="if the page is in Policy mode, switch it to Operator before waiting")
    p.add_argument("--force", action="store_true")
    asyncio.run(record(p.parse_args()))


if __name__ == "__main__":
    main()
