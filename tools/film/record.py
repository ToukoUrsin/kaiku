#!/usr/bin/env python3
"""Record a Kaiku web session and all resources needed to replay it offline."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import mimetypes
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp


RESOURCE_RE = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']|(?:import|from)\s*[\"']([^\"']+)[\"']|url\(\s*[\"']?([^\"')]+)")


def parse_at(raw: str) -> dict[str, Any]:
    try:
        sec, name, value = raw.split(":", 2)
        value_obj = json.loads(value)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"--at must be T:NAME:JSONVALUE (got {raw!r})") from exc
    if float(sec) < 0:
        raise argparse.ArgumentTypeError("--at time must be non-negative")
    return {"t": float(sec), "name": name, "value": value_obj, "raw": raw}


def asset_name(path: str) -> str:
    clean = path.split("?", 1)[0].lstrip("/") or "index.html"
    # Keep paths readable while preventing traversal and collisions from odd URLs.
    clean = "/".join(p for p in clean.split("/") if p not in ("", ".", ".."))
    if not clean:
        clean = "index.html"
    return clean


async def fetch_asset(session: aiohttp.ClientSession, base: str, url: str, root: Path,
                     assets: dict[str, dict[str, Any]], pending: set[str]) -> None:
    absolute = urljoin(base, url)
    parsed = urlparse(absolute)
    base_parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or (parsed.netloc and parsed.netloc != base_parsed.netloc):
        return
    key = parsed.path or "/"
    if parsed.query:
        key += "?" + parsed.query
    if key in assets or key in pending:
        return
    pending.add(key)
    try:
        async with session.get(absolute, headers={"Accept-Encoding": "identity"}) as resp:
            body = await resp.read()
            rel = asset_name(key)
            # Query-bearing resources are rare here; hash the suffix to keep a stable filename.
            if "?" in key:
                rel += "." + hashlib.sha256(key.encode()).hexdigest()[:10]
            target = root / "assets" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            assets[key] = {
                "file": str(Path("assets") / rel),
                "status": resp.status,
                "content_type": resp.headers.get("Content-Type", mimetypes.guess_type(rel)[0] or "application/octet-stream"),
                "headers": {k: v for k, v in resp.headers.items() if k.lower() in {"content-type", "cache-control", "etag"}},
                "bytes": len(body),
            }
            if resp.status == 200 and ("text" in resp.headers.get("Content-Type", "") or rel.endswith((".html", ".js", ".mjs", ".css", ".json"))):
                text = body.decode("utf-8", "replace")
                for match in RESOURCE_RE.finditer(text):
                    dep = next((x for x in match.groups() if x), None)
                    if dep and not dep.startswith(("data:", "#", "javascript:", "mailto:")):
                        await fetch_asset(session, absolute, dep, root, assets, pending)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: resource fetch failed {absolute}: {exc}", file=sys.stderr)
    finally:
        pending.discard(key)


async def record(args: argparse.Namespace) -> None:
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    manifest_path = out / "recording.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path} exists; use --force or choose another output directory")

    parsed = urlparse(args.url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SystemExit("URL must be an http(s) page URL")
    ws_url = ("wss" if parsed.scheme == "https" else "ws") + "://" + parsed.netloc + "/api/ws"
    base = f"{parsed.scheme}://{parsed.netloc}/"
    commands = sorted(args.at or [], key=lambda x: x["t"])
    train_count = sum(c["name"] == "train" for c in commands)
    if train_count > 1:
        raise SystemExit("refusing to send train more than once")

    assets: dict[str, dict[str, Any]] = {}
    pending: set[str] = set()
    ws_messages = out / "messages.jsonl"
    started_wall = time.time()
    counts = {"text": 0, "binary": 0}
    last_command = 0.0

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Capture the page and its complete static dependency graph, plus the API payloads the page uses.
        for first in ("/", "/api/scene", "/api/pack/lod0.bin", "/api/pack/lod1.bin", "/api/pack/lod2.bin"):
            await fetch_asset(session, base, first, out, assets, pending)
        scene_meta = assets.get("/api/scene")
        if scene_meta:
            try:
                scene = json.loads((out / scene_meta["file"]).read_text())
                for tex in scene.get("textures", []):
                    if isinstance(tex, dict) and tex.get("file"):
                        await fetch_asset(session, base, "/api/tex/" + str(tex["file"]), out, assets, pending)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: could not enumerate textures: {exc}", file=sys.stderr)

        # The requested window begins when the stream is opened, after the HTTP cache is ready.
        t0 = time.monotonic()
        print(f"recording {args.seconds:g}s from {args.url}", flush=True)
        async with session.ws_connect(ws_url, heartbeat=20, max_msg_size=16 * 1024 * 1024) as ws:
            log = ws_messages.open("w", encoding="utf-8")
            async def send_command(command: dict[str, Any]) -> None:
                nonlocal last_command
                await ws.send_str(json.dumps({"type": "cmd", "name": command["name"], "value": command["value"]}, separators=(",", ":")))
                last_command = command["t"]
                print(f"  command @{command['t']:.3f}s {command['name']}={json.dumps(command['value'])}", flush=True)

            async def command_loop() -> None:
                for command in commands:
                    delay = command["t"] - (time.monotonic() - t0)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if time.monotonic() - t0 <= args.seconds:
                        await send_command(command)

            cmd_task = asyncio.create_task(command_loop())
            deadline = t0 + args.seconds
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    now = time.monotonic() - t0
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        text = msg.data
                        log.write(json.dumps({"t": now, "kind": "text", "data": text}, separators=(",", ":")) + "\n")
                        log.flush()
                        counts["text"] += 1
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        data = bytes(msg.data)
                        log.write(json.dumps({"t": now, "kind": "binary", "data": base64.b64encode(data).decode("ascii")}, separators=(",", ":")) + "\n")
                        log.flush()
                        counts["binary"] += 1
                        # Protocol ACK: type byte 1, uint32 sequence little-endian at bytes 4..8.
                        if len(data) >= 8:
                            await ws.send_bytes(bytes((1,)) + data[4:8])
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        print(f"websocket closed by server: {msg.type}", file=sys.stderr)
                        break
            finally:
                cmd_task.cancel()
                await asyncio.gather(cmd_task, return_exceptions=True)
                # Leave the shared demo in its safe, normal state even if the requested window ended in policy.
                try:
                    await ws.send_str(json.dumps({"type": "cmd", "name": "mode", "value": "gloves"}, separators=(",", ":")))
                    await ws.send_str(json.dumps({"type": "cmd", "name": "pause", "value": False}, separators=(",", ":")))
                except Exception:
                    pass
            log.close()

    manifest = {
        "format": 1,
        "source_url": args.url,
        "ws_url": ws_url,
        "seconds_requested": args.seconds,
        "seconds_wall": time.monotonic() - t0,
        "recorded_at": started_wall,
        "commands": [{k: c[k] for k in ("t", "name", "value")} for c in commands],
        "messages": "messages.jsonl",
        "message_counts": counts,
        "assets": assets,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {out} ({counts['binary']} binary, {counts['text']} text messages; {len(assets)} HTTP assets)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("--seconds", type=float, default=150.0)
    p.add_argument("--at", action="append", type=parse_at, default=[])
    p.add_argument("--force", action="store_true", help="replace an existing recording directory")
    args = p.parse_args()
    asyncio.run(record(args))


if __name__ == "__main__":
    main()
