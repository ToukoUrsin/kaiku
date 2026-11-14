"""Live web viewer server: streams the joint positions of every environment to browsers.

    from kaiku.web.server import ViewerServer
    v = ViewerServer(model, n_envs, host="0.0.0.0", port=8650, title="Kaiku")
    v.start()                                   # background thread with its own asyncio loop
    v.publish(qpos, env_state=None, hud=None)   # from the simulation loop, 30 to 60 times a second
    v.on_message(callback)                      # callback(dict) for commands sent by the browser
    v.send_event({"type": "toast", "text": "Episode saved"})
    v.stop()

``publish`` only copies the array and wakes the server thread; it never waits for a client. Every
client has its own pace: a frame is sent only when the previous ones were acknowledged quickly
enough, otherwise it is skipped and the client later gets one delta from the last frame it did
receive straight to the newest state. Nothing is ever queued behind a slow connection.

HTTP routes: ``/`` and static files, ``/api/scene`` (JSON), ``/api/pack/lod{k}.bin`` (meshes),
``/api/tex/{name}`` (textures), ``/api/ws`` (the stream).
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import mimetypes
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
from aiohttp import WSMsgType, web

from . import protocol
from .scene_export import N_LODS, SceneExport, export_scene

log = logging.getLogger("kaiku.web")

STATIC_DIR = Path(__file__).parent / "static"
STATE_IDLE, STATE_RUNNING, STATE_SUCCESS, STATE_FAILED, STATE_LEAD = range(5)

MAX_INFLIGHT = 6          # frames a client may have unacknowledged
HUD_MIN_INTERVAL = 0.1    # seconds between hud updates to one client
_COMPRESSIBLE = (".html", ".js", ".mjs", ".css", ".json", ".svg", ".txt", ".map")


class _Frame:
    __slots__ = ("seq", "q16", "state", "t_ms")

    def __init__(self, seq: int, q16: np.ndarray, state: np.ndarray, t_ms: int):
        self.seq, self.q16, self.state, self.t_ms = seq, q16, state, t_ms


class _Client:
    """One browser connection and its pacing state. Only touched on the server thread."""

    def __init__(self, cid: int, ws: web.WebSocketResponse, max_hz: float):
        self.id = cid
        self.ws = ws
        self.wake = asyncio.Event()
        self.outbox: deque[str] = deque(maxlen=256)   # control messages, oldest dropped if a client stalls
        self.base: _Frame | None = None               # last frame sent (deltas are against it)
        self.need_key = True
        self.paused = False
        self.max_hz = max_hz
        self.user_hz = max_hz
        self.interval = 1.0 / max_hz                  # current pacing, grows when the link is slow
        self.last_send = 0.0
        self.inflight: deque[tuple[int, float]] = deque()   # (seq, time sent)
        self.rtts: deque[tuple[float, float]] = deque()     # (time, rtt) of the last seconds
        self.rtt_min = 0.02
        self.last_congested = 0.0
        self.hud_version = -1
        self.hud_sent = 0.0
        self.timer: asyncio.TimerHandle | None = None
        self.frames = 0
        self.bytes = 0
        self.skipped = 0
        self.connected_at = time.monotonic()

    def lag_limit(self) -> float:
        """How old the oldest unacknowledged frame may be before we stop sending new ones."""
        return min(0.6, max(0.06, 2.0 * self.rtt_min + 0.04))

    def on_ack(self, seq: int, now: float) -> None:
        rtt = None
        while self.inflight and self.inflight[0][0] <= seq:
            s, t = self.inflight.popleft()
            if s == seq:
                rtt = now - t
        if rtt is None:
            return
        self.rtts.append((now, rtt))
        while self.rtts and self.rtts[0][0] < now - 10.0:
            self.rtts.popleft()
        self.rtt_min = min(r for _, r in self.rtts)
        if rtt < 0.6 * self.lag_limit():
            self.interval = max(1.0 / min(self.max_hz, self.user_hz), self.interval * 0.92)


class ViewerServer:
    """Serves the viewer page and streams ``qpos`` of ``n_envs`` environments sharing one ``model``."""

    def __init__(self, model, n_envs: int, host: str = "0.0.0.0", port: int = 8650, title: str = "Kaiku",
                 *, cache_dir: str | None = None, max_hz: float = 60.0):
        if n_envs < 1 or n_envs > 65535:
            raise ValueError("n_envs must be between 1 and 65535")
        self.n_envs = int(n_envs)
        self.nq = int(model.nq)
        self.host = host
        self.port = int(port)
        self.title = title
        self.max_hz = float(max_hz)
        self.export: SceneExport = export_scene(model, cache_dir)
        self._quant = self.export.quantizer

        scene = dict(self.export.scene)
        scene.update(title=title, n_envs=self.n_envs)
        self._scene_json = json.dumps(scene, separators=(",", ":")).encode()
        self._scene_gz = gzip.compress(self._scene_json, 6)

        self._state = np.zeros(self.n_envs, dtype=np.uint8)
        self._state[0] = STATE_LEAD
        self._pending: tuple | None = None        # newest publish() not yet picked up by the server thread
        self._wake_scheduled = False
        self._frame: _Frame | None = None
        self._seq = 0
        self._enc: dict[int, bytes] = {}          # encoded payloads of the current frame, by base seq
        self._enc_seq = -1
        self._hud: dict | None = None
        self._hud_json = ""
        self._hud_version = 0
        self._callbacks: list[Callable[[dict], Any]] = []
        self._clients: dict[int, _Client] = {}
        self._next_cid = 1
        self._static_cache: dict[str, tuple[float, int, bytes, bytes | None]] = {}
        self._t0 = time.monotonic()
        self._published = 0

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_evt: asyncio.Event | None = None
        self._started = threading.Event()
        self._start_error: BaseException | None = None
        self._meshes: asyncio.Future | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------------
    # public API (thread safe)

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "", "::") else self.host
        return f"http://{host}:{self.port}/"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "ViewerServer":
        """Start serving on a background thread. Raises ``OSError`` if the port cannot be bound."""
        with self._lock:
            if self.running:
                return self
            self._started.clear()
            self._start_error = None
            self._thread = threading.Thread(target=self._run, name="kaiku-viewer", daemon=True)
            self._thread.start()
        self._started.wait()
        if self._start_error is not None:
            self._thread.join()
            self._thread = None
            raise self._start_error
        return self

    def stop(self) -> None:
        """Close every connection, free the port and join the server thread."""
        with self._lock:
            thread, loop = self._thread, self._loop
            if thread is None:
                return
            if loop is not None and self._stop_evt is not None:
                try:
                    loop.call_soon_threadsafe(self._stop_evt.set)
                except RuntimeError:
                    pass
            thread.join(timeout=10.0)
            self._thread = None

    def __enter__(self) -> "ViewerServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def publish(self, qpos, env_state=None, hud: dict | None = None) -> None:
        """Hand over the newest state. Returns immediately; never blocks on the network.

        qpos:       float array ``(n_envs, model.nq)`` (``(nq,)`` is accepted when ``n_envs == 1``).
        env_state:  optional ``(n_envs,)`` uint8: 0 idle, 1 running, 2 success, 3 failed, 4 lead.
                    ``None`` keeps the previous states (initially env 0 is the lead, the rest idle).
        hud:        optional small JSON-able dict shown as live stats. ``None`` keeps the previous one.
        """
        q = np.array(qpos, dtype=np.float64)
        if q.shape != (self.n_envs, self.nq):
            if q.size != self.n_envs * self.nq:
                raise ValueError(f"qpos has shape {q.shape}, expected ({self.n_envs}, {self.nq})")
            q = q.reshape(self.n_envs, self.nq)
        st = None
        if env_state is not None:
            st = np.array(env_state, dtype=np.uint8).reshape(-1)
            if st.shape[0] != self.n_envs:
                raise ValueError(f"env_state has {st.shape[0]} entries, expected {self.n_envs}")
        old = self._pending
        if old is not None:           # an unsent publish is replaced; keep what it carried along
            if st is None:
                st = old[1]
            if hud is None:
                hud = old[2]
        self._pending = (q, st, hud)
        self._published += 1
        loop = self._loop
        if loop is not None and not self._wake_scheduled:
            self._wake_scheduled = True
            try:
                loop.call_soon_threadsafe(self._on_publish)
            except RuntimeError:      # loop is shutting down
                self._wake_scheduled = False

    def on_message(self, callback: Callable[[dict], Any]) -> Callable[[dict], Any]:
        """Register ``callback(dict)`` for commands from browsers (buttons, keys, focus changes).

        Called on the server thread, so keep it short and thread safe (put the dict on a queue).
        The dict is what the page sent, plus ``"client"``: the id of the connection.
        Built-in page messages: ``{"type": "key", "key": "r", ...}``, ``{"type": "button", "id": ...}``,
        ``{"type": "focus", "env": 3}``. Usable as a decorator.
        """
        self._callbacks.append(callback)
        return callback

    def send_event(self, event: dict) -> None:
        """Push a one-off event to every connected page (not replayed to pages that connect later).

        The page understands ``{"type": "toast", "text": ..., "level": "info"|"warn"|"error"}``,
        ``{"type": "focus", "env": k}`` and ``{"type": "shot", "name": "overview"|"lead"}``; anything
        else is handed to page modules listening for it.
        """
        text = json.dumps({"_": "event", "d": event}, separators=(",", ":"))
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._broadcast, text)
        except RuntimeError:
            pass

    def stats(self) -> dict:
        """Counters for logging and benchmarks."""
        clients = [{"id": c.id, "frames": c.frames, "bytes": c.bytes, "skipped": c.skipped,
                    "rtt_ms": round(c.rtt_min * 1e3, 2), "hz": round(1.0 / c.interval, 1),
                    "seconds": round(time.monotonic() - c.connected_at, 2)} for c in list(self._clients.values())]
        return {"published": self._published, "seq": self._seq, "clients": clients}

    # ------------------------------------------------------------------------------------------
    # server thread

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except BaseException as e:  # noqa: BLE001 - reported to start()
            if not self._started.is_set():
                self._start_error = e
            else:
                log.exception("viewer server crashed")
        finally:
            self._loop = None
            self._started.set()
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    async def _serve(self) -> None:
        self._stop_evt = asyncio.Event()
        app = web.Application()
        app.router.add_get("/api/scene", self._h_scene)
        app.router.add_get("/api/pack/lod{lod:\\d}.bin", self._h_pack)
        app.router.add_get("/api/tex/{name}", self._h_tex)
        app.router.add_get("/api/ws", self._h_ws)
        app.router.add_get("/api/stats", self._h_stats)
        app.router.add_get("/{path:.*}", self._h_static)
        runner = web.AppRunner(app, handle_signals=False, access_log=None, shutdown_timeout=1.0)
        await runner.setup()
        try:
            site = web.TCPSite(runner, self.host, self.port, reuse_address=True)
            await site.start()
            if runner.addresses:                                      # port 0: find what we got
                self.port = runner.addresses[0][1]
            # the mesh pack is built (or found in the cache) off the loop; requests for it wait
            self._meshes = asyncio.ensure_future(asyncio.get_running_loop().run_in_executor(None, self.export.ensure_meshes))
            self._meshes.add_done_callback(lambda f: f.cancelled() or f.exception())
            self._started.set()
            if self._pending is not None:
                self._on_publish()
            await self._stop_evt.wait()
        finally:
            for c in list(self._clients.values()):
                await c.ws.close(code=1001, message=b"server stopping")
            await runner.cleanup()
            if self._meshes is not None and not self._meshes.done():
                await asyncio.wait([self._meshes])

    # -- frames ---------------------------------------------------------------------------------

    def _on_publish(self) -> None:
        self._wake_scheduled = False
        item, self._pending = self._pending, None
        if item is None:
            return
        q, st, hud = item
        if st is not None:
            self._state = st
        if hud is not None and hud != self._hud:
            try:
                self._hud_json = json.dumps({"_": "hud", "d": hud}, separators=(",", ":"), default=_jsonable)
                self._hud = hud
                self._hud_version += 1
            except (TypeError, ValueError):
                log.warning("hud is not JSON-able, ignored")
        self._seq += 1
        t_ms = int((time.monotonic() - self._t0) * 1000.0)
        self._frame = _Frame(self._seq, self._quant.quantize(q), self._state, t_ms)
        for c in self._clients.values():
            c.wake.set()

    def _payload(self, c: _Client, f: _Frame) -> bytes:
        if self._enc_seq != f.seq:
            self._enc.clear()
            self._enc_seq = f.seq
        key = -1 if (c.need_key or c.base is None) else c.base.seq
        data = self._enc.get(key)
        if data is None:
            if key < 0:
                data = protocol.encode_key(f.seq, f.t_ms, f.q16, f.state)
            else:
                b = c.base
                state = None if (b.state is f.state or np.array_equal(b.state, f.state)) else f.state
                data = protocol.encode_delta(f.seq, b.seq, f.t_ms, f.q16, b.q16, state)
            self._enc[key] = data
        return data

    def _gate(self, c: _Client, now: float) -> float | None:
        """0: send a frame now. >0: try again in that many seconds. None: wait for an event."""
        f = self._frame
        if f is None or c.paused or (c.base is not None and c.base.seq == f.seq and not c.need_key):
            return None
        if c.inflight:
            if len(c.inflight) >= MAX_INFLIGHT or now - c.inflight[0][1] > c.lag_limit():
                # the link is behind: skip frames until it catches up, and slow the pace a little
                if now - c.last_congested > 0.25:
                    c.last_congested = now
                    c.interval = min(1.0, c.interval * 1.5)
                c.skipped += 1
                return None
        wait = c.interval - (now - c.last_send)
        return wait if wait > 0.25 * c.interval else 0.0      # tolerate jitter in the publish cadence

    async def _sender(self, c: _Client) -> None:
        """The only writer of this client's socket."""
        ws = c.ws
        loop = asyncio.get_running_loop()
        while not ws.closed:
            await c.wake.wait()
            c.wake.clear()
            while c.outbox:
                await ws.send_str(c.outbox.popleft())
            now = loop.time()
            later = None
            if c.hud_version != self._hud_version and self._hud_json:
                if now - c.hud_sent >= HUD_MIN_INTERVAL:
                    c.hud_version, c.hud_sent = self._hud_version, now
                    await ws.send_str(self._hud_json)
                else:
                    later = HUD_MIN_INTERVAL - (now - c.hud_sent)
            wait = self._gate(c, now)
            if wait == 0.0:
                f = self._frame
                data = self._payload(c, f)
                c.base, c.need_key = f, False
                c.last_send = now
                c.inflight.append((f.seq, now))
                c.frames += 1
                c.bytes += len(data)
                await ws.send_bytes(data)
            elif wait is not None:
                later = wait if later is None else min(later, wait)
            if later is not None:
                if c.timer is not None:
                    c.timer.cancel()
                c.timer = loop.call_later(later, c.wake.set)

    def _broadcast(self, text: str) -> None:
        for c in self._clients.values():
            c.outbox.append(text)
            c.wake.set()

    # -- handlers -------------------------------------------------------------------------------

    async def _h_ws(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(heartbeat=20.0, compress=False, max_msg_size=1 << 20)
        await ws.prepare(request)
        sock = request.transport.get_extra_info("socket") if request.transport else None
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        c = _Client(self._next_cid, ws, self.max_hz)
        self._next_cid += 1
        self._clients[c.id] = c
        c.outbox.append(json.dumps({"_": "hello", "client": c.id, "model_hash": self.export.model_hash,
                                    "n_envs": self.n_envs, "nq": self.nq, "max_hz": self.max_hz}))
        c.wake.set()
        sender = asyncio.ensure_future(self._sender(c))
        loop = asyncio.get_running_loop()
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    data = msg.data
                    if len(data) == protocol.ACK.size and data[0] == protocol.MSG_ACK:
                        c.on_ack(protocol.ACK.unpack(data)[1], loop.time())
                        c.wake.set()
                elif msg.type == WSMsgType.TEXT:
                    self._on_text(c, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
        except (asyncio.CancelledError, ConnectionError):
            pass
        finally:
            self._clients.pop(c.id, None)
            if c.timer is not None:
                c.timer.cancel()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        return ws

    def _on_text(self, c: _Client, text: str) -> None:
        try:
            msg = json.loads(text)
        except ValueError:
            return
        if not isinstance(msg, dict):
            return
        kind = msg.get("_")
        if kind is None:
            msg["client"] = c.id
            for cb in list(self._callbacks):
                try:
                    cb(msg)
                except Exception:  # noqa: BLE001 - a callback must not take the stream down
                    log.exception("on_message callback failed")
            return
        if kind == "ping":
            c.outbox.append(json.dumps({"_": "pong", "c": msg.get("c"), "rtt_ms": round(c.rtt_min * 1e3, 2),
                                        "hz": round(1.0 / c.interval, 1), "skipped": c.skipped}))
        elif kind == "rate":      # the page asks for at most this many frames a second
            try:
                c.user_hz = min(self.max_hz, max(1.0, float(msg.get("hz", self.max_hz))))
            except (TypeError, ValueError):
                return
            c.interval = max(c.interval, 1.0 / c.user_hz)
        elif kind == "pause":     # hidden tab: stop sending, resume from the newest state
            c.paused = bool(msg.get("on", True))
            if not c.paused:
                c.inflight.clear()
        elif kind == "key":       # the page lost track of its base frame
            c.need_key = True
            c.inflight.clear()
        c.wake.set()

    async def _h_scene(self, request: web.Request) -> web.Response:
        return self._bytes_response(request, self._scene_json, self._scene_gz, "application/json", "no-cache")

    async def _h_stats(self, request: web.Request) -> web.Response:
        return web.json_response(self.stats())

    async def _h_pack(self, request: web.Request) -> web.StreamResponse:
        lod = int(request.match_info["lod"])
        if lod >= N_LODS:
            raise web.HTTPNotFound()
        try:
            await asyncio.shield(self._meshes)
        except Exception as e:  # noqa: BLE001
            log.exception("mesh pack build failed")
            raise web.HTTPInternalServerError(text=f"mesh pack build failed: {e}") from e
        gz = "gzip" in request.headers.get("Accept-Encoding", "")
        path = self.export.pack_path(lod, gz=gz)
        headers = {"Content-Type": "application/octet-stream", "Cache-Control": "public, max-age=31536000, immutable",
                   "Vary": "Accept-Encoding"}
        if gz:
            headers["Content-Encoding"] = "gzip"
        data = await asyncio.get_running_loop().run_in_executor(None, path.read_bytes)
        return web.Response(body=data, headers=headers)

    async def _h_tex(self, request: web.Request) -> web.Response:
        png = self.export.textures.get(request.match_info["name"])
        if png is None:
            raise web.HTTPNotFound()
        return web.Response(body=png, content_type="image/png",
                            headers={"Cache-Control": "public, max-age=31536000, immutable"})

    async def _h_static(self, request: web.Request) -> web.Response:
        rel = request.match_info["path"] or "index.html"
        path = (STATIC_DIR / rel).resolve()
        if STATIC_DIR.resolve() not in path.parents or not path.is_file():
            raise web.HTTPNotFound()
        st = path.stat()
        hit = self._static_cache.get(rel)
        if hit is None or hit[0] != st.st_mtime or hit[1] != st.st_size:
            raw = path.read_bytes()
            gz = gzip.compress(raw, 6) if path.suffix in _COMPRESSIBLE else None
            hit = (st.st_mtime, st.st_size, raw, gz)
            self._static_cache[rel] = hit
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".js", ".mjs"):
            ctype = "text/javascript"
        fixed = rel.startswith(("vendor/", "fonts/"))
        cache = "public, max-age=604800" if fixed else "no-cache"
        return self._bytes_response(request, hit[2], hit[3], ctype, cache, etag=f'"{int(st.st_mtime)}-{st.st_size}"')

    @staticmethod
    def _bytes_response(request: web.Request, raw: bytes, gz: bytes | None, ctype: str, cache: str,
                        etag: str | None = None) -> web.Response:
        headers = {"Content-Type": ctype + ("; charset=utf-8" if ctype.startswith("text/") or ctype.endswith("json") else ""),
                   "Cache-Control": cache, "Vary": "Accept-Encoding"}
        if etag:
            headers["ETag"] = etag
            if request.headers.get("If-None-Match") == etag:
                return web.Response(status=304, headers=headers)
        if gz is not None and "gzip" in request.headers.get("Accept-Encoding", ""):
            headers["Content-Encoding"] = "gzip"
            return web.Response(body=gz, headers=headers)
        return web.Response(body=raw, headers=headers)


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-able: {type(o).__name__}")
