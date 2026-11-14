"""Background RGMP client: connects to a driver, keeps device definitions, hands out data frames.

One daemon thread owns the socket. It reconnects with exponential backoff, so a consumer can be
started before the Rokoko driver (or the emulator, or a replay) and survives its restarts. The
caller never touches the network: it either registers callbacks, which run on the reader thread,
or pulls the newest frame per device and group with `latest()`.

Fault policy: only a framing error (a nonsensical msg_len) drops the connection, because then the
byte stream cannot be resynchronised. A definition that fails validation, or a data frame that does
not match its group's layout, is skipped and logged once per device, so one odd device cannot take
the gloves down with it. Frame types we do not know are ignored, as the spec asks.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .protocol import (
    DATA_HEADER,
    DEFAULT_PORT,
    PREFIX_DATA,
    PREFIX_DEFINITION,
    PREFIX_DISCONNECT,
    Definition,
    FrameSplitter,
    Group,
    ProtocolError,
    decode_disconnect,
)

_log = logging.getLogger(__name__)

_RECV_BYTES = 1 << 16
_KEEPALIVE = (("TCP_KEEPIDLE", 2), ("TCP_KEEPALIVE", 2), ("TCP_KEEPINTVL", 1), ("TCP_KEEPCNT", 3))
"""Probe after 2 s of silence, give up after 3 more. KEEPIDLE is the Linux name, KEEPALIVE the macOS one."""


@dataclass(frozen=True, slots=True)
class DataFrame:
    """One received data frame, still packed, with the layout needed to read it."""

    definition: Definition
    group: Group
    timestamp_us: int       # capture time on the definition's `timestamp_epoch`
    payload: bytes          # the frame payload; stream values start after the 16-byte data header
    t_recv: float           # host `time.monotonic()` when the bytes arrived

    def values(self) -> tuple[Any, ...]:
        """All stream values as a flat tuple; index it with `Stream.slot` or `Stream.read`."""
        return self.group.decode(self.payload)


class RgmpClient:
    """Receives RGMP from `host:port` on a background thread.

    Callbacks (all optional, all invoked on the reader thread, so keep them short):
      on_definition(definition)         a device announced itself, or re-sent its definition
      on_data(frame)                    a data frame that matches a known device and group
      on_disconnect(device_id)          a disconnect frame arrived, or the connection was lost
      on_frame(prefix, payload, t_recv) every frame exactly as received, unknown types included

    An exception raised by a callback is logged and does not stop the stream.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        *,
        on_definition: Callable[[Definition], None] | None = None,
        on_data: Callable[[DataFrame], None] | None = None,
        on_disconnect: Callable[[int], None] | None = None,
        on_frame: Callable[[int, bytes, float], None] | None = None,
        reconnect_min_s: float = 0.05,
        reconnect_max_s: float = 1.0,
        connect_timeout_s: float = 1.0,
    ) -> None:
        self._address = (host, port)
        self._on_definition = on_definition
        self._on_data = on_data
        self._on_disconnect = on_disconnect
        self._on_frame = on_frame
        self._reconnect_min_s = reconnect_min_s
        self._reconnect_max_s = reconnect_max_s
        self._connect_timeout_s = connect_timeout_s
        self._lock = threading.Lock()                       # guards the four fields below
        self._socket: socket.socket | None = None
        self._connected = False
        self._definitions: dict[int, Definition] = {}
        self._latest: dict[int, list[DataFrame | None]] = {}   # device_id -> newest frame per group index
        self._warned: set[tuple[int | None, str]] = set()   # reader thread only: what was already logged
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> RgmpClient:
        """Start the reader thread and return self. Does not wait for a connection."""
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="rgmp-client", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        """Close the connection and join the reader thread."""
        self._stop.set()
        with self._lock:
            live = self._socket
        if live is not None:
            try:
                live.shutdown(socket.SHUT_RDWR)             # wakes the reader out of recv()
            except OSError:
                pass                                        # the peer closed first
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def __enter__(self) -> RgmpClient:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    @property
    def connected(self) -> bool:
        """True while the TCP connection to the server is up."""
        return self._connected

    def definitions(self) -> dict[int, Definition]:
        """Snapshot of the devices currently announced, keyed by device_id."""
        with self._lock:
            return dict(self._definitions)

    def latest(self, device_id: int, group: str = "default") -> DataFrame | None:
        """The newest data frame of one device's group, or None if there is none (yet, or any more)."""
        with self._lock:
            definition = self._definitions.get(device_id)
            target = definition.group(group) if definition is not None else None
            return self._latest[device_id][target.index] if target is not None else None

    def _run(self) -> None:
        backoff = self._reconnect_min_s
        while not self._stop.is_set():
            if self._session():
                backoff = self._reconnect_min_s             # the server was alive: retry quickly
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, self._reconnect_max_s)

    def _session(self) -> bool:
        """Connect and read until the connection ends. True if at least one frame arrived."""
        try:
            sock = socket.create_connection(self._address, timeout=self._connect_timeout_s)
        except OSError as error:
            _log.debug("RGMP connect to %s:%d failed: %s", *self._address, error)
            return False
        self._warned.clear()
        received = False
        with sock:
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)   # notice a peer that vanished
            for name, value in _KEEPALIVE:
                if hasattr(socket, name):
                    sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
            with self._lock:
                self._socket = sock
                self._connected = True
            _log.info("RGMP connected to %s:%d", *self._address)
            splitter = FrameSplitter()
            buffer = bytearray(_RECV_BYTES)
            view = memoryview(buffer)
            try:
                while not self._stop.is_set():
                    count = sock.recv_into(buffer)
                    if count == 0:
                        break                               # orderly close by the server
                    t_recv = time.monotonic()
                    for prefix, payload in splitter.feed(view[:count]):
                        received = True
                        try:
                            self._dispatch(prefix, payload, t_recv)
                        except Exception as error:
                            if self._first_time(prefix, type(error).__name__):
                                _log.exception("RGMP callback failed on frame type %d", prefix)
            except ProtocolError as error:
                _log.warning("RGMP stream from %s:%d dropped: %s", *self._address, error)
            except OSError as error:
                _log.info("RGMP connection to %s:%d lost: %s", *self._address, error)
            finally:
                self._forget_all()
        return received

    def _dispatch(self, prefix: int, payload: bytes, t_recv: float) -> None:
        if self._on_frame is not None:
            self._on_frame(prefix, payload, t_recv)
        if prefix == PREFIX_DATA:
            self._handle_data(payload, t_recv)
        elif prefix == PREFIX_DEFINITION:
            self._handle_definition(payload)
        elif prefix == PREFIX_DISCONNECT:
            self._forget(decode_disconnect(payload))
        # Any other prefix is a frame type newer than this client; the spec says to skip it.

    def _handle_definition(self, payload: bytes) -> None:
        try:
            definition = Definition.from_payload(payload)
        except ProtocolError as error:
            _log.warning("ignoring a device with an invalid definition: %s", error)
            return
        with self._lock:                                    # a re-sent definition starts a new layout
            self._definitions[definition.device_id] = definition
            self._latest[definition.device_id] = [None] * len(definition.groups)
        _log.info("RGMP device %d: %s", definition.device_id, definition.device_type)
        if self._on_definition is not None:
            self._on_definition(definition)

    def _handle_data(self, payload: bytes, t_recv: float) -> None:
        problem = device_id = None
        if len(payload) < DATA_HEADER.size:
            problem = "a data frame shorter than its header"
        else:
            device_id, group_id, timestamp_us = DATA_HEADER.unpack_from(payload)
            definition = self._definitions.get(device_id)
            if definition is None:
                problem = "data that arrived before its stream definition"
            elif group_id >= len(definition.groups):
                problem = f"data for undeclared group {group_id}"
            elif len(payload) != definition.groups[group_id].payload_size:
                problem = f"group {group_id} frames of {len(payload)} bytes that do not fit the declared layout"
        if problem is not None:
            if self._first_time(device_id, problem):
                _log.warning("RGMP device %s: skipping %s", device_id, problem)
            return
        frame = DataFrame(definition, definition.groups[group_id], timestamp_us, payload, t_recv)
        with self._lock:
            self._latest[device_id][group_id] = frame
        if self._on_data is not None:
            self._on_data(frame)

    def _first_time(self, subject: int | None, problem: str) -> bool:
        """True once per connection for each (subject, problem), so a fault repeating at frame rate logs once."""
        if (subject, problem) in self._warned:
            return False
        self._warned.add((subject, problem))
        return True

    def _forget(self, device_id: int) -> None:
        with self._lock:
            self._definitions.pop(device_id, None)
            self._latest.pop(device_id, None)
        if self._on_disconnect is not None:
            self._on_disconnect(device_id)

    def _forget_all(self) -> None:
        """The connection is gone, so every device is: report each one as disconnected."""
        with self._lock:
            self._socket = None
            self._connected = False
            device_ids = list(self._definitions)
        for device_id in device_ids:
            try:
                self._forget(device_id)
            except Exception:
                _log.exception("RGMP disconnect callback failed for device %d", device_id)
