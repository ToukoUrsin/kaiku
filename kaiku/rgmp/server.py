"""Small multi-client RGMP server: fans already-encoded frames out to every connected client.

It stands in for the Rokoko driver behind the emulator and the replayer. Like the driver it
accepts any number of clients, and a client that connects mid-session first receives every
current stream definition and only then live data.

Latency: `send()` writes to the client sockets on the caller's thread, with TCP_NODELAY and no
queue in between. A client that cannot take a frame within `send_timeout_s` is disconnected
instead of buffered: motion data that is late is worthless, a half-sent frame cannot be repaired,
and a reconnecting client gets fresh definitions anyway. The default of 0 never waits at all.
"""
from __future__ import annotations

import json
import select
import socket
import threading

from .protocol import (
    DEFAULT_PORT,
    FRAME_HEADER,
    PREFIX_DEFINITION,
    PREFIX_DISCONNECT,
    ProtocolError,
    decode_disconnect,
)

_GREETING_TIMEOUT_S = 1.0
"""How long a new client may take to accept the definitions, which can exceed an empty send buffer."""


def _defined_device(frame: bytes) -> int:
    """device_id of a definition frame. Deliberately not validated further: a replay must be able
    to serve whatever a real driver sent, including definitions our own parser would reject."""
    try:
        return json.loads(frame[FRAME_HEADER.size:])["device_id"]
    except (ValueError, KeyError, TypeError) as error:
        raise ProtocolError(f"definition frame without a device_id: {error!r}") from error


class RgmpServer:
    """Serves RGMP on `host:port`; pass port 0 to let the OS pick one and read it from `port`."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, *, send_timeout_s: float = 0.0) -> None:
        self._address = (host, port)
        self._send_timeout_s = send_timeout_s
        self._lock = threading.Lock()                       # guards the fields below and orders all sends
        self._clients: tuple[socket.socket, ...] = ()       # replaced, never mutated: safe to iterate
        self._definitions: dict[int, bytes] = {}            # device_id -> its current definition frame
        self._has_client = threading.Event()
        self._sockets: tuple[socket.socket, ...] = ()       # listener and the wake-up pair, while running
        self._thread: threading.Thread | None = None
        self.port = port

    def start(self) -> RgmpServer:
        """Bind, listen and start accepting; returns self. Raises OSError if the port is taken."""
        if self._thread is None:
            listener = socket.create_server(self._address)   # sets SO_REUSEADDR: restart on the same port
            wake_reader, wake_writer = socket.socketpair()
            self.port = listener.getsockname()[1]
            self._sockets = (listener, wake_reader, wake_writer)
            self._thread = threading.Thread(
                target=self._accept, args=(listener, wake_reader), name="rgmp-server", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Close the listener and every client connection, and forget all definitions."""
        if self._thread is None:
            return
        self._sockets[-1].send(b"\0")                       # wakes the accept thread out of select()
        self._thread.join()
        self._thread = None
        with self._lock:
            for sock in (*self._sockets, *self._clients):
                sock.close()
            self._sockets = self._clients = ()
            self._definitions.clear()
            self._has_client.clear()

    def __enter__(self) -> RgmpServer:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    @property
    def client_count(self) -> int:
        """Connected clients. One that vanished silently is only noticed on the next `send`."""
        return len(self._clients)

    def wait_for_client(self, timeout: float | None = None) -> bool:
        """Block until at least one client is connected; False if `timeout` seconds pass first."""
        return self._has_client.wait(timeout)

    def device_ids(self) -> list[int]:
        """Devices that are currently defined, i.e. announced and not yet disconnected."""
        with self._lock:
            return list(self._definitions)

    def send(self, frame: bytes) -> None:
        """Broadcast one complete frame (outer header included) to every client.

        Definition and disconnect frames also update what a later client is greeted with: the
        newest definition per device, minus the devices that have disconnected.
        """
        prefix = FRAME_HEADER.unpack_from(frame)[0]
        with self._lock:
            if prefix == PREFIX_DEFINITION:
                self._definitions[_defined_device(frame)] = frame
            elif prefix == PREFIX_DISCONNECT:
                self._definitions.pop(decode_disconnect(frame[FRAME_HEADER.size:]), None)
            for client in self._clients:
                try:
                    client.sendall(frame)
                except OSError:                             # gone, or too slow to stay real-time
                    self._drop(client)

    def _accept(self, listener: socket.socket, wake: socket.socket) -> None:
        while True:
            readable, _, _ = select.select((listener, wake), (), ())
            if wake in readable:
                return
            try:
                client, _ = listener.accept()
            except OSError:
                continue                                    # the client gave up before we got to it
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.settimeout(_GREETING_TIMEOUT_S)
            with self._lock:                                # no data frame may overtake the definitions
                try:
                    for definition in self._definitions.values():
                        client.sendall(definition)
                    client.settimeout(self._send_timeout_s)
                except OSError:
                    client.close()
                    continue
                self._clients += (client,)
                self._has_client.set()

    def _drop(self, client: socket.socket) -> None:
        """Forget a client; the caller holds the lock."""
        client.close()
        self._clients = tuple(other for other in self._clients if other is not client)
        if not self._clients:
            self._has_client.clear()
