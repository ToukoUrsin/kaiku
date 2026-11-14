"""Serve a recording made by `kaiku.rgmp.record` as a live RGMP stream, with its original timing.

    python -m kaiku.rgmp.replay session.rgmp --port 12276 [--loop] [--speed 1.0] [--wait]

Frames go out byte for byte as recorded, each at its recorded arrival time divided by `--speed`.
Timing is scheduled against absolute deadlines, so it does not drift over a long recording.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from ._cli import stop_on_signals
from .protocol import DEFAULT_PORT, encode_disconnect
from .record import read_recording
from .server import RgmpServer

_SEND_TIMEOUT_S = 1.0
"""A replay prefers completeness to punctuality: at --speed 50 a client may briefly fall behind."""


def replay(
    path: str | Path,
    server: RgmpServer,
    *,
    speed: float = 1.0,
    loop: bool = False,
    stop: threading.Event | None = None,
) -> int:
    """Send the frames of a recording through a started `server`; returns how many were sent.

    Blocks until the recording ends or `stop` is set. With `loop` it starts over forever, and
    between passes every recorded device is disconnected first: its timestamps are about to jump
    back, which the spec only allows for a new session.
    """
    stop = stop or threading.Event()
    sent = 0
    while True:
        sent_before = sent
        started = time.monotonic()
        for arrival_ns, frame in read_recording(path):
            if stop.wait(max(0.0, started + arrival_ns / (1e9 * speed) - time.monotonic())):
                return sent
            server.send(frame)
            sent += 1
        if not loop or sent == sent_before:
            return sent
        for device_id in server.device_ids():
            server.send(encode_disconnect(device_id))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay an RGMP recording as a live stream.")
    parser.add_argument("path", help="recording made by kaiku.rgmp.record")
    parser.add_argument("--host", default="127.0.0.1", help="address to serve on (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to serve on (default: %(default)s)")
    parser.add_argument("--loop", action="store_true", help="start over when the recording ends")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed factor (default: %(default)s)")
    parser.add_argument("--wait", action="store_true", help="start playing when the first client connects")
    args = parser.parse_args(argv)
    if args.speed <= 0:
        parser.error("--speed must be positive")
    stop = stop_on_signals()
    with RgmpServer(args.host, args.port, send_timeout_s=_SEND_TIMEOUT_S) as server:
        print(f"replaying {args.path} on {args.host}:{server.port}, Ctrl-C to stop", file=sys.stderr)
        while args.wait and not stop.is_set() and not server.wait_for_client(0.1):
            pass
        sent = replay(args.path, server, speed=args.speed, loop=args.loop, stop=stop)
    print(f"replayed {sent} frames", file=sys.stderr)


if __name__ == "__main__":
    main()
