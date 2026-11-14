"""Record an RGMP stream to disk frame for frame, so real glove sessions can be replayed later.

    python -m kaiku.rgmp.record --host 127.0.0.1 --port 12276 --out session.rgmp [--seconds 30]

File format, version 1, little endian:

    header   magic "KAIKURGM", version:u32, started_unix_ns:i64
    record   arrival_ns:i64, then the frame exactly as it was on the wire
             (msg_prefix:u32, msg_len:u32, payload[msg_len])

`arrival_ns` counts from the first recorded frame. Every frame is stored untouched and unparsed,
unknown frame types included, so `kaiku.rgmp.replay` reproduces the driver's stream byte for
byte at frame level. Each record is flushed as it is written: after Ctrl-C, a crash or a full
disk the file is valid up to the last frame, and the reader ignores a truncated final record.
"""
from __future__ import annotations

import argparse
import struct
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from ._cli import stop_on_signals
from .client import RgmpClient
from .protocol import DEFAULT_PORT, FRAME_HEADER

MAGIC = b"KAIKURGM"
VERSION = 1
_FILE_HEADER = struct.Struct("<8sIq")       # magic, version, wall clock at start (unix ns)
_RECORD_HEADER = struct.Struct("<qII")      # arrival_ns, then the frame's own header: msg_prefix, msg_len
_POLL_S = 0.02


class RecordingWriter:
    """Appends frames to a recording file. `write` may be called from another thread than `close`."""

    def __init__(self, path: str | Path) -> None:
        self._file = open(path, "wb")
        self._file.write(_FILE_HEADER.pack(MAGIC, VERSION, time.time_ns()))
        self._file.flush()
        self._lock = threading.Lock()
        self._first_arrival: float | None = None
        self.frames = 0

    def write(self, prefix: int, payload: bytes, t_recv: float) -> None:
        """Store one frame. The signature is `RgmpClient`'s `on_frame` callback."""
        with self._lock:
            if self._file.closed:
                return
            if self._first_arrival is None:
                self._first_arrival = t_recv
            arrival_ns = round((t_recv - self._first_arrival) * 1e9)
            self._file.write(_RECORD_HEADER.pack(arrival_ns, prefix, len(payload)))
            self._file.write(payload)
            self._file.flush()
            self.frames += 1

    def elapsed_s(self) -> float:
        """Seconds since the first frame arrived; 0.0 until then."""
        first = self._first_arrival
        return 0.0 if first is None else time.monotonic() - first

    def close(self) -> None:
        with self._lock:
            self._file.close()


def read_recording(path: str | Path) -> Iterator[tuple[int, bytes]]:
    """Yield (arrival_ns, complete wire frame) for every record of a recording, in order.

    Raises ValueError if the file is not a recording of a known version. A record cut short at
    the end of the file is dropped silently: that is what an interrupted recording looks like.
    """
    with open(path, "rb") as file:
        header = file.read(_FILE_HEADER.size)
        if len(header) < _FILE_HEADER.size or not header.startswith(MAGIC):
            raise ValueError(f"{path} is not a kaiku RGMP recording")
        version = _FILE_HEADER.unpack(header)[1]
        if version != VERSION:
            raise ValueError(f"{path} is recording format version {version}, this reader knows {VERSION}")
        while True:
            head = file.read(_RECORD_HEADER.size)
            if len(head) < _RECORD_HEADER.size:
                return
            arrival_ns, _, length = _RECORD_HEADER.unpack(head)
            payload = file.read(length)
            if len(payload) < length:
                return
            yield arrival_ns, head[-FRAME_HEADER.size:] + payload


def record(
    host: str,
    port: int,
    out: str | Path,
    *,
    seconds: float | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Record the RGMP stream at `host:port` into `out` and return the number of frames stored.

    Runs until `stop` is set or, if `seconds` is given, until that much time has passed since the
    first frame (so waiting for the driver or the gloves does not eat into the recording). The
    client reconnects by itself, so a driver restart mid-recording just shows up as new definitions.
    """
    stop = stop or threading.Event()
    writer = RecordingWriter(out)
    try:
        with RgmpClient(host, port, on_frame=writer.write):
            while not stop.wait(_POLL_S):
                if seconds is not None and writer.elapsed_s() >= seconds:
                    break
    finally:
        writer.close()
    return writer.frames


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record an RGMP stream to a file, until Ctrl-C or --seconds.")
    parser.add_argument("--host", default="127.0.0.1", help="RGMP server host (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="RGMP server port (default: %(default)s)")
    parser.add_argument("--out", required=True, help="recording file to write, e.g. session.rgmp")
    parser.add_argument("--seconds", type=float, help="stop this long after the first frame (default: run until Ctrl-C)")
    args = parser.parse_args(argv)
    print(f"recording {args.host}:{args.port} to {args.out}, Ctrl-C to stop", file=sys.stderr)
    frames = record(args.host, args.port, args.out, seconds=args.seconds, stop=stop_on_signals())
    print(f"recorded {frames} frames to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
