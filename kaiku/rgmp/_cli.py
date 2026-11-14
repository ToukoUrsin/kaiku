"""Shared plumbing of the `python -m kaiku.rgmp.*` command line tools."""
from __future__ import annotations

import signal
import threading


def stop_on_signals() -> threading.Event:
    """An event that Ctrl-C or SIGTERM sets, so a tool winds down (flushes files, sends disconnect
    frames, closes sockets) instead of being torn out of whatever it was doing. Main thread only."""
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    return stop
