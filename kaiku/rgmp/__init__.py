"""RGMP (Rokoko General Motion Protocol) for kaiku: Smartgloves in, `GloveFrame` out.

    protocol   wire format: framing, data types, definitions, data-frame layouts (no I/O)
    client     RgmpClient: background, reconnecting receiver with callbacks and a pull API
    glove      GloveDecoder / GloveStream: a glove's `default` group as kaiku.types.GloveFrame
    server     RgmpServer: multi-client fan-out used by the emulator and the replayer
    emulator   a synthetic Smartgloves II pair       (python -m kaiku.rgmp.emulator)
    record     store a stream frame for frame        (python -m kaiku.rgmp.record)
    replay     serve a recording with its timing     (python -m kaiku.rgmp.replay)

The three command line modules are not imported here, so `python -m` runs them cleanly.
"""
from .client import DataFrame, RgmpClient
from .glove import GloveDecoder, GloveStream
from .protocol import DEFAULT_PORT, Definition, Group, ProtocolError, Stream, stream_key
from .server import RgmpServer

__all__ = [
    "DEFAULT_PORT",
    "DataFrame",
    "Definition",
    "GloveDecoder",
    "GloveStream",
    "Group",
    "ProtocolError",
    "RgmpClient",
    "RgmpServer",
    "Stream",
    "stream_key",
]
