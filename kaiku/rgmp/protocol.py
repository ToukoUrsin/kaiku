"""RGMP v2 wire format: framing, data types, stream definitions and data-frame layouts.

RGMP (Rokoko General Motion Protocol) is a little-endian, length-prefixed stream over TCP::

    frame      msg_prefix:u32  msg_len:u32  payload[msg_len]
    prefix 1   stream definition: UTF-8 JSON describing one device and its stream groups
    prefix 2   data frame: device_id:u32  group_id:u32  timestamp_us:u64, then the value of
               every stream of that group, tightly packed in definition order
    prefix 3   device disconnect: device_id:u32

This module is pure computation (no sockets, no threads), so the client, the server, the recorder
and the emulator share one implementation of the format. It is written from the public
specification (ref/rokoko-docs/reference_rgmp.txt), not from Rokoko's example client.
"""
from __future__ import annotations

import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

PREFIX_DEFINITION = 1
PREFIX_DATA = 2
PREFIX_DISCONNECT = 3

DEFAULT_PORT = 12276
"""TCP port the Rokoko SDK driver serves RGMP on."""

MAX_FRAME_BYTES = 16 * 1024 * 1024
"""Sanity bound on msg_len. Real frames are a few kilobytes; anything larger means we lost framing."""

FRAME_HEADER = struct.Struct("<II")       # msg_prefix, msg_len
DATA_HEADER = struct.Struct("<IIQ")       # device_id, group_id, timestamp_us
_DISCONNECT = struct.Struct("<I")         # device_id
_FRAME_AND_DATA_HEADER = "<IIIIQ"

_BASE_FORMATS = {"INT32": "i", "UINT32": "I", "INT64": "q", "UINT64": "Q", "FLOAT": "f", "DOUBLE": "d"}
_DATA_TYPE_RE = re.compile(r"(INT32|UINT32|INT64|UINT64|FLOAT|DOUBLE)(?:\[(\d+)(?:,\s*(\d+))?\])?")

StreamKey = tuple[str, str, str | None, str | None]
"""(measure_type, target_frame, reference_frame, custom_label): the identity of a stream in its group."""


class ProtocolError(ValueError):
    """The peer sent something that violates the RGMP v2 specification."""


@dataclass(frozen=True, slots=True)
class DataType:
    """A parsed `data_type` such as ``UINT32``, ``FLOAT[7]`` or ``DOUBLE[3, 3]``."""

    base: str                  # INT32 | UINT32 | INT64 | UINT64 | FLOAT | DOUBLE
    shape: tuple[int, ...]     # () scalar, (n,) vector, (rows, cols) matrix in row-major order
    count: int                 # number of base values
    size: int                  # bytes on the wire
    fmt: str                   # `struct` code for all values, without the byte-order prefix

    @property
    def is_integer(self) -> bool:
        return self.base not in ("FLOAT", "DOUBLE")


@lru_cache(maxsize=None)
def parse_data_type(text: str) -> DataType:
    """Parse a `data_type` string; raises ProtocolError for anything the spec does not allow."""
    match = _DATA_TYPE_RE.fullmatch(text.strip())
    if match is None:
        raise ProtocolError(f"unrecognised data_type {text!r}")
    base, *dims = match.groups()
    shape = tuple(int(dim) for dim in dims if dim is not None)
    if 0 in shape:
        raise ProtocolError(f"data_type {text!r} has a zero dimension")
    count = 1
    for dim in shape:
        count *= dim
    fmt = f"{count}{_BASE_FORMATS[base]}"
    return DataType(base, shape, count, struct.calcsize("<" + fmt), fmt)


def stream_key(
    measure_type: str,
    target_frame: str,
    reference_frame: str | None = None,
    custom_label: str | None = None,
) -> StreamKey:
    """Build the lookup key of a stream.

    The spec treats an omitted `reference_frame` and one equal to `target_frame` as the same
    thing, so both normalise to None. Apart from that the match is strict: asking without a
    reference frame never finds a stream that declared one.
    """
    if reference_frame == target_frame:
        reference_frame = None
    return (measure_type, target_frame, reference_frame or None, custom_label or None)


@dataclass(frozen=True, slots=True)
class Stream:
    """One stream of a group, with its position inside that group's data frames."""

    measure_type: str
    target_frame: str
    reference_frame: str | None      # None: the data is intrinsic to target_frame
    custom_label: str | None         # set exactly when measure_type is CUSTOM
    data_type: DataType
    offset: int                      # bytes from the first stream value of the data frame
    slot: int                        # index of this stream's first value in `Group.decode()`
    flags: Mapping[str, int]         # STATUS_FLAGS only: flag name -> bit index

    @property
    def key(self) -> StreamKey:
        return (self.measure_type, self.target_frame, self.reference_frame, self.custom_label)

    def mask(self, flag: str) -> int:
        """Bit mask of a named status flag, or 0 when this stream does not declare it."""
        bit = self.flags.get(flag)
        return 0 if bit is None else 1 << bit

    def read(self, values: Sequence[Any]) -> Any:
        """This stream's value out of `Group.decode()`: a number, or a tuple for arrays."""
        if not self.data_type.shape:
            return values[self.slot]
        return tuple(values[self.slot:self.slot + self.data_type.count])


def _parse_entry(entry: Mapping[str, Any]) -> tuple[StreamKey, DataType, dict[str, int]]:
    """Validate the fields shared by stream and static_data entries."""
    measure_type, label = entry["measure_type"], entry.get("custom_label") or None
    if (measure_type == "CUSTOM") != (label is not None):
        raise ProtocolError(
            f"{measure_type} stream for {entry['target_frame']!r}: custom_label is required on "
            "CUSTOM streams and forbidden on every other measure type"
        )
    key = stream_key(measure_type, entry["target_frame"], entry.get("reference_frame"), label)
    flags = {name: int(bit) for bit, name in (entry.get("bit_mapping") or {}).items()}
    return key, parse_data_type(entry["data_type"]), flags


class Group:
    """One stream group of a device, i.e. the layout of one kind of data frame.

    The whole group is compiled into a single `struct.Struct`, so a frame is decoded or encoded
    in one C call. `struct` with an explicit byte order never pads, which is exactly the spec's
    "tightly packed" rule: a DOUBLE that follows a FLOAT simply starts 4 bytes off an 8-byte
    boundary, and values are copied out rather than read through an aligned pointer.
    """

    __slots__ = ("index", "name", "expected_rate_hz", "streams", "size", "_by_key", "_values", "_frame")

    def __init__(self, index: int, entry: Mapping[str, Any]) -> None:
        self.index = index
        self.name: str = entry["name"]
        self.expected_rate_hz = float(entry["expected_rate_hz"])   # 0 means event driven
        streams: list[Stream] = []
        self._by_key: dict[StreamKey, Stream] = {}
        offset = slot = 0
        for stream_entry in entry["streams"]:
            key, data_type, flags = _parse_entry(stream_entry)
            if key in self._by_key:
                raise ProtocolError(f"group {self.name!r} declares stream {key} twice")
            stream = Stream(*key, data_type, offset, slot, flags)
            streams.append(stream)
            self._by_key[key] = stream
            offset += data_type.size
            slot += data_type.count
        self.streams = tuple(streams)
        self.size = offset                                          # bytes of stream values per frame
        layout = "".join(stream.data_type.fmt for stream in streams)
        self._values = struct.Struct("<" + layout)
        self._frame = struct.Struct(_FRAME_AND_DATA_HEADER + layout)

    @property
    def payload_size(self) -> int:
        """Exact payload length of a data frame of this group (data header + stream values)."""
        return DATA_HEADER.size + self.size

    def find(
        self,
        measure_type: str,
        target_frame: str,
        reference_frame: str | None = None,
        custom_label: str | None = None,
    ) -> Stream | None:
        """Look a stream up by its full key (see `stream_key`); None when it is not declared."""
        return self._by_key.get(stream_key(measure_type, target_frame, reference_frame, custom_label))

    def require(
        self,
        measure_type: str,
        target_frame: str,
        reference_frame: str | None = None,
        custom_label: str | None = None,
    ) -> Stream:
        """Like `find`, but a missing stream is a ProtocolError naming the key."""
        key = stream_key(measure_type, target_frame, reference_frame, custom_label)
        stream = self._by_key.get(key)
        if stream is None:
            raise ProtocolError(f"group {self.name!r} has no stream {key}")
        return stream

    def zeros(self) -> list[float]:
        """A flat value list for `encode`, all zero, with an int in every integer slot."""
        return [
            0 if stream.data_type.is_integer else 0.0
            for stream in self.streams
            for _ in range(stream.data_type.count)
        ]

    def encode(self, device_id: int, timestamp_us: int, values: Sequence[float]) -> bytes:
        """Build a complete data frame (outer header included) from flat values in slot order."""
        return self._frame.pack(PREFIX_DATA, self.payload_size, device_id, self.index, timestamp_us, *values)

    def decode(self, payload: bytes | bytearray | memoryview) -> tuple[Any, ...]:
        """Unpack every stream value of a data-frame payload into one flat tuple.

        Index the result with `Stream.slot`, or use `Stream.read`. The payload must be exactly
        this group's size: a mismatch means the layout we computed is not the sender's.
        """
        if len(payload) != self.payload_size:
            raise ProtocolError(
                f"group {self.name!r} data frame is {len(payload)} bytes, layout says {self.payload_size}"
            )
        return self._values.unpack_from(payload, DATA_HEADER.size)


class Definition:
    """A parsed Stream Definition Frame: everything the sender declared about one device."""

    __slots__ = (
        "document", "device_id", "device_type", "timestamp_epoch", "device_info", "static_data", "groups",
    )

    def __init__(self, document: Mapping[str, Any]) -> None:
        try:
            version = str(document["protocol_version"])
            if version.split(".")[0] != "2":
                raise ProtocolError(f"unsupported RGMP protocol_version {version!r}")
            device_id = document["device_id"]
            if type(device_id) is not int or not 0 <= device_id <= 0xFFFFFFFF:
                raise ProtocolError(f"device_id {device_id!r} is not a 32-bit unsigned integer")
            self.document = document                                   # the JSON as received
            self.device_id: int = device_id
            self.device_type: str = document["device_type"]
            self.timestamp_epoch: str = document["timestamp_epoch"]    # "unix_epoch" | "device_boot"
            self.device_info: Mapping[str, Any] = document.get("device_info") or {}
            self.static_data: dict[StreamKey, Any] = {
                _parse_entry(entry)[0]: entry["value"] for entry in document.get("static_data") or ()
            }
            self.groups = tuple(Group(index, entry) for index, entry in enumerate(document["groups"]))
        except ProtocolError:
            raise
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ProtocolError(f"malformed stream definition: {error!r}") from error

    @classmethod
    def from_payload(cls, payload: bytes | bytearray | memoryview) -> Definition:
        """Parse the payload of a definition frame."""
        try:
            document = json.loads(bytes(payload).decode("utf-8"))
        except ValueError as error:
            raise ProtocolError(f"stream definition is not UTF-8 JSON: {error}") from error
        return cls(document)

    def encode(self) -> bytes:
        """Build the complete definition frame (outer header included)."""
        text = json.dumps(self.document, separators=(",", ":"), ensure_ascii=False)
        return encode_frame(PREFIX_DEFINITION, text.encode("utf-8"))

    def group(self, name: str) -> Group | None:
        """The group with this name, or None. Data frames address groups by `Group.index`."""
        return next((group for group in self.groups if group.name == name), None)


def encode_frame(prefix: int, payload: bytes) -> bytes:
    """Wrap a payload in the outer header; also the way to emit frame types this module does not know."""
    return FRAME_HEADER.pack(prefix, len(payload)) + payload


def encode_disconnect(device_id: int) -> bytes:
    """Build a complete Device Disconnect frame."""
    return FRAME_HEADER.pack(PREFIX_DISCONNECT, _DISCONNECT.size) + _DISCONNECT.pack(device_id)


def decode_disconnect(payload: bytes | bytearray | memoryview) -> int:
    """The device_id carried by a Device Disconnect payload."""
    if len(payload) < _DISCONNECT.size:
        raise ProtocolError(f"disconnect frame of {len(payload)} bytes carries no device_id")
    return _DISCONNECT.unpack_from(payload)[0]


class FrameSplitter:
    """Reassembles frames from a TCP byte stream, however the network chunked it."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[tuple[int, bytes]]:
        """Add received bytes and return every frame they complete, as (msg_prefix, payload).

        Bytes of an unfinished frame stay buffered for the next call. A msg_len beyond
        MAX_FRAME_BYTES raises ProtocolError: the stream is out of sync and must be dropped.
        """
        buffer = self._buffer
        buffer += chunk
        frames: list[tuple[int, bytes]] = []
        start = 0
        with memoryview(buffer) as view:
            while len(view) - start >= FRAME_HEADER.size:
                prefix, length = FRAME_HEADER.unpack_from(view, start)
                if length > MAX_FRAME_BYTES:
                    raise ProtocolError(f"frame type {prefix} claims {length} bytes; stream is out of sync")
                end = start + FRAME_HEADER.size + length
                if end > len(view):
                    break
                frames.append((prefix, bytes(view[start + FRAME_HEADER.size:end])))
                start = end
        del buffer[:start]
        return frames
