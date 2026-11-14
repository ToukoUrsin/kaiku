"""RGMP wire format: data types, layouts, stream keys and the three frame types (no sockets)."""
from __future__ import annotations

import json
import struct

import pytest

from kaiku.rgmp.protocol import (
    DATA_HEADER,
    FRAME_HEADER,
    MAX_FRAME_BYTES,
    PREFIX_DATA,
    PREFIX_DEFINITION,
    PREFIX_DISCONNECT,
    Definition,
    FrameSplitter,
    ProtocolError,
    decode_disconnect,
    encode_disconnect,
    encode_frame,
    parse_data_type,
    stream_key,
)


def stream(data_type: str, measure_type: str, target: str, reference: str | None = None, **extra) -> dict:
    entry = {"data_type": data_type, "measure_type": measure_type, "target_frame": target, **extra}
    if reference is not None:
        entry["reference_frame"] = reference
    return entry


def document(streams: list[dict], **overrides) -> dict:
    return {
        "protocol_name": "RGMP",
        "protocol_version": "2.0.0",
        "device_id": 7,
        "device_type": "Test Device",
        "timestamp_epoch": "unix_epoch",
        "groups": [{"name": "default", "expected_rate_hz": 100, "streams": streams}],
        **overrides,
    }


@pytest.mark.parametrize(
    ("text", "shape", "count", "size"),
    [
        ("INT32", (), 1, 4),
        ("UINT32", (), 1, 4),
        ("INT64", (), 1, 8),
        ("UINT64[1]", (1,), 1, 8),
        ("FLOAT[7]", (7,), 7, 28),
        ("DOUBLE[3, 3]", (3, 3), 9, 72),      # as written in the spec's prose
        ("DOUBLE[3,3]", (3, 3), 9, 72),       # as the JSON schema's pattern requires
        ("INT32[2,5]", (2, 5), 10, 40),
    ],
)
def test_data_types_parse_to_shape_and_size(text, shape, count, size):
    data_type = parse_data_type(text)
    assert (data_type.shape, data_type.count, data_type.size) == (shape, count, size)
    assert data_type.is_integer == ("INT" in text)


@pytest.mark.parametrize("text", ["FLOAT[0]", "FLOAT[3,0]", "FLOAT[0,3]", "FLOAT[]", "FLOAT[1,2,3]", "float", "BYTE", "FLOAT[-1]"])
def test_invalid_data_types_are_protocol_errors(text):
    with pytest.raises(ProtocolError):
        parse_data_type(text)


def test_layout_is_tightly_packed_with_unaligned_64_bit_values():
    """A DOUBLE after a FLOAT starts 4 bytes off an 8-byte boundary; nothing may be padded."""
    group = Definition(document([
        stream("FLOAT", "CUSTOM", "a", custom_label="scalar"),
        stream("DOUBLE", "CUSTOM", "a", custom_label="unaligned double"),
        stream("INT32[3]", "CUSTOM", "a", custom_label="ints"),
        stream("UINT64", "STATUS_FLAGS", "a", bit_mapping={"63": "top"}),
        stream("INT64[2,2]", "CUSTOM", "a", custom_label="matrix"),
        stream("UINT32", "CUSTOM", "a", custom_label="u32"),
        stream("FLOAT[2, 3]", "CUSTOM", "a", custom_label="float matrix"),
        stream("DOUBLE[2]", "POSITION", "a", "world"),
    ])).groups[0]
    assert [s.offset for s in group.streams] == [0, 4, 12, 24, 32, 64, 68, 92]
    assert [s.slot for s in group.streams] == [0, 1, 2, 5, 6, 10, 11, 17]
    assert group.size == 108

    values = [
        0.5, 1 / 3, -1, 0, 2**31 - 1, 2**64 - 1, -(2**63), -1, 0, 2**63 - 1,
        2**32 - 1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1e-300, -1e300,
    ]
    frame = group.encode(0xFFFFFFFF, 2**64 - 1, values)
    by_hand = struct.pack("<IIIIQ" "f" "d" "3i" "Q" "4q" "I" "6f" "2d", PREFIX_DATA, 16 + 108, 0xFFFFFFFF, 0, 2**64 - 1, *values)
    assert frame == by_hand

    payload = frame[FRAME_HEADER.size:]
    assert DATA_HEADER.unpack_from(payload) == (0xFFFFFFFF, 0, 2**64 - 1)
    decoded = group.decode(payload)
    assert list(decoded) == values
    matrix = group.require("CUSTOM", "a", None, "matrix")
    assert matrix.read(decoded) == (-(2**63), -1, 0, 2**63 - 1)      # row-major
    assert group.require("CUSTOM", "a", None, "unaligned double").read(decoded) == 1 / 3
    flags = group.require("STATUS_FLAGS", "a")
    assert flags.read(decoded) & flags.mask("top") and flags.mask("not declared") == 0


def test_decode_rejects_a_payload_that_does_not_fit_the_layout():
    group = Definition(document([stream("FLOAT[3]", "POSITION", "a")])).groups[0]
    frame = group.encode(7, 1, group.zeros())
    with pytest.raises(ProtocolError):
        group.decode(frame[FRAME_HEADER.size:-4])
    with pytest.raises(ProtocolError):
        group.decode(frame[FRAME_HEADER.size:] + b"\0\0\0\0")


def test_stream_lookup_is_strict_on_every_key_field():
    group = Definition(document([
        stream("FLOAT[3]", "POSITION", "hand"),                          # reference omitted
        stream("FLOAT[3]", "POSITION", "hand", "LTP_ENU"),
        stream("FLOAT[3]", "POSITION", "hand", "LTP_NED"),
        stream("FLOAT[3]", "LINEAR_VELOCITY", "hand", "hand"),           # reference spelled out as itself
        stream("FLOAT", "CUSTOM", "hand", "LTP_ENU", custom_label="battery_pct"),
        stream("FLOAT", "CUSTOM", "hand", "LTP_ENU", custom_label="emf_residual"),
    ])).groups[0]
    intrinsic, enu, ned, velocity, battery, residual = group.streams

    assert group.find("POSITION", "hand") is intrinsic
    assert group.find("POSITION", "hand", "hand") is intrinsic           # same meaning as omitted
    assert group.find("POSITION", "hand", "LTP_ENU") is enu
    assert group.find("POSITION", "hand", "LTP_NED") is ned
    assert group.find("POSITION", "hand", "world") is None
    assert group.find("LINEAR_VELOCITY", "hand") is velocity
    assert group.find("LINEAR_VELOCITY", "hand", "LTP_ENU") is None      # declared intrinsic only
    assert group.find("ORIENTATION", "hand") is None
    assert group.find("POSITION", "head") is None
    assert group.find("CUSTOM", "hand", "LTP_ENU", "battery_pct") is battery
    assert group.find("CUSTOM", "hand", "LTP_ENU", "emf_residual") is residual
    assert group.find("CUSTOM", "hand", "LTP_ENU") is None               # the label is part of the key
    assert group.find("CUSTOM", "hand", None, "battery_pct") is None
    assert velocity.key == stream_key("LINEAR_VELOCITY", "hand") == ("LINEAR_VELOCITY", "hand", None, None)
    with pytest.raises(ProtocolError, match="LTP_NED"):
        group.require("ORIENTATION", "hand", "LTP_NED")


@pytest.mark.parametrize(
    "doc",
    [
        document([stream("FLOAT[3]", "POSITION", "a", "b"), stream("FLOAT[3]", "POSITION", "a", "b")]),
        document([stream("FLOAT[3]", "POSITION", "a"), stream("DOUBLE[3]", "POSITION", "a", "a")]),
        document([stream("FLOAT", "CUSTOM", "a")]),
        document([stream("FLOAT[3]", "POSITION", "a", custom_label="nope")]),
        document([stream("FLOAT[0]", "POSITION", "a")]),
        document([stream("UINT32", "STATUS_FLAGS", "a", bit_mapping={"first": "is_tracking"})]),
        document([{"data_type": "FLOAT", "measure_type": "POSITION"}]),
        document([], groups=None),
        document([], device_id=2**32),
        document([], device_id=-1),
        document([], device_id="7"),
        document([], device_id=True),
        document([], protocol_version="3.0.0"),
        {"protocol_version": "2.0.0", "device_id": 1},
        ["not", "an", "object"],
    ],
    ids=[
        "duplicate key", "duplicate via explicit self reference", "CUSTOM without label", "label on standard type",
        "bad data_type", "bad bit index", "stream without target", "no groups", "device_id too large",
        "negative device_id", "string device_id", "boolean device_id", "unknown major version",
        "missing required fields", "not an object",
    ],
)
def test_invalid_definitions_are_protocol_errors(doc):
    with pytest.raises(ProtocolError):
        Definition(doc)
    with pytest.raises(ProtocolError):
        Definition.from_payload(json.dumps(doc).encode())


def test_definition_payload_must_be_utf8_json():
    with pytest.raises(ProtocolError):
        Definition.from_payload(b"\xff\xfe not json")
    with pytest.raises(ProtocolError):
        Definition.from_payload(b'{"protocol_version": "2.0.0"')


def test_definition_frame_round_trip():
    doc = document(
        [
            stream("FLOAT[7]", "TRANSFORM", "pelvis", "LTP_ENU"),
            stream("UINT32", "STATUS_FLAGS", "pelvis", bit_mapping={"0": "is_tracking", "5": "has_error"}),
            stream("FLOAT[3]", "SOME_FUTURE_MEASURE", "pelvis"),         # new measure types must not break us
        ],
        device_id=2**32 - 1,
        device_info={"friendly_name": "Käsine (vasen) 🧤", "hand": "left"},
        static_data=[stream("FLOAT[3, 3]", "CUSTOM", "pelvis", custom_label="calibration", value=list(range(9)))],
    )
    doc["groups"].append({"name": "events", "expected_rate_hz": 0, "streams": [stream("INT64", "CUSTOM", "pelvis", custom_label="tick")]})

    frame = Definition(doc).encode()
    [(prefix, payload)] = FrameSplitter().feed(frame)
    assert prefix == PREFIX_DEFINITION and FRAME_HEADER.unpack_from(frame) == (PREFIX_DEFINITION, len(payload))
    assert not payload.endswith(b"\0")

    definition = Definition.from_payload(payload)
    assert definition.document == doc
    assert (definition.device_id, definition.device_type, definition.timestamp_epoch) == (2**32 - 1, "Test Device", "unix_epoch")
    assert definition.device_info["friendly_name"] == "Käsine (vasen) 🧤"
    assert definition.static_data[stream_key("CUSTOM", "pelvis", None, "calibration")] == list(range(9))
    assert [(g.index, g.name, g.expected_rate_hz, g.payload_size) for g in definition.groups] == [
        (0, "default", 100.0, 16 + 28 + 4 + 12),
        (1, "events", 0.0, 16 + 8),
    ]
    assert definition.group("events") is definition.groups[1] and definition.group("missing") is None
    assert definition.groups[0].require("STATUS_FLAGS", "pelvis").flags == {"is_tracking": 0, "has_error": 5}


def test_disconnect_frame_round_trip():
    frame = encode_disconnect(0xFFFFFFFE)
    assert frame == struct.pack("<III", PREFIX_DISCONNECT, 4, 0xFFFFFFFE)
    assert decode_disconnect(frame[FRAME_HEADER.size:]) == 0xFFFFFFFE
    with pytest.raises(ProtocolError):
        decode_disconnect(b"\x01\x02")


@pytest.mark.parametrize("chunk", [1, 3, 7, 8, 9, 64, 10_000])
def test_splitter_reassembles_frames_from_any_chunking(chunk):
    group = Definition(document([stream("DOUBLE[3]", "POSITION", "a"), stream("UINT32", "CUSTOM", "a", custom_label="n")])).groups[0]
    frames = [
        Definition(document([stream("FLOAT", "CUSTOM", "a", custom_label="x")])).encode(),
        group.encode(7, 1000, [1.0, 2.0, 3.0, 4]),
        encode_frame(99, b"a frame type from the future"),
        encode_frame(98, b""),                                           # empty payloads are legal framing
        group.encode(7, 2000, [5.0, 6.0, 7.0, 8]),
        encode_disconnect(7),
    ]
    stream_bytes = b"".join(frames)
    splitter = FrameSplitter()
    received = []
    for start in range(0, len(stream_bytes), chunk):
        received += splitter.feed(stream_bytes[start:start + chunk])
    assert [encode_frame(prefix, payload) for prefix, payload in received] == frames
    assert [prefix for prefix, _ in received] == [PREFIX_DEFINITION, PREFIX_DATA, 99, 98, PREFIX_DATA, PREFIX_DISCONNECT]
    assert splitter.feed(b"") == []


def test_splitter_refuses_an_absurd_length_instead_of_buffering_forever():
    splitter = FrameSplitter()
    assert splitter.feed(FRAME_HEADER.pack(PREFIX_DATA, MAX_FRAME_BYTES)[:7]) == []
    with pytest.raises(ProtocolError):
        FrameSplitter().feed(FRAME_HEADER.pack(PREFIX_DATA, MAX_FRAME_BYTES + 1))
