"""Smartgloves decoding and the emulated definition, checked against the public docs (no sockets)."""
from __future__ import annotations

import copy
import json
import re

import numpy as np
import pytest

from kaiku.rgmp.client import DataFrame
from kaiku.rgmp.emulator import DEVICE_IDS, RATE_HZ, glove_definition
from kaiku.rgmp.glove import GloveDecoder
from kaiku.rgmp.protocol import FRAME_HEADER, Definition, ProtocolError, stream_key
from kaiku.types import FINGERS

SENSORS = [f"{finger}_finger" for finger in FINGERS] + ["arm"]

# The data_type and measure_type patterns of ref/rokoko-docs/schema.json.
DATA_TYPE = re.compile(r"^(INT32|UINT32|INT64|UINT64|FLOAT|DOUBLE)(\[\d+(,\d+)?\])?$")
MEASURE_TYPE = re.compile(
    r"^(POSITION|ORIENTATION|TRANSFORM|ANGULAR_VELOCITY|LINEAR_VELOCITY|LINEAR_ACCELERATION"
    r"|PROPER_ACCELERATION|MAGNETIC_FIELD|STATUS_FLAGS|CUSTOM)$"
)


def keys(entries: list[dict]) -> set:
    return {
        stream_key(e["measure_type"], e["target_frame"], e.get("reference_frame"), e.get("custom_label"))
        for e in entries
    }


def flag_names(entries: list[dict], target: str, reference: str | None = None) -> set[str]:
    [entry] = [
        e for e in entries
        if e["measure_type"] == "STATUS_FLAGS" and e["target_frame"] == target and e.get("reference_frame") == reference
    ]
    return set(entry["bit_mapping"].values())


def decode(document: dict, values: dict, timestamp_us: int = 1_000_000, t_recv: float = 12.5):
    """Pack `values` (stream key -> value, everything else zero) into a default-group frame and decode it."""
    definition = Definition(document)
    group = definition.group("default")
    flat = group.zeros()
    for key, value in values.items():
        stream = group.require(*key)
        if stream.data_type.shape:
            flat[stream.slot:stream.slot + stream.data_type.count] = value
        else:
            flat[stream.slot] = value
    payload = group.encode(definition.device_id, timestamp_us, flat)[FRAME_HEADER.size:]
    return GloveDecoder(definition).decode(DataFrame(definition, group, timestamp_us, payload, t_recv))


def bits(document: dict, target: str, reference: str | None, *names: str) -> int:
    """The flag word with the named flags set, according to the definition's own bit_mapping."""
    stream = Definition(document).group("default").require("STATUS_FLAGS", target, reference)
    return sum(stream.mask(name) for name in names)


@pytest.mark.parametrize("side", ["left", "right"])
def test_emulated_definition_declares_what_the_docs_list(side):
    document = glove_definition(side, DEVICE_IDS[side])
    assert json.loads(json.dumps(document)) == document
    assert document["protocol_name"] == "RGMP" and document["protocol_version"] == "2.0.0"
    assert document["device_type"] == "Smartgloves II"
    assert document["timestamp_epoch"] == "device_boot"
    assert document["device_info"] == {"friendly_name": f"Smartgloves II ({side})", "hand": side}

    default, emf_only = document["groups"]
    assert (default["name"], emf_only["name"]) == ("default", "emf_only")
    assert default["expected_rate_hz"] == emf_only["expected_rate_hz"] == RATE_HZ == pytest.approx(83.33, abs=0.005)

    documented = {stream_key("ORIENTATION", "hub", "LTP_NED"), stream_key("STATUS_FLAGS", "hub")}
    documented |= {stream_key(measure, "hub_imu") for measure in ("ANGULAR_VELOCITY", "PROPER_ACCELERATION")}
    documented |= {
        stream_key("TRANSFORM", "arm_sensor", "coil_pro"),
        stream_key("CUSTOM", "arm_sensor", "coil_pro", "relative_emf_residual"),
        stream_key("STATUS_FLAGS", "arm_sensor", "coil_pro"),
    }
    for sensor in SENSORS:
        documented |= {
            stream_key("TRANSFORM", f"{sensor}_sensor", "hub"),
            stream_key("ANGULAR_VELOCITY", f"{sensor}_imu"),
            stream_key("PROPER_ACCELERATION", f"{sensor}_imu"),
            stream_key("STATUS_FLAGS", f"{sensor}_sensor"),
        }
    assert keys(default["streams"]) == documented and len(default["streams"]) == len(documented)

    emf_documented = set()
    for sensor in SENSORS:
        emf_documented |= {
            stream_key("TRANSFORM", f"{sensor}_sensor", "hub"),
            stream_key("CUSTOM", f"{sensor}_sensor", "hub", "relative_emf_residual"),
            stream_key("STATUS_FLAGS", f"{sensor}_sensor"),
        }
    assert keys(emf_only["streams"]) == emf_documented and len(emf_only["streams"]) == len(emf_documented)

    # The "which stream declares which flag" table.
    imu_saturation = {"is_accelerometer_saturated", "is_gyroscope_saturated"}
    for sensor in SENSORS:
        assert flag_names(default["streams"], f"{sensor}_sensor") == {"is_emf_saturated"} | imu_saturation
        assert flag_names(emf_only["streams"], f"{sensor}_sensor") == {"is_emf_saturated"}
    assert flag_names(default["streams"], "hub") == {"is_magnetometer_saturated", "is_heading_reliable"} | imu_saturation
    assert flag_names(default["streams"], "arm_sensor", "coil_pro") == imu_saturation | {
        "is_emf_saturated", "not_is_accelerometer_used", "is_coil_pro_present", "emf_flip_decided",
    }

    static = document["static_data"]
    assert keys(static) == (
        {stream_key("TRANSFORM", "hub_imu", "hub")}
        | {stream_key("TRANSFORM", f"{sensor}_imu", f"{sensor}_sensor") for sensor in SENSORS}
        | {stream_key("CUSTOM", frame, None, "OBB") for frame in ["hub", "coil_pro", *(f"{s}_sensor" for s in SENSORS)]}
    )
    for entry in static:
        expected = ("FLOAT[10]", 10) if entry["measure_type"] == "CUSTOM" else ("FLOAT[7]", 7)
        assert (entry["data_type"], len(entry["value"])) == expected

    for entry in [*default["streams"], *emf_only["streams"], *static]:
        assert DATA_TYPE.match(entry["data_type"]) and MEASURE_TYPE.match(entry["measure_type"])
        assert ("custom_label" in entry) == (entry["measure_type"] == "CUSTOM")
        assert ("bit_mapping" in entry) == (entry["measure_type"] == "STATUS_FLAGS")
    Definition(document)                                                 # and our own parser accepts it


def test_magnetometer_stream_is_opt_in():
    plain = glove_definition("left", 1)["groups"][0]["streams"]
    with_field = glove_definition("left", 1, magnetometer=True)["groups"][0]["streams"]
    assert keys(with_field) - keys(plain) == {stream_key("MAGNETIC_FIELD", "hub")}


def test_frame_is_decoded_into_kaiku_conventions():
    document = glove_definition("right", 42)
    values = {stream_key("TRANSFORM", f"{finger}_finger_sensor", "hub"): [i, i + 0.25, i + 0.5, 0.0, 0.0, 0.6, 0.8] for i, finger in enumerate(FINGERS)}
    values |= {
        # Same target frame twice: only the reference frame tells the two arm poses apart.
        stream_key("TRANSFORM", "arm_sensor", "hub"): [0.0, 0.09, 0.0, 1.0, 0.0, 0.0, 0.0],
        stream_key("TRANSFORM", "arm_sensor", "coil_pro"): [0.5, -0.25, 0.75, 0.0, 1.0, 0.0, 0.0],
        stream_key("ORIENTATION", "hub", "LTP_NED"): [0.5, -0.5, 0.5, -0.5],
        stream_key("STATUS_FLAGS", "arm_sensor", "coil_pro"): bits(document, "arm_sensor", "coil_pro", "is_coil_pro_present", "emf_flip_decided"),
        stream_key("STATUS_FLAGS", "hub"): bits(document, "hub", None, "is_heading_reliable"),
    }
    frame = decode(document, values, timestamp_us=123_456_789, t_recv=77.25)

    assert (frame.side, frame.device_id, frame.t_us, frame.t_recv) == ("right", 42, 123_456_789, 77.25)
    assert list(frame.fingers) == list(FINGERS)
    for i, finger in enumerate(FINGERS):
        pose = frame.fingers[finger]
        assert pose.pos.tolist() == [i, i + 0.25, i + 0.5] and pose.valid
        assert pose.quat.tolist() == pytest.approx([0.8, 0.0, 0.0, 0.6])     # wire x,y,z,w -> w,x,y,z
        assert pose.pos.dtype == pose.quat.dtype == np.float64
    assert frame.arm_in_hub.pos.tolist() == pytest.approx([0.0, 0.09, 0.0]) and frame.arm_in_hub.quat.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert frame.arm_in_coil.pos.tolist() == [0.5, -0.25, 0.75] and frame.arm_in_coil.quat.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert frame.hub_quat_ned.tolist() == [-0.5, 0.5, -0.5, 0.5]
    assert (frame.coil_present, frame.coil_flip_decided, frame.heading_reliable) == (True, True, True)


def test_flags_decide_validity_and_coil_presence():
    document = glove_definition("left", 1)
    identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    values = {stream_key("TRANSFORM", "arm_sensor", "coil_pro"): [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]}

    absent = decode(document, values)
    assert absent.arm_in_coil is None and not absent.coil_present
    assert not absent.coil_flip_decided and not absent.heading_reliable
    assert all(pose.valid for pose in absent.fingers.values()) and absent.arm_in_hub.valid

    # Flags that do not concern the pose leave it valid; EMF saturation and non-finite numbers do not.
    values |= {
        stream_key("STATUS_FLAGS", "arm_sensor", "coil_pro"): bits(document, "arm_sensor", "coil_pro", "is_coil_pro_present", "is_emf_saturated"),
        stream_key("STATUS_FLAGS", "index_finger_sensor"): bits(document, "index_finger_sensor", None, "is_emf_saturated"),
        stream_key("STATUS_FLAGS", "ring_finger_sensor"): bits(document, "ring_finger_sensor", None, "is_gyroscope_saturated", "is_accelerometer_saturated"),
        stream_key("STATUS_FLAGS", "arm_sensor"): bits(document, "arm_sensor", None, "is_emf_saturated"),
        stream_key("TRANSFORM", "little_finger_sensor", "hub"): [float("nan"), *identity[1:]],
        stream_key("TRANSFORM", "thumb_finger_sensor", "hub"): [*identity[:6], float("inf")],
    }
    frame = decode(document, values)
    assert {finger: pose.valid for finger, pose in frame.fingers.items()} == {
        "thumb": False, "index": False, "middle": True, "ring": True, "little": False,
    }
    assert not frame.arm_in_hub.valid
    assert frame.coil_present and frame.arm_in_coil is not None and not frame.arm_in_coil.valid
    assert frame.arm_in_coil.pos.tolist() == [1.0, 2.0, 3.0]
    assert not frame.coil_flip_decided


def test_decoder_depends_on_no_order_width_or_bit_position():
    """Reorder the streams, widen the values and move every flag bit: the decoded frame must not change."""
    plain = glove_definition("left", 5, magnetometer=True)
    rearranged = copy.deepcopy(plain)
    rearranged["device_type"] = "Smartgloves"                             # first generation: no emf_only group
    del rearranged["groups"][1]
    streams = rearranged["groups"][0]["streams"]
    streams.reverse()
    for entry in streams:
        if entry["measure_type"] in ("TRANSFORM", "ORIENTATION"):
            entry["data_type"] = entry["data_type"].replace("FLOAT", "DOUBLE")
        elif entry["measure_type"] == "STATUS_FLAGS":
            entry["data_type"] = "UINT64"
            entry["bit_mapping"] = {str(63 - int(bit)): name for bit, name in entry["bit_mapping"].items()}

    def values_for(document: dict) -> dict:
        values = {
            stream_key("TRANSFORM", f"{finger}_finger_sensor", "hub"): [0.25 * i, -0.5, 0.125, 0.5, 0.5, -0.5, 0.5]
            for i, finger in enumerate(FINGERS)
        }
        return values | {
            stream_key("TRANSFORM", "arm_sensor", "hub"): [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
            stream_key("TRANSFORM", "arm_sensor", "coil_pro"): [1.5, 0.25, -0.75, 0.0, 0.0, 1.0, 0.0],
            stream_key("ORIENTATION", "hub", "LTP_NED"): [0.0, 0.0, 1.0, 0.0],
            stream_key("STATUS_FLAGS", "hub"): bits(document, "hub", None, "is_heading_reliable", "is_gyroscope_saturated"),
            stream_key("STATUS_FLAGS", "middle_finger_sensor"): bits(document, "middle_finger_sensor", None, "is_emf_saturated"),
            stream_key("STATUS_FLAGS", "arm_sensor", "coil_pro"): bits(document, "arm_sensor", "coil_pro", "is_coil_pro_present", "emf_flip_decided", "not_is_accelerometer_used"),
        }

    a, b = decode(plain, values_for(plain)), decode(rearranged, values_for(rearranged))
    assert a.heading_reliable and a.coil_present and a.coil_flip_decided and not a.fingers["middle"].valid
    assert (b.side, b.coil_present, b.coil_flip_decided, b.heading_reliable) == (a.side, True, True, True)
    assert b.hub_quat_ned.tolist() == a.hub_quat_ned.tolist() == [0.0, 0.0, 0.0, 1.0]
    for got, want in zip([*b.fingers.values(), b.arm_in_hub, b.arm_in_coil], [*a.fingers.values(), a.arm_in_hub, a.arm_in_coil]):
        assert got.pos.tolist() == want.pos.tolist() and got.quat.tolist() == want.quat.tolist()
        assert got.valid == want.valid


def test_first_generation_glove_without_optional_flags_decodes():
    document = glove_definition("right", 9)
    document["device_type"] = "Smartgloves"
    del document["groups"][1]
    for entry in document["groups"][0]["streams"]:
        if entry["measure_type"] == "STATUS_FLAGS" and entry["target_frame"] == "hub":
            entry["bit_mapping"] = {"0": "is_accelerometer_saturated", "1": "is_gyroscope_saturated"}
    frame = decode(document, {stream_key("STATUS_FLAGS", "hub"): 0xFFFFFFFF})
    assert frame.side == "right" and not frame.heading_reliable


def mutated(change) -> dict:
    document = glove_definition("left", 1)
    change(document)
    return document


def drop_stream(measure_type: str, target: str):
    def change(document: dict) -> None:
        streams = document["groups"][0]["streams"]
        streams[:] = [s for s in streams if (s["measure_type"], s["target_frame"]) != (measure_type, target)]
    return change


def retype_flags(document: dict) -> None:
    for entry in document["groups"][0]["streams"]:
        if entry["measure_type"] == "STATUS_FLAGS":
            entry["data_type"] = "FLOAT"


@pytest.mark.parametrize(
    "document",
    [
        mutated(lambda d: d.update(device_type="Smartsuit Pro II")),
        mutated(lambda d: d.update(device_info={"friendly_name": "Smartgloves II"})),
        mutated(lambda d: d["device_info"].update(hand="both")),
        mutated(lambda d: d["groups"][0].update(name="raw")),
        mutated(drop_stream("TRANSFORM", "thumb_finger_sensor")),
        mutated(drop_stream("ORIENTATION", "hub")),
        mutated(lambda d: d["groups"][0]["streams"][0].update(data_type="FLOAT[6]")),
        mutated(retype_flags),
    ],
    ids=[
        "not a glove", "no hand", "unknown hand", "no default group", "missing finger pose",
        "missing hub orientation", "pose with six values", "flags that are not integers",
    ],
)
def test_decoder_refuses_what_it_cannot_decode_faithfully(document):
    with pytest.raises(ProtocolError):
        GloveDecoder(Definition(document))
