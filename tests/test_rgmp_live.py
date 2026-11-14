"""Client, server, emulator and GloveStream over real loopback sockets, on ephemeral ports."""
from __future__ import annotations

import itertools
import json
import logging
import math
import socket
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from kaiku.rgmp import GloveStream, RgmpClient, RgmpServer
from kaiku.rgmp.emulator import DEVICE_IDS, PERIOD_US, Emulator, StillSource, glove_definition
from kaiku.rgmp.protocol import (
    DATA_HEADER,
    FRAME_HEADER,
    PREFIX_DATA,
    PREFIX_DEFINITION,
    Definition,
    encode_disconnect,
    encode_frame,
)
from kaiku.types import SIDES, GloveFrame, SensorPose

HOST = "127.0.0.1"
LEFT, RIGHT = DEVICE_IDS["left"], DEVICE_IDS["right"]


@pytest.fixture(autouse=True)
def no_leaked_threads():
    before = set(threading.enumerate())
    yield
    assert [thread.name for thread in threading.enumerate() if thread not in before] == []


def wait_until(condition, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.002)


def on_wire(array: np.ndarray) -> list[float]:
    """What float64 values read back as after travelling as RGMP FLOAT (float32)."""
    return np.asarray(array, dtype=np.float32).astype(np.float64).tolist()


def assert_same_pose(got: SensorPose | None, want: SensorPose | None) -> None:
    if want is None:
        assert got is None
    else:
        assert (got.pos.tolist(), got.quat.tolist(), got.valid) == (on_wire(want.pos), on_wire(want.quat), want.valid)


def assert_same_glove(got: GloveFrame, want: GloveFrame) -> None:
    assert got.side == want.side
    assert got.hub_quat_ned.tolist() == on_wire(want.hub_quat_ned)
    assert list(got.fingers) == list(want.fingers)
    for finger, pose in want.fingers.items():
        assert_same_pose(got.fingers[finger], pose)
    assert_same_pose(got.arm_in_hub, want.arm_in_hub)
    assert_same_pose(got.arm_in_coil, want.arm_in_coil)
    assert (got.coil_present, got.coil_flip_decided, got.heading_reliable) == (
        want.coil_present, want.coil_flip_decided, want.heading_reliable,
    )


def glove_frame(definition: Definition, timestamp_us: int, x: float = 0.0) -> bytes:
    """A default-group data frame whose Coil Pro pose sits at (x, 0, 0) and is flagged as tracked."""
    group = definition.group("default")
    values = group.zeros()
    pose = group.require("TRANSFORM", "arm_sensor", "coil_pro")
    values[pose.slot:pose.slot + 7] = (x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    flags = group.require("STATUS_FLAGS", "arm_sensor", "coil_pro")
    values[flags.slot] = flags.mask("is_coil_pro_present")
    return group.encode(definition.device_id, timestamp_us, values)


def unused_port() -> int:
    with socket.create_server((HOST, 0)) as probe:
        return probe.getsockname()[1]


def test_emulated_gloves_arrive_as_glove_frames():
    source = StillSource()
    frames: dict[str, list[GloveFrame]] = {side: [] for side in SIDES}
    with Emulator(source, HOST, 0) as emulator:
        port = emulator.port
        with GloveStream(HOST, port, on_frame=lambda frame: frames[frame.side].append(frame)) as gloves:
            wait_until(lambda: all(len(frames[side]) >= 40 for side in SIDES))
            assert gloves.connected
            for side in SIDES:
                received = list(frames[side])
                newest = gloves.latest(side)
                assert_same_glove(newest, source.sample(side, 0.0))
                assert newest.device_id == DEVICE_IDS[side]
                assert 0.0 <= time.monotonic() - newest.t_recv < 10.0

                # Device clock: exactly one 12 ms period per sample, and in step with the host clock.
                steps = np.diff([frame.t_us for frame in received])
                assert (steps > 0).all() and not (steps % PERIOD_US).any()
                device_s = (received[-1].t_us - received[0].t_us) / 1e6
                host_s = received[-1].t_recv - received[0].t_recv
                assert abs(device_s - host_s) < 0.1
                assert 60.0 < gloves.rate_hz(side) < 110.0
            assert frames["left"][0].t_us != frames["right"][0].t_us       # each glove counts from its own boot
        assert not gloves.connected
        assert gloves.latest("left") is None and gloves.rate_hz("left") == 0.0
    with pytest.raises(OSError):
        socket.create_connection((HOST, port), timeout=1.0)                # the port is released


class ScriptedSource:
    """Left: no coil, heading unknown, thumb saturated, ring sensor missing. Right: every other sample only."""

    def __init__(self) -> None:
        still = StillSource()
        left = still.sample("left", 0.0)
        fingers = {**left.fingers, "thumb": replace(left.fingers["thumb"], valid=False)}
        del fingers["ring"]
        self.left = replace(
            left, fingers=fingers, arm_in_coil=None, coil_present=False, coil_flip_decided=False, heading_reliable=False
        )
        self.right = still.sample("right", 0.0)

    def sample(self, side: str, t: float) -> GloveFrame | None:
        if side == "left":
            return self.left
        return self.right if round(t * 1e6 / PERIOD_US) % 2 == 0 else None


def test_source_flags_gaps_and_missing_sensors_reach_the_client():
    source = ScriptedSource()
    right: list[GloveFrame] = []
    with Emulator(source, HOST, 0) as emulator:
        with GloveStream(HOST, emulator.port, on_frame=lambda f: right.append(f) if f.side == "right" else None) as gloves:
            wait_until(lambda: len(right) >= 10 and gloves.latest("left") is not None)
            left = gloves.latest("left")
    ring = left.fingers.pop("ring")
    assert not ring.valid and ring.quat.tolist() == [1.0, 0.0, 0.0, 0.0]   # sent as an invalid identity pose
    assert_same_glove(left, source.left)
    assert not left.fingers["thumb"].valid and left.fingers["index"].valid and left.arm_in_coil is None
    assert set(np.diff([frame.t_us for frame in right]).tolist()) == {2 * PERIOD_US}


class FingersUp:
    """Both hands vertical: the hub is pitched +90 degrees about X, so +Y (towards the wrist) points down."""

    def __init__(self) -> None:
        upright = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
        self._frames = {side: replace(StillSource().sample(side, 0.0), hub_quat_ned=upright) for side in SIDES}

    def sample(self, side: str, t: float) -> GloveFrame:
        return self._frames[side]


def test_emulated_raw_streams_are_consistent_with_the_pose():
    with Emulator(FingersUp(), HOST, 0, magnetometer=True) as emulator, RgmpClient(HOST, emulator.port) as client:
        wait_until(lambda: client.latest(LEFT) is not None and client.latest(LEFT, "emf_only") is not None)
        frame, emf_frame = client.latest(LEFT), client.latest(LEFT, "emf_only")
        assert client.latest(LEFT, "no such group") is None and client.latest(12345) is None
    group, values = frame.group, frame.values()
    emf_group, emf_values = emf_frame.group, emf_frame.values()
    assert frame.timestamp_us == emf_frame.timestamp_us or abs(frame.timestamp_us - emf_frame.timestamp_us) % PERIOD_US == 0

    # At rest an accelerometer reads 1 g against gravity: -Y here, for the hub and the parallel index sensor.
    for imu in ("hub_imu", "index_finger_imu", "arm_imu"):
        assert group.require("PROPER_ACCELERATION", imu).read(values) == pytest.approx((0.0, -9.80665, 0.0), abs=1e-5)
        assert group.require("ANGULAR_VELOCITY", imu).read(values) == (0.0, 0.0, 0.0)
    thumb = group.require("PROPER_ACCELERATION", "thumb_finger_imu").read(values)     # yawed 45 degrees in the hand
    assert thumb == pytest.approx((-9.80665 * math.sqrt(0.5), -9.80665 * math.sqrt(0.5), 0.0), abs=1e-5)
    assert group.require("MAGNETIC_FIELD", "hub").read(values) == pytest.approx((0.15, 0.50, 0.0), abs=1e-6)
    assert group.require("CUSTOM", "arm_sensor", "coil_pro", "relative_emf_residual").read(values) == 1.5

    for sensor in ("thumb_finger_sensor", "arm_sensor"):
        pose = group.require("TRANSFORM", sensor, "hub").read(values)
        assert emf_group.require("TRANSFORM", sensor, "hub").read(emf_values) == pose
        assert emf_group.require("CUSTOM", sensor, "hub", "relative_emf_residual").read(emf_values) == 1.5
        assert emf_group.require("STATUS_FLAGS", sensor).read(emf_values) == 0


def test_client_joining_mid_session_gets_current_definitions_then_data():
    left = Definition(glove_definition("left", 1))
    right = Definition(glove_definition("right", 2))
    gone = Definition(glove_definition("right", 3))
    newer_left = Definition(glove_definition("left", 1, magnetometer=True))
    with RgmpServer(HOST, 0) as server:
        for frame in (left.encode(), right.encode(), gone.encode(), glove_frame(left, 1), encode_disconnect(3), newer_left.encode()):
            server.send(frame)                                              # nobody is listening yet
        assert not server.wait_for_client(0.0) and server.device_ids() == [1, 2]

        first_seen: list[int] = []
        with RgmpClient(HOST, server.port, on_frame=lambda prefix, *_: first_seen.append(prefix)) as first:
            assert server.wait_for_client(30.0)
            server.send(glove_frame(right, 2, x=0.25))
            wait_until(lambda: first.latest(2) is not None)
            assert first_seen == [PREFIX_DEFINITION, PREFIX_DEFINITION, PREFIX_DATA]
            assert set(first.definitions()) == {1, 2} and first.latest(1) is None
            assert first.definitions()[1].group("default").find("MAGNETIC_FIELD", "hub") is not None

            with GloveStream(HOST, server.port) as second:                  # a second client, later still
                wait_until(lambda: server.client_count == 2)
                server.send(glove_frame(newer_left, 3, x=0.5))
                wait_until(lambda: second.latest("left") is not None and first.latest(1) is not None)
                assert second.latest("left").arm_in_coil.pos.tolist() == [0.5, 0.0, 0.0]
                assert second.latest("right") is None                       # it joined after that frame
                assert (first.latest(1).timestamp_us, first.latest(2).timestamp_us) == (3, 2)
    assert server.client_count == 0 and server.device_ids() == []


def test_client_started_before_the_server_connects_when_it_appears():
    port = unused_port()
    with GloveStream(HOST, port) as gloves:
        time.sleep(0.2)                                                     # a few refused connection attempts
        assert not gloves.connected and gloves.latest("left") is None
        with Emulator(port=port):
            wait_until(lambda: gloves.latest("left") is not None and gloves.latest("right") is not None)
            assert gloves.connected


def test_client_reconnects_after_the_server_restarts():
    first = Emulator(port=0).start()
    port = first.port
    with GloveStream(HOST, port) as gloves:
        wait_until(lambda: all(gloves.latest(side) is not None for side in SIDES))
        first_seen = gloves.latest("left").t_us
        wait_until(lambda: gloves.latest("left").t_us - first_seen > 500_000)   # the first session is half a second old
        before = gloves.latest("left")
        first.stop()
        wait_until(lambda: not gloves.connected and all(gloves.latest(side) is None for side in SIDES))
        with Emulator(port=port):
            wait_until(lambda: all(gloves.latest(side) is not None for side in SIDES))
            assert gloves.connected and gloves.latest("left").t_recv > before.t_recv
            assert gloves.latest("left").t_us < before.t_us                 # a new session: the device clock restarted


def test_resent_definition_and_disconnect_frame_start_and_end_sessions():
    plain = Definition(glove_definition("left", 11))
    wider = Definition(glove_definition("left", 11, magnetometer=True))
    assert plain.group("default").payload_size != wider.group("default").payload_size
    with RgmpServer(HOST, 0) as server, GloveStream(HOST, server.port) as gloves:
        assert server.wait_for_client(30.0)
        server.send(plain.encode())
        server.send(glove_frame(plain, 1, x=1.0))
        wait_until(lambda: gloves.latest("left") is not None)
        assert gloves.latest("left").arm_in_coil.pos[0] == 1.0

        server.send(wider.encode())
        wait_until(lambda: gloves.latest("left") is None)                   # the old layout's frame is history
        server.send(glove_frame(plain, 2, x=2.0))                           # stale layout: must be skipped
        server.send(glove_frame(wider, 3, x=3.0))
        wait_until(lambda: gloves.latest("left") is not None)
        assert (gloves.latest("left").t_us, gloves.latest("left").arm_in_coil.pos[0]) == (3, 3.0)

        server.send(encode_disconnect(11))
        wait_until(lambda: gloves.latest("left") is None)
        assert gloves.client.definitions() == {} and gloves.connected


def test_unknown_frames_bad_definitions_and_stray_data_are_skipped():
    glove = Definition(glove_definition("right", 21))
    suit = Definition({
        "protocol_name": "RGMP", "protocol_version": "2.0.0", "device_id": 30, "device_type": "Smartsuit Pro II",
        "timestamp_epoch": "unix_epoch",
        "groups": [{"name": "default", "expected_rate_hz": 100, "streams": [
            {"data_type": "FLOAT[3]", "measure_type": "POSITION", "target_frame": "pelvis", "reference_frame": "LTP_ENU"},
        ]}],
    })
    frames = [
        encode_frame(99, b"a frame type from the future"),
        encode_frame(PREFIX_DEFINITION, b"{ not json"),
        encode_frame(PREFIX_DEFINITION, json.dumps({**glove.document, "device_id": 22, "protocol_version": "9.0"}).encode()),
        encode_frame(PREFIX_DATA, b"\x01\x02"),                             # shorter than its own header
        glove_frame(glove, 1),                                              # before its definition
        suit.encode(),
        suit.groups[0].encode(30, 5, [1.0, 2.0, 3.0]),
        glove.encode(),
        encode_frame(PREFIX_DATA, DATA_HEADER.pack(21, 5, 2)),              # undeclared group
        encode_frame(PREFIX_DATA, DATA_HEADER.pack(21, 0, 3) + bytes(8)),   # does not fit the layout
        encode_frame(3, b""),                                               # disconnect without a device
        glove_frame(glove, 7, x=0.75),
    ]
    raw: list[bytes] = []
    with RgmpServer(HOST, 0) as server:
        tap = RgmpClient(HOST, server.port, on_frame=lambda prefix, payload, _: raw.append(encode_frame(prefix, payload)))
        with tap, GloveStream(HOST, server.port) as gloves:
            wait_until(lambda: server.client_count == 2)
            for frame in frames:
                server.send(frame)
            wait_until(lambda: gloves.latest("right") is not None and len(raw) == len(frames))
            assert raw == frames                                            # the raw hook sees everything, untouched
            assert gloves.latest("right").t_us == 7 and gloves.latest("left") is None
            assert set(tap.definitions()) == {21, 30}
            assert tap.latest(30).group.streams[0].read(tap.latest(30).values()) == (1.0, 2.0, 3.0)
            assert tap.connected and gloves.connected


def test_frames_survive_tiny_tcp_segments():
    glove = Definition(glove_definition("left", 4))
    stream = glove.encode() + glove_frame(glove, 5, x=0.5) + encode_frame(77, b"") + glove_frame(glove, 6, x=1.5)
    sizes = itertools.cycle((1, 2, 3, 5, 7, 11))
    with socket.create_server((HOST, 0)) as listener:
        listener.settimeout(30.0)
        with GloveStream(HOST, listener.getsockname()[1]) as gloves:
            peer, _ = listener.accept()
            with peer:
                peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                start = 0
                while start < len(stream):
                    size = next(sizes)
                    peer.sendall(stream[start:start + size])
                    start += size
                wait_until(lambda: gloves.latest("left") is not None and gloves.latest("left").t_us == 6)
                assert gloves.latest("left").arm_in_coil.pos[0] == 1.5


def test_lost_framing_drops_the_connection_and_reconnects():
    glove = Definition(glove_definition("left", 4))
    announced: list[Definition] = []
    gone: list[int] = []
    with RgmpServer(HOST, 0) as server:
        server.send(glove.encode())
        with RgmpClient(HOST, server.port, on_definition=announced.append, on_disconnect=gone.append) as client:
            wait_until(lambda: len(announced) == 1)
            server.send(FRAME_HEADER.pack(PREFIX_DATA, 0xFFFFFFF0))         # a length no frame can have
            wait_until(lambda: len(announced) == 2 and client.connected)    # back, and greeted again
            assert gone == [4]                                              # the drop was reported per device
            server.send(glove_frame(glove, 9))
            wait_until(lambda: client.latest(4) is not None)


def test_a_stalled_client_is_dropped_without_starving_the_others():
    blob = encode_frame(99, bytes(64 * 1024))
    received: list[int] = []
    with RgmpServer(HOST, 0) as server:
        stalled = socket.socket()
        stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        with stalled, RgmpClient(HOST, server.port, on_frame=lambda prefix, *_: received.append(prefix)):
            stalled.connect((HOST, server.port))                            # connects, then never reads
            wait_until(lambda: server.client_count == 2)
            sent = 0
            while server.client_count == 2:
                assert sent < 2000, "the stalled client was never dropped"
                server.send(blob)
                sent += 1
                wait_until(lambda: len(received) == sent)                   # the healthy client keeps up
            assert server.client_count == 1


def test_a_failing_callback_does_not_stop_the_stream(caplog):
    def explode(frame) -> None:
        raise RuntimeError("consumer bug")

    with caplog.at_level(logging.ERROR, logger="kaiku.rgmp.client"):
        with Emulator(port=0) as emulator, RgmpClient(HOST, emulator.port, on_data=explode) as client:
            wait_until(lambda: client.latest(LEFT) is not None)
            first = client.latest(LEFT).timestamp_us
            wait_until(lambda: client.latest(LEFT).timestamp_us >= first + 5 * PERIOD_US)
            assert client.latest(RIGHT) is not None
    assert len([record for record in caplog.records if "callback failed" in record.message]) == 1


def test_start_and_stop_are_idempotent_and_restartable():
    with Emulator(port=0) as emulator:
        emulator.start()
        gloves = GloveStream(HOST, emulator.port)
        gloves.stop()                                                       # never started: nothing to do
        for _ in range(2):
            assert gloves.start() is gloves.start()
            wait_until(lambda: gloves.latest("left") is not None)
            gloves.stop()
            gloves.stop()
            assert not gloves.connected and gloves.latest("left") is None
    emulator.stop()
