"""Smartgloves over RGMP: turn a glove's `default` group into `kaiku.types.GloveFrame`.

Each glove is its own RGMP device; `device_info.hand` says which side it is. Its `default` group
(83.33 Hz) carries, among other streams, one TRANSFORM per finger sensor and for the arm sensor
relative to the `hub`, the hub ORIENTATION in LTP_NED, and a second arm sensor TRANSFORM relative
to the `coil_pro`. See ref/rokoko-docs/reference_products_smartgloves.txt.

Nothing here depends on stream order, byte offsets, value width or flag bit positions: streams
are found by their full key and flag bits by name in each stream's `bit_mapping`, as the docs
require. The only conversion applied is the quaternion order, x,y,z,w on the wire to w,x,y,z.
"""
from __future__ import annotations

import logging
import operator
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from kaiku.types import FINGERS, SIDES, GloveFrame, SensorPose

from .client import DataFrame, RgmpClient
from .protocol import DEFAULT_PORT, Definition, Group, ProtocolError, Stream

_log = logging.getLogger(__name__)

GLOVE_DEVICE_TYPES = ("Smartgloves", "Smartgloves II")
DEFAULT_GROUP = "default"
HUB = "hub"
COIL = "coil_pro"
NED = "LTP_NED"
ARM_SENSOR = "arm_sensor"
FINGER_SENSORS = tuple(f"{finger}_finger_sensor" for finger in FINGERS)

EMF_SATURATED = "is_emf_saturated"
COIL_PRESENT = "is_coil_pro_present"
FLIP_DECIDED = "emf_flip_decided"
HEADING_RELIABLE = "is_heading_reliable"

_POSE_ORDER = (0, 1, 2, 6, 3, 4, 5)     # wire x,y,z,qx,qy,qz,qw -> x,y,z,qw,qx,qy,qz
_QUAT_ORDER = (3, 0, 1, 2)              # wire qx,qy,qz,qw -> qw,qx,qy,qz
_RATE_WINDOW_S = 1.0
_STALE_S = 3.0

_Flag = tuple[int, int]                 # (value slot of the STATUS_FLAGS stream, bit mask)


def _sized(stream: Stream, count: int) -> Stream:
    if stream.data_type.count != count:
        raise ProtocolError(f"stream {stream.key} has {stream.data_type.count} values, expected {count}")
    return stream


def _flag(group: Group, flag: str, target_frame: str, reference_frame: str | None = None) -> _Flag | None:
    """Locate a named flag through the stream's bit_mapping; None if the device does not declare it."""
    stream = group.find("STATUS_FLAGS", target_frame, reference_frame)
    if stream is None or not stream.mask(flag):
        return None
    if stream.data_type.shape or not stream.data_type.is_integer:
        raise ProtocolError(f"status flags {stream.key} are not a scalar integer")
    return stream.slot, stream.mask(flag)


def _is_set(values: Sequence[Any], flag: _Flag | None) -> bool:
    return flag is not None and values[flag[0]] & flag[1] != 0


class GloveDecoder:
    """Decodes the `default` group frames of one glove, for one stream definition.

    Everything that depends on the definition is resolved here, once. Per frame the work is one
    struct unpack, one gather into a float64 array and a handful of flag tests; the SensorPose
    arrays are views into that one array.
    """

    def __init__(self, definition: Definition) -> None:
        if definition.device_type not in GLOVE_DEVICE_TYPES:
            raise ProtocolError(f"device type {definition.device_type!r} is not a Smartglove")
        side = definition.device_info.get("hand")
        if side not in SIDES:
            raise ProtocolError(f"glove {definition.device_id} reports hand {side!r}")
        group = definition.group(DEFAULT_GROUP)
        if group is None:
            raise ProtocolError(f"glove {definition.device_id} has no {DEFAULT_GROUP!r} group")
        self.side: str = side
        self.device_id = definition.device_id
        self.group = group
        # The emulator marks its definitions explicitly; real Rokoko definitions do not carry
        # this private field, so this is a conservative stream-level distinction.
        self.emulated = bool(definition.document.get("kaiku_emulated") or definition.device_info.get("kaiku_emulated")
                             or definition.device_info.get("emulated"))
        self.firmware = definition.device_info.get("firmware") or definition.device_info.get("firmware_version")

        # (target, reference of its pose, reference of its flags), in the row order of the decoded
        # pose matrix: five fingers, arm in hub, arm in coil. Sensor flags are intrinsic to the
        # sensor (no reference frame); only the Coil Pro solution has flags of its own, keyed by
        # the same arm_sensor / coil_pro pair as its pose.
        poses = [(sensor, HUB, None) for sensor in (*FINGER_SENSORS, ARM_SENSOR)] + [(ARM_SENSOR, COIL, COIL)]
        slots = [
            _sized(group.require("TRANSFORM", sensor, reference), 7).slot + i
            for sensor, reference, _ in poses
            for i in _POSE_ORDER
        ]
        hub = _sized(group.require("ORIENTATION", HUB, NED), 4)
        self._pick = operator.itemgetter(*slots, *(hub.slot + i for i in _QUAT_ORDER))
        self._saturated = [_flag(group, EMF_SATURATED, sensor, flags_in) for sensor, _, flags_in in poses]
        self._coil_present = _flag(group, COIL_PRESENT, ARM_SENSOR, COIL)
        self._flip_decided = _flag(group, FLIP_DECIDED, ARM_SENSOR, COIL)
        self._heading_reliable = _flag(group, HEADING_RELIABLE, HUB)

    def decode(self, frame: DataFrame) -> GloveFrame:
        """Decode a data frame of `self.group` (the caller checks `frame.group is decoder.group`).

        A pose is `valid` when its numbers are finite and its sensor is not EMF-saturated.
        `arm_in_coil` is None while the glove does not see a Coil Pro: the stream is always
        present, but the docs call its content unreliable until `is_coil_pro_present` is set.
        """
        values = frame.values()
        numbers = np.array(self._pick(values), dtype=np.float64)
        poses = numbers[:-4].reshape(-1, 7)
        finite = np.isfinite(poses).all(axis=1).tolist()
        sensors = [
            SensorPose(row[:3], row[3:], ok and not _is_set(values, flag))
            for row, ok, flag in zip(poses, finite, self._saturated)
        ]
        coil_present = _is_set(values, self._coil_present)
        return GloveFrame(
            side=self.side,
            device_id=self.device_id,
            t_us=frame.timestamp_us,
            t_recv=frame.t_recv,
            hub_quat_ned=numbers[-4:],
            fingers=dict(zip(FINGERS, sensors)),
            arm_in_hub=sensors[-2],
            arm_in_coil=sensors[-1] if coil_present else None,
            coil_present=coil_present,
            coil_flip_decided=_is_set(values, self._flip_decided),
            heading_reliable=_is_set(values, self._heading_reliable),
        )


class GloveStream:
    """Live Smartgloves: the newest `GloveFrame` per side, fed by a background `RgmpClient`.

    Works unchanged against the Rokoko driver, the emulator and a replay. Devices that are not
    Smartgloves are ignored. When a glove disconnects, or the connection drops, its side reads
    None again, so a consumer never keeps acting on a frame from a glove that is gone.

    `on_frame`, if given, is called with every decoded frame on the reader thread.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        *,
        on_frame: Callable[[GloveFrame], None] | None = None,
    ) -> None:
        self.client = RgmpClient(
            host, port, on_definition=self._on_definition, on_data=self._on_data, on_disconnect=self._on_disconnect
        )
        self._on_frame = on_frame
        self._decoders: dict[int, GloveDecoder] = {}        # reader thread only
        self._lock = threading.Lock()                       # guards the two dicts below
        self._latest: dict[str, GloveFrame] = {}
        self._arrivals: dict[str, deque[float]] = {side: deque() for side in SIDES}
        self._emulated = False
        self._firmware: dict[str, str] = {}

    def start(self) -> GloveStream:
        """Start receiving in the background and return self. Does not wait for a glove."""
        self.client.start()
        return self

    def stop(self) -> None:
        self.client.stop()

    def __enter__(self) -> GloveStream:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    @property
    def connected(self) -> bool:
        """True while the TCP connection to the RGMP server is up (gloves may still be absent)."""
        return self.client.connected

    def latest(self, side: str) -> GloveFrame | None:
        """The newest frame of the "left" or "right" glove; None while that glove is not streaming."""
        with self._lock:
            frame = self._latest.get(side)
            # A TCP connection can remain open after a radio link has gone quiet. Do not keep
            # commanding from the last good hand indefinitely.
            if frame is None or time.monotonic() - frame.t_recv > _STALE_S:
                return None
            return frame

    def rate_hz(self, side: str) -> float:
        """Measured frame rate of one side over the last second (nominally 83.33); 0.0 when stalled."""
        with self._lock:
            arrivals = self._arrivals[side]
            if len(arrivals) < 2 or time.monotonic() - arrivals[-1] > _RATE_WINDOW_S:
                return 0.0
            span = arrivals[-1] - arrivals[0]
            return (len(arrivals) - 1) / span if span > 0 else 0.0

    @property
    def emulated(self) -> bool:
        with self._lock:
            return bool(self._emulated)

    @property
    def firmware(self) -> dict[str, str]:
        with self._lock:
            return dict(self._firmware)

    def age_s(self, side: str) -> float | None:
        with self._lock:
            frame = self._latest.get(side)
            return None if frame is None else max(0.0, time.monotonic() - frame.t_recv)

    def _on_definition(self, definition: Definition) -> None:
        self._on_disconnect(definition.device_id)           # a re-sent definition is a new session
        if definition.device_type not in GLOVE_DEVICE_TYPES:
            return
        try:
            decoder = GloveDecoder(definition)
            self._decoders[definition.device_id] = decoder
            with self._lock:
                self._emulated = self._emulated or decoder.emulated
                if decoder.firmware:
                    self._firmware[decoder.side] = str(decoder.firmware)
        except ProtocolError as error:
            _log.warning("ignoring glove %d: %s", definition.device_id, error)

    def _on_data(self, frame: DataFrame) -> None:
        decoder = self._decoders.get(frame.definition.device_id)
        if decoder is None or frame.group is not decoder.group:
            return
        glove = decoder.decode(frame)
        with self._lock:
            self._latest[decoder.side] = glove
            arrivals = self._arrivals[decoder.side]
            arrivals.append(glove.t_recv)
            while glove.t_recv - arrivals[0] > _RATE_WINDOW_S:
                arrivals.popleft()
        if self._on_frame is not None:
            self._on_frame(glove)

    def _on_disconnect(self, device_id: int) -> None:
        decoder = self._decoders.pop(device_id, None)
        if decoder is not None:
            with self._lock:
                self._latest.pop(decoder.side, None)
                self._arrivals[decoder.side].clear()
