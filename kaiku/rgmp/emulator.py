"""A synthetic Smartgloves II pair that speaks RGMP like the Rokoko driver, for work without gloves.

    python -m kaiku.rgmp.emulator --port 12276 [--source module:factory]

Each glove is its own RGMP device and announces what the public docs list for a Smartgloves II
(ref/rokoko-docs/reference_products_smartgloves.txt): `device_info` with `friendly_name` and
`hand`, the `default` and `emf_only` groups at 250/3 Hz, every documented stream and status flag,
the static IMU offsets and bounding boxes, and `device_boot` timestamps.

What a glove does comes from a pluggable `GloveSource`. A `GloveFrame` holds no IMU data, so the
streams it cannot fill are made consistent with a hand that is momentarily at rest: gyroscopes
read zero and accelerometers read gravity in their sensor's current orientation. The `emf_only`
group repeats the fused poses with a small constant residual.

The docs do not pin down data types, stream order, bit positions or static values; the choices
here are plausible stand-ins, and `kaiku.rgmp.glove` deliberately depends on none of them.
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys
import threading
import time
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from kaiku.types import FINGERS, SIDES, GloveFrame, SensorPose

from ._cli import stop_on_signals
from .glove import (
    ARM_SENSOR,
    COIL,
    COIL_PRESENT,
    DEFAULT_GROUP,
    EMF_SATURATED,
    FINGER_SENSORS,
    FLIP_DECIDED,
    HEADING_RELIABLE,
    HUB,
    NED,
)
from .protocol import DEFAULT_PORT, Definition, encode_disconnect
from .server import RgmpServer

RATE_HZ = 250 / 3
PERIOD_US = 12_000                  # exactly 1 / RATE_HZ
DEVICE_IDS = {"left": 0xA1100001, "right": 0xA1100002}

_PERIOD_S = PERIOD_US / 1e6
_MAX_LAG_S = 0.25                   # further behind than this (suspend, debugger): skip instead of bursting
_BOOT_US = {"left": 95_310_000, "right": 88_140_000}    # each glove's clock age when the emulator starts
_SENSORS = (*FINGER_SENSORS, ARM_SENSOR)
_RESIDUAL = "relative_emf_residual"
_RESIDUAL_PERCENT = 1.5
_REST_ACCELERATION = (0.0, 0.0, -9.80665)   # what an accelerometer at rest reads, in LTP_NED (m/s^2)
_EARTH_FIELD = (0.15, 0.0, 0.50)            # Helsinki's magnetic field in LTP_NED (Gauss)
_NO_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

# Bit positions deliberately differ between streams, as the docs warn they do on the real device.
_SENSOR_FLAGS = (EMF_SATURATED, "is_accelerometer_saturated", "is_gyroscope_saturated")
_HUB_FLAGS = ("is_magnetometer_saturated", "is_accelerometer_saturated", "is_gyroscope_saturated", HEADING_RELIABLE)
_COIL_FLAGS = (
    "is_accelerometer_saturated", "is_gyroscope_saturated", EMF_SATURATED, "not_is_accelerometer_used",
    COIL_PRESENT, FLIP_DECIDED,
)

# Static data: where each IMU chip sits in its housing, and the housing's half extents (metres).
_IMU_OFFSET = (0.0, 0.002, -0.001)
_HALF_EXTENTS = {
    HUB: (0.022, 0.028, 0.007),
    **{sensor: (0.006, 0.009, 0.003) for sensor in FINGER_SENSORS},
    ARM_SENSOR: (0.011, 0.016, 0.005),
    COIL: (0.13, 0.13, 0.05),
}


class GloveSource(Protocol):
    """Where the emulator gets its motion from.

    `sample(side, t)` returns the state of the "left" or "right" glove `t` seconds after the
    emulator started, or None while that glove has nothing to send (its frame is skipped, as if
    the glove were off). It runs on the emulator thread, once per side every 12 ms, so it must
    not block.

    Poses are in Rokoko's glove frames with kaiku's w,x,y,z quaternions, exactly as `GloveFrame`
    documents them. `device_id`, `t_us` and `t_recv` are ignored: the emulator stamps the wire
    itself. A sensor that is missing or not `valid` goes out flagged `is_emf_saturated`, and
    `arm_in_coil` is only sent as tracked while `coil_present` is true.
    """

    def sample(self, side: str, t: float) -> GloveFrame | None: ...


class StillSource:
    """Built-in source: two flat, open hands that never move.

    The hands lie palm down, fingers towards +X of both LTP_NED (north) and the Coil Pro, 0.8 m
    in front of the coil (outside its 0.5 m saturation zone) and 0.4 m apart. In a flat glove all
    sensor frames are parallel to the hub's, so only the splayed thumb is rotated.
    """

    def __init__(self) -> None:
        self._frames = {side: self._flat_hand(side) for side in SIDES}

    def sample(self, side: str, t: float) -> GloveFrame:
        return self._frames[side]

    @staticmethod
    def _flat_hand(side: str) -> GloveFrame:
        mirror = -1.0 if side == "left" else 1.0            # sensor x in the hub frame: +X is to the right
        level = np.array([1.0, 0.0, 0.0, 0.0])
        yaw_90 = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])     # +Y (towards the wrist) points south
        thumb = np.array([math.cos(math.pi / 8), 0.0, 0.0, -mirror * math.sin(math.pi / 8)])
        positions = {                                       # right hand, metres in the hub frame
            "thumb": (-0.045, -0.030, 0.010),
            "index": (-0.028, -0.095, -0.004),
            "middle": (-0.006, -0.102, -0.004),
            "ring": (0.015, -0.096, -0.004),
            "little": (0.035, -0.082, -0.004),
        }
        return GloveFrame(
            side=side,
            device_id=DEVICE_IDS[side],
            t_us=0,
            t_recv=0.0,
            hub_quat_ned=yaw_90,
            fingers={
                finger: SensorPose(np.array([mirror * x, y, z]), thumb if finger == "thumb" else level)
                for finger, (x, y, z) in positions.items()
            },
            arm_in_hub=SensorPose(np.array([0.0, 0.090, -0.004]), level),
            arm_in_coil=SensorPose(np.array([0.80, mirror * 0.20, -0.20]), yaw_90),
            coil_present=True,
            coil_flip_decided=True,
            heading_reliable=True,
        )


def _imu(frame: str) -> str:
    """The IMU frame inside a housing: thumb_finger_sensor -> thumb_finger_imu, hub -> hub_imu."""
    return frame.removesuffix("_sensor") + "_imu"


def _entry(
    data_type: str, measure_type: str, target_frame: str, reference_frame: str | None = None, **extra: Any
) -> dict[str, Any]:
    entry = {"data_type": data_type, "measure_type": measure_type, "target_frame": target_frame}
    if reference_frame is not None:
        entry["reference_frame"] = reference_frame
    return entry | extra


def _flags(target_frame: str, names: Sequence[str], reference_frame: str | None = None) -> dict[str, Any]:
    mapping = {str(bit): name for bit, name in enumerate(names)}
    return _entry("UINT32", "STATUS_FLAGS", target_frame, reference_frame, bit_mapping=mapping)


def glove_definition(side: str, device_id: int, *, magnetometer: bool = False) -> dict[str, Any]:
    """The Stream Definition document of one emulated Smartgloves II glove.

    `magnetometer` adds the hub MAGNETIC_FIELD stream, which the real driver only declares when
    an experimental feature is on.
    """
    default: list[dict[str, Any]] = []
    emf_only: list[dict[str, Any]] = []
    for sensor in _SENSORS:
        default += [
            _entry("FLOAT[7]", "TRANSFORM", sensor, HUB),
            _entry("FLOAT[3]", "ANGULAR_VELOCITY", _imu(sensor)),
            _entry("FLOAT[3]", "PROPER_ACCELERATION", _imu(sensor)),
            _flags(sensor, _SENSOR_FLAGS),
        ]
        emf_only += [
            _entry("FLOAT[7]", "TRANSFORM", sensor, HUB),
            _entry("FLOAT", "CUSTOM", sensor, HUB, custom_label=_RESIDUAL),
            _flags(sensor, (EMF_SATURATED,)),
        ]
    default += [
        _entry("FLOAT[4]", "ORIENTATION", HUB, NED),
        _entry("FLOAT[3]", "ANGULAR_VELOCITY", _imu(HUB)),
        _entry("FLOAT[3]", "PROPER_ACCELERATION", _imu(HUB)),
        *([_entry("FLOAT[3]", "MAGNETIC_FIELD", HUB)] if magnetometer else []),
        _flags(HUB, _HUB_FLAGS),
        _entry("FLOAT[7]", "TRANSFORM", ARM_SENSOR, COIL),
        _entry("FLOAT", "CUSTOM", ARM_SENSOR, COIL, custom_label=_RESIDUAL),
        _flags(ARM_SENSOR, _COIL_FLAGS, COIL),
    ]
    static_data = [
        *(
            _entry("FLOAT[7]", "TRANSFORM", _imu(frame), frame, value=[*_IMU_OFFSET, 0.0, 0.0, 0.0, 1.0])
            for frame in (HUB, *_SENSORS)
        ),
        *(
            _entry("FLOAT[10]", "CUSTOM", frame, custom_label="OBB", value=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, *half])
            for frame, half in _HALF_EXTENTS.items()
        ),
    ]
    return {
        "protocol_name": "RGMP",
        "protocol_version": "2.0.0",
        "kaiku_emulated": True,
        "device_id": device_id,
        "device_type": "Smartgloves II",
        "timestamp_epoch": "device_boot",
        "device_info": {"friendly_name": f"Smartgloves II ({side})", "hand": side},
        "static_data": static_data,
        "groups": [
            {"name": DEFAULT_GROUP, "expected_rate_hz": RATE_HZ, "streams": default},
            {"name": "emf_only", "expected_rate_hz": RATE_HZ, "streams": emf_only},
        ],
    }


def _compose(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    """Hamilton product of two w,x,y,z quaternions: the orientation `b`, given in frame `a`, in a's parent."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _seen_from(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    """A world vector `v` expressed in the frame whose orientation in the world is `q` (w,x,y,z)."""
    w, x, y, z = q
    vx, vy, vz = v
    return (
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y + w * z) * vy + 2 * (x * z - w * y) * vz,
        2 * (x * y - w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z + w * x) * vz,
        2 * (x * z + w * y) * vx + 2 * (y * z - w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
    )


def _wire_pose(pose: SensorPose | None) -> tuple[float, ...]:
    """A pose as the wire wants it: x,y,z then the quaternion as x,y,z,w."""
    if pose is None:
        return _NO_POSE
    qw, qx, qy, qz = pose.quat.tolist()
    return (*pose.pos.tolist(), qx, qy, qz, qw)


class _Glove:
    """One emulated glove: its definition, plus the value buffers its frames are packed from."""

    def __init__(self, side: str, magnetometer: bool) -> None:
        self.side = side
        self.definition = Definition(glove_definition(side, DEVICE_IDS[side], magnetometer=magnetometer))
        self._default, self._emf_only = default, emf_only = self.definition.groups
        self._values = default.zeros()
        self._emf_values = emf_only.zeros()
        # Per sensor: slots of its pose, accelerometer and flags, then of its emf_only pose and flags.
        self._sensors = [
            (
                default.require("TRANSFORM", sensor, HUB).slot,
                default.require("PROPER_ACCELERATION", _imu(sensor)).slot,
                default.require("STATUS_FLAGS", sensor).slot,
                emf_only.require("TRANSFORM", sensor, HUB).slot,
                emf_only.require("STATUS_FLAGS", sensor).slot,
            )
            for sensor in _SENSORS
        ]
        self._sensor_saturated = default.require("STATUS_FLAGS", _SENSORS[0]).mask(EMF_SATURATED)
        self._emf_saturated = emf_only.require("STATUS_FLAGS", _SENSORS[0]).mask(EMF_SATURATED)
        self._hub = default.require("ORIENTATION", HUB, NED).slot
        self._hub_acceleration = default.require("PROPER_ACCELERATION", _imu(HUB)).slot
        field = default.find("MAGNETIC_FIELD", HUB)
        self._hub_field = None if field is None else field.slot
        hub_flags = default.require("STATUS_FLAGS", HUB)
        self._hub_flags, self._heading_reliable = hub_flags.slot, hub_flags.mask(HEADING_RELIABLE)
        self._coil = default.require("TRANSFORM", ARM_SENSOR, COIL).slot
        coil_flags = default.require("STATUS_FLAGS", ARM_SENSOR, COIL)
        self._coil_flags = coil_flags.slot
        self._coil_present, self._coil_flipped, self._coil_saturated = (
            coil_flags.mask(flag) for flag in (COIL_PRESENT, FLIP_DECIDED, EMF_SATURATED)
        )
        self._values[default.require("CUSTOM", ARM_SENSOR, COIL, _RESIDUAL).slot] = _RESIDUAL_PERCENT
        for sensor in _SENSORS:
            self._emf_values[emf_only.require("CUSTOM", sensor, HUB, _RESIDUAL).slot] = _RESIDUAL_PERCENT

    def encode(self, frame: GloveFrame, tick: int) -> tuple[bytes, bytes]:
        """The `default` and `emf_only` data frames of this glove for one sample."""
        values, emf_values = self._values, self._emf_values
        hub = frame.hub_quat_ned.tolist()
        poses = (*(frame.fingers.get(finger) for finger in FINGERS), frame.arm_in_hub)
        for (pose_at, acceleration_at, flags_at, emf_pose_at, emf_flags_at), pose in zip(self._sensors, poses):
            wire = _wire_pose(pose)
            saturated = pose is None or not pose.valid
            in_ned = _compose(hub, (wire[6], wire[3], wire[4], wire[5]))
            values[pose_at:pose_at + 7] = emf_values[emf_pose_at:emf_pose_at + 7] = wire
            values[acceleration_at:acceleration_at + 3] = _seen_from(in_ned, _REST_ACCELERATION)
            values[flags_at] = self._sensor_saturated if saturated else 0
            emf_values[emf_flags_at] = self._emf_saturated if saturated else 0

        values[self._hub:self._hub + 4] = (hub[1], hub[2], hub[3], hub[0])
        values[self._hub_acceleration:self._hub_acceleration + 3] = _seen_from(hub, _REST_ACCELERATION)
        if self._hub_field is not None:
            values[self._hub_field:self._hub_field + 3] = _seen_from(hub, _EARTH_FIELD)
        values[self._hub_flags] = self._heading_reliable if frame.heading_reliable else 0

        coil = frame.arm_in_coil if frame.coil_present else None
        values[self._coil:self._coil + 7] = _wire_pose(coil)
        values[self._coil_flags] = (
            (self._coil_present if coil is not None else 0)
            | (self._coil_saturated if coil is not None and not coil.valid else 0)
            | (self._coil_flipped if frame.coil_flip_decided else 0)
        )

        device_id = self.definition.device_id
        timestamp_us = _BOOT_US[self.side] + tick * PERIOD_US
        return (
            self._default.encode(device_id, timestamp_us, values),
            self._emf_only.encode(device_id, timestamp_us, emf_values),
        )


class Emulator:
    """Serves a synthetic Smartgloves II pair on `host:port` (port 0: any free port, see `port`).

    Samples are taken and sent on one background thread against absolute 12 ms deadlines, so the
    average rate is exactly 250/3 Hz and only the OS wake-up jitter (around 0.1 ms) shows. Each
    glove's `timestamp_us` advances by exactly 12000 per sample on its own device-boot clock.
    """

    def __init__(
        self,
        source: GloveSource | None = None,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        *,
        magnetometer: bool = False,
    ) -> None:
        self._source = source or StillSource()
        self._server = RgmpServer(host, port)
        self._gloves = [_Glove(side, magnetometer) for side in SIDES]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The TCP port being served; valid after `start`."""
        return self._server.port

    def start(self) -> Emulator:
        """Open the port, announce both gloves and start streaming; returns self."""
        if self._thread is None:
            self._server.start()
            for glove in self._gloves:
                self._server.send(glove.definition.encode())
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="rgmp-emulator", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop streaming, tell the clients both gloves are gone, and close the port."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        for glove in self._gloves:
            self._server.send(encode_disconnect(glove.definition.device_id))
        self._server.stop()

    def __enter__(self) -> Emulator:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _run(self) -> None:
        started = time.monotonic()
        tick = 0
        while True:
            late = time.monotonic() - started - tick * _PERIOD_S
            if late > _MAX_LAG_S:
                tick += int(late / _PERIOD_S)               # drop the samples we slept through
            if self._stop.wait(max(0.0, -late)):
                return
            for glove in self._gloves:
                frame = self._source.sample(glove.side, tick * _PERIOD_S)
                if frame is not None:
                    for wire in glove.encode(frame, tick):
                        self._server.send(wire)
            tick += 1


def load_source(spec: str) -> GloveSource:
    """Build a source from "package.module:factory", where `factory()` returns a `GloveSource`."""
    module, _, factory = spec.partition(":")
    return getattr(importlib.import_module(module), factory)()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emulate a Smartgloves II pair over RGMP.")
    parser.add_argument("--host", default="127.0.0.1", help="address to serve on (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to serve on (default: %(default)s)")
    parser.add_argument("--source", metavar="MODULE:FACTORY", help="motion source (default: two still, flat hands)")
    parser.add_argument("--magnetometer", action="store_true", help="also declare the hub MAGNETIC_FIELD stream")
    args = parser.parse_args(argv)
    source = load_source(args.source) if args.source else None
    stop = stop_on_signals()
    with Emulator(source, args.host, args.port, magnetometer=args.magnetometer) as emulator:
        print(f"emulating a Smartgloves II pair on {args.host}:{emulator.port}, Ctrl-C to stop", file=sys.stderr)
        stop.wait()


if __name__ == "__main__":
    main()
