"""Hands already solved by Rokoko: the stream of ``rkk-hand-solver`` (Device SDK), RGMP v2, port 12277.

``rkk-hand-solver --emit world`` reads the gloves from the driver and publishes every joint of each hand as a
transform, in the group ``joints_world``. This module keeps the newest frame per side as joint positions in
the kaiku hand frame, ready for ``kaiku.hand.solved``.

Kaiku has a glove solver of its own (``kaiku.hand.solve``), written from the documented sensor layout. The
solver here is Rokoko's, tuned on real gloves, so when it is running the fingers come from it.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from kaiku.hand.geometry import quat_to_mat
from kaiku.hand.glove_model import HAND_FROM_GLOVE
from kaiku.types import SIDES

from .client import DataFrame, RgmpClient
from .protocol import Definition, ProtocolError

_log = logging.getLogger(__name__)

SOLVED_PORT = 12277
SOLVED_DEVICE_TYPE = "solved_hand"
GROUP = "joints_world"
# The chain of every digit as [joint 0, joint 1, joint 2, tip], in FINGERS order. Rokoko numbers the digits
# from the thumb (finger1) to the little finger (finger5); the thumb has no medial bone.
_THUMB = ("metacarpal", "proximal", "distal", "tip")
_FINGER = ("proximal", "medial", "distal", "tip")
CHAINS = tuple(tuple(f"finger{n}_{bone}" for bone in (_THUMB if n == 1 else _FINGER)) for n in range(1, 6))
_RATE_WINDOW_S = 1.0


@dataclass(frozen=True, slots=True)
class SolvedHand:
    """One hand as Rokoko solved it."""

    side: str
    t_recv: float               # host ``time.monotonic()`` when the frame arrived
    points: np.ndarray          # (5, 4, 3) joint centres and fingertip of every digit, kaiku hand frame, metres


class SolvedDecoder:
    """Reads the ``joints_world`` frames of one solved hand."""

    def __init__(self, definition: Definition) -> None:
        if definition.device_type != SOLVED_DEVICE_TYPE:
            raise ProtocolError(f"device type {definition.device_type!r} is not a solved hand")
        side = definition.device_info.get("hand")
        if side not in SIDES:
            raise ProtocolError(f"solved hand without a side: {side!r}")
        group = definition.group(GROUP)
        if group is None:
            raise ProtocolError(f"no {GROUP} group; start rkk-hand-solver with --emit world")
        slots = {s.target_frame: s.slot for s in group.streams if s.measure_type == "TRANSFORM"}
        try:
            wrist = slots[f"{side}_hand"]
            chains = [[slots[f"{side}_{name}"] for name in chain] for chain in CHAINS]
        except KeyError as missing:
            raise ProtocolError(f"solved hand without joint {missing}") from None
        self.side = side
        self.group = group
        self._wrist = wrist
        self._points = np.asarray(chains)[:, :, None] + np.arange(3)             # (5, 4, 3) value indices
        self._axes = HAND_FROM_GLOVE[side]

    def decode(self, frame: DataFrame) -> SolvedHand:
        values = np.asarray(frame.values(), dtype=float)
        w = self._wrist
        x, y, z, s = values[w + 3:w + 7]                                         # x y z w on the wire
        # Rokoko's hand bone has the glove's axes (+X right, +Y to the wrist, +Z down); out of its frame
        # and into kaiku's (+x to the fingertips, +z out of the back of the hand).
        turn = self._axes @ quat_to_mat((s, x, y, z)).T
        return SolvedHand(self.side, frame.t_recv, (values[self._points] - values[w:w + 3]) @ turn.T)


class SolvedStream:
    """Live solved hands: the newest `SolvedHand` per side, fed by a background `RgmpClient`.

    Quiet when nothing listens on the port: it keeps trying and every side reads None, so a consumer falls
    back to its own solver.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = SOLVED_PORT) -> None:
        self.client = RgmpClient(host, port, on_definition=self._on_definition, on_data=self._on_data,
                                 on_disconnect=self._on_disconnect)
        self._decoders: dict[int, SolvedDecoder] = {}        # reader thread only
        self._lock = threading.Lock()
        self._latest: dict[str, SolvedHand] = {}
        self._arrivals: dict[str, deque[float]] = {side: deque() for side in SIDES}

    def start(self) -> SolvedStream:
        self.client.start()
        return self

    def stop(self) -> None:
        self.client.stop()

    @property
    def connected(self) -> bool:
        return self.client.connected

    def latest(self, side: str, fresh_s: float = 0.25) -> SolvedHand | None:
        """The newest solved hand of one side; None when there is none or it is older than ``fresh_s``."""
        with self._lock:
            hand = self._latest.get(side)
        if hand is None or time.monotonic() - hand.t_recv > fresh_s:
            return None
        return hand

    def rate_hz(self, side: str) -> float:
        with self._lock:
            arrivals = self._arrivals[side]
            if len(arrivals) < 2 or time.monotonic() - arrivals[-1] > _RATE_WINDOW_S:
                return 0.0
            span = arrivals[-1] - arrivals[0]
            return (len(arrivals) - 1) / span if span > 0 else 0.0

    def _on_definition(self, definition: Definition) -> None:
        self._on_disconnect(definition.device_id)
        if definition.device_type != SOLVED_DEVICE_TYPE:
            return
        try:
            self._decoders[definition.device_id] = SolvedDecoder(definition)
        except ProtocolError as error:
            _log.warning("ignoring solved hand %d: %s", definition.device_id, error)

    def _on_data(self, frame: DataFrame) -> None:
        decoder = self._decoders.get(frame.definition.device_id)
        if decoder is None or frame.group is not decoder.group:
            return
        hand = decoder.decode(frame)
        if not np.isfinite(hand.points).all():
            return
        with self._lock:
            self._latest[decoder.side] = hand
            arrivals = self._arrivals[decoder.side]
            arrivals.append(hand.t_recv)
            while hand.t_recv - arrivals[0] > _RATE_WINDOW_S:
                arrivals.popleft()

    def _on_disconnect(self, device_id: int) -> None:
        decoder = self._decoders.pop(device_id, None)
        if decoder is not None:
            with self._lock:
                self._latest.pop(decoder.side, None)
                self._arrivals[decoder.side].clear()
