"""Hands solved by Rokoko's rkk-hand-solver: its stream, the fingers read from it, and the operator's range."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from kaiku.hand import Calibration, Teleop
from kaiku.hand.geometry import quat_from_rotvec, quat_to_mat
from kaiku.hand.glove_model import HAND_FROM_GLOVE
from kaiku.hand.model import HandModel
from kaiku.hand.solved import hand_length, joints_from_points, with_fingers
from kaiku.rgmp import RgmpServer
from kaiku.rgmp.client import DataFrame
from kaiku.rgmp.protocol import Definition, ProtocolError
from kaiku.rgmp.solved import CHAINS, GROUP, SOLVED_DEVICE_TYPE, SolvedDecoder, SolvedStream
from kaiku.teleop.retarget import GAP_KNOTS, HOLD_GAP, Retarget, closure_from_gap
from kaiku.types import SIDES, HumanHand

HOST = "127.0.0.1"
EYE = np.array([1.0, 0.0, 0.0, 0.0])


def some_joints(model: HandModel, rng: np.random.Generator) -> np.ndarray:
    """A hand inside the model's limits whose first bones are not curled straight down."""
    joints = rng.uniform(model.lower, model.upper)
    joints[:, 0] = np.minimum(joints[:, 0], np.radians(70.0))
    return joints


def solved_definition(side: str, device_id: int, *, group: str = GROUP) -> Definition:
    """What rkk-hand-solver --emit world announces for one hand: 26 joint transforms against LTP_NED."""
    joints = ["fore_arm", "hand"]
    for number in (5, 4, 3, 2, 1):                                   # little finger first, as the solver sends them
        bones = ("metacarpal", "proximal", "distal", "tip") if number == 1 else \
            ("metacarpal", "proximal", "medial", "distal", "tip")
        joints += [f"finger{number}_{bone}" for bone in bones]
    streams = [{"data_type": "FLOAT[7]", "measure_type": "TRANSFORM", "target_frame": f"{side}_{joint}",
                "reference_frame": "LTP_NED"} for joint in joints]
    return Definition({
        "protocol_name": "RGMP", "protocol_version": "2.0.0", "device_id": device_id,
        "device_type": SOLVED_DEVICE_TYPE, "timestamp_epoch": "device_boot", "device_info": {"hand": side},
        "groups": [{"name": group, "expected_rate_hz": 83.33, "streams": streams}],
    })


def solved_values(definition: Definition, side: str, points: np.ndarray, wrist_pos: np.ndarray,
                  wrist_quat: np.ndarray) -> list[float]:
    """The values of one frame for a hand whose joints are at ``points`` (kaiku hand frame), with the hand
    bone (Rokoko's axes) at ``wrist_pos``, ``wrist_quat`` (wxyz) in the world."""
    group = definition.group(GROUP)
    world_from_bone = quat_to_mat(wrist_quat)
    values = [0.0] * sum(stream.data_type.count for stream in group.streams)
    w, x, y, z = wrist_quat
    where = {f"{side}_hand": wrist_pos}
    for chain, digit in zip(CHAINS, points):
        for name, point in zip(chain, digit):
            where[f"{side}_{name}"] = wrist_pos + world_from_bone @ HAND_FROM_GLOVE[side].T @ point
    for stream in group.streams:
        pos = where.get(stream.target_frame, np.array([9.0, 9.0, 9.0]))        # joints kaiku does not read
        values[stream.slot:stream.slot + 7] = [*pos, x, y, z, w]
    return values


def frame_of(definition: Definition, values: list[float], t_recv: float = 0.0) -> DataFrame:
    group = definition.group(GROUP)
    wire = group.encode(definition.device_id, 1000, values)
    return DataFrame(definition, group, 1000, wire[8:], t_recv)                # past the outer frame header


@pytest.mark.parametrize("side", SIDES)
def test_joints_come_back_from_the_points_of_any_hand_the_model_can_make(side):
    model, rng = HandModel(side), np.random.default_rng(3)
    for _ in range(200):
        joints = some_joints(model, rng)
        np.testing.assert_allclose(joints_from_points(model, model.points(joints)), joints, atol=1e-9)


@pytest.mark.parametrize("side", SIDES)
def test_bone_lengths_of_the_other_skeleton_do_not_matter(side):
    model, other, rng = HandModel(side), HandModel(side, 0.21), np.random.default_rng(4)
    other.lengths = other.lengths * rng.uniform(0.8, 1.3, size=(5, 3))
    joints = some_joints(model, rng)
    np.testing.assert_allclose(joints_from_points(model, other.points(joints)), joints, atol=1e-9)


def test_a_digit_curled_straight_down_counts_as_pointing_ahead():
    model = HandModel("right")
    joints = np.zeros((5, 4))
    joints[1] = [np.radians(88.0), 0.3, 0.2, 0.2]
    got = joints_from_points(model, model.points(joints))
    assert got[1, 3] == 0.0 and abs(got[1, 0] - np.radians(88.0)) < np.radians(12.0)


@pytest.mark.parametrize("side", SIDES)
def test_decoder_brings_the_joints_into_the_hand_frame_wherever_the_hand_is(side):
    model, rng = HandModel(side, 0.19), np.random.default_rng(5)
    definition = solved_definition(side, 77)
    decoder = SolvedDecoder(definition)
    points = model.points(some_joints(model, rng))
    for _ in range(5):
        quat = quat_from_rotvec(rng.normal(size=3))
        values = solved_values(definition, side, points, rng.normal(size=3), quat)
        hand = decoder.decode(frame_of(definition, values, 12.5))
        assert (hand.side, hand.t_recv, hand.points.shape) == (side, 12.5, (5, 4, 3))
        np.testing.assert_allclose(hand.points, points, atol=2e-6)             # float32 on the wire


def test_decoder_refuses_what_is_not_a_solved_hand():
    good = solved_definition("left", 1)
    with pytest.raises(ProtocolError, match="not a solved hand"):
        SolvedDecoder(Definition({**good.document, "device_type": "Smartgloves II"}))
    with pytest.raises(ProtocolError, match="without a side"):
        SolvedDecoder(Definition({**good.document, "device_info": {}}))
    with pytest.raises(ProtocolError, match="--emit world"):
        SolvedDecoder(solved_definition("left", 1, group="joints_local"))
    short = dict(good.document)
    short["groups"] = [{**good.document["groups"][0], "streams": good.document["groups"][0]["streams"][:10]}]
    with pytest.raises(ProtocolError, match="without joint"):
        SolvedDecoder(Definition(short))


def test_with_fingers_keeps_the_wrist_and_takes_the_other_solvers_gaps_at_our_size():
    model, theirs = HandModel("right"), HandModel("right", 0.21)
    joints = some_joints(model, np.random.default_rng(6))
    own = HumanHand("right", 1.0, np.array([0.1, 0.2, 0.3]), EYE, True, model.tips(np.zeros((5, 4))),
                    np.zeros((5, 4)), np.zeros(5), np.ones(4))
    points = theirs.points(joints)
    assert hand_length(points) == pytest.approx(0.21)
    hand = with_fingers(own, model, points)
    assert hand.wrist_pos is own.wrist_pos and hand.has_position and hand.t == 1.0
    np.testing.assert_allclose(hand.joints, joints, atol=1e-9)
    tips = points[:, 3] * (model.hand_length / 0.21)
    np.testing.assert_allclose(hand.gaps, np.linalg.norm(tips[1:] - tips[0], axis=1))
    np.testing.assert_allclose(hand.pinch, hand.gaps)
    np.testing.assert_allclose(hand.tips, model.tips(joints))


def test_teleop_carries_measured_gaps_through_and_leaves_the_models_alone():
    model = HandModel("left")
    flat = np.zeros((5, 4))
    tips = model.tips(flat)
    own = {side: HumanHand(side, 0.0, np.zeros(3), EYE, True, HandModel(side).tips(flat), flat, np.zeros(5),
                           np.linalg.norm(tips[1:] - tips[0], axis=1)) for side in SIDES}
    teleop = Teleop(Calibration.capture(own["left"], own["right"]))
    plain = teleop.update(own, 0.0).hands["left"]
    assert plain.gaps is None
    np.testing.assert_allclose(plain.pinch, own["left"].pinch)
    measured = {**own, "left": replace(own["left"], gaps=np.array([0.02, 0.03, 0.04, 0.05]))}
    for step in range(1, 200):
        got = teleop.update(measured, step / 83.0).hands["left"]
    np.testing.assert_allclose(got.pinch, [0.02, 0.03, 0.04, 0.05], atol=1e-4)
    np.testing.assert_allclose(got.gaps, got.pinch)


def test_gap_range_grips_from_the_first_try_and_lets_go_with_a_flat_hand():
    rng_ = Retarget().gap_range()
    # A glove that reads 28 mm with the fingers together and 70 mm with the hand flat.
    rng_.open(np.full(4, 0.070))
    assert closure_from_gap(rng_(np.full(4, 0.070))[0]) <= 0.0                 # flat hand: open, lets go
    assert closure_from_gap(rng_(np.full(4, 0.034))[0]) >= 1.0                 # closing grips before any pinch
    assert closure_from_gap(rng_(np.full(4, 0.028))[0]) > 1.15                 # fingers together: squeeze
    np.testing.assert_allclose(rng_.low, 0.028)
    assert closure_from_gap(rng_(np.full(4, 0.070))[0]) == -1.0                # and the flat hand is now flat
    assert 0.028 < 0.028 + (HOLD_GAP - GAP_KNOTS[0]) / 3.0 < 0.040             # the grip sits just above the pinch


def test_gap_range_leaves_a_glove_that_reads_true_as_it_is():
    retarget = Retarget()
    rng_ = retarget.gap_range()
    rng_.open(rng_.flat)
    rng_(rng_.closed)                                                          # the operator has pinched once
    for gap in (0.012, 0.035, 0.06, 0.125):
        index, middle = gap, float(np.interp(gap, retarget._index_gap, retarget._middle_gap))
        got = rng_(np.array([index, middle, gap, gap]))
        np.testing.assert_allclose(got[:2], [index, middle], atol=1e-9)


def test_solved_stream_over_a_socket_reads_both_hands_and_forgets_stale_ones():
    left, right = solved_definition("left", 11), solved_definition("right", 12)
    models = {"left": HandModel("left"), "right": HandModel("right")}
    points = {side: models[side].points(some_joints(models[side], np.random.default_rng(7))) for side in SIDES}
    with RgmpServer(HOST, 0) as server:
        stream = SolvedStream(HOST, server.port).start()
        try:
            assert stream.latest("left") is None and stream.rate_hz("left") == 0.0
            assert server.wait_for_client(10.0)
            server.send(left.encode())
            server.send(right.encode())
            deadline = time.monotonic() + 10.0
            tick = 0
            while (stream.latest("left") is None or stream.latest("right") is None or stream.rate_hz("left") < 5):
                assert time.monotonic() < deadline
                for definition, side in ((left, "left"), (right, "right")):
                    values = solved_values(definition, side, points[side], np.zeros(3), EYE)
                    server.send(definition.group(GROUP).encode(definition.device_id, tick * 12000, values))
                tick += 1
                time.sleep(0.01)
            for side in SIDES:
                np.testing.assert_allclose(stream.latest(side).points, points[side], atol=2e-6)
            assert stream.connected
            time.sleep(0.12)
            assert stream.latest("left", fresh_s=0.1) is None                  # too old to act on
            assert stream.latest("left", fresh_s=5.0) is not None
        finally:
            stream.stop()


class _Seat:
    """The little of a session that a glove operator needs before it commands anything."""
    home_pinch = np.zeros((2, 3))
    home_rot = np.stack([np.eye(3), np.eye(3)])


class _Raisable:
    """Still, flat hands that can be lifted, seen by a Coil Pro whose Z points up or down."""

    def __init__(self, z_down: bool) -> None:
        from kaiku.rgmp.emulator import StillSource
        self.still, self.z_down, self.lift = StillSource(), z_down, 0.0

    def sample(self, side: str, t: float):
        from kaiku.hand.geometry import quat_mul
        from kaiku.types import SensorPose
        frame = self.still.sample(side, t)
        pose = frame.arm_in_coil
        # The still hands are palm down with their sensors' +Z (down) along the coil's +Z: a coil with Z down.
        pos, quat = pose.pos - [0.0, 0.0, self.lift], pose.quat
        if not self.z_down:
            pos, quat = pos * [1.0, -1.0, -1.0], quat_mul([0.0, 1.0, 0.0, 0.0], quat)
        return replace(frame, arm_in_coil=SensorPose(pos, quat))


@pytest.mark.parametrize("z_down", [False, True])
def test_calibration_finds_out_which_way_up_the_coil_is(z_down):
    from kaiku.rgmp.emulator import Emulator
    from kaiku.teleop.operator import GloveOperator

    source = _Raisable(z_down)
    emulator = Emulator(source, HOST, 0).start()
    operator = GloveOperator(_Seat(), HOST, emulator.port, solver="off")
    try:
        def solved() -> dict:
            deadline = time.monotonic() + 10.0
            while True:
                frames = {side: operator.stream.latest(side) for side in SIDES}
                if None not in frames.values() and all(
                        abs(f.arm_in_coil.pos[2]) == pytest.approx(0.20 + source.lift, abs=1e-6)
                        for f in frames.values()):
                    return {side: operator._solve(side, frame) for side, frame in frames.items()}
                assert time.monotonic() < deadline
                time.sleep(0.005)

        operator.hands = solved()
        assert operator.calibrate()
        low = operator.hands["left"].wrist_pos[2]
        np.testing.assert_allclose(operator.ranges["left"].high, operator.hands["left"].pinch)
        source.lift = 0.10
        assert solved()["left"].wrist_pos[2] - low == pytest.approx(0.10, abs=1e-4)     # up is up either way
    finally:
        operator.close()
        emulator.stop()


def test_calibration_waits_for_flat_hands():
    from kaiku.rgmp.emulator import Emulator, StillSource
    from kaiku.teleop.operator import GloveOperator

    emulator = Emulator(StillSource(), HOST, 0).start()
    operator = GloveOperator(_Seat(), HOST, emulator.port, solver="off")
    try:
        deadline = time.monotonic() + 10.0
        while operator.stream.latest("left") is None or operator.stream.latest("right") is None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        hands = {side: operator._solve(side, operator.stream.latest(side)) for side in SIDES}
        on_edge = quat_from_rotvec([np.pi / 2, 0.0, 0.0])                       # a hand turned thumb-up
        operator.hands = {**hands, "left": replace(hands["left"], wrist_quat=on_edge)}
        assert not operator.calibrate() and operator.teleop is None
        operator.hands = hands
        assert operator.calibrate()
    finally:
        operator.close()
        emulator.stop()
