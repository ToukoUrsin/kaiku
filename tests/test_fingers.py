"""One hand's fingers from a camera's landmarks or from one Smartglove, down the gloves' own path."""
from __future__ import annotations

import time

import numpy as np

import kaiku.app.visitor as visitor_module
from kaiku.app.session import Session
from kaiku.app.visitor import DEPTH_FULL, VisitorOperator
from kaiku.hand.model import HandModel
from kaiku.hand.poses import HandPoses
from kaiku.hand.synth import DEFAULT_HOME, hand_to_glove_frame
from kaiku.rgmp.emulator import Emulator
from kaiku.teleop.fingers import GloveFingers, solve_landmarks

# A right hand held up to a camera, palm towards it, fingers up: the hand's x, y, z in the camera's axes.
UP = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MODEL = HandModel("right")
POSES = HandPoses(MODEL)


def _landmarks(aperture: float | None, rng=None, noise: float = 0.0) -> np.ndarray:
    joints = POSES.open if aperture is None else POSES.grip(aperture)
    points = np.vstack([np.zeros((1, 3)), MODEL.points(joints).reshape(20, 3)]) @ UP.T
    return points if rng is None else points + rng.normal(0.0, noise, points.shape)


def _gap(aperture: float | None) -> float:
    joints = POSES.open if aperture is None else POSES.grip(aperture)
    tips = MODEL.tips(joints)
    return float(np.linalg.norm(tips[1] - tips[0]))


def test_landmarks_read_as_the_hand_they_came_from():
    for aperture in (None, 0.10, 0.06, 0.03):
        world, want = _landmarks(aperture), _gap(aperture)
        hand, _ = solve_landmarks("right", world, True, MODEL, 0.0)
        assert abs(float(hand.pinch[0]) - want) < 0.004
        assert not hand.has_position                          # where the hand is, is not the fingers' to say
        # the same hand for the left arm, and the same hand when the camera calls it a left one
        left, _ = solve_landmarks("left", world, True, HandModel("left"), 0.0)
        wrong, _ = solve_landmarks("right", world, False, MODEL, 0.0)
        assert abs(float(left.pinch[0]) - want) < 0.004
        assert abs(float(wrong.pinch[0]) - want) < 0.004
    opened, _ = solve_landmarks("right", _landmarks(None), True, MODEL, 0.0)
    closed, _ = solve_landmarks("right", _landmarks(0.02), True, MODEL, 0.0)
    assert float(closed.curl[1]) > float(opened.curl[1]) + 0.3   # the index finger reads as curled


def test_a_persons_own_grasp_sorts_and_the_echoes_follow(monkeypatch):
    """Nothing scripted in the grasp: 21 noisy landmarks a tick, the hand's place and how near it is."""
    rng = np.random.default_rng(1)
    s = Session(4, seed=1)
    s.new_episode(*s.sample_scene(3))
    v = VisitorOperator(s)
    v.begin()
    clock = [time.monotonic()]
    monkeypatch.setattr(visitor_module.time, "monotonic", lambda: clock[0])
    state = {"x": 0.28, "y": 0.0, "depth": 0.0, "aperture": None}

    def tick():
        clock[0] += 1 / 50
        v.set_command({"x": state["x"], "y": state["y"], "grip": False, "input": "camera", "own": True,
                       "world": _landmarks(state["aperture"], rng, 0.002).round(4).reshape(-1).tolist(),
                       "right": True, "depth": state["depth"]})
        s.tick(v.update(s.observe()))

    def run(until, limit):
        for _ in range(limit):
            tick()
            if until():
                return

    lay = s.layouts[0]
    for k in sorted(range(len(lay.placements)), key=lambda k: lay.placements[k].pos[0]):
        ob = s.scene.by_name[lay.placements[k].name]
        box = s.scene.bin_centers[lay.mapping[ob.color]][:2]
        part = lambda: s.observe().obj_pos[0, ob.index]
        over = lambda xy, tol=0.012: np.linalg.norm(s.observe().pinch[0, v.side, :2] - xy) < tol
        p = part()[:2].copy()
        state.update(x=float(p[0]), y=float(p[1]), depth=0.0, aperture=None)
        run(lambda: over(p), 400)
        run(lambda: False, 40)                                # the palm turns to the part
        state["depth"] = DEPTH_FULL * 1.1                     # the hand goes towards the camera: down
        run(lambda: s.observe().pinch[0, v.side, 2] < s.scene.rest_z + 0.012, 300)
        run(lambda: False, 10)
        for aperture in np.linspace(0.11, 0.02, 15):          # the fingers close
            state["aperture"] = float(aperture)
            tick()
        run(lambda: part()[2] > s.scene.rest_z + 0.05, 200)
        assert v.hud()["own"] and v.fingers == "camera"
        state["depth"] = 0.0
        start = np.array([state["x"], state["y"]])
        for i in range(60):                                   # carried to its bin
            q = start + (box - start) * (i + 1) / 60
            state.update(x=float(q[0]), y=float(q[1]))
            tick()
        run(lambda: over(v.command, 0.02), 300)
        run(lambda: False, 25)
        for aperture in np.linspace(0.02, 0.12, 12):          # and the fingers open
            state["aperture"] = float(aperture)
            tick()
        state["aperture"] = None
        run(lambda: False, 60)
    assert int(s.progress()[0]) == len(lay.placements)
    assert "own grasp" in v.who() and "camera" in v.who()
    s.finish_lead()
    while not s.settled and s.tick_no < 12000:
        tick()
    r = s.result()
    assert r.success.all() and int(r.clipped.sum()) == 0
    s.close()


class _OneGlove:
    """A right Smartglove with no Coil Pro: fingers and turn, no position. The left hand has none."""

    def __init__(self):
        self.aperture: float | None = None

    def sample(self, side, t):
        if side != "right":
            return None
        joints = POSES.open if self.aperture is None else POSES.grip(self.aperture)
        return hand_to_glove_frame("right", joints, DEFAULT_HOME["right"], t, coil_present=False)


def _until(check, timeout=10.0):
    end = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < end, "not in time"
        time.sleep(0.01)


def test_one_glove_without_a_coil_is_the_fingers_of_either_arm():
    source = _OneGlove()
    with Emulator(source, port=0) as emulator:
        glove = GloveFingers("127.0.0.1", emulator.port, solver="off")
        try:
            _until(lambda: glove.hand("right", time.monotonic()) is not None)
            flat = glove.hand("right", time.monotonic())
            assert not flat.has_position and glove.hud()["side"] == "right" and not glove.hud()["coil"]
            assert abs(float(flat.pinch[0]) - _gap(None)) < 0.02
            source.aperture = 0.03
            _until(lambda: float(glove.hand("left", time.monotonic()).pinch[0]) < 0.05)   # mirrored onto the left arm
            assert glove.hand("left", time.monotonic()).side == "left"
        finally:
            glove.close()


def test_one_gloves_pinch_grasps_for_a_visitor_with_a_pointer():
    source = _OneGlove()
    s = Session(1, seed=7, reachable_only=True)
    with Emulator(source, port=0) as emulator:
        glove = GloveFingers("127.0.0.1", emulator.port, solver="off")
        v = VisitorOperator(s, glove=glove)
        try:
            _until(lambda: glove.hand("right", time.monotonic()) is not None)
            placement = min(s.layouts[0].placements, key=lambda p: p.pos[0])
            ob = s.scene.by_name[placement.name]
            box = s.scene.bin_centers[s.layouts[0].mapping[ob.color]]

            def run(until, limit, x, y):
                for _ in range(limit):
                    v.set_command({"x": float(x), "y": float(y), "grip": False})   # the pointer never presses
                    s.tick(v.update(s.observe()))
                    if until():
                        return True
                return False

            run(lambda: False, 60, *placement.pos[:2])
            assert v.fingers == "glove" and v.phase == "idle"
            source.aperture = 0.02                                # thumb to index finger: the grasp
            assert run(lambda: v.phase == "carry", 900, *placement.pos[:2])
            run(lambda: False, 200, *box[:2])
            source.aperture = None                                # the hand opens over the bin
            assert run(lambda: s.progress()[0] >= 1, 900, *box[:2])
            assert "one Smartglove" in v.who() and "pointer" in v.who()
            hud = v.hud()
            assert hud["fingers"] == "glove" and hud["glove"]["side"] == "right"
        finally:
            v.close()
    s.close()


class _CoilGlove:
    """A right Smartglove on a Coil Pro, moved by the test: where the hand is and how far it is closed."""

    def __init__(self):
        self.aperture: float | None = None
        self.pos = np.array(DEFAULT_HOME["right"][0], float)

    def sample(self, side, t):
        if side != "right":
            return None
        joints = POSES.open if self.aperture is None else POSES.grip(self.aperture)
        return hand_to_glove_frame("right", joints, (self.pos.copy(), DEFAULT_HOME["right"][1]), t)


def test_one_glove_on_a_coil_places_the_hand_and_does_the_grasp():
    source = _CoilGlove()
    home = source.pos.copy()
    s = Session(1, seed=7, reachable_only=True)
    with Emulator(source, port=0) as emulator:
        glove = GloveFingers("127.0.0.1", emulator.port, solver="off")
        v = VisitorOperator(s, glove=glove)
        try:
            _until(lambda: glove.hand("right", time.monotonic()) is not None)

            def run(until, limit):
                for _ in range(limit):
                    s.tick(v.update(s.observe()))
                    time.sleep(0.004)                             # the glove sends in real time
                    if until():
                        return True
                return False

            assert run(lambda: v.anchor is not None, 1500)        # flat and still: that is the middle
            assert v.input == "coil" and v.own and v.hud()["placed"]
            scale = v.rt.scale

            def hand_to(xy, height, steps=120):
                """Move the person's hand so the robot's is asked to ``xy``, ``height`` over the hover height."""
                want = home + np.array([xy[0] - 0.28, xy[1], height]) / scale
                start = source.pos.copy()
                for i in range(steps):
                    source.pos = start + (want - start) * (i + 1) / steps
                    run(lambda: False, 1)

            placement = min(s.layouts[0].placements, key=lambda p: p.pos[0])
            ob = s.scene.by_name[placement.name]
            box = s.scene.bin_centers[s.layouts[0].mapping[ob.color]]
            part = lambda: s.observe().obj_pos[0, ob.index]
            hand_to(part()[:2], 0.0)
            run(lambda: np.linalg.norm(s.observe().pinch[0, v.side, :2] - part()[:2]) < 0.012, 500)
            run(lambda: False, 50)                                # the palm turns to the part
            hand_to(part()[:2], -0.2, 80)                         # down onto it
            run(lambda: s.observe().pinch[0, v.side, 2] < s.scene.rest_z + 0.012, 400)
            source.aperture = 0.02
            run(lambda: False, 60)
            hand_to(part()[:2], 0.03, 80)                         # the person lifts: nothing lifts for them
            assert run(lambda: part()[2] > s.scene.rest_z + 0.05, 300)
            hand_to(box[:2], 0.03)
            run(lambda: np.linalg.norm(s.observe().pinch[0, v.side, :2] - v.command) < 0.02, 500)
            run(lambda: False, 40)
            source.aperture = None
            assert run(lambda: s.progress()[0] >= 1, 400)
            assert "Coil Pro" in v.who() and "own grasp" in v.who()
        finally:
            v.close()
    s.close()


def _replay_real_hand(name: str, monkeypatch) -> np.ndarray:
    """A real person's hand (tests/fixtures/real_hand) down the camera path: [t, robot middle, robot index]
    per tick, the first joint of each finger as their hand asks it of the Dex3 (0 straight, -1 closed)."""
    import json
    from pathlib import Path

    rows = [json.loads(line) for line in (Path(__file__).parent / "fixtures" / "real_hand" / name).read_text().splitlines()]
    clock = [time.monotonic()]
    monkeypatch.setattr(visitor_module.time, "monotonic", lambda: clock[0])
    s = Session(2, seed=1)
    layout, followers = s.sample_scene(3)
    s.new_episode(layout, followers)
    v = VisitorOperator(s)
    v.begin()
    out, k, t = [], 0, 0.0
    try:
        while t < rows[-1]["t"]:
            while k + 1 < len(rows) and rows[k + 1]["t"] <= t:
                k += 1
            clock[0] += 1 / 50
            t += 1 / 50
            v.set_command({"x": 0.28, "y": 0.0, "grip": False, "input": "camera", "own": False,
                           "world": np.array(rows[k]["world"]).reshape(-1).tolist(),
                           "right": rows[k]["label"] != "Right", "depth": 0.0})
            s.tick(v.update(s.observe()))
            out.append([t, v.posture[3], v.posture[5]])        # what their fingers ask of the Dex3, left-hand order
    finally:
        s.close()
    return np.array(out)


def test_a_real_persons_fingers_move_the_robots_one_at_a_time(monkeypatch):
    """A fist that opens thumb first, then the index finger, then the middle finger, filmed from above:
    the robot's hand is closed for the fist, its index finger opens when the person's does while its
    middle finger stays closed, and with the hand open both are open."""
    a = _replay_real_hand("counting_left.jsonl", monkeypatch)
    at = lambda lo, hi: a[(a[:, 0] >= lo) & (a[:, 0] < hi), 1:].mean(axis=0)      # (middle, index)
    fist, index_out, both_out, open_hand = at(0.5, 2.0), at(2.6, 3.0), at(4.4, 5.6), at(7.0, 13.0)
    assert fist[0] < -0.9 and fist[1] < -0.9                  # closed
    assert index_out[1] > index_out[0] + 0.15                 # the index finger alone has opened
    assert both_out[0] > fist[0] + 0.3 and both_out[1] > fist[1] + 0.3
    assert open_hand[0] > -0.45 and open_hand[1] > -0.45      # open


def test_a_real_flat_hand_is_not_a_pinch(monkeypatch):
    """The other hand stays a fist for seven seconds and then opens. Seen from above, the camera reads
    its flat hand's thumb and index finger as 60 mm apart, less than its fist's: nothing may close."""
    a = _replay_real_hand("counting_right.jsonl", monkeypatch)
    fist = a[(a[:, 0] >= 0.5) & (a[:, 0] < 6.5), 1:].mean(axis=0)
    flat = a[(a[:, 0] >= 11.0) & (a[:, 0] < 13.0), 1:].mean(axis=0)
    assert fist[0] < -0.9 and fist[1] < -0.9
    assert flat[0] > -0.45 and flat[1] > -0.45
