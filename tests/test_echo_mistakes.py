"""A person makes mistakes, and the echoes must still sort their own tables.

Each run here is a visitor's scene with one mistake in it: a part put down on the mat and taken
again, a press that takes nothing, a pinch let go before the lift, a part let go against a bin's
rim, parts taken in another order, a part put aside while another is sorted first.
"""
from __future__ import annotations

import numpy as np
import pytest

from kaiku.app.session import Session
from kaiku.app.visitor import VisitorOperator

MISTAKES = ("clean", "table", "edge", "miss", "cancel", "rim", "order", "aside")
AHEAD = 0.34            # m from the robot: as far towards the bins as its hand still takes a part from the table


class Hand:
    """A scripted person at the pointer: goes to a part, presses, drags, lets go."""

    def __init__(self, s: Session, v: VisitorOperator):
        self.s, self.v = s, v
        self.lay = s.layouts[0]
        self.at = np.array(v.command, float)

    def run(self, where, grip: bool, limit: int, until=lambda: False) -> bool:
        for _ in range(limit):
            xy = where() if callable(where) else where
            self.at = np.array(xy, float)
            self.v.set_command({"x": float(xy[0]), "y": float(xy[1]), "grip": grip})
            self.s.tick(self.v.update(self.s.observe()))
            if until():
                return True
        return False

    def drag(self, to, grip: bool, ticks: int = 30) -> None:
        start = self.at.copy()
        for i in range(ticks):
            self.run(start + (np.asarray(to, float)[:2] - start) * (i + 1) / ticks, grip, 1)

    def part(self, k: int):
        ob = self.s.scene.by_name[self.lay.placements[k].name]
        return ob, self.s.scene.bin_centers[self.lay.mapping[ob.color]][:2]

    def where(self, k: int) -> np.ndarray:
        return self.s.observe().obj_pos[0, self.part(k)[0].index, :2].copy()

    def pick(self, k: int, presses: int = 3) -> bool:
        """The pointer comes to the part; pressed, the hand goes down and lifts it. A press that took
        nothing is let go and made again, as a person does."""
        self.drag(self.where(k), False, 20)
        for _ in range(presses):
            self.run(lambda: self.where(k), False, 25)
            self.run(lambda: self.where(k), True, 500, lambda: self.v.phase == "carry")
            self.run(self.at, True, 20)                      # is the part in the hand?
            if self.v.phase == "carry":
                return True
        return False

    def put(self, to, settle: int = 30) -> None:
        """Dragged there in a second, held a moment, let go."""
        self.drag(to, True)
        self.run(to, True, settle)
        self.run(to, False, 400, lambda: self.v.phase == "idle")
        self.run(to, False, 20)

    def sort(self, k: int) -> None:
        if self.pick(k):
            self.put(self.part(k)[1])

    def free(self, k: int, rest: list[int]) -> np.ndarray:
        """A free place on the mat for part ``k``: clear of the parts still on it, towards its own bin."""
        others = [self.where(j) for j in rest]
        box = self.part(k)[1]
        spots = [np.array([x, y]) for x in (0.28, 0.30) for y in np.linspace(-0.17, 0.17, 35)]
        room = lambda q: min([0.10] + [float(np.linalg.norm(q - o)) for o in others])
        return max(spots, key=lambda q: room(q) - 0.1 * abs(q[1] - box[1]))

    def lies(self, k: int) -> dict:
        ob = self.part(k)[0]
        return self.v.where(ob.index, self.s.observe().obj_pos[0, ob.index])


def drive(s: Session, v: VisitorOperator, mistake: str) -> None:
    """Sort the lead's scene with one ``mistake`` in it, made on the first part taken."""
    h = Hand(s, v)
    sc, lay = s.scene, s.layouts[0]
    order = sorted(range(len(lay.placements)), key=lambda k: lay.placements[k].pos[0])
    if mistake == "order":
        order = order[::-1]
    first, rest = order[0], order[1:]
    box = h.part(first)[1]
    if mistake == "table":
        # put down on a free place on the mat, on the way to its bin, and taken again from there
        h.pick(first)
        h.put(h.free(first, rest))
    elif mistake == "edge":
        # put down in front of its bin, as far out as the hand still takes a part again
        h.pick(first)
        h.put(np.array([AHEAD, float(np.clip(box[1], -0.19, 0.19))]))
    elif mistake == "miss":
        # a press beside the part: nothing under the hand
        beside = h.where(first) + [-0.03, 0.0]
        others = [h.where(k) for k in rest]
        for dy in (0.07, -0.07):
            spot = h.where(first) + [-0.03, dy]
            if min(np.linalg.norm(spot - o) for o in others) > 0.06 and abs(spot[1]) < 0.22:
                beside = spot
        h.drag(beside, False, 20)
        h.run(beside, True, 30)
        h.run(beside, False, 10)
    elif mistake == "cancel":
        # pressed on the part, and let go when the fingers have closed, before the lift
        h.run(lambda: h.where(first), False, 25)
        h.run(lambda: h.where(first), True, 500, lambda: v.job is not None and v.job.label == "lift")
        h.run(lambda: h.where(first), False, 45)
    elif mistake == "rim":
        # let go against the front rim of its bin: the part ends in front of it, or half on it
        h.pick(first)
        lo, _ = sc.bin_inner(lay.mapping[h.part(first)[0].color])
        h.put(np.array([lo[0] - 0.012, float(np.clip(box[1], -0.19, 0.19))]))
        h.run(h.at, False, 45)
    elif mistake == "aside":
        # put down on the mat, and another part is sorted first
        h.pick(first)
        h.put(h.free(first, rest))
        if rest:
            h.sort(rest[0])
            rest = rest[1:]
    if mistake in ("table", "edge", "rim", "aside") and not h.lies(first)["reach"]:
        first = None                                   # it went into a bin after all, or out of reach
    for k in ([] if first is None else [first]) + rest:
        h.sort(k)


def play(seed: int, mistake: str, n: int = 16):
    """One visitor's scene with ``mistake`` in it. Returns the session's result and whether the lead sorted."""
    s = Session(n, seed=seed)
    try:
        v = VisitorOperator(s)
        drive(s, v, mistake)
        led = bool(v.done)
        s.finish_lead()
        while not s.settled and s.tick_no < 12000:
            s.tick(s.home_command())
        return s.result(), led
    finally:
        s.close()


@pytest.mark.parametrize("mistake,seed", [("table", 1), ("cancel", 2), ("rim", 3), ("aside", 4), ("order", 2),
                                          ("miss", 3)])
def test_one_mistake_does_not_cost_the_echoes(mistake, seed):
    r, led = play(seed, mistake)
    assert led and r.success[0], "the scripted person did not sort the lead's own table"
    assert r.clipped.sum() == 0
    assert r.success[1:].sum() >= 12, f"{mistake}: only {int(r.success[1:].sum())} of 15 echoes sorted their own table"
