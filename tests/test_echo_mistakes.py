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


def drive(s: Session, v: VisitorOperator, mistake: str, then=None) -> None:
    """Sort the lead's scene with one ``mistake`` in it, made on the first part taken. ``then(hand, part)``
    is called after a part was put down by mistake, before it is taken again."""
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
    if then is not None:
        then(h, first)
    if mistake in ("table", "edge", "rim", "aside") and not h.lies(first)["reach"]:
        first = None                                   # it went into a bin after all, or out of reach
    for k in ([] if first is None else [first]) + rest:
        h.sort(k)


def play(seed: int, mistake: str, n: int = 16, then=None):
    """One visitor's scene with ``mistake`` in it. Returns the session's result, whether the lead sorted,
    how many parts each robot sorted, and the echo's own account of the grasps it left out."""
    s = Session(n, seed=seed)
    try:
        v = VisitorOperator(s)
        drive(s, v, mistake, then)
        led = bool(v.done)
        s.finish_lead()
        while not s.settled and s.tick_no < 12000:
            s.tick(s.home_command())
        return s.result(), led, s.progress(), (s.echo.left_out.copy(), s.echo.out_of_reach.copy())
    finally:
        s.close()


# Echoes of 15 that sorted their own table before each grasp was its own piece: 3, 10, 9, 12, 15, 15.
@pytest.mark.parametrize("mistake,seed,least", [("edge", 2, 13), ("table", 1, 12), ("aside", 1, 13), ("cancel", 2, 14),
                                                ("order", 2, 14), ("miss", 2, 14)])
def test_one_mistake_does_not_cost_the_echoes(mistake, seed, least):
    r, led, _, _ = play(seed, mistake)
    assert led and r.success[0], "the scripted person did not sort the lead's own table"
    assert r.clipped.sum() == 0
    assert r.success[1:].sum() >= least, f"{mistake}: only {int(r.success[1:].sum())} of 15 echoes sorted their own table"


def test_a_part_the_lead_cannot_take_again_costs_the_echoes_only_that_part():
    """Let go against the rim, the lead's part ends where its hand cannot take it again. The lead sorts the
    rest, and so do the echoes (1 of 15 did before)."""
    r, led, got, _ = play(2, "rim")
    assert not led and r.clipped.sum() == 0
    assert got[0] == len(r.layouts[0].placements) - 1
    assert (got[1:] >= got[0]).sum() >= 12, f"only {int((got[1:] >= got[0]).sum())} of 15 echoes sorted what the lead sorted"


def test_an_echo_with_no_partner_for_a_grasp_leaves_it_out_and_fails_alone():
    """The lead takes a part again that it had put down. One echo's own part already lies in its bin: it has
    nothing to redo and stays successful. Another's lies where its arm cannot take it: it says so, sorts
    the rest of its table, and nobody else is spoiled."""
    seen = {}

    def then(h: Hand, k: int) -> None:
        s, sc = h.s, h.s.scene
        h.run(h.at, False, 90)                                   # the echoes have put theirs down too
        seen["k"] = k
        for env, where in ((1, "bin"), (2, "far")):
            ob = sc.by_name[s.layouts[env].placements[k].name]
            lo, hi = sc.bin_inner(s.layouts[env].mapping[ob.color])
            at = np.r_[(lo[:2] + hi[:2]) / 2, lo[2] + sc.height / 2] if where == "bin" else \
                np.r_[0.37, s.observe().obj_pos[env, ob.index, 1], sc.rest_z]
            address = s.twin.obj_qadr[ob.index]
            s.sim.data[env].qpos[address:address + 3] = at
            s.sim.data[env].qvel[:] = 0.0
        s._obs = None

    r, led, got, (left_out, out_of_reach) = play(3, "table", then=then)
    k, parts = seen["k"], len(r.layouts[0].placements)
    assert led and r.success[0] and r.clipped.sum() == 0
    assert r.success[1] and left_out[0, k] and not out_of_reach[0, k]          # already in its bin: nothing to redo
    assert not r.success[2] and out_of_reach[1, k] and got[2] == parts - 1     # says so, and fails alone
    assert left_out[2:].sum() == 0 and r.success[3:].sum() >= 12
