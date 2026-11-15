"""A visitor drives the lead robot with a pointer: the lead sorts, the echoes redo it, one driver at a time."""
from __future__ import annotations

import time

import numpy as np

from kaiku.app.run import Runner
from kaiku.app.session import Session
from kaiku.app.visitor import VisitorOperator


def _ticks(session, operator, count):
    for _ in range(count):
        session.tick(operator.update(session.observe()))


def test_pointer_visitor_sorts_lead_without_clips():
    s = Session(1, seed=7, reachable_only=True)
    v = VisitorOperator(s)
    for placement in list(s.layouts[0].placements):
        v.set_command({"x": placement.pos[0], "y": placement.pos[1], "grip": True})
        _ticks(s, v, 450)
        ob = s.scene.by_name[placement.name]
        c = s.scene.bin_centers[s.layouts[0].mapping[ob.color]]
        v.set_command({"x": c[0], "y": c[1], "grip": True})
        _ticks(s, v, 180)
        v.set_command({"x": c[0], "y": c[1], "grip": False})
        _ticks(s, v, 300)
    assert v.done
    assert s.progress()[0] == len(s.layouts[0].placements)
    assert s.clipped[0] == 0
    # what the driver's page draws: every part with the bin it belongs in, and nothing left on the mat
    table = v.hud()["table"]
    assert len(table["parts"]) == len(s.layouts[0].placements) and set(table["bins"]) == set(s.scene.bin_centers)
    assert not any(part["on_mat"] for part in table["parts"])
    v.set_command({"x": -5.0, "y": 9.0, "grip": False})                 # a pointer far off the table stays on it
    assert np.all(v.command >= v.reach[0]) and np.all(v.command <= v.reach[1])
    s.close()


def _drive(s: Session, v: VisitorOperator) -> None:
    """Sort the lead's scene the way a visitor does: press on a part, drag it to its bin, let go."""
    lay = s.layouts[0]

    def run(until, limit, where, grip):
        for _ in range(limit):
            x, y = where()
            v.set_command({"x": float(x), "y": float(y), "grip": grip})
            s.tick(v.update(s.observe()))
            if until():
                return

    for k in sorted(range(len(lay.placements)), key=lambda k: lay.placements[k].pos[0]):
        ob = s.scene.by_name[lay.placements[k].name]
        box = s.scene.bin_centers[lay.mapping[ob.color]][:2]
        part = lambda: s.observe().obj_pos[0, ob.index, :2]
        run(lambda: False, 25, part, False)                               # the pointer comes to the part
        run(lambda: v.phase == "carry", 500, part, True)                  # pressed: the hand goes down and lifts
        start = part().copy()
        for i in range(30):                                               # dragged to the bin in a second
            q = start + (box - start) * (i + 1) / 30
            v.set_command({"x": float(q[0]), "y": float(q[1]), "grip": True})
            s.tick(v.update(s.observe()))
        run(lambda: False, 30, lambda: box, True)
        run(lambda: v.phase == "idle", 400, lambda: box, False)           # let go: into the bin


def test_the_echoes_redo_what_a_visitor_did():
    s = Session(16, seed=1)
    v = VisitorOperator(s)
    _drive(s, v)
    parts = len(s.layouts[0].placements)
    assert v.done and s.progress()[0] == parts
    s.finish_lead()
    while not s.settled and s.tick_no < 9000:
        s.tick(s.home_command())
    r = s.result()
    assert r.clipped.sum() == 0
    assert r.success[0]
    assert r.success[1:].sum() >= 12, f"only {int(r.success[1:].sum())} of 15 echoes sorted their own table"
    s.close()


def test_a_glove_on_this_machine_drives_from_the_first_scene(tmp_path):
    """With one glove plugged in, nothing the stand-in did is recorded ahead of the person's scenes."""
    r = Runner(n=2, port=0, seed=7, threads=1, mode="visitor", glove_fingers="127.0.0.1:1", record=tmp_path)
    try:
        assert r.phase == "hold"
        t0 = time.monotonic()
        while r.phase != "run" and time.monotonic() - t0 < 30.0:
            r._step()
        assert r.mode == "visitor" and r.phase == "run"
        assert not (tmp_path / "episodes.jsonl").exists()
    finally:
        r.close()


def test_one_visitor_drives_at_a_time():
    r = Runner(n=2, port=0, seed=7, threads=1)
    try:
        say = lambda client, name, value=None: r.commands.put({"type": "cmd", "name": name, "value": value, "client": client})

        def step(until, limit_s=60.0):
            t0 = time.monotonic()
            while not until() and time.monotonic() - t0 < limit_s:
                r._commands()
                r._step()
            assert until()

        say(5, "drive_start")
        r._commands()
        assert r.driver_client == 5 and r.want_mode == "visitor"
        say(9, "drive_start")                                             # a second visitor waits
        say(9, "mode", "policy")                                          # and their buttons do nothing
        say(9, "pause", True)
        r._commands()
        assert r.driver_client == 5 and r.want_mode == "visitor" and not r.paused
        step(lambda: r.mode == "visitor" and r.phase == "run")
        before = r.visitor.command.copy()
        say(9, "hand", {"x": 0.30, "y": 0.10, "grip": True})              # not the driver's hand
        r._commands()
        assert np.array_equal(r.visitor.command, before) and not r.visitor.grip
        say(5, "hand", {"x": 0.30, "y": 0.10, "grip": False})
        r._commands()
        assert np.allclose(r.visitor.command, [0.30, 0.10])
        r.commands.put({"type": "disconnect", "client": 9})               # somebody else leaves: nothing changes
        r._commands()
        assert r.driver_client == 5
        r.commands.put({"type": "disconnect", "client": 5})               # the driver leaves: the stand-in is back
        r._commands()
        assert r.driver_client is None and r.want_mode == "gloves"
        step(lambda: r.mode == "gloves" and r.phase == "run")
        say(9, "pause", True)                                             # and buttons count again
        r._commands()
        assert r.paused
    finally:
        r.close()


def test_the_page_is_told_where_a_part_that_was_let_go_came_to_rest():
    """Let go over the table, a part is on the table, and the page is not told it went into a bin."""
    s = Session(1, seed=7, reachable_only=True)
    v = VisitorOperator(s)
    try:
        lay = s.layouts[0]
        placement = lay.placements[0]
        ob = s.scene.by_name[placement.name]
        own = lay.mapping[ob.color]
        at = np.array(placement.pos[:2], float)
        say = lambda xy, grip: v.set_command({"x": float(xy[0]), "y": float(xy[1]), "grip": grip})

        def run(until, limit=600):
            for _ in range(limit):
                s.tick(v.update(s.observe()))
                if until():
                    return
            assert until()

        say(at, True)
        run(lambda: v.phase == "carry")
        assert v.hud()["landed"] is None
        say(at, False)                                                    # let go where it was picked: not over a bin
        run(lambda: v.phase == "release", 5)
        assert v.hud()["into"] is None                                    # the page knows it is not going into a bin
        run(lambda: v.phase == "idle" and v.let_go is None)
        landed = v.hud()["landed"]
        assert landed == {"where": "table", "color": ob.color, "bin": own, "reach": True}
        assert s.progress()[0] == 0

        say(at, True)                                                     # again, and this time into its bin
        run(lambda: v.phase == "carry")
        assert v.hud()["landed"] is None                                  # a part in the hand: the last one is old news
        box = s.scene.bin_centers[own][:2]
        say(box, True)
        _ticks(s, v, 180)
        say(box, False)
        run(lambda: v.phase == "release", 5)
        assert v.hud()["into"] == own
        run(lambda: v.phase == "idle" and v.let_go is None)
        assert v.hud()["landed"]["where"] == "in" and s.progress()[0] == 1
        v.landed["at"] -= 60.0                                            # a while later it is no longer news
        assert v.hud()["landed"] is None

        # every place a part can lie has its own word
        lo, hi = s.scene.bin_inner(own)
        inside = (lo + hi) / 2
        other = next(b for b in s.scene.bin_centers if b != own)
        olo, ohi = s.scene.bin_inner(other)
        rim = np.r_[lo[0] - s.scene.bin_wall - 0.03, inside[1], s.scene.rest_z]      # on the table, against its wall
        assert v.where(ob.index, inside)["where"] == "in"
        assert v.where(ob.index, (olo + ohi) / 2) == {"where": "wrong", "color": ob.color, "bin": own, "reach": False}
        assert v.where(ob.index, rim)["where"] == "beside"
        assert v.where(ob.index, np.r_[at, s.scene.rest_z])["where"] == "table"
        assert v.where(ob.index, np.r_[at, s.scene.counter_z - 0.5])["where"] == "off"
    finally:
        s.close()


def test_an_empty_hand_is_not_carried_to_a_bin_and_a_new_turn_starts_with_nobodys_choices():
    s = Session(1, seed=7, reachable_only=True)
    v = VisitorOperator(s)
    placement = s.layouts[0].placements[0]
    ob = s.scene.by_name[placement.name]
    v.set_command({"x": placement.pos[0], "y": placement.pos[1], "grip": True})
    for _ in range(600):
        s.tick(v.update(s.observe()))
        if v.phase == "carry":
            break
    assert v.phase == "carry" and v.hud()["holding"]
    # the part is taken out of the hand (it slipped, or the grasp never had it)
    address = s.twin.obj_qadr[ob.index]
    s.sim.data[0].qpos[address:address + 3] = [placement.pos[0], placement.pos[1], s.scene.rest_z]
    s.sim.data[0].qvel[:] = 0.0
    for _ in range(40):
        s.tick(v.update(s.observe()))
    told = v.hud()
    assert v.phase == "idle" and v.held == -1                      # nothing is carried on to a bin
    assert told["dropped"] and not told["holding"] and told["over"] is None
    _ticks(s, v, 5)
    assert v.phase == "idle"                                       # the held button does not grasp again by itself
    v.set_command({"x": placement.pos[0], "y": placement.pos[1], "grip": False})
    _ticks(s, v, 2)
    assert not v.again                                             # a new press does

    v.set_command({"x": 0.3, "y": 0.0, "grip": True, "own": True})
    assert v.own and v.asked
    v.turn()                                                       # another visitor: not the last one's grasp
    assert not v.own and not v.asked and not v.grip and v.world is None
    s.close()
