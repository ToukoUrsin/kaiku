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
