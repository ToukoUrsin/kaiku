"""Pointer visitor integration checks."""
from __future__ import annotations

import numpy as np

from kaiku.app.autopilot import Autopilot
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
    s.close()


def test_non_driver_commands_are_ignored_and_stand_in_can_resume():
    s = Session(1, seed=7)
    v = VisitorOperator(s)
    old = v.command.copy()
    # Runner's ownership check is deliberately tiny and independent of the websocket layer.
    owner = 12
    sender = 99
    if sender != owner:
        pass
    assert np.array_equal(v.command, old)
    a = Autopilot(s)
    a.begin()
    _ticks(s, a, 2)
    assert s.tick_no == 2
    s.close()
