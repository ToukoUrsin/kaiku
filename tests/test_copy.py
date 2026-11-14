"""The comparison: one scene where every robot is sent what the lead is sent, with no echo."""
import numpy as np

from kaiku.app.autopilot import Autopilot
from kaiku.app.run import Runner
from kaiku.app.session import Session


def _scene(s: Session, sampler_scene, copy: bool):
    lead, followers = sampler_scene
    s.new_episode(lead, followers, copy=copy)
    op = Autopilot(s)
    op.begin()
    while not s.settled and s.tick_no < 4000:
        if op.done:
            s.finish_lead()
        s.tick(op.update(s.observe()))
    return s.result()


def test_a_copy_of_the_leads_joints_sorts_few_tables_and_the_echo_most():
    s = Session(16, seed=3)
    try:
        scene = s.sample_scene(3)                              # the same tables twice
        copied = _scene(s, scene, copy=True)
        echoed = _scene(s, scene, copy=False)
    finally:
        s.close()
    assert copied.success[0] and echoed.success[0]             # the operator's own robot sorts its table both times
    assert copied.success[1:].sum() <= 7                       # the others reach for parts that lie elsewhere
    assert echoed.success[1:].sum() >= 13
    assert copied.clipped.sum() == 0 and echoed.clipped.sum() == 0


def test_the_runner_shows_one_scene_without_the_echo_and_counts_none_of_it(tmp_path):
    r = Runner(n=4, port=0, seed=7, threads=1, mode="autopilot", record=tmp_path, hold_s=0.0)
    try:
        said = []
        r.toast = lambda text, level="info": said.append(text)
        for _ in range(4000):                                  # one ordinary scene first
            r._step()
            if r.tot["episodes"] == 1:
                break
        assert r.tot["episodes"] == 1
        saved = r.rec.demos
        r.commands.put({"type": "cmd", "name": "copy_scene", "client": 1})
        r._commands()
        assert r.copy_next and r.hud()["copy"]["next"]
        r.next.result()                                        # its tables are drawn in another process
        for _ in range(6000):
            r._step()
            if r.compare is not None:
                break
        assert r.compare is not None and r.compare["robots"] == 3
        assert r.s.copy and r.hud()["copy"]["on"]              # still the scene everybody is looking at
        assert r.tot["episodes"] == 1 and r.rec.demos == saved # not a demonstration, not in the totals
        assert any("Without Kaiku" in text for text in said)
        tables = [[p.pos.copy() for p in lay.placements] for lay in r.s.layouts]
        r.commands.put({"type": "cmd", "name": "copy_scene", "client": 2})
        r._commands()
        assert not r.copy_next                                 # not twice in a row
        r.hold_until = 0.0
        for _ in range(6000):                                  # then the same tables once more, echoed and counted
            r._step()
            if r.tot["episodes"] == 2:
                break
        assert r.tot["episodes"] == 2 and r.compare["echo"] is not None
        assert not r.s.copy and not r.hud()["copy"]["on"]
        again = [[p.pos.copy() for p in lay.placements] for lay in r.s.layouts]
        assert all(np.allclose(a, b) for x, y in zip(tables, again) for a, b in zip(x, y))
        assert any("Same tables with Kaiku" in text for text in said)
    finally:
        r.close()


def test_a_session_of_ones_own_starts_over_and_a_shared_one_does_not():
    r = Runner(n=2, port=0, seed=7, threads=1, mode="autopilot", hold_s=0.0)
    try:
        first = [p.pos.copy() for p in r.next.result()[0].placements]
        for _ in range(4000):
            r._step()
            if r.tot["episodes"] == 1:
                break
        assert r.tot["episodes"] == 1
        r.commands.put({"type": "cmd", "name": "reset", "client": 1})
        r._commands()
        assert r.tot["episodes"] == 0 and not r.history
        again = [p.pos.copy() for p in r.next.result()[0].placements]
        assert len(first) == len(again) and all(np.allclose(a, b) for a, b in zip(first, again))
        r.shared = True
        r.tot["episodes"] = 3
        r.commands.put({"type": "cmd", "name": "reset", "client": 1})
        r._commands()
        assert r.tot["episodes"] == 3
    finally:
        r.close()
