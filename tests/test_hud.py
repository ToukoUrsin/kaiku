"""What the runner tells the page: who is driving, which scene this is, how long training takes."""
import json
import time

from kaiku.app import run as run_module
from kaiku.app import trainer as trainer_module
from kaiku.app.run import BASELINES, Runner, baselines
from kaiku.app.trainer import Trainer


def _step(r, until, limit_s=60.0):
    t0 = time.monotonic()
    while not until() and time.monotonic() - t0 < limit_s:
        r._commands()
        r._step()
    assert until()


def test_the_page_is_told_what_drives_and_a_pointer_is_never_a_glove():
    r = Runner(n=2, port=0, seed=7, threads=1)
    try:
        say = lambda client, name, value=None: r.commands.put({"type": "cmd", "name": name, "value": value, "client": client})
        operator = r.hud()["operator"]
        assert operator["drive"] == "stand-in" and "driver" not in operator      # the scripted hand in emulated gloves

        say(5, "drive_start")
        r._commands()
        operator = r.hud()["operator"]                                         # asked for; the tables are still being laid
        assert operator["driver"] == 5 and operator["starting"] and operator["drive"] == "stand-in"

        _step(r, lambda: r.mode == "visitor" and r.phase == "run")
        say(5, "hand", {"x": 0.30, "y": 0.10, "grip": False})
        r._commands()
        operator = r.hud()["operator"]
        assert operator["drive"] == "pointer" and not operator["starting"]
        for claim in ("left", "right", "calibrated"):                          # no glove rows, no calibration
            assert claim not in operator
        assert "pointer" in r._who() and "glove" not in r._who().lower()

        say(5, "hand", {"x": 0.30, "y": 0.10, "grip": False, "input": "camera"})
        r._commands()
        assert r.hud()["operator"]["drive"] == "camera"
        r.visitor.fingers, r.visitor.input = "glove", "pointer"                # one Smartglove's fingers, the pointer's place
        assert r._drive(r.visitor.hud()) == "glove-pointer"
        r.visitor.input = "coil"
        assert r._drive(r.visitor.hud()) == "glove-coil"

        say(5, "drive_end")
        r._commands()
        _step(r, lambda: r.mode == "gloves" and r.phase == "run")
        assert r.hud()["operator"]["drive"] == "stand-in"
    finally:
        r.close()


def test_autopilot_says_autopilot():
    r = Runner(n=2, port=0, seed=7, threads=1, mode="autopilot")
    try:
        operator = r.hud()["operator"]
        assert operator["drive"] == "autopilot" and not operator["left"]["connected"]
    finally:
        r.close()


def test_a_scene_keeps_its_number_until_the_next_starts_and_a_cut_one_is_not_called_sorted():
    r = Runner(n=2, port=0, seed=7, threads=1, mode="autopilot", hold_s=0.0)
    try:
        r.hold_until = float("inf")                            # stay in the pause between scenes to look at it
        assert r.hud()["episode"] == {**r.hud()["episode"], "index": 1, "end": None}
        for _ in range(4000):
            r._step()
            if r.tot["episodes"] == 1:
                break
        assert r.tot["episodes"] == 1 and r.phase == "hold"
        r.hold_until = float("inf")
        episode = r.hud()["episode"]                           # scene 1 has just ended: still scene 1, not 2
        assert episode["index"] == 1 and episode["end"] == "sorted" and r.hud()["status"] == "settling"

        r.hold_until = 0.0
        _step(r, lambda: r.phase == "run")
        episode = r.hud()["episode"]
        assert episode["index"] == 2 and episode["end"] is None
        r.commands.put({"type": "cmd", "name": "next_scene", "client": 1})    # cut short: nobody's result
        r._commands()
        r.hold_until = float("inf")
        episode = r.hud()["episode"]
        assert episode["index"] == 2 and episode["end"] == "cut" and r.tot["episodes"] == 1
        r.hold_until = 0.0
        _step(r, lambda: r.phase == "run")
        assert r.hud()["episode"] == {**r.hud()["episode"], "index": 2, "end": None}   # its number again, as a new scene
    finally:
        r.close()


def test_the_comparison_with_a_replay_is_served_from_the_file_that_measured_it(tmp_path):
    measured = json.loads(BASELINES.read_text())
    served = baselines()
    assert set(served["arms"]) == {"plain", "mimicgen", "kaiku"}
    for name, arm in served["arms"].items():
        source = measured["policy"]["arms"][name]
        assert arm["clean"] == source["clean"] and arm["scenes"] == source["scenes"]
        assert abs(arm["yield"] - source["clean"] / source["scenes"]) < 1e-3
        assert arm["low"] <= arm["yield"] <= arm["high"]
    assert served["tables"] == measured["policy"]["heldout_scenes"] and served["runs"] == measured["policy"]["seeds_per_arm"]
    assert served["made"]["mimicgen"]["clean"] == measured["generation"]["mimicgen"]["clean"]
    assert baselines(tmp_path / "missing.json") is None        # no file: the page shows nothing, not a guess

    r = Runner(n=2, port=0, seed=7, threads=1, mode="autopilot")
    try:
        assert r.hud()["policy"]["baselines"] == served
    finally:
        r.close()


def test_buttons_are_said_to_be_held_while_a_comparison_runs():
    r = Runner(n=2, port=0, seed=7, threads=1, mode="autopilot", hold_s=0.0)
    try:
        assert not r.hud()["copy"]["busy"]
        r.copy_at = time.monotonic() - run_module.COPY_EVERY_S - 1.0
        r.commands.put({"type": "cmd", "name": "copy_scene", "client": 1})
        r._commands()
        assert r.hud()["copy"]["busy"]
    finally:
        r.close()


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _run(trainer, clock, paces, about=None):
    """A training run whose pace changes from stage to stage: ``paces`` is (share of the bar, seconds it takes).
    Returns what was said to be left, every ten seconds."""
    trainer.about_min = about
    trainer.jobs = {w: {"stage": "training", "progress": 0.0} for w in ("kaiku", "plain")}
    trainer.error, trainer.finished = None, False
    trainer.started, trainer.total_s = clock(), None
    said, done = [], 0.0
    for share, seconds in paces:
        for i in range(int(seconds // 10)):
            clock.t += 10.0
            now = done + share * (i + 1) / (seconds // 10)
            for job in trainer.jobs.values():
                job["progress"] = now
            said.append((clock.t - trainer.started, now, trainer.status()["eta_s"]))
        done += share
    return said


def test_the_time_left_in_training_does_not_grow_while_the_bar_moves(tmp_path):
    """The hosted run: files read fast, training slower, testing slowest, about 25 minutes in all. The old
    estimate went from the pace of the last stretch and grew from 16 to 24 minutes between 19% and 36%."""
    paces = [(0.10, 60.0), (0.70, 1020.0), (0.20, 420.0)]      # 25 minutes
    total = sum(seconds for _, seconds in paces)
    for about in (25.0, None, 8.0):                            # measured before, never measured, measured badly
        clock = _Clock()
        trainer = Trainer(tmp_path, tmp_path / "policies")
        trainer.clock = clock
        said = [(t, p, eta) for t, p, eta in _run(trainer, clock, paces, about) if eta is not None]
        assert said, "nothing was ever said"
        if about:
            assert said[0][0] <= 10.0                          # a machine that has been timed says so from the start
        worst = max(b[2] - a[2] for a, b in zip(said, said[1:]))
        assert worst <= 45.0, f"the time left grew by {worst:.0f} s in ten seconds (about={about})"
        for t, p, eta in said:
            if p >= 0.5:                                       # half way: within a few minutes of the truth
                assert abs(eta - (total - t)) <= 300.0, (about, t, p, eta)
        assert said[-1][2] <= 60.0
        if about == 25.0:                                      # a good measurement stays good all the way
            assert all(abs(eta - (total - t)) <= 150.0 for t, p, eta in said)


def test_a_finished_run_is_what_the_next_one_is_said_to_take(tmp_path):
    out = tmp_path / "policies"
    out.mkdir()
    clock = _Clock()
    trainer = Trainer(tmp_path, out)
    assert trainer.about_min is None                           # never timed here: the page promises nothing
    trainer.clock = clock
    trainer.started = clock()
    trainer._measured(1260.0)
    assert trainer.about_min == 21.0
    assert Trainer(tmp_path, out).about_min == 21.0            # and after a restart
    assert Trainer(tmp_path, out, about_min=25.0).about_min == 25.0   # what somebody measured and handed in counts first


def test_rented_gpus_and_this_machine_are_timed_apart(tmp_path):
    """A run on rented GPUs takes minutes, the same run here six times that: neither may set the other's promise."""
    out = tmp_path / "policies"
    out.mkdir()
    clock = _Clock()
    trainer = Trainer(tmp_path, out, "modal", about_min=25.0)
    trainer.clock = clock
    assert trainer.rented and trainer.about == trainer_module.MODAL_ABOUT_MIN
    trainer.using, trainer.started = "modal", clock()
    trainer._measured(300.0)                                   # the rented run took five minutes
    assert trainer.about == 5.0 and trainer.about_min == 25.0
    trainer._measured(2.0)                                     # a run that did nothing is not a measurement
    assert trainer.about == 5.0

    # Rented, the time left starts from the rented minutes; done again here, from this machine's.
    for using, total in (("modal", 300.0), ("local", 1500.0)):
        trainer.using = using
        said = _run(trainer, clock, [(1.0, total)], about=25.0)
        assert abs(said[0][2] - (total - 10.0)) <= 30.0, (using, said[0])
        assert max(b[2] - a[2] for a, b in zip(said, said[1:])) <= 45.0


def test_the_next_visitor_waits_in_line_and_each_page_hears_only_about_its_own_turn():
    r = Runner(n=2, port=0, seed=7, threads=1)
    said = []
    r.viewer.send_event = said.append
    try:
        say = lambda client, name, value=None: r.commands.put({"type": "cmd", "name": name, "value": value, "client": client})
        say(5, "drive_start")
        say(6, "drive_start")                                  # somebody already drives: 6 is next
        say(6, "drive_start")                                  # asking twice is still once
        r._commands()
        operator = r.hud()["operator"]
        assert operator["driver"] == 5 and operator["queue"] == [6] and operator["turn_left_s"] > 0
        assert [e["to"] for e in said] == [5] and all(e["key"] == "turn" for e in said)   # nobody else is told "your turn"

        say(5, "drive_end")                                    # handed back: 6 has the robots, and one word each
        r._commands()
        operator = r.hud()["operator"]
        assert operator["driver"] == 6 and operator["queue"] == []
        assert [(e["to"], "ended" in e["text"]) for e in said[1:]] == [(5, True), (6, False)]

        say(7, "drive_start")
        r.commands.put({"type": "disconnect", "client": 7})   # a page that closes leaves the line
        r._commands()
        assert r.hud()["operator"]["queue"] == []
    finally:
        r.close()


def test_a_visitor_is_told_before_and_after_the_scene_is_replaced(monkeypatch):
    monkeypatch.setattr(run_module, "VISITOR_SCENE_S", 1.0)   # a short scene, so the test does not wait 150 s
    r = Runner(n=2, port=0, seed=7, threads=1, hold_s=0.0)
    try:
        r.commands.put({"type": "cmd", "name": "drive_start", "client": 5})
        r._commands()
        _step(r, lambda: r.mode == "visitor" and r.phase == "run")
        operator = r.hud()["operator"]
        assert 0 < operator["scene_left_s"] and not operator["expired"] and not operator["dealt"]
        r.last_visitor_command = time.monotonic() + 1e6       # the visitor is still there
        _step(r, lambda: r._visitor_expired)
        assert r.hud()["operator"]["expired"]
        _step(r, lambda: r.phase == "run" and not r._visitor_expired)
        operator = r.hud()["operator"]
        assert operator["dealt"] and not operator["expired"] and operator["driver"] == 5
    finally:
        r.close()


def test_a_pinch_over_nothing_is_said_and_nothing_moves_for_it():
    from kaiku.app.session import Session
    from kaiku.app.visitor import VisitorOperator
    s = Session(1, seed=7, reachable_only=True)
    v = VisitorOperator(s)
    try:
        lo, _ = v.reach
        v.set_command({"x": float(lo[0]), "y": float(lo[1]), "grip": True})    # a corner with no part in it
        for _ in range(10):
            s.tick(v.update(s.observe()))
        hud = v.hud()
        assert hud["missed"] and not hud["dropped"] and v.phase == "idle"
        v.set_command({"x": float(lo[0]), "y": float(lo[1]), "grip": False})
        v.missed_at -= 10.0                                    # a while later it is no longer news
        s.tick(v.update(s.observe()))
        assert not v.hud()["missed"]
    finally:
        s.close()


def test_the_page_carries_no_training_time_of_its_own():
    """The runner says what a run takes (policy.about_min, policy.eta_s). A number typed into the page could disagree."""
    import re
    static = run_module.HERE / "web" / "static" / "js"
    for name in ("hud.js", "sample.js"):
        text = (static / name).read_text()
        assert not re.search(r"about_min\s*(\?\?|:|\|\|)\s*\d", text), name
        assert not re.search(r"eta_s\s*(\?\?|:|\|\|)\s*\d", text), name
