"""What the runner tells the page: who is driving, which scene this is, how long training takes."""
import json
import time

from kaiku.app import run as run_module
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
