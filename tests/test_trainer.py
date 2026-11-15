"""The Train button's two places to train: rented GPUs, and this machine when those fail or run out."""
import json
import sys
import time

from kaiku.app import trainer as T

DONE = 'import json\nfor w in ("kaiku", "plain"):\n    print("KAIKU " + json.dumps({"which": w, "stage": "done", "progress": 1.0, "success": 0.5}))'
FAIL = 'import json\nprint("KAIKU " + json.dumps({"stage": "error", "text": "no credit"}))\nraise SystemExit(1)'


class Fake(T.Trainer):
    def __init__(self, *a, rented_works=True, **kw):
        super().__init__(*a, **kw)
        self.rented_works, self.ran = rented_works, []

    def _cmd(self):
        self.ran.append(self.using)
        return [sys.executable, "-c", DONE if self.using == "local" or self.rented_works else FAIL]


def _wait(t):
    for _ in range(200):
        if t.finished:
            return t.status()
        time.sleep(0.05)
    raise AssertionError("the run never ended")


def test_a_rented_run_that_fails_is_done_again_on_this_machine(tmp_path):
    t = Fake(tmp_path, tmp_path / "out", "modal", about_min=25, rented_works=False)
    assert t.about == T.MODAL_ABOUT_MIN
    assert t.start()
    st = _wait(t)
    assert t.ran == ["modal", "local"] and st["state"] == "done" and st["error"] is None


def test_rented_runs_are_counted_and_run_out(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MODAL_RUNS", 1)
    t = Fake(tmp_path, tmp_path / "out", "modal", about_min=25)
    assert t.start() and _wait(t)["state"] == "done"
    assert t.about == 25                                   # the next one trains here, and the page says how long that takes
    assert t.start() and _wait(t)["state"] == "done"
    assert t.ran == ["modal", "local"]
    again = Fake(tmp_path, tmp_path / "out", "modal", about_min=25)      # a restart does not forget the count
    assert again.about == 25


def test_training_here_is_left_as_it_was(tmp_path):
    t = Fake(tmp_path, tmp_path / "out", "local", about_min=25)
    assert t.about == 25 and t.start() and _wait(t)["state"] == "done" and t.ran == ["local"]


def test_the_page_promises_the_slowest_of_the_last_rented_runs(tmp_path):
    """A warm rented run takes four minutes and a cold one six: the number must not flip with every run."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "timing.json").write_text(json.dumps({"minutes": 25.0, "rented_minutes": 3.6}))     # as an older build left it
    t = T.Trainer(tmp_path, out, "modal", about_min=25)
    assert t.rented_min == T.MODAL_ABOUT_MIN
    t.using = "modal"
    for seconds in (216.0, 230.0, 240.0):
        t._measured(seconds)
    assert t.rented_min == T.MODAL_ABOUT_MIN            # the cold run is still among the last five
    t._measured(250.0)
    assert t.rented_min == round(250.0 / 60.0, 1)       # five warm runs in a row and it is forgotten
    again = T.Trainer(tmp_path, out, "modal", about_min=25)
    assert again.rented_min == t.rented_min and again.about_min == 25


def test_a_rented_run_that_waits_for_its_machines_gives_no_time_and_then_an_honest_one(tmp_path):
    """Eight minutes at nothing done is a wait for a GPU, not the pace of the run."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "timing.json").write_text(json.dumps({"minutes": 25.0, "rented_runs": [3.7, 7.5, 3.7, 3.7]}))
    t = T.Trainer(tmp_path, out, "modal", about_min=25)
    now = [1000.0]
    t.clock = lambda: now[0]
    t.using, t.started = "modal", now[0]
    t.jobs = {w: {"stage": "loading", "progress": 0.0} for w in ("kaiku", "plain")}
    now[0] += 30.0
    st = t.status()
    assert st["stage"] == "loading" and 400 < st["eta_s"] <= 450          # the usual promise at first
    for _ in range(45):
        now[0] += 10.0
        st = t.status()
    assert st["stage"] == "starting" and st["eta_s"] is None              # 8 minutes in: no countdown to nothing
    for progress, after in ((0.2, 12.0), (0.37, 12.0), (0.6, 20.0)):
        now[0] += after
        t.jobs = {w: {"stage": "training", "progress": progress} for w in ("kaiku", "plain")}
        st = t.status()
        assert st["stage"] == "training" and st["eta_s"] is not None and st["eta_s"] < 480   # never "18 minutes left"
