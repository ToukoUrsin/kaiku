"""The Train button's two places to train: rented GPUs, and this machine when those fail or run out."""
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
