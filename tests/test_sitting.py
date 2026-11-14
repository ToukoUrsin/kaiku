from __future__ import annotations

import json

import numpy as np

from kaiku.app.sitting import bundle_session, finish, validate


def _session(tmp_path):
    session = tmp_path / "sitting"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"source": "emulated", "start_time": 1}))
    (session / "DONE").write_text("{}\n")
    np.savez_compressed(session / "ep000000.npz",
                        lead_pinch=np.zeros((4, 2, 3), np.float32),
                        lead_rot=np.tile(np.eye(3), (4, 2, 1, 1)).astype(np.float32),
                        lead_hand=np.zeros((4, 14), np.float32),
                        hands=np.zeros((4, 2, 21, 3), np.float32))
    (session / "episodes.jsonl").write_text(json.dumps({
        "env": 0, "source": "human", "file": "ep000000.npz", "row": 0, "time": 1,
        "layout": [], "mapping": {},
    }) + "\n")
    return session


def test_sitting_validates_and_bundles_lead_commands(tmp_path):
    session = _session(tmp_path)
    ok, verdict, rows = validate(session)
    assert ok and "clean lead scenes" in verdict
    out = bundle_session(session, rows, tmp_path / "bundle")
    data = np.load(out / "ep000000.npz")
    assert data["lead_pinch"].shape == (4, 2, 3)
    assert json.loads((out / "session.json").read_text())["source"] == "emulated"


def test_a_sitting_reports_only_what_was_measured(tmp_path, monkeypatch):
    session = _session(tmp_path)
    from kaiku.app import sitting
    monkeypatch.setattr(sitting, "REPO", tmp_path / "repo")
    monkeypatch.setattr(sitting, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(sitting, "RUNS", tmp_path / "runs")
    assert finish(session, study=False) == 0
    summary = (session / "SITTING.md").read_text()
    assert "Study: not run" in summary and "%" not in summary          # no number nobody measured
    assert "not a real glove session" in summary                        # an emulated one changes no words
    assert not (tmp_path / "bundle").exists()                           # and is not what the demo replays
    assert not (tmp_path / "repo" / "kaiku" / "policy" / "pretrained" / "study_human.json").exists()


def test_a_real_sitting_becomes_the_replay(tmp_path, monkeypatch):
    session = _session(tmp_path)
    (session / "session.json").write_text(json.dumps({"source": "smartgloves", "glove_count": 1, "coil_present": False}))
    stand_in = {"env": 0, "episode": 0, "source": "human", "operator": "stand-in (emulated smartgloves)",
                "file": "ep000000.npz", "row": 0}
    gloved = {**stand_in, "episode": 1, "operator": "visitor (one Smartglove's fingers, position from the pointer, assisted grasp)"}
    echo = {**gloved, "env": 1, "source": "echo", "operator": "echo", "row": 1}
    (session / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (stand_in, gloved, echo)))
    from kaiku.app import sitting
    monkeypatch.setattr(sitting, "REPO", tmp_path / "repo")
    monkeypatch.setattr(sitting, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(sitting, "RUNS", tmp_path / "runs")
    assert finish(session, study=False) == 0
    summary = (session / "SITTING.md").read_text()
    assert "real Smartgloves, 1 glove(s)" in summary and "SUBMISSION.gloves.md" in summary
    assert summary.startswith("1 scenes sorted by the lead, 2 demonstrations")     # the stand-in's scene is not a glove's
    (session / "episodes.jsonl").write_text(json.dumps(stand_in) + "\n")
    assert finish(session, study=False) == 1                                       # nothing a glove drove: not a sitting
    assert (tmp_path / "bundle" / "ep000000.npz").exists()


def test_a_fit_is_read_from_what_it_said():
    from kaiku.app.sitting import read_fit
    text = ('noise\nKAIKU {"which": "plain", "stage": "training", "progress": 0.5}\n'
            'KAIKU {"which": "plain", "stage": "done", "success": 0.03, "humans": 30, "demos": 30}\n'
            '\x01\x01\x01KAIKU {"which": "kaiku", "stage": "done", "success": 0.81, "humans": 30, "demos": 1300}\n')
    said = read_fit(text)
    assert said["plain"]["success"] == 0.03 and said["kaiku"]["demos"] == 1300
