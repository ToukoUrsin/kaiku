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


def test_sitting_smoke_writes_study_and_summary(tmp_path, monkeypatch):
    session = _session(tmp_path)
    from kaiku.app import sitting
    monkeypatch.setattr(sitting, "REPO", tmp_path / "repo")
    monkeypatch.setattr(sitting, "BUNDLE", tmp_path / "bundle")
    assert finish(session, n=2, steps=2, scenes=1, film=False, local=True) == 0
    summary = (session / "SITTING.md").read_text()
    assert "Deploy:" in summary

