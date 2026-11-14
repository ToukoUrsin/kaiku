from __future__ import annotations

import json

import numpy as np

from kaiku.teleop.operator import ReplayOperator


class _Session:
    home_pinch = np.zeros((2, 3))
    home_rot = np.stack([np.eye(3), np.eye(3)])

    def sample_lead(self):
        raise AssertionError("a recorded scene should be used")


def test_replay_uses_recorded_lead_and_hands(tmp_path):
    layout = [{"name": "obj_cube_red_0", "pos": [0.4, 0.1, 0.8], "yaw": 0.0}]
    (tmp_path / "episodes.jsonl").write_text(json.dumps({
        "env": 0, "source": "human", "file": "ep000000.npz", "row": 0,
        "layout": layout, "mapping": {"red": "left"}, "time": 1.0,
    }) + "\n")
    np.savez(tmp_path / "ep000000.npz", lead_pinch=np.ones((2, 2, 3)),
             lead_rot=np.tile(np.eye(3), (2, 2, 1, 1)), lead_hand=np.zeros((2, 14)),
             hands=np.zeros((2, 2, 21, 3)))
    op = ReplayOperator(_Session(), tmp_path)
    assert len(op.rows) == 1
    assert op.scene().placements[0].name == "obj_cube_red_0"
    op.begin()
    first = op.update(None)
    assert np.all(first.pinch == 1)
    assert op.hud()["source"] == "recorded"

