import json
from pathlib import Path

import numpy as np

from kaiku.data.export import export


def test_export_tiny_recorder_round_trip(tmp_path: Path):
    rec = tmp_path / "rec"
    rec.mkdir()
    n = 5
    state = np.zeros((1, n, 28), np.float32)
    state[0, :, 0] = np.arange(n, dtype=np.float32)
    action = state + 10
    parts = np.zeros((1, n, 3, 7), np.float32)
    parts[..., 3] = 1
    np.savez(rec / "ep000000.npz", state=state, action=action, parts=parts, span=np.array([[1, 4]]))
    row = {"demo": 0, "file": "ep000000.npz", "row": 0, "episode": 3, "source": "human",
           "frames": 3, "success": True, "clipped": 0, "instruction": "Sort parts.",
           "task": "sort_fixed", "mapping": {"red": "left"}, "layout": [], "operator": "test"}
    (rec / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    result = export(rec, tmp_path / "out")
    assert result["episodes"] == 1 and result["frames"] == 3
    for root in (result["groot"], result["lerobot"]):
        assert (root / "meta" / "info.json").exists()
        assert (root / "meta" / "aalto_episodes.jsonl").exists()
        import pyarrow.parquet as pq
        table = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
        assert table.column_names[:3] == ["observation.state", "observation.arm_dq", "action"]
        np.testing.assert_allclose(np.asarray(table["observation.state"].to_pylist()), state[0, 1:4])
        np.testing.assert_allclose(np.asarray(table["action"].to_pylist()), action[0, 1:4])
        np.testing.assert_allclose(np.asarray(table["observation.arm_dq"].to_pylist())[:, 0], [30, 30, 30])
        assert json.loads((root / "meta" / "aalto_episodes.jsonl").read_text())["clipped_steps"] == 0

