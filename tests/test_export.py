import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")          # the export extra
from kaiku.data.export import export  # noqa: E402


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
        assert table.column_names[:4] == ["observation.state", "observation.arm_dq", "observation.environment_state", "action"]
        np.testing.assert_allclose(np.asarray(table["observation.state"].to_pylist()), state[0, 1:4])
        np.testing.assert_allclose(np.asarray(table["action"].to_pylist()), action[0, 1:4])
        np.testing.assert_allclose(np.asarray(table["observation.arm_dq"].to_pylist())[:, 0], [30, 30, 30])
        assert json.loads((root / "meta" / "aalto_episodes.jsonl").read_text())["clipped_steps"] == 0



def test_download_is_the_last_scene_only(tmp_path: Path):
    """The page's download: one zip with the last scene's demonstrations in both layouts."""
    import zipfile

    from kaiku.data.export import bundle, latest_rows
    rec = tmp_path / "rec"
    rec.mkdir()
    n, lines = 6, []
    for scene, robots in enumerate((2, 3)):
        state = np.zeros((robots, n, 28), np.float32)
        parts = np.zeros((robots, n, 3, 7), np.float32)
        parts[..., 3] = 1
        name = f"ep{scene:06d}.npz"
        np.savez(rec / name, state=state, action=state + 1, parts=parts, span=np.array([[0, n]] * robots))
        for k in range(robots):
            lines.append(json.dumps({"demo": len(lines), "file": name, "row": k, "episode": scene,
                                    "source": "human" if k == 0 else "echo", "frames": n, "success": True,
                                    "clipped": 0, "instruction": "Sort parts.", "layout": []}))
    (rec / "episodes.jsonl").write_text("\n".join(lines) + '\n{"demo": 5, "file": "ep0000')    # a row being written
    rows = latest_rows(rec)
    assert [r["file"] for r in rows] == ["ep000001.npz"] * 3
    assert len(latest_rows(rec, scenes=2)) == 5
    got = bundle(rec, tmp_path / "dl" / "scene.zip", rows=rows)
    assert got["episodes"] == 3 and got["frames"] == 3 * n
    names = zipfile.ZipFile(got["zip"]).namelist()
    for layout in ("lerobot", "groot"):
        assert f"kaiku-demonstrations/{layout}/meta/info.json" in names
        assert sum(x.startswith(f"kaiku-demonstrations/{layout}/data/") for x in names) == 3


def test_export_says_what_it_ships_and_who_drove(tmp_path: Path):
    """No camera stream is listed when no video is written, the parts' poses are in every frame,
    and an episode names who drove it instead of calling every operator a human."""
    import pyarrow.parquet as pq

    from kaiku.data.export import PARTS, PARTS_MAX
    rec = tmp_path / "rec"
    rec.mkdir()
    n = 4
    state = np.zeros((2, n, 28), np.float32)
    parts = np.zeros((2, n, 3, 7), np.float32)
    parts[..., 0] = np.arange(3) + 1.0           # part k sits at x = k + 1
    parts[..., 3] = 1
    np.savez(rec / "ep000000.npz", state=state, action=state, parts=parts, span=np.array([[0, n]] * 2))
    layout = [{"name": f"p{k}", "color": "red", "shape": "cube", "pos": [k + 1.0, 0, 0], "yaw": 0} for k in range(3)]
    rows = [{"demo": k, "file": "ep000000.npz", "row": k, "episode": 0, "source": "human" if k == 0 else "echo",
             "operator": "stand-in (emulated smartgloves)" if k == 0 else "echo", "frames": n, "success": True,
             "clipped": 0, "instruction": "Sort parts.", "layout": layout} for k in range(2)]
    (rec / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = export(rec, tmp_path / "out")
    for root in (result["groot"], result["lerobot"]):
        info = json.loads((root / "meta" / "info.json").read_text())
        assert info["total_videos"] == 0 and "video_path" not in info
        assert not any(f["dtype"] == "video" for f in info["features"].values())
        assert not (root / "videos").exists()
        assert info["features"][PARTS]["shape"] == [PARTS_MAX * 7]
        table = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
        assert set(info["features"]) == set(table.column_names)          # every listed feature is there, and no other
        poses = np.asarray(table[PARTS].to_pylist()).reshape(n, PARTS_MAX, 7)
        np.testing.assert_allclose(poses[:, :3, 0], [[1, 2, 3]] * n)
        assert not poses[:, 3:].any()                                    # the parts this scene does not have
        side = [json.loads(x) for x in (root / "meta" / "aalto_episodes.jsonl").read_text().splitlines()]
        assert [s["source"] for s in side] == ["operator", "echo"]
        assert side[0]["driven_by"] == "stand-in (emulated smartgloves)" and side[0]["parts"] == 3
    assert json.loads((result["groot"] / "meta" / "modality.json").read_text())["video"] == {}
