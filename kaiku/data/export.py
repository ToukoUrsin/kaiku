"""Export Kaiku recorder rows to the challenge's LeRobot and GR00T layouts.

The recorder stores one compressed array per operator session and a JSONL row for each
successful robot. This module makes one dataset episode per row, clips it to ``span``,
derives arm velocities, and optionally renders the twin cameras by kinematic replay.
The v3 writer is deliberately dependency-light: when the optional ``lerobot`` package is
not installed it writes the same feature metadata and parquet/video layout directly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kaiku.data.recorder import FPS

H, W = 480, 640
CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
VIDEO_KEYS = tuple(f"observation.images.{x}" for x in CAMS)
GROUPS = {"left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 21), "right_hand": (21, 28)}
ROBOT_TYPE = "unitree_g1_29dof_dex3"
ANNOTATION = "annotation.human.task_description"


def arm_velocity(q: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Estimate measured arm velocity with a central difference at the recorder rate."""
    q = np.asarray(q, dtype=np.float32)
    out = np.empty_like(q)
    if len(q) < 2:
        out.fill(0)
    else:
        dt = 1.0 / float(fps)
        out[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
        out[0] = (q[1] - q[0]) / dt
        out[-1] = (q[-1] - q[-2]) / dt
    return out


def _jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in rows))


def _names(prefixes: tuple[str, str], arm_names: list[str], hand_names: list[str]) -> list[str]:
    return [f"{prefixes[0]}.{x}" for x in arm_names] + [f"{prefixes[1]}.{x}" for x in hand_names]


def _feature_info(arm_names: list[str], hand_names: list[str], depth: bool = False) -> dict:
    names = _names(("arm_q", "hand_q"), arm_names, hand_names)
    action_names = _names(("arm_q_target", "hand_q_target"), arm_names, hand_names)
    f = {
        "observation.state": {"dtype": "float32", "shape": [28], "names": names},
        "observation.arm_dq": {"dtype": "float32", "shape": [14], "names": [f"arm_dq.{x}" for x in arm_names]},
        "action": {"dtype": "float32", "shape": [28], "names": action_names},
    }
    for key in VIDEO_KEYS:
        f[key] = {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channels"],
                  "info": {"video.height": H, "video.width": W, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                           "video.fps": FPS, "video.channels": 3, "has_audio": False}}
    if depth:
        f["observation.images.head_depth"] = {"dtype": "video", "shape": [H, W, 1],
                                               "names": ["height", "width", "channels"],
                                               "info": {"is_depth_map": True, "depth_unit": "mm"}}
    f.update({"timestamp": {"dtype": "float32", "shape": [1], "names": None},
              "frame_index": {"dtype": "int64", "shape": [1], "names": None},
              "episode_index": {"dtype": "int64", "shape": [1], "names": None},
              "index": {"dtype": "int64", "shape": [1], "names": None},
              "task_index": {"dtype": "int64", "shape": [1], "names": None},
              ANNOTATION: {"dtype": "int64", "shape": [1], "names": None},
              "next.reward": {"dtype": "float32", "shape": [1], "names": None},
              "next.done": {"dtype": "bool", "shape": [1], "names": None}})
    return f


def _columns(state, action, dq, task_index, episode_index, start):
    n = len(state)
    return {"observation.state": state.astype(np.float32), "observation.arm_dq": dq.astype(np.float32),
            "action": action.astype(np.float32), "timestamp": (np.arange(n) / FPS).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64), "episode_index": np.full(n, episode_index, np.int64),
            "index": np.arange(start, start + n, dtype=np.int64), "task_index": np.full(n, task_index, np.int64),
            ANNOTATION: np.full(n, task_index, np.int64), "next.reward": np.zeros(n, np.float32),
            "next.done": np.arange(n) == n - 1}


def _write_parquet(path: Path, cols: dict) -> None:
    arrays = {}
    for key, value in cols.items():
        value = np.asarray(value)
        arrays[key] = pa.array(value.tolist(), type=pa.list_(pa.float32())) if value.ndim == 2 else pa.array(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path)


def _encode_video(frames: list[np.ndarray], path: Path, depth: bool = False) -> None:
    """Encode RGB frames with ffmpeg without creating a second image tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pix = "gray16le" if depth else "rgb24"
    arr = np.asarray(frames)
    if depth:
        arr = arr.astype(np.uint16)
    else:
        arr = arr.astype(np.uint8)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix,
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-frames:v", str(len(arr)), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-g", "2", "-bf", "0", "-crf", "18", "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=arr.tobytes(), check=True)


def _stats(cols: dict, videos: dict[str, list[np.ndarray]], n: int) -> dict:
    out = {}
    for key, value in cols.items():
        a = np.asarray(value, dtype=np.float64)
        axis = 0
        keep = a.ndim == 1
        out[key] = {k: np.asarray(getattr(a, k)(axis=axis, keepdims=keep)).tolist()
                    for k in ("min", "max", "mean", "std")}
        out[key]["q01"] = np.quantile(a, .01, axis=axis, keepdims=keep).tolist()
        out[key]["q10"] = np.quantile(a, .10, axis=axis, keepdims=keep).tolist()
        out[key]["q50"] = np.quantile(a, .50, axis=axis, keepdims=keep).tolist()
        out[key]["q90"] = np.quantile(a, .90, axis=axis, keepdims=keep).tolist()
        out[key]["q99"] = np.quantile(a, .99, axis=axis, keepdims=keep).tolist()
        out[key]["count"] = [n]
    for key, frames in videos.items():
        a = np.asarray(frames, dtype=np.float32) / (65535.0 if a_dtype_is_depth(frames) else 255.0)
        if a.ndim == 4:
            a = a.transpose(0, 3, 1, 2)
        out[key] = {k: np.asarray(getattr(a, k)(axis=(0, 2, 3), keepdims=True)).tolist()
                    for k in ("min", "max", "mean", "std")}
        out[key]["count"] = [n]
    return out


def a_dtype_is_depth(frames) -> bool:
    return len(frames) > 0 and np.asarray(frames[0]).dtype == np.uint16


class CameraRenderer:
    """Kinematic MuJoCo renderer for one demonstration at a time."""

    def __init__(self):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco
        from kaiku.sim.twin import build_twin
        self.mujoco = mujoco
        self.twin = build_twin(with_scene=True)
        self.data = mujoco.MjData(self.twin.model)
        self.renderer = mujoco.Renderer(self.twin.model, height=H, width=W)
        self.cam_ids = {x: mujoco.mj_name2id(self.twin.model, mujoco.mjtObj.mjOBJ_CAMERA, x)
                        for x in CAMS}

    def render(self, state: np.ndarray, parts: np.ndarray, layout: list[dict]) -> dict[str, list[np.ndarray]]:
        tw, d, mj = self.twin, self.data, self.mujoco
        names = {x["name"]: i for i, x in enumerate(layout)}
        for frame, poses in zip(state, parts):
            d.qpos[tw.arm_qadr], d.qpos[tw.hand_qadr] = frame[:14], frame[14:]
            for name, i in names.items():
                if name not in tw.scene.by_name:
                    continue
                j = tw.scene.by_name[name].index
                a = tw.obj_qadr[j]
                d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = poses[i, :3], poses[i, 3:7]
            mj.mj_forward(tw.model, d)
            for cam in CAMS:
                self.renderer.update_scene(d, camera=cam)
                yield cam, self.renderer.render().copy()


def _load_rows(src: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(x) for x in (src / "episodes.jsonl").read_text().splitlines() if x.strip()]
    return rows[:limit] if limit else rows


def latest_rows(recorder_dir: str | Path, scenes: int = 1) -> list[dict]:
    """The rows of the last ``scenes`` scenes a recorder saved, read from the end of its list."""
    side = Path(recorder_dir) / "episodes.jsonl"
    if not side.exists():
        return []
    with open(side, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 400_000 * scenes))                       # a scene's rows are about 60 kB
        tail = f.read().decode("utf-8", "replace").splitlines()
    if size > 400_000 * scenes:
        tail = tail[1:]                                               # the first line is cut
    rows = []
    for line in tail:
        try:
            rows.append(json.loads(line))
        except ValueError:                                            # a row being written
            continue
    files = list(dict.fromkeys(r["file"] for r in rows))[-scenes:]
    return [r for r in rows if r["file"] in files]


def bundle(recorder_dir: str | Path, zip_path: str | Path, *, rows: list[dict] | None = None, scenes: int = 1) -> dict:
    """One zip with the last scenes as ``lerobot/`` and ``groot/``, for a download."""
    import tempfile
    import zipfile
    rows = rows if rows is not None else latest_rows(recorder_dir, scenes)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kaiku-export-") as tmp:
        out = Path(tmp) / "kaiku-demonstrations"
        got = export(recorder_dir, out, rows=rows)
        part = zip_path.with_suffix(".part")
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(tmp))
        part.replace(zip_path)
    return {"episodes": got["episodes"], "frames": got["frames"], "zip": zip_path, "bytes": zip_path.stat().st_size}


def _prepare_episode(src: Path, row: dict):
    z = np.load(src / row["file"])
    k = int(row.get("row", 0))
    state, action, parts = z["state"][k], z["action"][k], z["parts"][k]
    a, b = map(int, z["span"][k])
    a, b = max(a, 0), min(b, len(state))
    if b <= a:
        raise ValueError(f"{row['file']} row {k} has empty span {a}:{b}")
    state, action, parts = state[a:b].astype(np.float32), action[a:b].astype(np.float32), parts[a:b].astype(np.float32)
    return state, action, parts, a, b


def _card(root: Path, count: int, frames: int, cameras: bool, depth: bool, fmt: str) -> None:
    camera_note = ("Cameras were rendered by kinematic replay in the Kaiku MuJoCo twin using the challenge calibration."
                   if cameras else
                   "Camera videos were not rendered in this export; rerun with --render-cameras on a machine with MuJoCo EGL.")
    depth_note = ("included as a zero-free head-depth stream when requested." if depth
                  else "not included (the recorder has no depth).")
    (root / "DATA_CARD.md").write_text(f"""# Kaiku demonstrations

This dataset contains {count} simulated Unitree G1 + Dex3 sorting demonstrations ({frames} frames at 30 Hz).
Each recorder row is one episode and is clipped to its recorded `span`; no kept row had a clipped safety command.
The operator is a scripted stand-in wearing emulated smartgloves unless the sidecar says otherwise. Echo rows are
kinematic fan-out copies with their own scene layout. `observation.arm_dq` is derived from measured arm angles with
a central difference at 30 Hz (one-sided differences at episode boundaries), because the recorder has no velocities.
{camera_note}
The GR00T copy is LeRobot v2.1 with `meta/modality.json`; the other copy is the LeRobot v3.0-compatible direct layout
written without the optional `lerobot` package. Depth is {depth_note}
""")


def export(recorder_dir: str | Path, out_dir: str | Path, *, limit: int | None = None,
           render_cameras: bool = False, include_depth: bool = False, overwrite: bool = False,
           operator_first: bool = False, rows: list[dict] | None = None) -> dict:
    """Write ``out_dir/groot`` and ``out_dir/lerobot`` and return episode/frame counts.

    ``rows`` are the recorder rows to write; without them, all of ``episodes.jsonl`` (up to ``limit``)."""
    src, out = Path(recorder_dir), Path(out_dir)
    if rows is None:
        rows = _load_rows(src, None if operator_first else limit)
    if operator_first:
        rows = sorted(rows, key=lambda row: (row.get("source") != "human", int(row.get("demo", 0))))
        if limit:
            rows = rows[:limit]
    if not rows:
        raise ValueError(f"no rows in {src / 'episodes.jsonl'}")
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    groot, v3 = out / "groot", out / "lerobot"
    for root in (groot, v3): (root / "meta").mkdir(parents=True, exist_ok=True)
    arm_names = [f"arm_{i}" for i in range(14)]
    hand_names = [f"hand_{i}" for i in range(14)]
    # Use the challenge joint names where the twin is available.
    try:
        from kaiku.sim.twin import build_twin
        names_twin = build_twin(with_scene=False)
        arm_names, hand_names = names_twin.arm_names, names_twin.hand_names
    except Exception:
        pass
    tasks, episodes, side, stats = {}, [], [], []
    total, renderer = 0, CameraRenderer() if render_cameras else None
    for ep, row in enumerate(rows):
        state, action, parts, a, b = _prepare_episode(src, row)
        instruction = row.get("instruction", "Sort the parts into the coloured bins.")
        task = tasks.setdefault(instruction, len(tasks))
        cols = _columns(state, action, arm_velocity(state[:, :14]), task, ep, total)
        videos = {}
        if renderer:
            frames = {k: [] for k in CAMS}
            for cam, image in renderer.render(state, parts, row.get("layout", [])):
                frames[cam].append(image)
            videos = {f"observation.images.{k}": v for k, v in frames.items()}
        for root, version in ((groot, "v2.1"), (v3, "v3.0")):
            chunk = ep // 1000
            _write_parquet(root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet", cols)
            for key, frames in videos.items():
                _encode_video(frames, root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep:06d}.mp4")
        episodes.append({"episode_index": ep, "tasks": [instruction], "length": b - a})
        side.append({"episode_index": ep, "source": row.get("source", "human"), "operator_episode": row.get("episode"),
                     "frames": [a, b], "clipped_steps": 0, "success": row.get("success", True), "task": row.get("task"),
                     "instruction": instruction, "mapping": row.get("mapping"), "layout": row.get("layout"),
                     "operator": row.get("operator"), "lag_s": row.get("lag_s")})
        stats.append({"episode_index": ep, "stats": _stats(cols, videos, b - a)})
        total += b - a
    features = _feature_info(arm_names, hand_names, depth=False)
    info_common = {"robot_type": ROBOT_TYPE, "total_episodes": len(episodes), "total_frames": total,
                   "total_tasks": len(tasks), "total_videos": len(episodes) * len(CAMS), "fps": FPS,
                   "splits": {"train": f"0:{len(episodes)}"}, "features": features}
    info_g = {"codebase_version": "v2.1", "chunks_size": 1000, "total_chunks": (len(episodes) + 999) // 1000,
              "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
              "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", **info_common}
    info_v = {"codebase_version": "v3.0", "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
              "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", **info_common}
    _jsonl(groot / "meta" / "episodes.jsonl", episodes); _jsonl(v3 / "meta" / "episodes.jsonl", episodes)
    tasks_rows = [{"task_index": i, "task": text} for text, i in tasks.items()]
    for root, info in ((groot, info_g), (v3, info_v)):
        _jsonl(root / "meta" / "tasks.jsonl", tasks_rows)
        _jsonl(root / "meta" / "episodes_stats.jsonl", stats)
        _jsonl(root / "meta" / "aalto_episodes.jsonl", side)
        (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
        if root == groot:
            (root / "meta" / "modality.json").write_text(json.dumps({"state": {k: {"start": a, "end": b} for k, (a, b) in GROUPS.items()},
                "action": {k: {"start": a, "end": b} for k, (a, b) in GROUPS.items()},
                "video": {k.split(".")[-1]: {"original_key": k} for k in VIDEO_KEYS},
                "annotation": {"human.task_description": {"original_key": "task_index"}}}, indent=2))
        _card(root, len(episodes), total, bool(renderer), False, "v2.1" if root == groot else "v3.0")
    return {"episodes": len(episodes), "frames": total, "groot": groot, "lerobot": v3}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recorder_dir"); ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--render-cameras", action="store_true")
    ap.add_argument("--operator-first", action="store_true")
    ap.add_argument("--depth", action="store_true", help="reserved for a future rendered head-depth stream")
    a = ap.parse_args(argv)
    result = export(a.recorder_dir, a.out_dir, limit=a.limit, render_cameras=a.render_cameras,
                    include_depth=a.depth, overwrite=a.overwrite, operator_first=a.operator_first)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in result.items()}))


if __name__ == "__main__":
    main()
