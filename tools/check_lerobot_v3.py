"""Export a recorder sample and prove that LeRobot v3 can read it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from kaiku.data.export import _prepare_episode, export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recorder_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.recorder_dir / "episodes.jsonl").read_text().splitlines() if line.strip()][:2]
    if len(rows) != 2:
        raise SystemExit("need two recorder rows")
    result = export(args.recorder_dir, args.output_dir, rows=rows, overwrite=True)

    # Some CPU-only wheels omit torchvision's optional NMS extension, while
    # LeRobot imports torchvision for image transforms even for tabular data.
    try:
        import torch

        lib = torch.library.Library("torchvision", "DEF")
        for name in ("nms", "qnms"):
            lib.define(f"{name}(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(repo_id="local/kaiku", root=result["lerobot"])
    assert len(dataset) == result["frames"], (len(dataset), result["frames"])
    table = pq.read_table(result["lerobot"] / "data/chunk-000/file-000.parquet")
    episode_indices = np.asarray(table["episode_index"].to_numpy())
    for episode_index, row in enumerate(rows):
        expected, _, _, _, _ = _prepare_episode(args.recorder_dir, row)
        positions = np.flatnonzero(episode_indices == episode_index)
        assert len(positions) == len(expected)
        for position in (positions[0], positions[-1]):
            item = dataset[int(position)]
            actual = item["observation.state"].detach().cpu().numpy()
            np.testing.assert_allclose(actual, expected[position - positions[0]], rtol=0, atol=1e-6)
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        dataset[index]
    print(f"loaded with lerobot version {getattr(__import__('lerobot'), '__version__', 'unknown')} yes")
    print(f"frames {len(dataset)} episodes {len(rows)} boundaries yes")


if __name__ == "__main__":
    main()
