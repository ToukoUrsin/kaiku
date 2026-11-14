"""The Unitree G1 model, fetched once.

    python -m kaiku.sim.fetch

Kaiku uses the G1 with Dex3 hands from MuJoCo Menagerie (BSD-3-Clause), pinned to one commit.
It is 39 MB of meshes, so it is not in this repository: the first run clones just that folder
into ``third_party/mujoco_menagerie``.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"
DEST = REPO / "third_party" / "mujoco_menagerie"
MODEL = DEST / "unitree_g1" / "g1_with_hands.xml"


def ensure(log=print) -> Path:
    """The model's path; clones it first if it is not there."""
    if MODEL.exists():
        return MODEL
    if not shutil.which("git"):
        raise SystemExit(f"The Unitree G1 model is missing and git is not installed.\n"
                         f"Copy the folder unitree_g1 of {URL} (commit {COMMIT[:8]}) to {DEST / 'unitree_g1'}.")
    log("Fetching the Unitree G1 model from MuJoCo Menagerie (39 MB, once)...")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    git = lambda *a, cwd=DEST: subprocess.run(["git", *a], cwd=cwd, check=True)       # noqa: E731
    if not (DEST / ".git").exists():
        git("clone", "--filter=blob:none", "--no-checkout", "--sparse", URL, str(DEST), cwd=DEST.parent)
    git("sparse-checkout", "set", "unitree_g1")
    git("checkout", "--quiet", COMMIT)
    if not MODEL.exists():
        raise SystemExit(f"The clone finished but {MODEL} is not there.")
    return MODEL


if __name__ == "__main__":
    print(ensure(lambda m: print(m, file=sys.stderr)))
