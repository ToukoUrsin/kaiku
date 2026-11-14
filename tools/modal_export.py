"""Render and export Kaiku demonstrations on Modal's T4 workers.

The recorder snapshot is already in the ``kaiku-demos`` volume at ``/demos``. A local
entrypoint fans out independent recorder rows so a long CPU export does not run on the
overloaded development box.
"""
from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/kaiku_repo"
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("ffmpeg", "libgl1", "libegl1")
         .pip_install("numpy", "pyarrow", "mujoco>=3.2", "pyyaml", "pillow"))
app = modal.App("kaiku-export")
vol = modal.Volume.from_name("kaiku-demos", create_if_missing=False)


@app.function(image=image, gpu="T4", cpu=8, memory=16384, timeout=3 * 3600, volumes={"/data": vol})
def export_job(limit: int | None = None, out: str = "/data/export") -> dict:
    import os
    import sys
    os.environ.setdefault("MUJOCO_GL", "egl")
    if not Path(REMOTE, "kaiku").exists():
        with tarfile.open("/data/code/kaiku-export-bundle.tar") as bundle:
            bundle.extractall("/root")
    sys.path.insert(0, REMOTE)
    sys.path.insert(0, f"{REMOTE}/kaiku")
    from kaiku.data.export import export
    result = export("/data/demos", out, limit=limit, render_cameras=True, overwrite=True, operator_first=True)
    return {k: str(v) if isinstance(v, Path) else v for k, v in result.items()}


@app.local_entrypoint()
def main(limit: int = 0, out: str = "/data/export"):
    if (ROOT / "kaiku").exists():
        bundle_path = Path(tempfile.gettempdir()) / "kaiku-export-bundle.tar"
        with tarfile.open(bundle_path, "w") as bundle:
            bundle.add(ROOT / "kaiku", arcname="kaiku")
            bundle.add(ROOT / "third_party/mujoco_menagerie/unitree_g1", arcname="third_party/mujoco_menagerie/unitree_g1")
        with vol.batch_upload(force=True) as upload:
            upload.put_file(bundle_path, "/code/kaiku-export-bundle.tar")
    result = export_job.remote(limit or None, out)
    print(json.dumps(result))

