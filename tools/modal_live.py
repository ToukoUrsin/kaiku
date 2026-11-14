"""The live demo on Modal: one container, one simulation, a public address.

    modal volume put kaiku-demos DEMOS_DIR /live1     # optional: demonstrations to start from
    modal deploy tools/modal_live.py

Everybody who opens the address watches (and steers) the same 48 robots. The demonstrations
are recorded onto the volume, so the Train button has them and a restart keeps them.
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/kaiku_repo"
G1 = "third_party/mujoco_menagerie/unitree_g1"
PORT = 8653
DEMOS = "/data/live1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "mujoco==4.0.1", "pyyaml", "torch", "scipy", "mink==1.4.0", "qpsolvers[daqp]",
                 "aiohttp", "websockets", "msgpack")
    .add_local_dir(ROOT / "kaiku", f"{REMOTE}/kaiku", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / G1, f"{REMOTE}/{G1}")
)
app = modal.App("kaiku-live", image=image)
vol = modal.Volume.from_name("kaiku-demos", create_if_missing=True)


@app.function(gpu="T4", cpu=32, memory=32768, timeout=24 * 3600, min_containers=1, max_containers=1,
              volumes={"/data": vol})
@modal.concurrent(max_inputs=500)
@modal.web_server(PORT, startup_timeout=300, label="kaiku")
def live():
    import subprocess
    import threading
    import time

    subprocess.Popen(["python", "-u", "-m", "kaiku.app.run", "--n", "48", "--port", str(PORT), "--record", DEMOS],
                     cwd=REMOTE)

    def keep():
        while True:
            time.sleep(120)
            vol.commit()

    threading.Thread(target=keep, daemon=True).start()
