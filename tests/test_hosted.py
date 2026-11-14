"""The hosted image has no off-screen renderer: importing the simulator must not ask for one."""
import os
import subprocess
import sys


def test_importing_the_simulator_leaves_the_renderer_choice_alone():
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    out = subprocess.run([sys.executable, "-c", "import os; import kaiku.app.run; print(os.environ.get('MUJOCO_GL'))"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip().splitlines()[-1] == "None"
