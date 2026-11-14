"""The head-camera tests render off screen: on Linux that needs EGL chosen before mujoco is imported."""
import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
