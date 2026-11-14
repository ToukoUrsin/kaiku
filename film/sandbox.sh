#!/bin/bash
# Inside a drun sandbox: install Chrome, Mesa (llvmpipe) and ffmpeg, then run shoot.py with the given arguments.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
t0=$SECONDS
sudo apt-get update -qq >/dev/null
sudo apt-get install -y -qq ffmpeg wget libegl1 libgl1-mesa-dri libgles2 libgbm1 >/dev/null
if ! command -v google-chrome >/dev/null; then
  wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  sudo apt-get install -y -qq /tmp/chrome.deb >/dev/null
fi
python3 -m pip install -q --break-system-packages playwright aiohttp pillow 2>/dev/null || python3 -m pip install -q playwright aiohttp pillow
echo "setup $((SECONDS - t0)) s, $(nproc) cpus, $(google-chrome --version)"
exec python3 film/shoot.py "$@"
