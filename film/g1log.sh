#!/bin/bash
# The challenge's own runner follows Kaiku's lead robot for 60 s through the official safety filter.
# Writes WORK/g1/runner.log and the runner's own summary; shot 8 shows its numbers from that file.
#   film/g1log.sh /bulk3/runs/robotteleop/film/work
set -euo pipefail
work=$(readlink -f "$1"); root=$(cd "$(dirname "$0")/.." && pwd); out=$work/g1
mkdir -p "$out"
cd "$root"
nice -n 10 .venv/bin/python -m kaiku.app.run --n 4 --host 127.0.0.1 --port 8661 --g1 8662 >"$out/kaiku.log" 2>&1 &
kaiku=$!
trap 'kill $kaiku 2>/dev/null || true' EXIT
until curl -sf -o /dev/null http://127.0.0.1:8661/; do sleep 1; done
sleep 8                                   # the stand-in is driving by now
cd ref/aaltonomous/repo
PYTHONPATH=. KAIKU_URDF="$root/third_party/aaltonomous_safety/g1_body29_hand14.urdf" nice -n 10 "$root/.venv/bin/python" - "$out" <<'PY' 2>&1 | tee "$out/runner.log"
import json, os, sys
from aaltonomous.interface import safety
safety.WorkspaceChecker.__init__.__defaults__ = (os.environ["KAIKU_URDF"],)      # the filter's own robot model, as in tests/test_g1.py
from aaltonomous.deploy import runner
from aaltonomous.interface.env import AaltoEnv
env = AaltoEnv(backend="mock", task="sort_fixed", seed=0, time_limit=60.0, realtime=True)
summaries, _ = runner.run("mock", "ws://127.0.0.1:8662", "kaiku", 1, out=sys.argv[1] + "/runs", time_limit=60.0, env=env)
s = summaries[0]
json.dump(s, open(sys.argv[1] + "/summary.json", "w"), indent=1, default=str)
print(json.dumps({k: s.get(k) for k in ("end_reason", "steps", "command_hz", "safety")}, default=str))
PY
