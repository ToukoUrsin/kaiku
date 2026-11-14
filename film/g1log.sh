#!/bin/bash
# The challenge's own runner (its command-line entry point, mock robot) follows a local Kaiku's lead
# robot for 60 s through the official safety filter. The runner shot is made from what this writes:
#   WORK/g1/runner.log     every line the runner printed, with its own clock; once a second, between
#                          its lines, the safety filter's own counters (logger "safety.filter")
#   WORK/g1/summary.json   the runner's summary of the episode
#   WORK/../rec/g1         the local Kaiku's stream over the same minute (film/rec.py), for the picture
#   film/g1log.sh /bulk3/runs/robotteleop/film/v3/work
set -euo pipefail
work=$(readlink -f "$1"); root=$(cd "$(dirname "$0")/.." && pwd); out=$work/g1
home=$root                                # a worktree borrows the main checkout's .venv and ref/
[ -d "$root/.venv" ] || home=$(git -C "$root" worktree list --porcelain | sed -n '1s/^worktree //p')
py=$home/.venv/bin/python
mkdir -p "$out" "$work/../rec"
cd "$root"
nice -n 10 "$py" -m kaiku.app.run --n 4 --host 127.0.0.1 --port 8661 --g1 8662 >"$out/kaiku.log" 2>&1 &
kaiku=$!
trap 'kill $kaiku 2>/dev/null || true' EXIT
until curl -sf -o /dev/null http://127.0.0.1:8661/; do sleep 1; done
sleep 8                                   # the stand-in is driving by now
nice -n 10 /usr/bin/python film/rec.py http://127.0.0.1:8661 "$work/../rec/g1" --seconds 70 --force >"$out/rec.log" 2>&1 &
rec=$!
sleep 3
cd "$home/ref/aaltonomous/repo"
PYTHONPATH=. KAIKU_URDF="$root/third_party/aaltonomous_safety/g1_body29_hand14.urdf" nice -n 10 "$py" - \
  --backend mock --policy ws://127.0.0.1:8662 --team kaiku --time-limit 60 --out "$out/runs" --export <<'PY' 2>&1 | tee "$out/runner.log"
import logging, os, sys, threading, time
from aaltonomous.interface import env, safety
safety.WorkspaceChecker.__init__.__defaults__ = (os.environ["KAIKU_URDF"],)      # the filter's own robot model, as in tests/test_g1.py
from aaltonomous.deploy import runner
made, init = [], env.AaltoEnv.__init__
def keep(self, *a, **k):                  # the runner makes its own robot; the film only reads its filter's counters
    init(self, *a, **k)
    made.append(self)
env.AaltoEnv.__init__ = keep
def tell():
    log = logging.getLogger("safety.filter")
    while True:
        time.sleep(1.0)
        if made:
            log.info("%s", made[0].filter.summary())
threading.Thread(target=tell, daemon=True).start()
print("aaltonomous.deploy.runner " + " ".join(sys.argv[1:]), flush=True)
runner.main(sys.argv[1:])                 # the runner's own main(), its own arguments
PY
wait $rec
tail -n 1 "$(ls -d "$out"/runs/kaiku/*/ | tail -n 1)summary.jsonl" >"$out/summary.json"
cat "$out/summary.json"
