#!/bin/bash
# From the box: record the Rokoko gloves plugged into the Mac, with nothing opening on its screen.
#
#   tools/booth.sh sync                       copy this checkout to the Mac's ~/kaiku
#   tools/booth.sh start [--emulated one|two] start Rokoko's driver, the hand solver and the recorders
#   tools/booth.sh status                     which gloves stream, at what rate, with a Coil Pro or not
#   tools/booth.sh stop                       stop, then pull the session to /bulk3/runs/robotteleop/sessions/
#   tools/booth.sh check [SESSION]            how the recorded fingers came through: closes counted, fist, flat hand
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MAC="${KAIKU_MAC:-macbook}"
OUT=/bulk3/runs/robotteleop/sessions
SSH=(ssh -o ConnectTimeout=10 "$MAC")

# The Mac drops a connection now and then: a dropped one (ssh's own 255) is tried again, twice.
mac() {
  local n=0 rc
  until "${SSH[@]}" "$@"; do
    rc=$?
    [ "$rc" -ne 255 ] && return "$rc"
    n=$((n + 1))
    [ "$n" -ge 3 ] && return 255
    sleep 2
  done
}

case "${1:-}" in
  sync)
    rsync -az --exclude .venv --exclude .git --exclude ref --exclude __pycache__ --exclude '*.egg-info' \
      --exclude .pytest_cache --exclude MORNING.md "$HERE/" "$MAC:kaiku/" && echo "synced to $MAC:~/kaiku"
    ;;
  start) mac "bash ~/kaiku/tools/booth_mac.sh start ${2:-} ${3:-}" ;;
  status) mac "bash ~/kaiku/tools/booth_mac.sh status" ;;
  stop)
    mac "bash ~/kaiku/tools/booth_mac.sh stop" || exit 1
    mkdir -p "$OUT"
    rsync -az "$MAC:kaiku-sessions/" "$OUT/" && echo "pulled to $OUT" && ls -d "$OUT"/booth-* | sort | tail -1
    "$0" check
    ;;
  check)                                       # what the newest recording says about the hand that was in the glove
    S="${2:-$(ls -d "$OUT"/booth-* | sort | tail -1)}"
    (cd "$HERE" && MUJOCO_GL=egl nice -n 10 .venv/bin/python tools/booth_check.py "$S" --speed 4 | grep '"said"')
    ;;
  *) echo "usage: tools/booth.sh sync | start [--emulated one|two] | status | stop | check [SESSION]"; exit 2 ;;
esac
