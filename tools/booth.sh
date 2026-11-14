#!/bin/bash
# From the box: record the Rokoko gloves plugged into the Mac, with nothing opening on its screen.
#
#   tools/booth.sh sync                       copy this checkout to the Mac's ~/kaiku
#   tools/booth.sh start [--emulated one|two] start Rokoko's driver, the hand solver and the recorders
#   tools/booth.sh status                     which gloves stream, at what rate, with a Coil Pro or not
#   tools/booth.sh stop                       stop, then pull the session to /bulk3/runs/robotteleop/sessions/
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MAC="${KAIKU_MAC:-macbook}"
OUT=/bulk3/runs/robotteleop/sessions
SSH=(ssh -o ConnectTimeout=10 "$MAC")

case "${1:-}" in
  sync)
    rsync -az --exclude .venv --exclude .git --exclude ref --exclude __pycache__ --exclude '*.egg-info' \
      --exclude .pytest_cache --exclude MORNING.md "$HERE/" "$MAC:kaiku/" && echo "synced to $MAC:~/kaiku"
    ;;
  start) "${SSH[@]}" "bash ~/kaiku/tools/booth_mac.sh start ${2:-} ${3:-}" ;;
  status) "${SSH[@]}" "bash ~/kaiku/tools/booth_mac.sh status" ;;
  stop)
    "${SSH[@]}" "bash ~/kaiku/tools/booth_mac.sh stop" || exit 1
    mkdir -p "$OUT"
    rsync -az "$MAC:kaiku-sessions/" "$OUT/" && echo "pulled to $OUT" && ls -dt "$OUT"/booth-* | head -1
    ;;
  *) echo "usage: tools/booth.sh sync | start [--emulated one|two] | status | stop"; exit 2 ;;
esac
