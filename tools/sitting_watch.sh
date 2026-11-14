#!/usr/bin/env bash
# On the box: whatever sitting finishes on the Mac is pulled here and python -m kaiku.app.sitting is
# run on it once (the replay, the study in sandboxes, SITTING.md). Light: one ssh a minute.
#
#   systemd-run --user --unit kaiku-sitting-watch tools/sitting_watch.sh
set -u
ROOT=/bulk3/runs/robotteleop/sessions
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCK=/tmp/kaiku-sitting-watch.lock
mkdir -p "$ROOT"
while true; do
  if mkdir "$LOCK" 2>/dev/null; then
    trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
    mapfile -t done_files < <(ssh -o ConnectTimeout=10 -o BatchMode=yes macbook 'ls -1 ~/kaiku-sessions/*/DONE 2>/dev/null' || true)
    for marker in "${done_files[@]}"; do
      [ -n "$marker" ] || continue
      name=$(basename "$(dirname "$marker")")
      target="$ROOT/$name"
      [ -f "$target/.sitting" ] && continue                   # looked at once already
      mkdir -p "$target"
      rsync -a "macbook:$(dirname "$marker")/" "$target/" || continue
      where="$target"                                          # a booth session keeps its scenes in demos/
      [ -f "$target/demos/session.json" ] && where="$target/demos"
      (cd "$REPO" && nice -n 10 .venv/bin/python -m kaiku.app.sitting "$where" >>"$target/after-sitting.log" 2>&1) || true
      date -u +%Y-%m-%dT%H:%M:%SZ > "$target/.sitting"
    done
    rmdir "$LOCK" 2>/dev/null || true
    trap - EXIT
  fi
  sleep 60
done
