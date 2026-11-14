#!/usr/bin/env bash
set -u
ROOT=/bulk3/runs/robotteleop/sessions
LOCK=/tmp/kaiku-sitting-watch.lock
mkdir -p "$ROOT"
while true; do
  if mkdir "$LOCK" 2>/dev/null; then
    trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
    mapfile -t done_files < <(ssh -o ConnectTimeout=10 macbook 'ls -1 ~/kaiku-sessions/*/DONE 2>/dev/null' || true)
    for marker in "${done_files[@]}"; do
      [ -n "$marker" ] || continue
      name=$(basename "$(dirname "$marker")")
      target="$ROOT/$name"
      [ -f "$target/SITTING.md" ] && continue
      mkdir -p "$target"
      rsync -a --ignore-existing "macbook:$(dirname "$marker")/" "$target/" || continue
      # light here; the study inside it runs in sandboxes
      (cd "$(dirname "$0")/.." && nice -n 10 .venv/bin/python -m kaiku.app.sitting "$target" >>"$target/after-sitting.log" 2>&1) || true
    done
    rmdir "$LOCK" 2>/dev/null || true
    trap - EXIT
  fi
  sleep 60
done

