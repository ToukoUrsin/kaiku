#!/usr/bin/env bash
# On the box: a safety net for the booth. When something new is plugged into the Mac by USB, start
# Rokoko's driver and the recorders (tools/booth.sh start) without anyone typing. If a glove streams,
# keep recording until it is unplugged, then stop (the session is pulled and looked at by
# tools/booth.sh stop and tools/sitting_watch.sh). If nothing streams within a minute, stop again and
# leave that device alone until it is unplugged. A recording somebody else started is never touched.
# Light: one ssh every 10 seconds. Opens nothing on the Mac's screen.
#
#   systemd-run --user --unit kaiku-glove-watch tools/glove_watch.sh
#
# For a test without hardware: KAIKU_FAKE_USB=/path/file (one device name per line stands in for the
# Mac's USB list) and KAIKU_WATCH_START="--emulated one".
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MAC="${KAIKU_MAC:-macbook}"
LOG="${KAIKU_WATCH_LOG:-/bulk3/runs/robotteleop/glove_watch.log}"
START_ARGS="${KAIKU_WATCH_START:-}"
NOT_A_GLOVE='iPhone|iPad|Keyboard|Mouse|Trackpad|Display|Monitor|Card Reader|Ethernet|Billboard|USB Receiver|Webcam|Headset|Mass Storage|Flash Drive|YubiKey|Hub$'
mkdir -p "$(dirname "$LOG")"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }

# The names of the USB devices on the Mac, one per line. Fails (and prints nothing) when the Mac is away.
usb() {
  if [ -n "${KAIKU_FAKE_USB:-}" ]; then cat "$KAIKU_FAKE_USB" 2>/dev/null; return 0; fi
  local out
  out="$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$MAC" 'ioreg -p IOUSB -w0' 2>/dev/null)" || return 1
  printf '%s\n' "$out" | sed -n 's/^[ |]*+-o \(.*\)@[0-9a-fA-F]*  *<class.*/\1/p' | grep -v -E '^Apple.*XHCI' | sort -u
}
recording() { ssh -o ConnectTimeout=8 -o BatchMode=yes "$MAC" 'test -L ~/kaiku-sessions/booth-current' 2>/dev/null; }
streams() { "$REPO/tools/booth.sh" status 2>/dev/null | grep -q -E '^(left|right): streaming'; }

ignored=""      # devices that turned out not to be a glove, until they are unplugged
active=""       # the device whose recording this script started
gone=0
say "watching $MAC for a glove"
while true; do
  sleep 10
  devs="$(usb)" || continue
  if [ -n "$active" ]; then
    if printf '%s\n' "$devs" | grep -q -x -F "$active"; then gone=0; continue; fi
    gone=$((gone + 1)); [ "$gone" -lt 2 ] && continue
    say "$active unplugged: stopping and pulling the session"
    "$REPO/tools/booth.sh" stop >>"$LOG" 2>&1 || say "stop failed (already stopped by somebody else?)"
    active=""; gone=0; continue
  fi
  # forget ignored devices that are gone
  ignored="$(printf '%s\n' "$ignored" | grep -x -F -f <(printf '%s\n' "$devs") 2>/dev/null || true)"
  new="$(printf '%s\n' "$devs" | grep -v -E "$NOT_A_GLOVE" | grep -v -x -F -f <(printf '%s\n' "${ignored:-__none__}") | head -1 || true)"
  [ -z "$new" ] && continue
  if recording; then
    say "$new plugged in, but a recording is already running: left to whoever started it"
    ignored="$(printf '%s\n%s' "$ignored" "$new")"; continue
  fi
  say "$new plugged in: starting Rokoko's driver and the recorders"
  # shellcheck disable=SC2086
  "$REPO/tools/booth.sh" start $START_ARGS >>"$LOG" 2>&1
  ok=0
  for _ in 1 2 3 4 5 6; do sleep 8; if streams; then ok=1; break; fi; done
  if [ "$ok" = 1 ]; then
    active="$new"; gone=0
    say "a glove streams from $new: recording until it is unplugged"
  else
    say "nothing streams from $new: stopping, and leaving it alone until it is unplugged"
    # so nobody takes the empty folder for a glove session
    ssh -o ConnectTimeout=8 -o BatchMode=yes "$MAC" 'S="$(readlink ~/kaiku-sessions/booth-current)" && echo no-glove > "$S/source"' 2>/dev/null
    "$REPO/tools/booth.sh" stop >>"$LOG" 2>&1 || true
    ignored="$(printf '%s\n%s' "$ignored" "$new")"
  fi
done
