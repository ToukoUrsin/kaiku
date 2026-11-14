#!/bin/bash
# Record whatever Rokoko gloves are plugged into this Mac. Runs on the Mac, started over ssh from the
# box (tools/booth.sh). Opens no window. Starts Rokoko's driver and hand solver, records the driver's
# raw bytes (--record-raw, replayable through the driver), the driver's RGMP stream and the solved
# stream, each into ~/kaiku-sessions/booth-YYYYMMDD-HHMMSS/. It also starts Kaiku with that one glove
# driving (page on http://127.0.0.1:8651, demonstrations saved into the same folder).
#
#   booth_mac.sh start [--emulated one|two]     booth_mac.sh status     booth_mac.sh stop
set -u
SDK="$HOME/.local/share/rokoko-device-sdk/bin"
REPO="$HOME/kaiku"
PY="$REPO/.venv/bin/python"
ROOT="$HOME/kaiku-sessions"
CUR="$ROOT/booth-current"
PORT=12276
SOLVED=12277
PAGE=8651

port_open() { nc -z 127.0.0.1 "$1" >/dev/null 2>&1; }

status() {
  if [ ! -L "$CUR" ]; then echo "nothing is recording"; return 1; fi
  S="$(readlink "$CUR")"
  cd "$REPO" || return 1
  "$PY" - "$S" "$PORT" "$SOLVED" <<'EOF'
import os, socket, sys, time
from kaiku.rgmp import GloveStream
folder, port, solved = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def up(p):
    try:
        socket.create_connection(("127.0.0.1", p), timeout=0.3).close(); return True
    except OSError:
        return False
print("driver", "up" if up(port) else "DOWN", "| solver", "up" if up(solved) else "DOWN",
      "| page", "http://127.0.0.1:8651" if up(8651) else "not up yet")
if up(port):
    g = GloveStream("127.0.0.1", port).start()
    time.sleep(3.0)
    for side in ("left", "right"):
        f = g.latest(side)
        hz = g.rate_hz(side)
        if f is None or hz < 1:
            print(f"{side}: no glove")
        else:
            print(f"{side}: streaming {hz:.0f} Hz, " + ("Coil Pro position" if f.coil_present else "no Coil Pro (fingers and turn only)"))
    g.stop()
for name in sorted(os.listdir(folder)):
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        print(f"{name}: {os.path.getsize(path) / 1e6:.2f} MB")
print("since start:", round(time.time() - os.path.getctime(os.path.join(folder, "started"))), "s")
EOF
}

case "${1:-}" in
  start)
    if [ -L "$CUR" ] && [ -f "$(readlink "$CUR")/solver.pid" ] && kill -0 "$(cat "$(readlink "$CUR")/solver.pid")" 2>/dev/null; then
      echo "already recording into $(readlink "$CUR")"; status; exit 0
    fi
    S="$ROOT/booth-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$S" && : > "$S/started"
    rm -f "$CUR" && ln -s "$S" "$CUR"
    ARGS=(--emit world)
    case "${2:-}:${3:-}" in
      --emulated:one) ARGS+=(--driver-arg=--pipeline-source --driver-arg=smartglove-emulated --driver-arg=--emulated-devices --driver-arg="$REPO/docs/emulated-rig-one.json"); echo emulated > "$S/source" ;;
      --emulated:*)   ARGS+=(--driver-arg=--pipeline-source --driver-arg=smartglove-emulated --driver-arg=--emulated-devices --driver-arg="$REPO/docs/emulated-rig.json"); echo emulated > "$S/source" ;;
      *)              ARGS+=(--driver-arg=--auto-stream-usb --driver-arg=--record-raw --driver-arg="$S/driver"); echo smartgloves > "$S/source" ;;
    esac
    cd "$S" || exit 1
    PATH="$SDK:$PATH" nohup "$SDK/rkk-hand-solver" "${ARGS[@]}" > "$S/solver.log" 2>&1 < /dev/null &
    echo $! > "$S/solver.pid"
    n=0; until port_open "$PORT" || [ $n -ge 20 ]; do sleep 1; n=$((n+1)); done
    if ! port_open "$PORT"; then echo "Rokoko's driver did not start:"; tail -5 "$S/solver.log"; exit 1; fi
    cd "$REPO" || exit 1
    nohup "$PY" -m kaiku.rgmp.record --host 127.0.0.1 --port "$PORT" --out "$S/raw.rgmp" > "$S/raw.log" 2>&1 < /dev/null &
    echo $! > "$S/raw.pid"
    n=0; until port_open "$SOLVED" || [ $n -ge 10 ]; do sleep 1; n=$((n+1)); done
    if port_open "$SOLVED"; then
      nohup "$PY" -m kaiku.rgmp.record --host 127.0.0.1 --port "$SOLVED" --out "$S/solved.rgmp" > "$S/solved.log" 2>&1 < /dev/null &
      echo $! > "$S/solved.pid"
    fi
    # The same glove drives sixteen robots on this Mac, for whoever opens http://127.0.0.1:8651 here. With
    # a Coil Pro the hand alone drives; without one the page's pointer or camera says where the hand is.
    nohup "$PY" -m kaiku.app.run --n 16 --host 127.0.0.1 --port "$PAGE" --mode visitor --one-glove "127.0.0.1:$PORT" \
      --record "$S/demos" > "$S/run.log" 2>&1 < /dev/null &
    echo $! > "$S/run.pid"
    echo "recording into $S"
    status
    ;;
  status) status ;;
  stop)
    if [ ! -L "$CUR" ]; then echo "nothing is recording"; exit 0; fi
    S="$(readlink "$CUR")"
    for name in run raw solved solver; do
      [ -f "$S/$name.pid" ] && kill "$(cat "$S/$name.pid")" 2>/dev/null
    done
    n=0; until ! kill -0 "$(cat "$S/solver.pid" 2>/dev/null)" 2>/dev/null || [ $n -ge 5 ]; do sleep 1; n=$((n+1)); done
    # the solver starts the driver as its own child: whatever is left of it goes too
    for pid in $(pgrep -x rokoko-sdk); do kill "$pid" 2>/dev/null; done
    rm -f "$CUR"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$S/DONE"
    echo "stopped: $S"; ls -l "$S" | awk 'NR>1 {print $5, $9}'
    ;;
  *) echo "usage: booth_mac.sh start [--emulated one|two] | status | stop"; exit 2 ;;
esac
