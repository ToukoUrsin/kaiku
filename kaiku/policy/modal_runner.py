"""Bridge a live Modal container to the deployed policy-training functions.

This is deliberately a small subprocess entry point.  The live simulation can keep its own
Python process responsive while this process owns the Modal client and forwards generator
heartbeats as the normal ``KAIKU {json}`` trainer messages.
"""
from __future__ import annotations

import argparse
import io
import json
import queue
import sys
import threading
import time
from pathlib import Path


def say(**message) -> None:
    print("KAIKU " + json.dumps(message), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--scenes", type=int, default=48)
    ap.add_argument("--last", type=int, default=16)
    ap.add_argument("--app", default="kaiku-policy")
    ap.add_argument("--function", default="fit_remote_stream")
    ap.add_argument("--volume", default="kaiku-demos")
    ap.add_argument("--upload", action="store_true",
                    help="the demonstrations are on this machine, not on the volume: send the latest ones first")
    a = ap.parse_args(argv)

    try:
        import modal
    except Exception as exc:  # make the failure visible through the trainer HUD
        say(stage="error", text=f"Modal client unavailable: {exc}")
        return 1

    run_id = f"live_{int(time.time() * 1000)}_{threading.get_native_id()}"
    data, sent = a.data, None
    try:
        volume = modal.Volume.from_name(a.volume)
        if a.upload:
            # The demo runs somewhere else: the scenes both policies will use go to the volume, under
            # a folder of this run's own, which is removed again when the run is over.
            names, rows = latest(Path(a.data), a.last)
            if not names:
                say(stage="error", text="No finished demonstrations to train on yet.")
                return 1
            for which in ("kaiku", "plain"):
                say(which=which, stage="loading", progress=0.0)
            sent = f"/sent/{run_id}"
            with volume.batch_upload(force=True) as up:
                for name in names:
                    up.put_file(Path(a.data) / name, f"{sent}/{name}")
                up.put_file(io.BytesIO(("\n".join(json.dumps(r) for r in rows) + "\n").encode()), f"{sent}/episodes.jsonl")
            data = f"/data{sent}"
        else:
            # The recorder writes to this mounted volume.  A commit is required before a remote
            # container can reload the latest episode and npz files.
            volume.commit()
    except Exception as exc:
        say(stage="error", text=f"Could not send the demonstrations: {repr(exc)[:300]}")
        return 1

    try:
        return train(a, modal, data, run_id)
    finally:
        if sent:
            try:
                volume.remove_file(sent, recursive=True)
            except Exception:
                pass


def latest(root: Path, last: int) -> tuple[list[str], list[dict]]:
    """The files of the operator's latest ``last`` sorted scenes, as the loader picks them, and their rows."""
    by_file: dict[str, list[dict]] = {}
    for line in (root / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            by_file.setdefault(row["file"], []).append(row)
    names = [n for n in sorted(by_file) if any(r["source"] == "human" for r in by_file[n]) and (root / n).exists()]
    names = names[-last:] if last else names
    return names, [r for n in names for r in by_file[n]]


def train(a, modal, data: str, run_id: str) -> int:
    fn = modal.Function.from_name(a.app, a.function)
    cfgs = {
        which: {
            "which": which,
            "name": f"{run_id}_{which}",
            "data": data,
            "steps": a.steps,
            "scenes": a.scenes,
            "last": a.last,
            # rollout uses this value for MuJoCo's batch worker pool; each remote job has
            # its own 16-CPU container.
            "threads": 16, "rollout_threads": 16,
        }
        for which in ("kaiku", "plain")
    }
    jobs: dict[str, dict] = {which: {"stage": "loading", "progress": 0.0} for which in cfgs}
    events: queue.Queue[tuple[str, dict]] = queue.Queue()

    def consume(which: str) -> None:
        try:
            for event in fn.remote_gen(cfgs[which]):
                if isinstance(event, dict):
                    events.put((which, event))
        except Exception as exc:
            events.put((which, {"kind": "error", "text": repr(exc)[:300]}))
        finally:
            events.put((which, {"kind": "stream_done"}))

    threads = [threading.Thread(target=consume, args=(which,), daemon=True, name=f"modal-{which}")
               for which in cfgs]
    for thread in threads:
        thread.start()

    results: dict[str, bytes] = {}
    infos: dict[str, dict] = {}
    errors: list[str] = []
    ended: set[str] = set()
    last_emit = time.monotonic()

    def emit(which: str) -> None:
        message = dict(jobs[which])
        message.pop("which", None)
        say(**message, which=which)

    while len(ended) < len(cfgs):
        try:
            which, event = events.get(timeout=1.0)
        except queue.Empty:
            # A loader or a GPU kernel can have no natural callback for a while.  Repeating the
            # last state keeps the browser meter and stage text alive at least once per second.
            for name, job in jobs.items():
                if name not in ended:
                    emit(name)
            last_emit = time.monotonic()
            continue

        kind = event.get("kind")
        if kind == "status":
            status = {k: v for k, v in event.items() if k not in ("kind", "which")}
            jobs[which].update(status)
            emit(which)
            last_emit = time.monotonic()
        elif kind == "heartbeat":
            # The function emits a heartbeat with its last known status.  Keep the local copy
            # authoritative in case an older deployment omitted some fields.
            jobs[which].update({k: v for k, v in event.items() if k not in ("kind", "which")})
            emit(which)
            last_emit = time.monotonic()
        elif kind == "result":
            payload = event.get("policy")
            if not isinstance(payload, (bytes, bytearray)):
                errors.append(f"{which}: remote result did not contain policy bytes")
            else:
                results[which] = bytes(payload)
                infos[which] = dict(event.get("info") or {})
                jobs[which].update({"stage": "done", "progress": 1.0, **infos[which]})
                emit(which)
        elif kind == "error":
            errors.append(f"{which}: {event.get('text', 'remote training failed')}")
            say(stage="error", text=errors[-1])
        elif kind == "stream_done":
            ended.add(which)

        # Protect against a malformed/quiet generator without flooding stdout.
        if time.monotonic() - last_emit >= 1.0:
            for name, job in jobs.items():
                if name not in ended:
                    emit(name)
            last_emit = time.monotonic()

    for thread in threads:
        thread.join(timeout=2)
    if errors or set(results) != set(cfgs):
        if not errors:
            errors.append("remote training ended without both policies")
        say(stage="error", text="; ".join(errors)[:500])
        return 1

    destination = Path(a.out)
    destination.mkdir(parents=True, exist_ok=True)
    for which, payload in results.items():
        temporary = destination / f".{which}.pt.part"
        temporary.write_bytes(payload)
        temporary.replace(destination / f"{which}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
