#!/usr/bin/env python3
"""Voiceover and music for the film, from film.json. Only lines whose words changed are made again.

    hsec exec --only ELEVENLABS_API_KEY -- python3 film/vo.py /bulk3/runs/robotteleop/film/work

Writes WORK/vo/<hash>.mp3 + .json (length and word timings, for the captions) and WORK/vo/music.mp3.
"""
import base64
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

FILM = Path(__file__).resolve().parent


def line_key(text: str, voice: dict) -> str:
    return hashlib.sha256(json.dumps([text, voice["id"], voice["model"], voice["speed"]]).encode()).hexdigest()[:12]


def spoken(film: dict, text: str) -> str:
    """A line with its numbers filled in from film.json."""
    return text.format(**{k: v for k, v in film["numbers"].items() if k != "about"})


def post(url: str, body: dict) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main() -> None:
    work = Path(sys.argv[1])
    film = json.loads((FILM / "film.json").read_text())
    voice = film["voice"]
    out = work / "vo"
    out.mkdir(parents=True, exist_ok=True)
    for shot in film["shots"]:
        for key in ("vo", "vo_real"):
            for line in shot.get(key, []):
                text = spoken(film, line["text"])
                k = line_key(text, voice)
                mp3, meta = out / f"{k}.mp3", out / f"{k}.json"
                if mp3.exists() and meta.exists():
                    continue
                j = json.loads(post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice['id']}/with-timestamps?output_format=mp3_44100_128", {
                    "text": text, "model_id": voice["model"],
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.2, "use_speaker_boost": True, "speed": voice["speed"]}}))
                mp3.write_bytes(base64.b64decode(j["audio_base64"]))
                al, words, cur = j["alignment"], [], None
                for ch, s, e in zip(al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]):
                    if ch.isspace():
                        if cur:
                            words.append(cur)
                        cur = None
                        continue
                    if cur is None:
                        cur = {"w": "", "s": s, "e": e}
                    cur["w"] += ch
                    cur["e"] = e
                if cur:
                    words.append(cur)
                meta.write_text(json.dumps({"text": text, "dur": al["character_end_times_seconds"][-1], "words": words}))
                print(f"{shot['id']:14s} {al['character_end_times_seconds'][-1]:5.2f} s  {text}", flush=True)
    music = out / "music.mp3"
    mkey = out / "music.key"
    want = hashlib.sha256(json.dumps(film["music"]["prompt"]).encode()).hexdigest()[:12]
    if "--no-music" not in sys.argv and (not music.exists() or not mkey.exists() or mkey.read_text() != want):
        music.write_bytes(post("https://api.elevenlabs.io/v1/music?output_format=mp3_44100_128", {
            "prompt": film["music"]["prompt"], "music_length_ms": int(film["music"]["seconds"] * 1000), "force_instrumental": True}))
        mkey.write_text(want)
        print("music written", flush=True)


if __name__ == "__main__":
    main()
