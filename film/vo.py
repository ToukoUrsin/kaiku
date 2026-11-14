#!/usr/bin/env python3
"""Voiceover and music for the film, from film.json. Only lines whose words changed are made again.

    hsec exec --only ELEVENLABS_API_KEY,JUNCTION_GEMINI_API_KEY,DEEPGRAM_API_KEY -- python3 film/vo.py /bulk3/runs/robotteleop/film/v3/work

Writes WORK/vo/<hash>.mp3 + .json (length and word timings, for the captions) and WORK/vo/music.mp3.

film.json's voice.provider picks who speaks: "elevenlabs" (the default; word timings come with the
sound) or "gemini" (Gemini's speech model; the word timings are Deepgram's transcription of the sound,
and the build stops if what was heard is not the line's own words). The music is ElevenLabs'.
"""
import base64
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

FILM = Path(__file__).resolve().parent


def line_key(text: str, voice: dict) -> str:
    who = [text, voice["id"], voice["model"], voice["speed"]]
    if voice.get("provider", "elevenlabs") != "elevenlabs":
        who += [voice["provider"], voice.get("instructions", "")]
    return hashlib.sha256(json.dumps(who).encode()).hexdigest()[:12]


def spoken(film: dict, text: str) -> str:
    """A line with its numbers filled in from film.json."""
    return text.format(**{k: v for k, v in film["numbers"].items() if k != "about"})


def post(url: str, body: dict | bytes, headers: dict) -> bytes:
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        why = re.sub(r"[A-Za-z0-9_*-]{20,}", "...", e.read()[:300].decode(errors="replace"))      # never a key, not even a masked one
        raise SystemExit(f"{url.split('/')[2]} said {e.code}: {why}")


def eleven(text: str, voice: dict) -> tuple[bytes, float, list[dict]]:
    j = json.loads(post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice['id']}/with-timestamps?output_format=mp3_44100_128", {
        "text": text, "model_id": voice["model"],
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.2, "use_speaker_boost": True, "speed": voice["speed"]}},
        {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}))
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
    return base64.b64decode(j["audio_base64"]), al["character_end_times_seconds"][-1], words


def plain(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def heard(audio: bytes) -> list[dict]:
    """The words in a spoken line and when each was said (Deepgram's transcription)."""
    j = json.loads(post("https://api.deepgram.com/v1/listen?model=nova-3&language=en", audio,
                        {"authorization": "Token " + os.environ["DEEPGRAM_API_KEY"], "content-type": "audio/mpeg"}))
    return [w for w in j["results"]["channels"][0]["alternatives"][0]["words"] if plain(w["word"])]


def gemini(text: str, voice: dict, tmp: Path) -> tuple[bytes, float, list[dict]]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ["JUNCTION_GEMINI_API_KEY"]
    j = json.loads(post(f"https://generativelanguage.googleapis.com/v1beta/models/{voice['model']}:generateContent", {
        "contents": [{"parts": [{"text": text}]}],          # the text alone: this model reads a direction aloud
        "generationConfig": {"responseModalities": ["AUDIO"],
                             "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice["id"]}}}}},
        {"x-goog-api-key": key}))
    part = next(p["inlineData"] for p in j["candidates"][0]["content"]["parts"] if "inlineData" in p)
    raw = tmp.with_suffix(".raw")                           # a WAV file, or bare 16-bit samples with the rate in the type
    raw.write_bytes(base64.b64decode(part["data"]))
    rate = re.search(r"rate=(\d+)", part["mimeType"])
    source = ["-f", "s16le", "-ar", rate[1], "-ac", "1"] if rate else []
    tempo = f"atempo={voice['speed']}," if voice["speed"] != 1 else ""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *source, "-i", str(raw),
                    "-af", f"{tempo}silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                    "apad=pad_dur=0.15", "-ar", "44100", "-ac", "1", "-b:a", "128k", str(tmp)], check=True)
    raw.unlink()
    audio = tmp.read_bytes()
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(tmp)],
                               capture_output=True, text=True, check=True).stdout)
    got = heard(audio)
    # The line's own words, each with the time the same word was heard at; a word heard otherwise
    # (a number as digits) takes its share of the gap between its neighbours.
    mine = text.split()
    bits = [(n, plain(x)) for n, w in enumerate(mine) for x in re.split(r"[-\u2013]", w) if plain(x)]   # thirty-two is heard as two words
    a, b = [x for _, x in bits], [plain(w["word"]) for w in got]
    same = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if same.ratio() < 0.85 or len(b) > len(a) + 3:
        raise SystemExit(f"the voice did not say the line: {text!r} was heard as {' '.join(w['word'] for w in got)!r}")
    at: list = [None] * len(mine)
    for i, j, n in same.get_matching_blocks():
        for k in range(n):
            w, was = bits[i + k][0], at[bits[i + k][0]]
            at[w] = (was[0] if was else got[j + k]["start"], got[j + k]["end"])
    words, i = [], 0
    while i < len(mine):
        if at[i]:
            words.append({"w": mine[i], "s": round(at[i][0], 3), "e": round(at[i][1], 3)})
            i += 1
            continue
        j = i
        while j < len(mine) and not at[j]:
            j += 1
        t0 = words[-1]["e"] if words else 0.0
        t1 = at[j][0] if j < len(mine) else dur
        total, run = sum(len(w) + 1 for w in mine[i:j]), 0
        for w in mine[i:j]:
            s = t0 + (t1 - t0) * run / total
            run += len(w) + 1
            words.append({"w": w, "s": round(s, 3), "e": round(t0 + (t1 - t0) * (run - 1) / total, 3)})
        i = j
    return audio, dur, words, same.ratio()


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
                if voice.get("provider", "elevenlabs") == "gemini":
                    best = None                             # the voice varies from one reading to the next: the closest of three
                    for _ in range(3):
                        got = gemini(text, voice, out / "line.tmp.mp3")
                        (out / "line.tmp.mp3").unlink()
                        best = got if best is None or got[3] > best[3] else best
                        if best[3] >= 0.999:
                            break
                    audio, dur, words, match = best
                    if match < 0.999:
                        print(f"  heard a little otherwise ({match:.2f}); listen to {k}.mp3", flush=True)
                else:
                    audio, dur, words = eleven(text, voice)
                mp3.write_bytes(audio)
                meta.write_text(json.dumps({"text": text, "dur": dur, "words": words}))
                print(f"{shot['id']:14s} {dur:5.2f} s  {text}", flush=True)
    music = out / "music.mp3"
    mkey = out / "music.key"
    want = hashlib.sha256(json.dumps(film["music"]["prompt"]).encode()).hexdigest()[:12]
    if "--no-music" not in sys.argv and (not music.exists() or not mkey.exists() or mkey.read_text() != want):
        music.write_bytes(post("https://api.elevenlabs.io/v1/music?output_format=mp3_44100_128", {
            "prompt": film["music"]["prompt"], "music_length_ms": int(film["music"]["seconds"] * 1000), "force_instrumental": True},
            {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}))
        mkey.write_text(want)
        print("music written", flush=True)


if __name__ == "__main__":
    main()
