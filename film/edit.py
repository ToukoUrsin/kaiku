#!/usr/bin/python
"""Cut the film from the takes, the voice and film.json.

    /usr/bin/python film/edit.py plan WORK      here: checks, the film's own type, and everything the render needs in WORK/edit
    python3 edit.py render [SHOT ...]           in WORK/edit, inside a drun sandbox: ffmpeg only (SHOT: cut only these shots)

plan stops the build when the film would say something that is not so: every number spoken or shown is
compared with what the page itself said in that take (CLIP.says.jsonl) and with the checked-in files
(study.json, baselines.json, the runner's own summary); the page's source line must say what is really
driving; no frame may show a broken page.

Shot 1 without a real clip is the hand film: the page's own camera picture (film/hand.py saved it with
the time of every picture) beside the robot's hand from the same run's stream. The picture shown in a
film frame is the last one the page had drawn by the moment that frame's robot state arrived, on one
clock. Nothing is shifted to look better. With a real clip (film/make.sh vN --shot2 FILE) shot 1 is that
clip, full frame, with the words film.json keeps under *_real: the only ones that may say gloves.

The cold open (film.json "open": a folder with open.mp4 and end.mp4, made elsewhere) goes in front as it is,
picture and sound, at whatever length the file has when the film is built; the end card goes last. Shot 1 then
starts on the open's last frame: the hand film (or the --shot2 clip) fills the frame with its hand where the
robot's hand was, and moves to its place. plan stops if the whole film passes film.json open.max seconds.
`plan WORK --no-open` builds the body alone.

Nothing the film's own type says is smaller than about 36 px at 1080p; what carries a number is larger.
"""
from __future__ import annotations

import datetime
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
W, H = 1920, 1080
TERM = 1180                      # shot 8: how wide the runner's log is; the robot it follows stays clear of it
INSET = (56, 210, 800, 600)      # shot 1: where the page's camera picture sits
GREEN, GREY = "#5ee08f", "#c9cdd3"


# ---------------------------------------------------------------- shared

def says(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stop(why: str) -> None:
    raise SystemExit("STOP: " + why)


def frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip().strip(","))


def plain(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def probe(path: Path) -> tuple[float, bool, int, int]:
    """A film file's picture length in seconds, whether it has sound, and its size."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,width,height:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    v = next(x for x in j["streams"] if x["codec_type"] == "video")
    return float(v.get("duration") or j["format"]["duration"]), any(x["codec_type"] == "audio" for x in j["streams"]), v["width"], v["height"]


def robot_hand(png: Path, guess: dict) -> tuple[float, float, float, str]:
    """Where the robot's hand is in the open's last frame (centre x, centre y, height, in film pixels): the bright
    shape on black, from its top down. film.json's values when the frame is not a hand on black."""
    import numpy as np
    from PIL import Image
    bright = np.asarray(Image.open(png).convert("L").resize((W, H))) > 60
    gw, gh = guess["size"]
    rows = np.flatnonzero(bright.sum(axis=1) >= 0.01 * W)
    if len(rows):
        top, th, cols = int(rows[0]), float(gh), []
        for _ in range(2):                                    # its width over the hand's own rows gives its height (same hand, same pose)
            cols = np.flatnonzero(bright[top:top + int(th)].sum(axis=0) >= 0.05 * th)
            if not len(cols):
                break
            th = (int(cols[-1]) - int(cols[0])) * gh / gw
        if len(cols) and 0.03 * H < top < 0.6 * H and 0.3 * W < (cols[0] + cols[-1]) / 2 < 0.7 * W and 0.5 * gh < th < 1.6 * gh:
            return float(cols[0] + cols[-1]) / 2, top + th / 2, th, "measured in the open's last frame"
    return float(guess["centre"][0]), float(guess["centre"][1]), float(gh), "film.json's: the open's last frame is not a bright hand on black"


def place(pw: float, ph: float, hand: list[float], to: tuple[float, float, float], grow: float) -> tuple[float, float, float, float]:
    """Where a picture pw x ph goes (x, y, w, h) so that its hand (centre x, centre y, height, as shares of the
    picture) sits on `to` (centre x, centre y, height in film pixels). It is enlarged until it fills the frame,
    but the hand is never more than `grow` times the height asked for."""
    hx, hy = hand[0] * pw, hand[1] * ph
    same = to[2] / (hand[2] * ph)
    cover = max(to[0] / hx, (W - to[0]) / (pw - hx), to[1] / hy, (H - to[1]) / (ph - hy))
    k = min(max(cover, same), grow * same)
    return to[0] - hx * k, to[1] - hy * k, pw * k, ph * k


def glide(a: tuple, b: tuple, t0: float, d: float) -> dict[str, str]:
    """ffmpeg expressions for a picture that eases from box a to box b (x, y, w, h) over d seconds from t0."""
    k = f"clip((t-{t0:.3f})/{d:.3f},0,1)"
    ease = f"({k}*{k}*(3-2*{k}))"
    f = lambda i: f"{b[i]:.2f}" if abs(a[i] - b[i]) < 0.01 else f"({a[i]:.2f}+({b[i] - a[i]:.2f})*{ease})"
    return {"x": f(0), "y": f(1), "w": f"2*trunc({f(2)}/2)", "h": f"2*trunc({f(3)}/2)"}


# ---------------------------------------------------------------- the film's own type (plan, on the box)

CSS = """*{margin:0;box-sizing:border-box}html,body{width:1920px;height:1080px;background:transparent;color:#fff;overflow:hidden;
font-family:Inter,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.scrim{position:absolute;left:0;top:0;height:230px;background:linear-gradient(180deg,rgba(8,9,11,.9),rgba(8,9,11,.55) 52%,rgba(8,9,11,0))}
.head{position:absolute;left:56px;top:38px;font-size:54px;font-weight:700;letter-spacing:-.028em;line-height:1.12;text-wrap:balance;
text-shadow:0 2px 18px rgba(0,0,0,.75)}
.head.col{font-size:46px;max-width:1026px}
.mark{position:absolute;right:56px;top:44px;font-size:34px;font-weight:600;padding:10px 26px 10px 22px;border-radius:999px;
background:rgba(12,13,15,.88);border:1.5px solid rgba(255,255,255,.3);display:flex;gap:14px;align-items:center}
.mark.foot{right:auto;top:auto;left:56px;bottom:44px}
.mark i{width:14px;height:14px;border-radius:50%;background:#5ee08f}
.ring{position:absolute;border:2px solid rgba(255,255,255,.45);border-radius:20px;box-shadow:0 0 0 18px rgba(8,9,11,.8)}
.under{position:absolute;background:rgba(8,9,11,.8);border-radius:0 0 26px 26px;padding:8px 20px 20px;font-size:36px;font-weight:600;
line-height:1.22;letter-spacing:-.014em}
.under small{display:block;font-size:30px;font-weight:500;color:#b9bdc4;margin-top:8px}
.pill{position:absolute;font-size:36px;font-weight:600;line-height:1.25;padding:12px 24px;border-radius:16px;background:rgba(8,9,11,.82)}
.side{position:absolute;left:0;top:0;bottom:0;width:1280px;background:linear-gradient(90deg,rgba(8,9,11,.9),rgba(8,9,11,.74) 60%,rgba(8,9,11,0))}
.line{position:absolute;left:56px;width:1060px;padding-left:28px;border-left:8px solid #5ee08f;font-size:46px;font-weight:650;line-height:1.18;
letter-spacing:-.02em;text-shadow:0 2px 14px rgba(0,0,0,.8)}
.dim{position:absolute;inset:0;background:rgba(8,9,11,.3)}
.shead{position:absolute;left:0;right:0;top:60px;text-align:center;font-size:54px;font-weight:700;letter-spacing:-.028em}
.big{position:absolute;top:176px;width:800px;text-align:center}
.big b{display:block;font-size:256px;font-weight:750;letter-spacing:-.05em;line-height:1;text-shadow:0 4px 30px rgba(0,0,0,.6)}
.big span{display:block;font-size:46px;font-weight:600;margin-top:6px;color:#eceef0}
.then{position:absolute;left:0;right:0;top:598px;text-align:center}
.then b{display:block;font-size:74px;letter-spacing:-.03em;line-height:1.1}
.then b i{font-style:normal;color:#5ee08f}
.then span{display:block;font-size:42px;color:#d5d8dd;margin-top:14px;font-weight:500}
.snote{position:absolute;left:210px;right:210px;bottom:48px;text-align:center;font-size:38px;line-height:1.3;color:#c9cdd3;font-weight:500;text-wrap:balance}
.low{position:absolute;left:0;right:0;bottom:0;height:430px;background:linear-gradient(180deg,rgba(8,9,11,0),rgba(8,9,11,.84) 36%,rgba(8,9,11,.92))}
.mrow{position:absolute;display:flex;align-items:baseline;gap:22px}
.mrow b{font-size:140px;font-weight:750;letter-spacing:-.05em;line-height:1;text-shadow:0 4px 30px rgba(0,0,0,.6)}
.mrow span{font-size:40px;font-weight:600;color:#eceef0;max-width:540px;line-height:1.12}
.mrow.pct b{font-size:104px}
.mrow.pct span{color:#d5d8dd;font-weight:500}
.low.tall{height:520px}
.lownote.two{bottom:26px;line-height:1.25;text-wrap:balance}
.lownote{position:absolute;left:56px;right:56px;bottom:42px;font-size:36px;font-weight:500;color:#c9cdd3}
.end{position:absolute;left:0;right:0;bottom:0;height:470px;background:linear-gradient(180deg,rgba(8,9,11,0),rgba(8,9,11,.88) 30%,rgba(8,9,11,.95));
display:flex;flex-direction:column;align-items:center;justify-content:flex-end;padding-bottom:40px;text-align:center}
.end .a{font-size:150px;font-weight:750;letter-spacing:-.045em;line-height:1.1}
.end .l{font-size:54px;font-weight:600;color:#eceef0;margin-top:2px}
.end .c{font-size:28px;color:#b0b5bc;margin-top:22px;font-weight:500}
.term{position:absolute;left:36px;top:186px;width:1108px;height:858px;border-radius:16px;background:#0b0c0e;border:1.5px solid #2a2d33;overflow:hidden;
box-shadow:0 20px 60px rgba(0,0,0,.6)}
.term .bar{height:52px;background:#17191d;border-bottom:1.5px solid #2a2d33;display:flex;align-items:center;gap:9px;padding:0 18px;
font:500 26px Inter,sans-serif;color:#aab0b8}.term .bar i{width:13px;height:13px;border-radius:50%;background:#3a3e45}.term .bar span{margin-left:10px}
.term .rows{position:absolute;left:0;right:0;bottom:0;top:52px;padding:18px 22px;display:flex;flex-direction:column-reverse;overflow:hidden}
.term .rows>div{margin-bottom:auto;flex:none}
.term p{font:22px/1.42 'DejaVu Sans Mono',monospace;color:#868c95;white-space:pre-wrap;overflow-wrap:anywhere;flex:none}
.term p.cmd{color:#d5d8dd}
.term .d{color:#565b63}.term .hi{color:#7ef0a8;font-weight:700;background:rgba(94,224,143,.14);border-radius:4px}
.zoom{position:absolute;left:60px;top:262px;width:1060px;padding:24px 28px 26px;border-radius:14px;background:#0b0c0e;border:2.5px solid #5ee08f;
font:700 44px/1.32 'DejaVu Sans Mono',monospace;color:#7ef0a8;white-space:pre-wrap;box-shadow:0 14px 50px rgba(0,0,0,.75)}
.count{position:absolute;top:40px;padding:16px 26px 20px;border-radius:16px;background:rgba(8,9,11,.88);border:1.5px solid rgba(94,224,143,.55)}
.count b{display:block;font-size:76px;font-weight:750;letter-spacing:-.035em;line-height:1.05;color:#5ee08f}
.count span{display:block;font-size:40px;font-weight:600;line-height:1.2;margin:8px 0 10px}"""


def draw(pages: dict[str, tuple[str, str]], out: Path) -> None:
    """Each page is (size 'WxH', body html); written as a PNG with a clear background."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        chrome = next((c for c in ("/usr/bin/google-chrome", "/usr/bin/google-chrome-stable") if os.path.exists(c)), None)
        browser = p.chromium.launch(headless=True, **({"executable_path": chrome} if chrome else {}))
        page = browser.new_context(viewport={"width": W, "height": H}).new_page()
        for name, (size, body) in pages.items():
            w, h = (int(x) for x in size.split("x"))
            page.set_content(f"<style>{CSS}</style>{body}")
            page.wait_for_timeout(120)
            page.screenshot(path=str(out / f"{name}.png"), omit_background=True, clip={"x": 0, "y": 0, "width": w, "height": h})
        browser.close()


def head(text: str, mark: str = "", width: int = W, col: bool = False) -> str:
    e = html.escape
    return (f"<div class=scrim style='width:{width}px'></div><div class='head{' col' if col else ''}' style='max-width:{width - 112 - (300 if mark else 0)}px'>{e(text)}</div>"
            + (f"<div class=mark><i></i>{e(mark)}</div>" if mark else ""))


def big(value: str, label: str, left: int, colour: str) -> str:
    return f"<div class=big style='left:{left}px;color:{colour}'><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>"


def terminal(lines: list[tuple[str, str]], title: str) -> str:
    """lines: (kind, text). The runner's own lines, as it printed them."""
    rows = []
    for kind, text in lines:
        t = html.escape(text, quote=False)
        t = re.sub(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)", r"<span class=d>\1</span>", t)
        t = re.sub(r"('steps': \d+, 'clipped_steps': \d+, 'mode': '\w+')", r"<span class=hi>\1</span>", t)
        t = re.sub(r"('ticks': \d+)", r"<span class=hi>\1</span>", t)
        t = re.sub(r"(--backend mock)", r"<span class=hi>\1</span>", t)
        rows.append(f"<p class='{kind}'>{t}</p>")
    return (f"<div class=term><div class=bar><i></i><i></i><i></i><span>{html.escape(title)}</span></div>"
            f"<div class=rows><div>{''.join(rows)}</div></div></div>")


# ---------------------------------------------------------------- plan (on the box)

def plan(work: Path, use_open: bool = True) -> None:
    sys.path.insert(0, str(HERE))
    from vo import line_key, spoken

    film = json.loads((HERE / "film.json").read_text())
    fps, clips, takes = film["fps"], work / "clips", film["takes"]
    n = film["numbers"]
    fill = lambda text: text.format(**{k: v for k, v in n.items() if k != "about"})
    e = html.escape
    kind = (clips / "shot2.kind").read_text().strip() if (clips / "shot2.kind").exists() else "standin"
    real = kind in ("real", "bare")                             # bare: our own hand at the laptop's camera, no glove
    sfx = "_bare" if kind == "bare" else "_real"
    facts: dict = {"shot1": kind, "shots": {}, "checks": []}
    ok = facts["checks"].append
    shots = {s["id"]: s for s in film["shots"]}
    dur = lambda s: s["dur_real"] if real and s.get("swappable") else s["dur"]

    def need(cond: bool, why: str) -> None:
        if not cond:
            stop(why)
        ok(why.split(" BUT ")[0])

    def page_said(name: str) -> list[dict]:
        rows = says(clips / f"{name}.says.jsonl")
        for r in rows:
            for key in ("banner-text", "conn-text"):
                if re.search(r"stopped sending|Reconnecting|Connecting to|Connection lost", r.get(key) or ""):
                    stop(f"{name} at {r['t']} s shows a broken page: {r[key]!r}")
            for key in ("caption", "tag-text", "source-text"):
                text = r.get(key) or ""
                if re.search(r"glove", text, re.I) and not re.search(r"emulated", text, re.I):
                    stop(f"{name} at {r['t']} s: the page says gloves without 'emulated': {text!r}")
        return rows

    # ---- every word the film itself says or shows: no long dashes, and no gloves outside the words kept for a real glove clip
    def words_of(value, key: str = ""):
        if isinstance(value, str):
            yield key, value
        elif isinstance(value, list):
            for v in value:
                yield from words_of(v, key)
        elif isinstance(value, dict):
            for k, v in value.items():
                if k not in ("id", "about", "clip"):
                    yield from words_of(v, k if not key.endswith("_real") else key)
    for s in film["shots"]:
        for key, text in words_of(s):
            if re.search(r"[\u2013\u2014]", text):
                stop(f"{s['id']}: {key} has a long dash: {text!r}")
            if re.search(r"glove", text, re.I) and not key.endswith("_real"):
                stop(f"{s['id']}: {key} claims gloves: {text!r}")
    need(all(c.get("speed", 1) == 1 for c in takes["clips"].values()), "every take is filmed at the speed it ran BUT film.json gives one a speed")

    # ---- voice: where every line and every word falls
    t, total, vo, srt = 0.0, round(sum(dur(s) for s in film["shots"]), 3), [], []
    for shot in film["shots"]:
        lines = shot.get("vo" + sfx, shot["vo"]) if shot.get("swappable") and real else shot.get("vo", [])
        for ln in lines:
            text = spoken(film, ln["text"])
            key = line_key(text, film["voice"])
            meta = json.loads((work / "vo" / f"{key}.json").read_text())
            if ln["at"] + meta["dur"] > dur(shot) - 0.15:
                stop(f"{shot['id']}: the line is {meta['dur']:.1f} s and does not fit in the shot: {text!r}")
            if vo and vo[-1]["at"] + vo[-1]["dur"] > t + ln["at"] - 0.2:
                stop(f"{shot['id']}: the line starts before the one before it has ended: {text!r}")
            vo.append({"file": f"{key}.mp3", "at": round(t + ln["at"], 3), "dur": meta["dur"], "shot": shot["id"], "text": text, "words": meta["words"]})
        facts["shots"][shot["id"]] = {"start": round(t, 3)}
        t += dur(shot)
    if not 85 <= total <= 107:
        stop(f"the film's body is {total} s; the brief says about 1:35 to 1:40 after the cold open")
    # ---- the cold open and the end card: made elsewhere, read now, at whatever length they have now
    op = film.get("open") if use_open else None
    t_open = t_end = 0.0
    cards: dict = {}
    if op:
        for name, key in (("open", "file"), ("end", "end")):
            if not op.get(key):
                continue
            src = Path(op["dir"]) / op[key]
            if not src.exists():
                stop(f"film.json names {src} as the film's {name} and it is not there. `film/make.sh vN --no-open` builds the body alone")
            secs, sound, pw, ph = probe(src)
            wav = Path(op["dir"]) / op.get("sound", "") if name == "open" and not sound and op.get("sound") else None
            cards[name] = {"src": str(src), "frames": int(round(secs * fps)), "seconds": round(int(round(secs * fps)) / fps, 4), "file_seconds": round(secs, 3),
                           "sound": f"{name}.mp4" if sound else "open.wav" if wav and wav.exists() else None}
        need("open" in cards, "the film starts with the cold open BUT film.json's open names no file")
        t_open, t_end = cards["open"]["seconds"], cards.get("end", {}).get("seconds", 0.0)
        whole = round(t_open + total + t_end, 3)
        if whole > op["max"]:
            stop(f"THE FILM IS {whole:.1f} s, OVER {op['max']} s: open {t_open:g} s ({cards['open']['src']}) + body {total:g} s + end card {t_end:g} s. "
                 f"Shorten shots in film.json by {whole - op['max']:.1f} s or more, or drop the end card (open.end)")
        if whole > op["aim"]:
            print(f"NOTE: the film is {whole:.1f} s; the aim is {op['aim']} s or less")
    facts.update(vo=vo, total=total, lead=t_open, tail=t_end, whole=round(t_open + total + t_end, 3), cards=cards)

    def cue(sid: str, words: str) -> float:
        """Seconds into the shot at which the voice starts saying `words`."""
        want = [plain(w) for w in fill(words).split()]
        cap = fill(words)[0].isupper()
        for v in vo:
            if v["shot"] != sid:
                continue
            said = v["words"]
            for i in range(len(said) - len(want) + 1):
                if [plain(w["w"]) for w in said[i:i + len(want)]] == want and (not cap or said[i]["w"][0].isupper()):
                    return round(v["at"] - facts["shots"][sid]["start"] + said[i]["s"], 3)
        stop(f"{sid}: the voice never says {words!r}")

    for v in vo:                                             # sidecar captions, one per sentence
        start = None
        for k, w in enumerate(v["words"]):
            start = w["s"] if start is None else start
            if re.search(r"[.!?:]$", w["w"]) or k == len(v["words"]) - 1:
                srt.append((t_open + v["at"] + start, t_open + v["at"] + w["e"], " ".join(x["w"] for x in v["words"] if start <= x["s"] <= w["s"])))
                start = None

    edit = work / "edit"
    if edit.exists():
        shutil.rmtree(edit)
    (edit / "in" / "inset").mkdir(parents=True)
    (edit / "out").mkdir()
    ins = edit / "in"
    pages: dict[str, tuple[str, str]] = {}
    recipe: dict[str, dict] = {}
    full = f"{W}x{H}"
    hold = "tpad=stop_mode=clone:stop_duration=3"
    clip = lambda name: ["-i", f"in/{name}.mp4"]
    used: list[str] = []

    def join(sid: str, names: list[str]) -> dict:
        """One shot from one take or from two cameras on the same take, cut where one ends and the other starts."""
        got = sum(frames(clips / f"{x}.mp4") for x in names)
        need(got >= int(round(dur(shots[sid]) * fps)), f"{sid}: its takes cover the shot's {dur(shots[sid])} s BUT they are only {got} frames")
        for a, b in zip(names, names[1:]):
            ca, cb = takes["clips"][a], takes["clips"][b]
            need(ca["rec"] == cb["rec"] and abs(ca["end"] - cb["start"]) < 1e-6,
                 f"{sid}: {b} starts at the second {a} ends, in the same recording: no time is skipped BUT film.json has {ca['end']} and {cb['start']}")
        used.extend(f"{x}.mp4" for x in names)
        graph = f"[0:v]{hold}[base]" if len(names) == 1 else "".join(f"[{i}:v]" for i in range(len(names))) + f"concat=n={len(names)}:v=1,{hold}[base]"
        return {"inputs": sum((clip(x) for x in names), []), "n": len(names), "graph": graph}

    # ---- the open, as it is now, and where the robot's hand is in its last frame
    to, H0, MV, DV = None, 0.0, 0.0, 0.0
    if op:
        H0, MV, DV = op["hold"], op["move"], op["dissolve"]
        for name, c in cards.items():
            shutil.copy(c["src"], ins / f"{name}.mp4")
            if int(round(probe(ins / f"{name}.mp4")[0] * fps)) != c["frames"]:
                stop(f"{c['src']} changed while the film was being built. Run film/make.sh again")
        if cards["open"]["sound"] == "open.wav":
            shutil.copy(Path(op["dir"]) / op["sound"], ins / "open.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-sseof", "-1", "-i", str(ins / "open.mp4"), "-vf", f"scale={W}:{H}",
                        "-update", "1", str(ins / "open-last.png")], check=True)
        *to, how = robot_hand(ins / "open-last.png", op["hand"])
        facts["match"] = {"robot_hand": [round(v) for v in to], "how": how}
    last = ["-loop", "1", "-framerate", str(fps), "-t", "5", "-i", "in/open-last.png"]
    # shot 1 comes out of the open's last frame on a plain cut: the open's own last frame is the last one shown, and the
    # shot's first frame is already the hand (the open's maker asks for a cut, no dissolve). A dissolve is drawn only when
    # film.json asks for one longer than a film frame; anything shorter could only repeat the open's last frame.
    out_of_open = lambda k: (f";[{k}:v]format=rgba,fade=t=out:st=0:d={DV}:alpha=1[ol];[pic][ol]overlay=0:0:enable='lt(t,{DV})'[base]"
                             if DV * fps > 1.5 else ";[pic]null[base]")
    words_at = round(H0 + MV - 0.1, 3) if op else 0          # the film's own type waits until the picture is in its place

    # ---- shot 1: the hand
    s, sid = shots["1-hand"], "1-hand"
    if real:
        got = frames(clips / "shot2.mp4")
        need(got >= int(round(dur(s) * fps)) - 1, f"shot 1 is a clip of a real hand, given with --shot2, {dur(s)} s of it BUT the clip is only {got / fps:.1f} s")
        used.append("shot2.mp4")
        _, _, pw, ph = probe(clips / "shot2.mp4")
        fit = ((W - pw) / 2, (H - ph) / 2, pw, ph)               # its place: the whole picture, over a blurred copy of itself
        hand = [float(v) for v in (clips / "shot2.hand").read_text().split(",")]
        box = place(pw, ph, hand, to, op["grow"]) if op else fit
        g = glide(box, fit, H0, MV)
        graph = (f"[0:v]{hold},split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=40:5,eq=brightness=-0.18:saturation=0.8[bg];"
                 f"[b]scale=w='{g['w']}':h='{g['h']}':eval=frame[fg];[bg][fg]overlay=x='{g['x']}':y='{g['y']}'[pic]")
        recipe[sid] = {"inputs": ["-i", "in/shot2.mp4"] + (last if op else []), "n": 2 if op else 1,
                       "over": [{"file": "ov-1", "a": words_at, "b": dur(s), **({"in": 0.4} if op else {})}],
                       "graph": graph + out_of_open(1) if op else graph.replace("[pic]", "[base]")}
        # A screen recording of the page has the bins and the person's own camera picture along its lower left:
        # the bare-hand label sits lower right, clear of both.
        where = "right:56px;bottom:52px;font-size:36px" if kind == "bare" else "left:56px;bottom:52px;font-size:40px"
        pages["ov-1"] = (full, head(s["head" + sfx]) + f"<div class=pill style='{where}'>{e(s['label' + sfx])}</div>")
        facts["match"] = {**facts.get("match", {}), "picture": [pw, ph], "hand_in_picture": hand, "starts_at": [round(v) for v in box], "ends_at": [round(v) for v in fit]}
    else:
        take = (work / takes["hand"]["take"]).resolve()
        t0 = json.loads((take / "rec" / "recording.json").read_text())["t0_epoch"]
        pics = sorted(says(take / "inset.jsonl"), key=lambda r: r["e"])
        reads = says(take / "hand.jsonl")
        c = takes["clips"]["1-robot"]
        rows = page_said("1-robot")
        need(all("hand, seen by a camera" in r.get("caption", "") for r in rows),
             "shot 1: the page's own line says a hand seen by a camera drives BUT a row of 1-robot.says.jsonl differs")
        mine = [r for r in reads if c["start"] <= r["e"] / 1000 - t0 <= c["end"]]
        need(bool(mine) and all(r["drive"] == "camera" and r["fingers"] == "camera" for r in mine),
             "shot 1: the runner said the camera drove the arm and the fingers for every camera frame BUT hand.jsonl differs")
        seen = sum(r["seen"] for r in mine) / len(mine)
        need(seen > 0.9, f"shot 1: the page saw the hand in {seen:.0%} of {len(mine)} camera frames BUT that is too few")
        k, worst, count = 0, 0.0, frames(clips / "1-robot.mp4")
        for f in range(count):
            want = t0 + c["start"] + f / fps                    # when this frame's robot state reached the recorder
            while k + 1 < len(pics) and pics[k + 1]["e"] <= want:
                k += 1
            worst = max(worst, abs(want - pics[k]["e"]))
            os.link(take / "inset" / f"{pics[k]['n']:06d}.jpg", ins / "inset" / f"{f:04d}.jpg")
        need(worst < 0.12, f"shot 1: every film frame has a camera picture drawn within {worst * 1000:.0f} ms before it BUT the gap is too long")
        need(6.0 <= dur(s) <= 8.0, f"shot 1: the hand film is on for {dur(s)} s BUT the brief says 6 to 8")
        import numpy as np
        from PIL import Image, ImageDraw
        x, y, w, h = INSET
        pw, ph = 956, 716                                        # the page's camera picture, without its edge
        mask = Image.new("L", (pw, ph), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=round(20 * pw / w), fill=255)
        mask.save(ins / "mask.png")
        pages["ov-1"] = (full, head(s["head"]) + f"<div class=ring style='left:{x}px;top:{y}px;width:{w}px;height:{h}px'></div>"
                         f"<div class=under style='left:{x - 18}px;top:{y + h + 17}px;width:{w + 36}px'>{e(s['label'])}<small>{e(n['hand_credit'])}</small></div>"
                         f"<div class=pill style='right:56px;top:140px'>{e(s['label_robot'])}</div>")
        base = join(sid, ["1-robot"])
        still = lambda name: ["-loop", "1", "-framerate", str(fps), "-t", "30", "-i", f"in/{name}.png"]
        if op:
            # The camera picture starts full frame, its hand where the robot's hand was in the open's last frame, and moves to its place
            # beside the robot's hand. Where it does not reach the frame's edge it runs softly into a blurred copy of itself.
            box = place(pw, ph, op["film_hand"], to, op["grow"])
            k = box[2] / pw
            gx, gy = np.meshgrid(np.arange(pw), np.arange(ph))
            soft = np.ones((ph, pw))
            feather = 110 / k
            for inside, dist in ((box[0] > 1, gx), (box[0] + box[2] < W - 1, pw - 1 - gx), (box[1] > 1, gy), (box[1] + box[3] < H - 1, ph - 1 - gy)):
                if inside:
                    soft = np.minimum(soft, np.clip(dist / feather, 0, 1))
            Image.fromarray((soft * np.asarray(mask)).astype("uint8")).save(ins / "mask-soft.png")
            g = glide(box, (x, y, w, h), H0, MV)
            size, at = f"scale=w='{g['w']}':h='{g['h']}':eval=frame", f"overlay=x='{g['x']}':y='{g['y']}'"
            graph = (f"[0:v]{hold}[r];[1:v]crop={pw}:{ph}:0:0,{hold},split=3[pa][pb][pc];[2:v]format=gray[mh];[3:v]format=gray[ms];"
                     f"[pc]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:3,format=rgba,fade=t=out:st={H0}:d={MV}:alpha=1[bg];"
                     f"[pa]format=rgba[pa1];[pa1][mh]alphamerge,fade=t=in:st={H0}:d={MV}:alpha=1,{size}[ha];"
                     f"[pb]format=rgba[pb1];[pb1][ms]alphamerge,{size}[sa];"
                     f"[r][bg]overlay=0:0[c0];[c0][ha]{at}[c1];[c1][sa]{at}[pic]" + out_of_open(4))
            recipe[sid] = {"inputs": base["inputs"] + ["-framerate", str(fps), "-i", "in/inset/%04d.jpg"] + still("mask") + still("mask-soft") + last,
                           "n": 5, "over": [{"file": "ov-1", "a": words_at, "b": dur(s), "in": 0.4}], "graph": graph}
            facts["match"].update(picture=[pw, ph], hand_in_picture=op["film_hand"], starts_at=[round(v) for v in box], ends_at=[x, y, w, h])
        else:
            recipe[sid] = {"inputs": base["inputs"] + ["-framerate", str(fps), "-i", "in/inset/%04d.jpg"] + still("mask"),
                           "n": 3, "over": [{"file": "ov-1", "a": 0, "b": dur(s)}],
                           "graph": f"[0:v]{hold}[r];[1:v]crop={pw}:{ph}:0:0,format=rgba[i];[2:v]format=gray[m];"
                                    f"[i][m]alphamerge,scale={w}:{h}[im];[r][im]overlay={x}:{y}:eof_action=repeat[base]"}
    if op:
        need(H0 + MV + 0.5 < dur(s), f"shot 1 starts on the open's last frame and its picture is in place {H0 + MV:g} s in BUT the shot is only {dur(s)} s")
        ok(f"shot 1 starts with its hand at {facts['match']['robot_hand'][:2]}, {facts['match']['robot_hand'][2]} px tall: where the robot's hand is, {facts['match']['how']}")
    else:
        recipe[sid]["fade_in"] = 0.4

    # ---- shot 2: what is simulated, said once
    s, sid = shots["2-sim"], "2-sim"
    rows = page_said("2-sim")
    need(all("scripted hand" in r.get("caption", "").lower() and "scripted hand" in (r.get("tag-text") or "").lower() and r.get("robots") == str(n["fleet_others"]) for r in rows),
         f"shot 2: the page's own line and its label on the hand say a scripted hand drives, with {n['fleet_others']} echoes BUT a row of 2-sim.says.jsonl differs")
    study = json.loads((ROOT / "kaiku/policy/pretrained/study.json").read_text())
    told = (ROOT / "SUBMISSION.md").read_text()
    # The text was cut to under 2,000 characters on 15.11: it says the same three things in fewer words, and that
    # a scripted hand drives when no visitor does is read off the page itself (rows, above).
    need("runs in a simulator" in told and "copy of Junction's G1 sorting station" in told and "the simulator tells the policy where the parts are" in told
         and "scripted hand" in told,
         "shot 2: the three lines about what is simulated are what SUBMISSION.md says BUT its words have changed")
    pages["ov-2"] = (full, "<div class=side></div>" + head(s["head"]).replace("<div class=scrim style='width:1920px'></div>", ""))
    recipe[sid] = {**join(sid, ["2-sim"]), "over": [{"file": "ov-2", "a": 0, "b": dur(s)}]}
    for k, (text, words) in enumerate(zip(s["lines"], s["cues"])):
        pages[f"ov-2{'abc'[k]}"] = (full, f"<div class=line style='top:{196 + 122 * k}px'>{e(text)}</div>")
        recipe[sid]["over"].append({"file": f"ov-2{'abc'[k]}", "a": max(0.0, cue(sid, words) - 0.1), "b": dur(s), "in": 0.25})

    if "3-fleet" in shots:                      # a cut that leaves this shot out of film.json
        # ---- shot 3: one driven, the rest echo
        s, sid = shots["3-fleet"], "3-fleet"
        rows = page_said("3-all")
        need(all(r.get("robots") == str(n["fleet_others"]) and "scripted hand" in r.get("caption", "").lower() and r.get("chain-count") == "1"
                 and r.get("chain-who") == "robot driven" for r in rows),
             f"shot 3: the page says a scripted hand drives 1 robot and {n['fleet_others']} echoes follow BUT a row of 3-all.says.jsonl differs")
        need(n["fleet"] == n["fleet_others"] + 1, f"the fleet is {n['fleet']} robots: one driven and {n['fleet_others']} echoes BUT film.json's numbers disagree")
        pages["ov-3"] = (full, head(fill(s["head"])))
        recipe[sid] = {**join(sid, ["3-all"]), "over": [{"file": "ov-3", "a": 0, "b": dur(s)}]}

    # ---- shots 4 and 5: the same tables without Kaiku and with it, three tables close and then all of them, at the speed they ran
    rows4c, rows4, rows5c, rows5 = page_said("4-close"), page_said("4-wide"), page_said("5-close"), page_said("5-wide")
    over = next((r for r in rows4 if (r.get("banner-text") or "").startswith("Without Kaiku,")), None)
    if not over:
        stop("shot 4: the page never said how many robots sorted their table without Kaiku")
    m = re.match(r"Without Kaiku, (\d+) of (\d+) robots sorted their table", over["banner-text"])
    without, robots = int(m[1]), int(m[2])
    need(all((r.get("banner-text") or "").startswith(f"Kaiku is off. All {robots} robots get the operator's joint commands") for r in rows4c + [r for r in rows4 if r["t"] < over["t"]]),
         "shot 4: until its result the page's banner says Kaiku is off and every robot gets the operator's joint commands BUT a row differs")
    done = next((r for r in rows5 if (r.get("banner-text") or "").startswith("With Kaiku,")), None)
    if not done:
        stop("shot 5: the page never said how many robots sorted their table with Kaiku")
    m = re.match(r"With Kaiku, (\d+) of (\d+) robots sorted their table\. Without: (\d+)", done["banner-text"])
    need(int(m[3]) == without and int(m[2]) == robots == n["fleet_others"],
         f"shots 4 and 5 are the same {robots} tables: without Kaiku {without}, with Kaiku {m[1]} BUT the two takes disagree")
    need(int(m[1]) > without, f"shot 5: with Kaiku {m[1]} of {robots} sorted BUT that is not more than without ({without})")
    need(all((r.get("banner-text") or "") == f"The same tables, Kaiku on. Without it: {without} of {robots}" for r in rows5c),
         f"shot 5, close: the page's banner says these are the same tables with Kaiku on, {without} of {robots} without it BUT a row differs")
    need(all("scripted hand" in r.get("caption", "").lower() for r in rows4c + rows4 + rows5c + rows5), "shots 4 and 5: the page's source line says a scripted hand drives BUT a row differs")
    saved = [int(r["demos"].replace(",", "")) for r in rows5 if r.get("demos")]
    need(len(saved) > 1 and saved[-1] > saved[0] and all(r.get("demos-label") == "demonstrations saved" for r in rows5),
         f"shot 5: the page's count of demonstrations saved rises as the tables are sorted ({saved and saved[0]:,} to {saved and saved[-1]:,}) BUT it does not")
    r4 = over["t"] - takes["clips"]["4-close"]["start"] - 0.25                    # the page is read twice a second
    r5 = done["t"] - takes["clips"]["5-close"]["start"] - 0.25
    facts["shots"]["4-without"].update(result_at=round(r4, 3), said=over["banner-text"])
    facts["shots"]["5-with"].update(result_at=round(r5, 3), said=done["banner-text"])
    for sid in ("4-without", "5-with"):
        s, k = shots[sid], sid[0]
        close = round((takes["clips"][f"{k}-close"]["end"] - takes["clips"][f"{k}-close"]["start"]), 3)
        pages[f"ov-{k}m"] = (full, f"<div class='mark foot'><i></i>{e(s['mark'])}</div>")
        recipe[sid] = {**join(sid, [f"{k}-close", f"{k}-wide"]), "over": [{"file": f"ov-{k}m", "a": 0, "b": dur(s) if sid == "4-without" else close, "hard": sid != "4-without"}]}
    s, sid = shots["5-with"], "5-with"
    at = cue(sid, s["pill_cue"]) - 0.15
    need(at > r5 - 1.0, f"shot 5: the film says every clean run is saved as the page's count rises ({r5:.1f} s into the shot) BUT the line comes at {at:.1f} s")
    pages["ov-5p"] = (full, f"<div class=pill style='left:56px;bottom:334px;border:1.5px solid rgba(94,224,143,.55)'>{e(s['pill'])}</div>")
    recipe[sid]["over"].append({"file": "ov-5p", "a": at, "b": dur(s), "in": 0.25})

    # ---- shot 6: the one number slide: the study, as the page's panel and the checked-in file have it
    row = {r["demos"]: r for r in study["curve"]}
    pct = lambda x: f"{round(x * 100)}%"
    a, b = row[n["study_demos"]], row[n["study_many"]]
    need(pct(a["plain"]) == n["study_alone"] and pct(a["kaiku"]) == n["study_with_echoes"] and pct(b["plain"]) == n["study_many_alone"] and a["kaiku"] > b["plain"],
         f"shot 6: study.json has {n['study_demos']} demonstrations at {n['study_alone']} alone and {n['study_with_echoes']} with echoes, {n['study_many']} alone at {n['study_many_alone']} "
         f"BUT the file says {pct(a['plain'])}, {pct(a['kaiku'])}, {pct(b['plain'])}")
    need(study["about"]["scenes"] == n["study_scenes"] and study["about"]["runs"] == n["study_runs"] and "stand-in" in study["about"]["operator"],
         f"shot 6: study.json is {n['study_runs']} runs on {n['study_scenes']} unseen scenes, demonstrations from the scripted stand-in BUT the file differs")
    panel = says(clips / "panel.png.says.jsonl")[0]
    for key, piece in (("claim-top", f"Trained on {n['study_demos']} demonstrations"), ("claim-main", f"{n['study_alone']} alone, {n['study_with_echoes']} with echoes"),
                       ("claim-more", f"{n['study_demos']} with echoes beat {n['study_many']} without, {n['study_many_alone']}"),
                       ("study-note", f"{n['study_runs']} training runs, tested on {n['study_scenes']} scenes")):
        need(piece in panel.get(key, ""), f"shot 6: the page's panel says {piece!r} BUT the panel in this take says {panel.get(key)!r}. "
             "Change film.json numbers, or record the page again (film/README.md)")
    s, sid = shots["6-study"], "6-study"
    need(dur(s) <= 13.0, f"shot 6: the one number slide is {dur(s)} s BUT it may be 13 at most (12 in v4; v5 also says whose demonstrations and that the echoes cost no one's time)")
    need(a["echoes"] == n["study_echoes"] and f"{a['echoes']:,}" == n["study_echoes_shown"],
         f"shot 6: study.json has {n['study_demos']} demonstrations bring {n['study_echoes_shown']} echoes BUT the file says {a['echoes']:,}")
    need("scripted hand" in fill(s["head"]) and "scripted hand" in fill(s["vo"][0]["text"]),
         "shot 6: the slide says, on screen and in the voice, that the study's demonstrations are the scripted hand's BUT one of them does not")
    c1, c2, c3 = cue(sid, s["cues"][0]), cue(sid, s["cues"][1]), cue(sid, s["then_cue"])
    mine = f"{n['study_demos']} + {n['study_echoes_shown']} echoes"
    need(mine in fill(s["then"]), f"shot 6: the line under the numbers counts the echoes ({mine}) BUT it says {fill(s['then'])!r}")
    then = e(fill(s["then"])).replace(e(mine), f"<i>{e(mine)}</i>")
    pages["ov-6"] = (full, f"<div class=shead>{e(fill(s['head']))}</div><div class=snote>{e(fill(s['note']))}</div>")
    pages["ov-6a"] = (full, big(fill(s["big"][0][0]), fill(s["big"][0][1]), 160, GREY))
    pages["ov-6b"] = (full, big(fill(s["big"][1][0]), fill(s["big"][1][1]), 960, GREEN))
    pages["ov-6c"] = (full, f"<div class=then><b>{then}</b><span>{e(fill(s['then_sub']))}</span></div>")
    page_said("6-back")
    base = join(sid, ["6-back"])
    recipe[sid] = {**base, "graph": base["graph"].replace("[base]", f",drawbox=x=0:y=0:w={W}:h={H}:color=black@0.66:t=fill[base]"),
                   "over": [{"file": "ov-6", "a": 0, "b": dur(s)}, {"file": "ov-6a", "a": c1, "b": dur(s), "in": 0.25},
                            {"file": "ov-6b", "a": c2, "b": dur(s), "in": 0.25}, {"file": "ov-6c", "a": c3, "b": dur(s), "in": 0.25}]}

    # ---- shot 7: the same 16 demonstrations, replayed two ways: one spoken line over robots
    base = json.loads((ROOT / "kaiku/policy/pretrained/baselines.json").read_text())
    arms = base["policy"]["arms"]
    need(pct(arms["mimicgen"]["yield"]) == n["matched_replay"] and pct(arms["kaiku"]["yield"]) == n["matched_kaiku"]
         and base["generation"]["kaiku"]["source_scenes"] == base["generation"]["mimicgen"]["source_scenes"] == n["matched_demos"]
         and base["policy"]["heldout_scenes"] == n["study_scenes"] and base["policy"]["seeds_per_arm"] == n["study_runs"],
         f"shot 7: baselines.json has the same {n['matched_demos']} demonstrations at {n['matched_replay']} with the replay and {n['matched_kaiku']} with Kaiku's echoes, "
         f"{n['study_runs']} runs on {n['study_scenes']} tables BUT the file says {pct(arms['mimicgen']['yield'])} and {pct(arms['kaiku']['yield'])}")
    need("re-implementation" in base["about"]["notes"], "shot 7: baselines.json says the replay is our own re-implementation BUT its notes do not")
    gen = base["generation"]
    need(gen["mimicgen"]["clean"] == n["matched_replay_kept"] and gen["kaiku"]["clean"] == n["matched_kaiku_kept"]
         and gen["mimicgen"]["attempts"] == gen["kaiku"]["attempts"] == n["matched_tables"] and gen["mimicgen"]["source_scenes"] == n["matched_demos"]
         and f"{n['matched_tables']:,}" == n["matched_tables_shown"],
         f"shot 7: on the same {n['matched_tables']} tables the replay kept {n['matched_replay_kept']} clean runs and Kaiku's echoes {n['matched_kaiku_kept']} "
         f"BUT baselines.json says {gen['mimicgen']['clean']} and {gen['kaiku']['clean']} of {gen['kaiku']['attempts']}")
    need(arms["mimicgen"]["pairs"] > arms["plain"]["pairs"] and arms["kaiku"]["pairs"] > arms["plain"]["pairs"],
         "shot 7: each policy was trained on the 16 demonstrations plus the runs kept, as the note says BUT baselines.json's training pairs say otherwise")
    bars = dict(panel.get("bars") or [])
    shown = {name: f"{round(float(v.rstrip('%')))}%" for name, v in bars.items()}
    need(shown.get("With MimicGen-style replays") == n["matched_replay"] and shown.get("With all of Kaiku's echoes") == n["matched_kaiku"]
         and f"another {n['matched_demos']} demonstrations" in panel.get("baselines-lede", "") and "our own MimicGen-style re-implementation" in panel.get("baselines-note", ""),
         f"shot 7: the page's panel shows {n['matched_replay']} with MimicGen-style replays and {n['matched_kaiku']} with Kaiku's echoes, and says the replay is our own "
         f"BUT the panel in this take says {bars!r}")
    s, sid = shots["7-matched"], "7-matched"
    cs = [cue(sid, c) for c in s["cues"]]                   # the two counts, then the two shares, each as the voice says it
    # each column's two numbers share one width, so its two labels start at the same place
    row = lambda v, label, left, colour, bottom, k="": (f"<div class='mrow {k}' style='left:{left}px;bottom:{bottom}px;color:{colour}'>"
                                                        f"<b style='min-width:{210 if left < W / 2 else 280}px'>{e(fill(v))}</b><span>{e(fill(label))}</span></div>")
    pages["ov-7"] = (full, head(fill(s["head"])) + f"<div class='low tall'></div><div class='lownote two'>{e(fill(s['note']))}</div>")
    (k1, l1, p1, q1), (k2, l2, p2, q2) = s["big"]
    pages["ov-7a"] = (full, row(k1, l1, 56, GREY, 290))
    pages["ov-7b"] = (full, row(k2, l2, 980, GREEN, 290))
    pages["ov-7c"] = (full, row(p1, q1, 56, GREY, 148, "pct"))
    pages["ov-7d"] = (full, row(p2, q2, 980, GREEN, 148, "pct"))
    page_said("7-robots")
    recipe[sid] = {**join(sid, ["7-robots"]), "over": [{"file": "ov-7", "a": 0, "b": dur(s)}]
                   + [{"file": f"ov-7{x}", "a": c, "b": dur(s), "in": 0.25} for x, c in zip("abcd", cs)]}

    if "8-runner" in shots:                      # a cut that leaves this shot out of film.json
        # ---- shot 8: the challenge's own runner: its log for five seconds at most, its own lines at the moments it printed them
        s, sid = shots["8-runner"], "8-runner"
        g1 = json.loads((work / "g1" / "summary.json").read_text())
        log = (work / "g1" / "runner.log").read_text().splitlines()
        need(g1["safety"]["mode"] == "ok" and g1["safety"]["steps"] == g1["ticks"] == n["runner_commands"] and g1["safety"]["clipped_steps"] == n["runner_clipped"],
             f"shot 8: the runner's summary says {n['runner_commands']} commands went through the safety filter and {n['runner_clipped']} were clipped BUT it says {g1['safety']} of {g1['ticks']} ticks")
        need("--backend mock" in log[0], "shot 8: the runner ran with its mock robot BUT the first line of runner.log does not say --backend mock")
        t0 = json.loads((work.parent / "rec" / "g1" / "recording.json").read_text())["t0_epoch"]
        timed, last = [], 0.0
        for text in log:
            m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) ", text)
            if m:
                last = datetime.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").timestamp() + int(m[2]) / 1000 - t0
            timed.append((last, "cmd" if not timed else "", re.sub(r"/\S+?/work(?=/g1)", "\u2026", text)))
        summary, line = next((t, text) for t, _, text in timed if " episode 0: " in text)
        on, held = takes["runner"]["on"], takes["runner"]["for"]
        need(held <= 5.0, f"shot 8: the runner's log is on screen for {held} s BUT the brief says 5 at most")
        lead = round((cue(sid, s["cut_cue"]) - 0.2) * fps) / fps
        need(on + 1.0 <= lead <= on + held - 1.8, f"shot 8: the runner's summary line arrives {lead - on:.1f} s after its log comes on and stays {on + held - lead:.1f} s BUT that is too early or too short to read")
        start = summary - lead                                   # the second of the recording the shot starts at
        c = takes["clips"]["8-robot"]
        need(c["start"] <= start and c["end"] >= start + dur(s), "shot 8: the robot beside the log is the same seconds of the same run BUT the take in film.json does not cover them")
        states = [(on, start + on)] + [(round(t - start, 3), t) for t, _, _ in timed if start + on < t < start + on + held]
        states = sorted({st[0]: st for st in states}.values())
        over8 = []
        for i, (at, now) in enumerate(states):
            lines = [(k, text) for t, k, text in timed if t <= now + 1e-6]
            pages[f"term-{i:02d}"] = (full, terminal(lines, "aaltonomous.deploy.runner --backend mock"))
            over8.append({"file": f"term-{i:02d}", "a": at, "b": states[i + 1][0] if i + 1 < len(states) else on + held, "hard": True})
        m = re.search(r"'safety': \{'steps': (\d+), 'clipped_steps': (\d+), 'mode': '(\w+)'", line)
        need(bool(m) and int(m[1]) == n["runner_commands"] and int(m[2]) == n["runner_clipped"] and m[3] == "ok",
             "shot 8: the line shown large is cut from the runner's own summary line and says the same as its summary file BUT it does not")
        pages["ov-8z"] = (full, f"<div class=zoom>{e(m[0], quote=False)}, \u2026}}</div>")
        over8.append({"file": "ov-8z", "a": lead, "b": on + held, "hard": True})
        pages["ov-8"] = (full, head(s["head"], width=TERM, col=True)
                         + f"<div class=pill style='left:{TERM + 24}px;bottom:40px;max-width:{W - TERM - 60}px;font-size:34px'>{e(s['label_robot'])}</div>")
        over8.insert(0, {"file": "ov-8", "a": 0, "b": dur(s)})
        need(f"{g1['safety']['steps']:,}" == n["runner_commands_shown"], f"shot 8: the count shown is the runner's own ({n['runner_commands_shown']}) BUT its summary says {g1['safety']['steps']:,}")
        pages["ov-8n"] = (full, f"<div class=count style='left:{TERM + 24}px;width:{W - TERM - 60}px'><b>{e(fill(s['count'][0]))}</b><span>{e(fill(s['count'][1]))}</span><b>{e(fill(s['count'][2]))}</b></div>")
        over8.append({"file": "ov-8n", "a": lead + 0.4, "b": dur(s), "in": 0.25})
        recipe[sid] = {"inputs": clip("8-robot"), "n": 1, "over": over8,
                       "graph": f"[0:v]trim=start={start - c['start']:.4f},setpts=PTS-STARTPTS,{hold}[base]"}
        facts["shots"][sid].update(log_on=[on, on + held], summary_at=lead, rec_from=round(start, 3), states=len(states))
        used.append("8-robot.mp4")
        page_said("8-robot")

    # ---- shot 9: nobody drives
    s, sid = shots["9-nobody"], "9-nobody"
    rows = page_said("9-close") + page_said("9-wide")
    need(all("Nobody is driving" in r.get("caption", "") and r.get("robots") == str(n["fleet"]) and r.get("chain-who") == "Nobody driving" for r in rows),
         f"shot 9: the page says nobody is driving {n['fleet']} robots BUT a row of 9-close or 9-wide.says.jsonl differs")
    pages["ov-9"] = (full, head(s["head"], mark=s["mark"]))
    ca, cb = takes["clips"]["9-close"], takes["clips"]["9-wide"]
    need(ca["rec"] == cb["rec"] and ca["end"] <= cb["start"], "shot 9: three tables and then all of them, from the same policy run, in order BUT film.json has them otherwise")
    got = frames(clips / "9-close.mp4") + frames(clips / "9-wide.mp4")
    need(got >= int(round(dur(s) * fps)), f"{sid}: its takes cover the shot's {dur(s)} s BUT they are only {got} frames")
    used.extend(["9-close.mp4", "9-wide.mp4"])
    recipe[sid] = {"inputs": clip("9-close") + clip("9-wide"), "n": 2, "graph": f"[0:v][1:v]concat=n=2:v=1,{hold}[base]", "over": [{"file": "ov-9", "a": 0, "b": dur(s)}]}

    # ---- shot 10: the policy still sorting alone, and the live page's address over it
    s, sid = shots["10-address"], "10-address"
    rows = page_said("10-policy")
    need(all("Nobody is driving" in r.get("caption", "") and r.get("chain-who") == "Nobody driving" for r in rows),
         "shot 10: the page says nobody is driving BUT a row of 10-policy.says.jsonl differs")
    end = next((r for r in rows if re.match(r"\d+ of \d+ robots sorted their scene with nobody driving", r.get("banner-text") or "")), None)
    need(end is not None and int(end["banner-text"].split()[2]) == n["fleet"], f"shot 10: the page says how many of its {n['fleet']} robots sorted their scene with nobody driving BUT it never did in this take")
    facts["shots"][sid]["said"] = end["banner-text"]
    credit = f"<div class=c>{e(n['hand_credit_long'])}</div>" if not real else ""
    pages["ov-10"] = (full, f"<div class=end><div class=a>{e(fill(s['card']['address']))}</div><div class=l>{e(s['card']['line'])}</div>{credit}</div>")
    recipe[sid] = {**join(sid, ["10-policy"]), "over": [{"file": "ov-10", "a": s["card"]["at"], "b": dur(s), "in": 0.5}]}

    # the planned silence: from the moment the page shows how few sorted their table without Kaiku
    q = facts["shots"]["4-without"]["start"] + facts["shots"]["4-without"]["result_at"]
    facts["quiet"] = [round(q, 3), round(min(q + 2.8, facts["shots"]["5-with"]["start"]), 3)]

    # ---- stage everything the render needs
    draw(pages, ins)
    for name in used:
        os.link(clips / name, ins / name)
    for v in vo:
        if not (ins / v["file"]).exists():
            os.link(work / "vo" / v["file"], ins / v["file"])
    os.link(work / "vo" / "music.mp3", ins / "music.mp3")
    facts["render"] = [{"id": s["id"], "dur": dur(s), **recipe[s["id"]]} for s in film["shots"]]
    shutil.copy(HERE / "film.json", edit / "film.json")
    shutil.copy(HERE / "edit.py", edit / "edit.py")
    (edit / "facts.json").write_text(json.dumps(facts, indent=1) + "\n")
    stamp = lambda s: f"{int(s // 3600):02d}:{int(s // 60 % 60):02d}:{int(s % 60):02d},{int(round(s % 1 * 1000)) % 1000:03d}"
    (edit / "out" / "film.srt").write_text("".join(f"{i + 1}\n{stamp(a)} --> {stamp(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(srt)))
    (edit / "run.sh").write_text("set -e\nexport DEBIAN_FRONTEND=noninteractive\ncommand -v ffmpeg >/dev/null || "
                                 "{ (sudo apt-get update -qq || apt-get update -qq) >/dev/null 2>&1; (sudo apt-get install -y -qq ffmpeg || apt-get install -y -qq ffmpeg) >/dev/null 2>&1; }\n"
                                 "python3 edit.py render\n")
    for line in facts["checks"]:
        print("ok  " + line)
    print(f"film {facts['whole']:g} s = open {t_open:g} + body {total:g} + end card {t_end:g}; shot 1 is the {'real clip' if real else 'hand film'}; "
          f"silence {t_open + facts['quiet'][0]:g} to {t_open + facts['quiet'][1]:g} s; ready in {edit}")


# ---------------------------------------------------------------- render (in the sandbox)

def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", *args], check=True)


def render(only: list[str]) -> None:
    """only: shot ids to cut on their own (out/sN.mp4), to look at a shot without cutting the film."""
    film, facts = json.loads(Path("film.json").read_text()), json.loads(Path("facts.json").read_text())
    fps, cards = film["fps"], facts.get("cards", {})
    # every part is cut the same way and says the same about its colour, so the parts join without being encoded again
    x264 = ["-an", "-r", str(fps), "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p", "-video_track_timescale", "15360",
            "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0"]

    def card(name: str) -> str:
        """The open or the end card: its own picture, nothing over it, at the film's frame rate, for as long as it is."""
        out, c = f"out/{name}.mp4", cards[name]
        ff("-i", f"in/{name}.mp4", "-vf", f"fps={fps},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p,"
           "tpad=stop_mode=clone:stop_duration=2", "-frames:v", str(c["frames"]), *x264, "-crf", "14", out)
        print(name, c["seconds"], "s", flush=True)
        return out

    parts = [card("open")] if "open" in cards and not only else []
    for i, r in enumerate(facts["render"]):
        if only and r["id"] not in only:
            continue
        dur, out = r["dur"], f"out/s{i + 1}.mp4"
        ins, g, last = list(r["inputs"]), [r["graph"]], "base"
        for k, o in enumerate(r["over"]):                    # the film's own type, each piece for its own seconds
            ins += ["-loop", "1", "-framerate", str(fps), "-t", f"{dur + 1:.3f}", "-i", f"in/{o['file']}.png"]
            a, b = o["a"], o["b"]
            chain = "format=rgba"
            if o.get("in"):
                chain += f",fade=t=in:st={a:.3f}:d={o['in']}:alpha=1"
            if o.get("out"):
                chain += f",fade=t=out:st={b - o['out']:.3f}:d={o['out']}:alpha=1"
            g.append(f"[{r['n'] + k}:v]{chain}[o{k}]")
            when = "" if a <= 0 and b >= dur else f":enable='between(t,{a:.3f},{b - 0.001:.3f})'" if o.get("hard") else f":enable='gte(t,{a:.3f})*lt(t,{b:.3f})'"
            g.append(f"[{last}][o{k}]overlay=0:0{when}[b{k}]")
            last = f"b{k}"
        tail = f"fade=t=in:st=0:d={r['fade_in']}," if r.get("fade_in") else ""
        g.append(f"[{last}]{tail}fps={fps},format=yuv420p,setsar=1[v]")
        ff(*ins, "-filter_complex", ";".join(g), "-map", "[v]", "-frames:v", str(int(round(dur * fps))), *x264, "-crf", "17", out)
        parts.append(out)
        print("shot", r["id"], flush=True)
    if only:
        return
    if "end" in cards:
        parts.append(card("end"))
    Path("out/list.txt").write_text("".join(f"file '{Path(p).name}'\n" for p in parts))
    ff("-f", "concat", "-safe", "0", "-i", "out/list.txt", "-c", "copy", "out/video.mp4")

    # ---- sound: the voice, and music that gets out of its way and stops for the planned silence
    total, music, (q0, q1) = facts["total"], film["music"], facts["quiet"]
    ins, g, labels = [], [], []
    for k, v in enumerate(facts["vo"]):
        ins += ["-i", f"in/{v['file']}"]
        g.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={int(v['at'] * 1000)}:all=1[v{k}]")
        labels.append(f"[v{k}]")
    nv = len(facts["vo"])
    ins += ["-i", "in/music.mp3"]
    g.append(f"{''.join(labels)}amix=inputs={nv}:normalize=0,apad=whole_dur={total},atrim=0:{total},asplit[vo][key]")
    g.append(f"[{nv}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{total},volume={music['gain_db']}dB,"
             f"volume='min(1,max(0,max(({q0}-t)/0.5,(t-{q1})/0.7)))':eval=frame,afade=t=in:st=0:d=0.6[m]")
    g.append(f"[m][key]sidechaincompress=threshold=0.015:ratio={music.get('duck', 6)}:attack=30:release=500[md]")
    g.append(f"[md][vo]amix=inputs=2:normalize=0,afade=t=out:st={total - 1.5}:d=1.5[a]")
    ff(*ins, "-filter_complex", ";".join(g), "-map", "[a]", "-c:a", "pcm_s16le", "out/mix.wav")

    def loud(path: str, extra: str = "") -> dict:
        r = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-vn", "-af", extra + "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True)
        return json.loads(r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1])
    gain = -16.0 - float(loud("out/mix.wav")["input_i"])
    wav = ["-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le"]
    ff("-i", "out/mix.wav", "-af", f"volume={gain:.2f}dB,alimiter=limit=0.84:level=disabled,apad=whole_dur={total},atrim=0:{total}", *wav, "out/body.wav")

    def sound(name: str) -> str:
        """The open's or the end card's own sound, at its own level, exactly as long as its picture; silence if it has none."""
        out, c = f"out/{name}.wav", cards[name]
        if c["sound"]:
            ff("-i", f"in/{c['sound']}", "-vn", "-af", f"aresample=48000,apad=whole_dur={c['seconds']},atrim=0:{c['seconds']}", *wav, out)
        else:
            ff("-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", str(c["seconds"]), *wav, out)
        return out
    sounds = ([sound("open")] if "open" in cards else []) + ["out/body.wav"] + ([sound("end")] if "end" in cards else [])
    whole = facts["whole"]
    ff("-i", "out/video.mp4", *sum((["-i", p] for p in sounds), []), "-filter_complex",
       "".join(f"[{k + 1}:a]" for k in range(len(sounds))) + f"concat=n={len(sounds)}:v=0:a=1[a]", "-map", "0:v", "-map", "[a]", "-c:v", "copy",
       "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-t", str(whole), "out/film.mp4")
    bad = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", "out/film.mp4", "-f", "null", "-"], capture_output=True, text=True).stderr.strip()
    if bad:
        raise SystemExit("STOP: the film does not play cleanly from start to end:\n" + bad[:2000])
    got = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", "out/film.mp4"], capture_output=True, text=True).stdout)
    if got > film.get("open", {}).get("max", 120) + 0.05 or abs(got - whole) > 0.2:
        raise SystemExit(f"STOP: the film came out {got:.2f} s; the plan said {whole} s and the limit is 2:00")
    after = loud("out/body.wav")
    r = subprocess.run(["ffmpeg", "-nostdin", "-i", "out/film.mp4", "-vn", "-af", "silencedetect=noise=-50dB:d=2.5", "-f", "null", "-"], capture_output=True, text=True)
    quiet = [[float(a), float(b)] for a, b in zip(re.findall(r"silence_start: ([\d.]+)", r.stderr), re.findall(r"silence_end: ([\d.]+)", r.stderr))]
    lead = facts.get("lead", 0.0)
    report = {"seconds": round(got, 3), "open_seconds": lead, "body_seconds": total, "end_card_seconds": facts.get("tail", 0.0),
              "open": cards.get("open", {}).get("src"), "open_lufs": float(loud("out/open.wav")["input_i"]) if "open" in cards else None,
              "body_gain_db": round(gain, 2), "body_lufs": float(after["input_i"]), "body_true_peak": float(after["input_tp"]),
              "planned_silence": [round(lead + q0, 3), round(lead + q1, 3)], "silences_over_2.5s": quiet, "shot1": facts["shot1"], "match": facts.get("match")}
    Path("out/report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "plan":
        plan(Path(sys.argv[2]).resolve(), use_open="--no-open" not in sys.argv)
    elif len(sys.argv) >= 2 and sys.argv[1] == "render":
        render(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
