#!/usr/bin/python
"""Cut the film from the takes, the voice and film.json.

    /usr/bin/python film/edit.py plan WORK      here: checks, the film's own type, and everything the render needs in WORK/edit
    python3 edit.py render                      in WORK/edit, inside a drun sandbox: ffmpeg only

plan stops the build when the film would say something that is not so: every number spoken or shown is
compared with what the page itself said in that take (CLIP.says.jsonl) and with the checked-in files
(study.json, baselines.json, the runner's own summary); the page's source line must say what is really
driving; no frame may show a broken page.

Shot 1 without a real clip is the hand film: the page's own camera picture (film/hand.py saved it with
the time of every picture) beside the robot's hand from the same run's stream. The picture shown in a
film frame is the last one the page had drawn by the moment that frame's robot state arrived, on one
clock. Nothing is shifted to look better.
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
LEFT, PANEL = 1138, 782          # shots 5, 6 and 9: the stage beside the page's own panel
TERM, ROBOT = 1180, 740          # shot 8: the runner's log beside the robot it follows
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


# ---------------------------------------------------------------- the film's own type (plan, on the box)

CSS = """*{margin:0;box-sizing:border-box}html,body{width:1920px;height:1080px;background:transparent;color:#fff;overflow:hidden;
font-family:Inter,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.scrim{position:absolute;left:0;top:0;height:230px;background:linear-gradient(180deg,rgba(8,9,11,.9),rgba(8,9,11,.55) 52%,rgba(8,9,11,0))}
.head{position:absolute;left:56px;top:38px;font-size:54px;font-weight:700;letter-spacing:-.028em;line-height:1.12;text-wrap:balance;
text-shadow:0 2px 18px rgba(0,0,0,.75)}
.head.col{font-size:46px;max-width:1026px}
.sub{font-size:31px;font-weight:500;letter-spacing:-.01em;color:#dfe2e6;margin-top:12px}
.mark{position:absolute;right:56px;top:44px;font-size:27px;font-weight:600;padding:10px 22px 10px 20px;border-radius:999px;
background:rgba(12,13,15,.88);border:1.5px solid rgba(255,255,255,.3);display:flex;gap:12px;align-items:center}
.mark i{width:12px;height:12px;border-radius:50%;background:#5ee08f}
.note{position:absolute;left:56px;bottom:44px;max-width:1026px;font-size:30px;line-height:1.34;color:#d5d8dd;font-weight:500;
text-shadow:0 1px 10px rgba(0,0,0,.8)}
.big{position:absolute;top:262px}.big b{display:block;font-size:214px;font-weight:750;letter-spacing:-.05em;line-height:1;
text-shadow:0 4px 30px rgba(0,0,0,.6)}
.big span{display:block;font-size:38px;font-weight:600;margin-top:8px;color:#eceef0;text-shadow:0 1px 10px rgba(0,0,0,.8)}
.then{position:absolute;left:56px;top:640px;padding-left:26px;border-left:8px solid #5ee08f}
.then b{display:block;font-size:62px;letter-spacing:-.03em;line-height:1.1}.then span{display:block;font-size:34px;color:#d5d8dd;margin-top:12px;font-weight:500}
.ring{position:absolute;border:2px solid rgba(255,255,255,.45);border-radius:20px;box-shadow:0 0 0 18px rgba(8,9,11,.8)}
.under{position:absolute;background:rgba(8,9,11,.8);border-radius:0 0 26px 26px;padding:10px 20px 20px;font-size:30px;font-weight:600;
line-height:1.28;letter-spacing:-.012em}
.under small{display:block;font-size:26px;font-weight:500;color:#b9bdc4;margin-top:8px}
.pill{position:absolute;font-size:28px;font-weight:600;padding:11px 22px;border-radius:14px;background:rgba(8,9,11,.8)}
.band{position:absolute;left:0;right:0;top:296px;height:488px;background:rgba(8,9,11,.88);display:flex;flex-direction:column;
align-items:center;justify-content:center;text-align:center}
.band .l{font-size:46px;font-weight:600;color:#dfe2e6}.band .a{font-size:150px;font-weight:750;letter-spacing:-.045em;line-height:1.12;margin-top:6px}
.band .c{font-size:25px;color:#b0b5bc;margin-top:28px;font-weight:500}
.term{position:absolute;left:36px;top:186px;width:1108px;height:858px;border-radius:16px;background:#0b0c0e;border:1.5px solid #2a2d33;overflow:hidden}
.term .bar{height:46px;background:#17191d;border-bottom:1.5px solid #2a2d33;display:flex;align-items:center;gap:9px;padding:0 18px;
font:500 20px Inter,sans-serif;color:#8d929a}.term .bar i{width:13px;height:13px;border-radius:50%;background:#3a3e45}.term .bar span{margin-left:10px}
.term .rows{position:absolute;left:0;right:0;bottom:0;top:46px;padding:18px 22px;display:flex;flex-direction:column-reverse;overflow:hidden;
  -webkit-mask-image:linear-gradient(to bottom,transparent 0,transparent 14px,#000 64px)}
.term .rows>div{margin-bottom:auto;flex:none}
.count{position:absolute;top:40px;padding:16px 24px 18px;border-radius:16px;background:rgba(8,9,11,.86);border:1.5px solid rgba(94,224,143,.55)}
.count b{display:block;font-size:64px;font-weight:750;letter-spacing:-.035em;line-height:1.05;color:#5ee08f}
.count span{display:block;font-size:29px;font-weight:600;margin-top:6px}
.term p{font:22px/1.42 'DejaVu Sans Mono',monospace;color:#aab0b8;white-space:pre-wrap;overflow-wrap:anywhere;flex:none}
.term p.cmd{color:#fff}
.term .d{color:#646a72}.term .hi{color:#7ef0a8;font-weight:700;background:rgba(94,224,143,.14);border-radius:4px}"""


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


def head(text: str, sub: str = "", mark: str = "", width: int = W, col: bool = False) -> str:
    e = html.escape
    return (f"<div class=scrim style='width:{width}px'></div><div class='head{' col' if col else ''}' style='max-width:{width - 112 - (260 if mark else 0)}px'>{e(text)}"
            + (f"<div class=sub>{e(sub)}</div>" if sub else "") + "</div>"
            + (f"<div class=mark><i></i>{e(mark)}</div>" if mark else ""))


def big(value: str, label: str, left: int, colour: str) -> str:
    return f"<div class=big style='left:{left}px;color:{colour}'><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>"


def terminal(lines: list[tuple[str, str]], title: str) -> str:
    """lines: (kind, text). The runner's own lines, with the numbers the voice names picked out."""
    rows = []
    for kind, text in lines:
        t = html.escape(text, quote=False)
        t = re.sub(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)", r"<span class=d>\1</span>", t)
        t = re.sub(r"('steps': \d+, 'clipped_steps': \d+, 'mode': '\w+')", r"<span class=hi>\1</span>", t)
        t = re.sub(r"('ticks': \d+)", r"<span class=hi>\1</span>", t)
        t = re.sub(r"(--backend mock)", r"<span class=hi>\1</span>", t)
        rows.append(f"<p class='{kind}'>{t}</p>")
    return (f"<div style='position:absolute;inset:0;background:#0f1012'></div><div class=term><div class=bar><i></i><i></i><i></i><span>{html.escape(title)}</span></div>"
            f"<div class=rows><div>{''.join(rows)}</div></div></div>")


# ---------------------------------------------------------------- plan (on the box)

def plan(work: Path) -> None:
    sys.path.insert(0, str(HERE))
    from vo import line_key, spoken

    film = json.loads((HERE / "film.json").read_text())
    fps, clips, takes = film["fps"], work / "clips", film["takes"]
    n = film["numbers"]
    fill = lambda text: text.format(**{k: v for k, v in n.items() if k != "about"})
    kind = (clips / "shot2.kind").read_text().strip() if (clips / "shot2.kind").exists() else "standin"
    facts: dict = {"shot1": kind, "shots": {}, "checks": []}
    ok = facts["checks"].append
    shots = {s["id"]: s for s in film["shots"]}

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

    # ---- every word the film itself says or shows: no gloves, no long dashes
    for s in film["shots"]:
        for key in ("head", "head_real", "sub", "label", "label_robot", "note", "then", "then_sub"):
            if re.search(r"[\u2013\u2014]|glove", s.get(key, ""), re.I):
                stop(f"{s['id']}: {key} has a long dash or claims gloves: {s[key]!r}")
        for ln in s.get("vo", []) + s.get("vo_real", []):
            if re.search(r"[\u2013\u2014]|glove", ln["text"], re.I):
                stop(f"{s['id']}: the voice has a long dash or claims gloves: {ln['text']!r}")

    # ---- voice: where every line and every word falls
    t, total, vo, srt = 0.0, sum(s["dur"] for s in film["shots"]), [], []
    for shot in film["shots"]:
        lines = shot.get("vo_real", shot["vo"]) if shot.get("swappable") and kind == "real" else shot.get("vo", [])
        for ln in lines:
            text = spoken(film, ln["text"])
            key = line_key(text, film["voice"])
            meta = json.loads((work / "vo" / f"{key}.json").read_text())
            if ln["at"] + meta["dur"] > shot["dur"] - 0.15:
                stop(f"{shot['id']}: the line is {meta['dur']:.1f} s and does not fit in the shot: {text!r}")
            vo.append({"file": f"{key}.mp3", "at": round(t + ln["at"], 3), "dur": meta["dur"], "shot": shot["id"], "text": text, "words": meta["words"]})
        facts["shots"][shot["id"]] = {"start": t}
        t += shot["dur"]
    if total > 115:
        stop(f"the film is {total} s; the brief says at most about 1:55")
    facts.update(vo=vo, total=total)

    def cue(sid: str, words: str) -> float:
        """Seconds into the shot at which the voice starts saying `words`."""
        first = plain(fill(words).split()[0])
        v = next(v for v in vo if v["shot"] == sid)
        for w in v["words"]:
            if plain(w["w"]) == first and (not fill(words)[0].isupper() or w["w"][0].isupper()):
                return round(v["at"] - facts["shots"][sid]["start"] + w["s"], 3)
        stop(f"{sid}: the voice never says {words!r}")

    for v in vo:                                             # sidecar captions, one per sentence
        start = None
        for k, w in enumerate(v["words"]):
            start = w["s"] if start is None else start
            if re.search(r"[.!?:]$", w["w"]) or k == len(v["words"]) - 1:
                srt.append((v["at"] + start, v["at"] + w["e"], " ".join(x["w"] for x in v["words"] if start <= x["s"] <= w["s"])))
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
    still = lambda name, dur: ["-loop", "1", "-framerate", str(fps), "-t", f"{dur + 1:.3f}", "-i", f"in/{name}.png"]
    used: list[str] = []

    # ---- shot 1: the hand
    s, sid = shots["1-hand"], "1-hand"
    if kind == "real":
        used.append("shot2.mp4")
        recipe[sid] = {"inputs": ["-i", "in/shot2.mp4"], "n": 1, "over": [{"file": "ov-1", "a": 0, "b": s["dur"]}],
                       "graph": f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,{hold}[base]"}
        pages["ov-1"] = (full, head(s["head_real"]))
        ok("shot 1 is a clip of a real hand, given with --shot2")
    else:
        take = (work / takes["hand"]["take"]).resolve()
        t0 = json.loads((take / "rec" / "recording.json").read_text())["t0_epoch"]
        pics = sorted(says(take / "inset.jsonl"), key=lambda r: r["e"])
        reads = says(take / "hand.jsonl")
        k, at, worst = 0, 0, 0.0
        for part in ("1-robot-a", "1-robot-b"):
            c = takes["clips"][part]
            rows = page_said(part)
            need(all("hand, seen by a camera" in r.get("caption", "") for r in rows),
                 f"shot 1, {part}: the page's own line says a hand seen by a camera drives BUT a row of {part}.says.jsonl differs")
            mine = [r for r in reads if c["start"] <= r["e"] / 1000 - t0 <= c["end"]]
            need(bool(mine) and all(r["drive"] == "camera" and r["fingers"] == "camera" for r in mine),
                 f"shot 1, {part}: the runner said the camera drove the arm and the fingers for every camera frame BUT hand.jsonl differs")
            seen = sum(r["seen"] for r in mine) / len(mine)
            need(seen > 0.9, f"shot 1, {part}: the page saw the hand in {seen:.0%} of {len(mine)} camera frames BUT that is too few")
            for f in range(frames(clips / f"{part}.mp4")):
                want = t0 + c["start"] + f / fps                # when this frame's robot state reached the recorder
                while k + 1 < len(pics) and pics[k + 1]["e"] <= want:
                    k += 1
                worst = max(worst, abs(want - pics[k]["e"]))
                os.link(take / "inset" / f"{pics[k]['n']:06d}.jpg", ins / "inset" / f"{at:04d}.jpg")
                at += 1
            used.append(f"{part}.mp4")
        need(at == int(round(s["dur"] * fps)), f"shot 1: the two angles are {at} frames, the shot's {s['dur']} s BUT they differ")
        need(worst < 0.12, f"shot 1: every film frame has a camera picture drawn within {worst * 1000:.0f} ms before it BUT the gap is too long")
        from PIL import Image, ImageDraw
        x, y, w, h = INSET
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=20, fill=255)
        mask.save(ins / "mask.png")
        e = html.escape
        pages["ov-1"] = (full, head(s["head"]) + f"<div class=ring style='left:{x}px;top:{y}px;width:{w}px;height:{h}px'></div>"
                         f"<div class=under style='left:{x - 18}px;top:{y + h + 17}px;width:{w + 36}px'>{e(s['label'])}<small>{e(n['hand_credit'])}</small></div>"
                         f"<div class=pill style='right:56px;top:150px'>{e(s['label_robot'])}</div>")
        recipe[sid] = {"inputs": clip("1-robot-a") + clip("1-robot-b") + ["-framerate", str(fps), "-i", "in/inset/%04d.jpg", "-loop", "1", "-framerate", str(fps), "-t", "30", "-i", "in/mask.png"],
                       "n": 4, "over": [{"file": "ov-1", "a": 0, "b": s["dur"]}],
                       "graph": f"[0:v][1:v]concat=n=2:v=1,{hold}[r];[2:v]crop=956:716:0:0,scale={w}:{h},format=rgba[i];[3:v]format=gray[m];"
                                f"[i][m]alphamerge[im];[r][im]overlay={x}:{y}:eof_action=repeat[base]"}
    recipe[sid]["fade_in"] = 0.4

    # ---- shot 2: one driven, the rest echo
    s, sid = shots["2-fleet"], "2-fleet"
    rows = page_said("2-fleet")
    need(all(r.get("robots") == str(n["fleet_others"]) and "scripted hand" in r.get("caption", "").lower() and r.get("chain-count") == "1"
             and r.get("chain-who") == "robot driven" for r in rows),
         f"shot 2: the page says a scripted hand drives 1 robot and {n['fleet_others']} echoes follow BUT a row of 2-fleet.says.jsonl differs")
    pages["ov-2"] = (full, head(fill(s["head"]), s["sub"]))
    recipe[sid] = {"inputs": clip("2-fleet"), "n": 1, "graph": f"[0:v]{hold}[base]", "over": [{"file": "ov-2", "a": 0, "b": s["dur"]}]}
    used.append("2-fleet.mp4")

    # ---- shots 3 and 4: the same tables without Kaiku and with it, at the speed they ran
    rows3, rows4 = page_said("3-without"), page_said("4-with")
    for name in ("3-without", "4-with", "7-nobody"):
        need(takes["clips"][name].get("speed", 1) == 1, f"{name} is filmed at the speed it ran BUT film.json gives it a speed")
    over = next((r for r in rows3 if (r.get("banner-text") or "").startswith("Without Kaiku,")), None)
    if not over:
        stop("shot 3: the page never said how many robots sorted their table without Kaiku")
    m = re.match(r"Without Kaiku, (\d+) of (\d+) robots sorted their table", over["banner-text"])
    without, robots = int(m[1]), int(m[2])
    done = next((r for r in rows4 if (r.get("banner-text") or "").startswith("With Kaiku,")), None)
    if not done:
        stop("shot 4: the page never said how many robots sorted their table with Kaiku")
    m = re.match(r"With Kaiku, (\d+) of (\d+) robots sorted their table\. Without: (\d+)", done["banner-text"])
    need(int(m[3]) == without and int(m[2]) == robots == n["fleet_others"],
         f"shots 3 and 4 are the same {robots} tables: without Kaiku {without}, with Kaiku {m[1]} BUT the two takes disagree")
    need(int(m[1]) > without, f"shot 4: with Kaiku {m[1]} of {robots} sorted BUT that is not more than without ({without})")
    line = next((r for r in rows4 if r.get("opening")), None)
    need(bool(line) and re.fullmatch(rf"Without Kaiku: {without} of {robots} tables sorted\. With Kaiku: {m[1]} of {robots}\.", line["opening"]) is not None,
         f"shot 4: the page says its own result over the scene ({line and line['opening']!r}) BUT it is missing or differs from its banner")
    need(all("scripted hand" in r.get("caption", "").lower() for r in rows3 + rows4), "shots 3 and 4: the page's source line says a scripted hand drives BUT a row differs")
    r3 = over["t"] - takes["clips"]["3-without"]["start"] - 0.25                  # the page is read twice a second
    r4 = line["t"] - takes["clips"]["4-with"]["start"] - 0.25
    facts["shots"]["3-without"].update(result_at=round(r3, 3), said=over["banner-text"])
    facts["shots"]["4-with"].update(result_at=round(r4, 3), said=line["opening"])
    for sid, until in (("3-without", None), ("4-with", r4 - 0.35)):
        s = shots[sid]
        pages[f"ov-{sid[0]}"] = (full, head(s["head"]))
        pages[f"ov-{sid[0]}m"] = (full, f"<div class=mark><i></i>{html.escape(s['mark'])}</div>")
        recipe[sid] = {"inputs": clip(sid), "n": 1, "graph": f"[0:v]{hold}[base]",
                       "over": [{"file": f"ov-{sid[0]}", "a": 0, "b": until or s["dur"], "out": 0.3 if until else 0},      # the page's own result takes the top of the frame
                                {"file": f"ov-{sid[0]}m", "a": 0, "b": s["dur"]}]}
        used.append(f"{sid}.mp4")

    # ---- shot 5: the study, as the page's panel and the checked-in file have it
    study = json.loads((ROOT / "kaiku/policy/pretrained/study.json").read_text())
    row = {r["demos"]: r for r in study["curve"]}
    pct = lambda x: f"{round(x * 100)}%"
    a, b = row[n["study_demos"]], row[n["study_many"]]
    need(pct(a["plain"]) == n["study_alone"] and pct(a["kaiku"]) == n["study_with_echoes"] and pct(b["plain"]) == n["study_many_alone"] and a["kaiku"] > b["plain"],
         f"shot 5: study.json has {n['study_demos']} demonstrations at {n['study_alone']} alone and {n['study_with_echoes']} with echoes, {n['study_many']} alone at {n['study_many_alone']} "
         f"BUT the file says {pct(a['plain'])}, {pct(a['kaiku'])}, {pct(b['plain'])}")
    need(study["about"]["scenes"] == n["study_scenes"] and study["about"]["runs"] == n["study_runs"] and "stand-in" in study["about"]["operator"],
         f"shot 5: study.json is {n['study_runs']} runs on {n['study_scenes']} unseen scenes, demonstrations from the scripted stand-in BUT the file differs")
    panel = says(clips / "5-panel.png.says.jsonl")[0]
    for key, piece in (("claim-top", f"Trained on {n['study_demos']} demonstrations"), ("claim-main", f"{n['study_alone']} alone, {n['study_with_echoes']} with echoes"),
                       ("claim-more", f"{n['study_demos']} with echoes beat {n['study_many']} without, {n['study_many_alone']}"),
                       ("study-note", f"{n['study_runs']} training runs, tested on {n['study_scenes']} scenes")):
        need(piece in panel.get(key, ""), f"shot 5: the page's panel says {piece!r} BUT the panel in this take says {panel.get(key)!r}. "
             "Change film.json numbers, or record the page again (film/README.md)")
    s, sid = shots["5-study"], "5-study"
    c1, c2, c3 = cue(sid, s["cues"][0]), cue(sid, s["cues"][1]), cue(sid, s["then_cue"])
    pages["ov-5"] = (full, head(fill(s["head"]), width=LEFT, col=True) + f"<div class=note>{html.escape(fill(s['note']))}</div>")
    pages["ov-5a"] = (full, big(fill(s["big"][0][0]), s["big"][0][1], 56, GREY))
    pages["ov-5b"] = (full, big(fill(s["big"][1][0]), s["big"][1][1], 590, GREEN))
    pages["ov-5c"] = (full, f"<div class=then><b>{html.escape(fill(s['then']))}</b><span>{html.escape(fill(s['then_sub']))}</span></div>")

    def beside(sid: str, left: str, pic: str, dim: float, over: list[dict]) -> None:
        from PIL import Image
        r, g, bl = Image.open(clips / f"{pic}.png").convert("RGB").getpixel((4, 4))
        recipe[sid] = {"inputs": clip(left) + still(pic, shots[sid]["dur"]), "n": 2, "over": over,
                       "graph": f"[0:v]{hold},drawbox=x=0:y=0:w={LEFT}:h={H}:color=black@{dim}:t=fill[l];"
                                f"[1:v]scale={PANEL}:{H}:force_original_aspect_ratio=decrease,pad={PANEL}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x{r:02x}{g:02x}{bl:02x},setsar=1[p];"
                                f"[l][p]hstack=inputs=2[base]"}
        used.extend([f"{left}.mp4", f"{pic}.png"])
        page_said(left)

    beside(sid, "5-left", "5-panel", 0.62, [{"file": "ov-5", "a": 0, "b": s["dur"]}, {"file": "ov-5a", "a": c1, "b": s["dur"], "in": 0.25},
                                            {"file": "ov-5b", "a": c2, "b": s["dur"], "in": 0.25}, {"file": "ov-5c", "a": c3, "b": s["dur"], "in": 0.25}])

    # ---- shot 6: the same 16 demonstrations, replayed two ways
    base = json.loads((ROOT / "kaiku/policy/pretrained/baselines.json").read_text())
    arms = base["policy"]["arms"]
    need(pct(arms["mimicgen"]["yield"]) == n["matched_replay"] and pct(arms["kaiku"]["yield"]) == n["matched_kaiku"]
         and base["generation"]["kaiku"]["source_scenes"] == base["generation"]["mimicgen"]["source_scenes"] == n["matched_demos"]
         and base["policy"]["heldout_scenes"] == n["study_scenes"] and base["policy"]["seeds_per_arm"] == n["study_runs"],
         f"shot 6: baselines.json has the same {n['matched_demos']} demonstrations at {n['matched_replay']} with the replay and {n['matched_kaiku']} with Kaiku's echoes, "
         f"{n['study_runs']} runs on {n['study_scenes']} tables BUT the file says {pct(arms['mimicgen']['yield'])} and {pct(arms['kaiku']['yield'])}")
    need("re-implementation" in base["about"]["notes"], "shot 6: baselines.json says the replay is our own re-implementation BUT its notes do not")
    bars = dict(panel.get("bars") or [])
    shown = {name: f"{round(float(v.rstrip('%')))}%" for name, v in bars.items()}
    need(shown.get("With MimicGen-style replays") == n["matched_replay"] and shown.get("With all of Kaiku's echoes") == n["matched_kaiku"]
         and f"another {n['matched_demos']} demonstrations" in panel.get("baselines-lede", "") and "our own MimicGen-style re-implementation" in panel.get("baselines-note", ""),
         f"shot 6: the page's panel shows {n['matched_replay']} with MimicGen-style replays and {n['matched_kaiku']} with Kaiku's echoes, and says the replay is our own "
         f"BUT the panel in this take says {bars!r}")
    s, sid = shots["6-matched"], "6-matched"
    c1, c2 = cue(sid, s["cues"][0]), cue(sid, s["cues"][1])
    pages["ov-6"] = (full, head(fill(s["head"]), width=LEFT, col=True) + f"<div class=note>{html.escape(fill(s['note']))}</div>")
    pages["ov-6a"] = (full, big(fill(s["big"][0][0]), s["big"][0][1], 56, GREY))
    pages["ov-6b"] = (full, big(fill(s["big"][1][0]), s["big"][1][1], 590, GREEN))
    beside(sid, "6-left", "6-panel", 0.62, [{"file": "ov-6", "a": 0, "b": s["dur"]}, {"file": "ov-6a", "a": c1, "b": s["dur"], "in": 0.25},
                                            {"file": "ov-6b", "a": c2, "b": s["dur"], "in": 0.25}])

    # ---- shot 7: nobody drives
    s, sid = shots["7-nobody"], "7-nobody"
    rows = page_said("7-nobody")
    need(all("Nobody is driving" in r.get("caption", "") and r.get("robots") == str(n["fleet"]) and r.get("chain-who") == "Nobody driving" for r in rows),
         f"shot 7: the page says nobody is driving {n['fleet']} robots BUT a row of 7-nobody.says.jsonl differs")
    end = next((r for r in rows if re.match(r"\d+ of \d+ robots sorted their scene with nobody driving", r.get("banner-text") or "")), None)
    need(end is not None, "shot 7: the page says how many robots sorted their scene with nobody driving BUT it never did in this take")
    facts["shots"][sid]["said"] = end["banner-text"]
    pages["ov-7"] = (full, head(s["head"], mark=s["mark"]))
    recipe[sid] = {"inputs": clip("7-nobody"), "n": 1, "graph": f"[0:v]{hold}[base]", "over": [{"file": "ov-7", "a": 0, "b": s["dur"]}]}
    used.append("7-nobody.mp4")

    # ---- shot 8: the challenge's own runner, its own lines at the moments it printed them
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
        timed.append((last, "cmd" if not timed else "", text.replace(str(work), "\u2026")))
    summary = next(t for t, _, text in timed if " episode 0: " in text)
    cut = round((cue(sid, s["cut_cue"]) - 0.2) * fps) / fps
    a_from, b_from = takes["runner"]["a_from"], summary - takes["runner"]["b_lead"]
    ca, cb = takes["clips"]["8-robot-a"], takes["clips"]["8-robot-b"]
    need(ca["start"] == a_from and ca["end"] >= a_from + cut and cb["start"] <= b_from and cb["end"] >= b_from + s["dur"] - cut,
         "shot 8: the robot beside the log is the same seconds of the same run BUT the takes in film.json do not cover them")
    states = [(0.0, "a", a_from)]
    states += [(round(t - a_from, 3), "a", t) for t, _, _ in timed if a_from < t < a_from + cut]
    states += [(cut, "b", b_from)] + [(round(cut + t - b_from, 3), "b", t) for t, _, _ in timed if b_from < t < b_from + s["dur"] - cut]
    states = sorted({st[0]: st for st in states}.values())
    over8 = [{"file": "ov-8", "a": 0, "b": s["dur"]}]
    for i, (at, part, now) in enumerate(states):
        lines = [(k, text) for t, k, text in timed if t <= now + 1e-6]
        pages[f"term-{i:02d}"] = (f"{TERM}x{H}", terminal(lines, "aaltonomous.deploy.runner, mock robot"))
        over8.insert(i, {"file": f"term-{i:02d}", "a": at, "b": states[i + 1][0] if i + 1 < len(states) else s["dur"], "hard": True})
    pages["ov-8"] = (full, head(s["head"], width=TERM, col=True)
                     + f"<div class=pill style='left:{TERM + 24}px;bottom:40px;max-width:{ROBOT - 48}px;font-size:25px;line-height:1.3'>{html.escape(s['label_robot'])}</div>")
    # the film skips the middle of the minute, and says so
    pages["ov-8g"] = (full, f"<div class=pill style='left:{TERM - 36 - 268}px;top:191px;width:260px;padding:5px 0;font-size:22px;border-radius:9px;text-align:center;background:#5ee08f;color:#06130b'>"
                            f"{html.escape(s['later'].format(seconds=round(b_from - a_from - cut)))}</div>")
    over8.append({"file": "ov-8g", "a": cut, "b": cut + 2.6, "out": 0.3})
    need(f"{g1['safety']['steps']:,}" == n["runner_commands_shown"], f"shot 8: the count shown is the runner's own ({n['runner_commands_shown']}) BUT its summary says {g1['safety']['steps']:,}")
    pages["ov-8n"] = (full, f"<div class=count style='left:{TERM + 24}px;width:{ROBOT - 60}px'><b>{html.escape(fill(s['count'][0]))}</b><span>{html.escape(fill(s['count'][1]))}</span></div>")
    over8.append({"file": "ov-8n", "a": cut + takes["runner"]["b_lead"], "b": s["dur"], "in": 0.25})
    recipe[sid] = {"inputs": clip("8-robot-a") + clip("8-robot-b"), "n": 2, "over": over8,
                   "graph": f"[0:v]trim=0:{cut:.4f},setpts=PTS-STARTPTS[a];[1:v]trim=start={b_from - cb['start']:.4f},setpts=PTS-STARTPTS,{hold}[b];[a][b]concat=n=2:v=1[r];"
                            f"color=c=0x0f1012:s={TERM}x{H}:r={fps}[c];[c][r]hstack=inputs=2:shortest=1[base]"}
    facts["shots"][sid].update(cut=cut, a_from=a_from, b_from=round(b_from, 3), states=len(states))
    used.extend(["8-robot-a.mp4", "8-robot-b.mp4"])
    page_said("8-robot-a"), page_said("8-robot-b")

    # ---- shot 9: the dataset
    s, sid = shots["9-dataset"], "9-dataset"
    ds = says(clips / "9-panel.png.says.jsonl")[0]
    need(bool(ds.get("ds-episodes")), "shot 9: the page's Dataset tab shows how many demonstrations are saved BUT it does not in this take")
    pages["ov-9"] = (full, head(s["head"], width=LEFT, col=True))
    beside(sid, "9-left", "9-panel", 0.2, [{"file": "ov-9", "a": 0, "b": s["dur"]}])

    # ---- shot 10: the live page and its address
    s, sid = shots["10-address"], "10-address"
    rows = page_said("10-page")
    need(all(r.get("conn-text") == "Live" for r in rows), "shot 10: the page says it is live BUT a row of 10-page.says.jsonl differs")
    credit = f"<div class=c>{html.escape(n['hand_credit_long'])}</div>" if kind == "standin" else ""
    pages["ov-10"] = (full, f"<div class=band><div class=l>{html.escape(s['card']['line'])}</div><div class=a>{html.escape(fill(s['card']['address']))}</div>{credit}</div>")
    recipe[sid] = {"inputs": clip("10-page"), "n": 1, "graph": f"[0:v]{hold}[base]", "over": [{"file": "ov-10", "a": s["card"]["at"], "b": s["dur"], "in": 0.6}]}
    used.append("10-page.mp4")

    # the planned silence: from the moment the page shows how few sorted their table without Kaiku
    q = facts["shots"]["3-without"]["start"] + facts["shots"]["3-without"]["result_at"]
    facts["quiet"] = [round(q, 3), round(min(q + 2.8, facts["shots"]["4-with"]["start"]), 3)]

    # ---- stage everything the render needs
    draw(pages, ins)
    for name in used:
        os.link(clips / name, ins / name)
    for v in vo:
        if not (ins / v["file"]).exists():
            os.link(work / "vo" / v["file"], ins / v["file"])
    os.link(work / "vo" / "music.mp3", ins / "music.mp3")
    facts["render"] = [{"id": s["id"], "dur": s["dur"], **recipe[s["id"]]} for s in film["shots"]]
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
    print(f"film {total:g} s, shot 1 is the {'real clip' if kind == 'real' else 'hand film'}, silence {facts['quiet'][0]:g} to {facts['quiet'][1]:g} s; ready in {edit}")


# ---------------------------------------------------------------- render (in the sandbox)

def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", *args], check=True)


def render() -> None:
    film, facts = json.loads(Path("film.json").read_text()), json.loads(Path("facts.json").read_text())
    fps = film["fps"]
    x264 = ["-an", "-r", str(fps), "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "15360"]
    parts = []
    for i, r in enumerate(facts["render"]):
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
        ff(*ins, "-filter_complex", ";".join(g), "-map", "[v]", "-frames:v", str(int(round(dur * fps))), *x264, out)
        parts.append(out)
        print("shot", r["id"], flush=True)
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
        r = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-af", extra + "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True)
        return json.loads(r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1])
    gain = -16.0 - float(loud("out/mix.wav")["input_i"])
    fix = f"volume={gain:.2f}dB,alimiter=limit=0.84:level=disabled,"
    ff("-i", "out/video.mp4", "-i", "out/mix.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", fix.rstrip(","),
       "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-t", str(total), "out/film.mp4")
    after = loud("out/film.mp4")
    r = subprocess.run(["ffmpeg", "-nostdin", "-i", "out/film.mp4", "-af", "silencedetect=noise=-50dB:d=2.5", "-f", "null", "-"], capture_output=True, text=True)
    quiet = [[float(a), float(b)] for a, b in zip(re.findall(r"silence_start: ([\d.]+)", r.stderr), re.findall(r"silence_end: ([\d.]+)", r.stderr))]
    report = {"seconds": total, "gain_db": round(gain, 2), "lufs": float(after["input_i"]), "true_peak": float(after["input_tp"]),
              "planned_silence": [q0, q1], "silences_over_2.5s": quiet, "shot1": facts["shot1"]}
    Path("out/report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "plan":
        plan(Path(sys.argv[2]).resolve())
    elif len(sys.argv) >= 2 and sys.argv[1] == "render":
        render()
    else:
        raise SystemExit(__doc__)
