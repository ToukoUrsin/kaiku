#!/usr/bin/python
"""Cut the film from the takes, the voice and film.json.

    /usr/bin/python film/edit.py plan WORK      here: checks, cards, and everything the render needs in WORK/edit
    python3 edit.py render                      in WORK/edit, inside a drun sandbox: ffmpeg only

plan stops the build when the film would say something the page did not: every number that is spoken
or shown is compared with what the page itself said in that take (CLIP.says.jsonl), the source line
must say what is really driving, and no frame may show a broken page.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 1920, 1080
LEFT, PANEL = 1138, 782          # shots 6 and 8: the stage beside the page's own panel


# ---------------------------------------------------------------- shared

def says(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stop(why: str) -> None:
    raise SystemExit("STOP: " + why)


def frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip().strip(","))


# ---------------------------------------------------------------- plan (on the box)

CARD_CSS = """*{margin:0;box-sizing:border-box}html,body{width:1920px;height:1080px;background:BG;color:#fff;
font-family:Inter,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.c{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;padding:0 160px}
.s{font-size:38px;color:#9a9da3;font-weight:500;letter-spacing:-.01em}
.b{font-size:150px;font-weight:700;letter-spacing:-.04em;line-height:1.08}
.m{font:500 26px 'DejaVu Sans Mono',monospace;color:#6f737a;margin-top:56px;white-space:pre-wrap}
.t{font-size:132px;font-weight:700;letter-spacing:-.04em}
.a{font-size:84px;font-weight:600;letter-spacing:-.03em;margin-top:28px}
.pill{position:absolute;left:40px;top:40px;font-size:30px;font-weight:600;padding:12px 24px;border-radius:999px;
background:rgba(12,13,15,.82);color:#fff}"""


def cards(film: dict, facts: dict, out: Path) -> None:
    from playwright.sync_api import sync_playwright
    g1 = facts["g1"]
    card = next(s for s in film["shots"] if s.get("card"))["card"]
    card = {k: v.format(**{a: b for a, b in film["numbers"].items() if a != "about"}) for k, v in card.items()}
    pages = {
        "card-log": ("#000", f"""<div class=c><div class=s>The humanoid challenge's own runner and safety filter. {g1['seconds']} seconds, mock robot.</div>
<div class=b style="margin-top:40px">{g1['commands']} commands</div><div class=b>{g1['clipped']} clipped</div>
<div class=m>{g1['line']}</div></div>"""),
        "card-address": ("#000", f"""<div class=c style="align-items:center;text-align:center"><div class=t>{card['title']}</div>
<div class=s style="margin-top:18px">{card['line']}</div><div class=a>{card['address']}</div></div>"""),
        "card-speed": ("transparent", "<div class=pill>2× speed</div>"),
    }
    with sync_playwright() as p:
        chrome = next((c for c in ("/usr/bin/google-chrome", "/usr/bin/google-chrome-stable") if os.path.exists(c)), None)
        browser = p.chromium.launch(headless=True, **({"executable_path": chrome} if chrome else {}))
        page = browser.new_context(viewport={"width": W, "height": H}).new_page()
        for name, (bg, body) in pages.items():
            page.set_content(f"<style>{CARD_CSS.replace('BG', bg)}</style>{body}")
            page.wait_for_timeout(150)
            page.screenshot(path=str(out / f"{name}.png"), omit_background=bg == "transparent")
        browser.close()


def plan(work: Path) -> None:
    sys.path.insert(0, str(HERE))
    from vo import line_key, spoken

    film = json.loads((HERE / "film.json").read_text())
    fps, clips, takes = film["fps"], work / "clips", film["takes"]["clips"]
    n = film["numbers"]
    kind = (clips / "shot2.kind").read_text().strip() if (clips / "shot2.kind").exists() else "standin"
    facts: dict = {"shot2": kind, "shots": {}, "checks": []}
    ok = facts["checks"].append

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

    # ---- what each take said
    used = ["1-all", "3-env1", "3-env7", "3-env19", "4a", "4b", "5a", "5b", "6-page", "6-left", "7-nobody", "8-left"]
    said = {name: page_said(name) for name in used}
    if kind == "standin":
        rows = page_said("shot2-standin")
        if not all(re.search(r"scripted hand", r.get("caption", ""), re.I) and "emulated" in r.get("caption", "") for r in rows):
            stop("shot 2 stand-in: the page's source line does not say a scripted hand in emulated gloves is driving")
        ok(f"shot 2 source line, every half second: {rows[0]['caption']!r}")

    def need(cond: bool, why: str) -> None:
        if not cond:
            stop(why)
        ok(why.split(" BUT ")[0])

    need(all(r.get("robots") == str(n["fleet_others"]) and "scripted hand" in r.get("caption", "").lower() for r in said["1-all"]),
         f"shot 1: the page says a scripted hand drives and {n['fleet_others']} robots repeat it BUT a row of 1-all.says.jsonl differs")

    # ---- shot 4 and 5: the counts are the page's own banner and headline
    a4, a5 = frames(clips / "4a.mp4") / fps, frames(clips / "5a.mp4") / fps
    over = next((r for r in said["4b"] if (r.get("banner-text") or "").startswith("Without Kaiku,")), None)
    if not over:
        stop("shot 4: the page never said how many robots sorted their table without Kaiku")
    m = re.match(r"Without Kaiku, (\d+) of (\d+) robots sorted their table", over["banner-text"])
    without, robots = int(m[1]), int(m[2])
    t4 = a4 + (over["t"] - takes["4b"]["start"]) / takes["4b"].get("speed", 1)
    done = next((r for r in said["5b"] if (r.get("banner-text") or "").startswith("With Kaiku,")), None)
    if not done:
        stop("shot 5: the page never said how many robots sorted their table with Kaiku")
    m = re.match(r"With Kaiku, (\d+) of (\d+) robots sorted their table\. Without: (\d+)", done["banner-text"])
    need(int(m[3]) == without and int(m[2]) == robots, f"shots 4 and 5 are the same {robots} tables: without Kaiku {without}, with Kaiku {m[1]} BUT the two takes disagree")
    need(int(m[1]) > without, f"shot 5: with Kaiku {m[1]} of {robots} sorted BUT that is not more than without ({without})")
    facts["shots"]["4-without"] = {"a": a4, "result_at": round(t4, 3), "said": over["banner-text"]}
    facts["shots"]["5-with"] = {"a": a5, "said": done["banner-text"]}

    # ---- shot 6: the study numbers spoken are the ones on the page's panel
    finding = says(clips / "6-panel.png.says.jsonl")[0].get("finding", "")
    for piece in (f"From {n['study_demos']} demonstrations", f"{n['study_with_echoes']} with echoes", f"{n['study_few']} with echoes beat {n['study_many']} without"):
        need(piece in finding, f"shot 6: the page's panel says {piece!r} BUT the panel in this take says {finding!r}. "
             "Change film.json numbers, or record the page again (film/README.md)")

    # ---- shot 7: nobody driving
    need(all("Nobody is driving" in r.get("caption", "") and r.get("robots") == str(n["fleet"]) and r.get("chain-count") == "0" for r in said["7-nobody"]),
         f"shot 7: the page says nobody is driving {n['fleet']} robots BUT a row of 7-nobody.says.jsonl differs")
    facts["shots"]["7-nobody"] = {"success": said["7-nobody"][-1].get("amplification"), "sorted": said["7-nobody"][-1].get("demos")}

    # ---- shot 8: the runner's own numbers
    g1 = json.loads((work / "g1" / "summary.json").read_text())
    line = (work / "g1" / "runner.log").read_text().strip().splitlines()[-1]
    need(g1["safety"]["mode"] == "ok" and g1["safety"]["steps"] == g1["ticks"], "shot 8: every command of the run went through the safety filter BUT the runner's summary says otherwise")
    facts["g1"] = {"commands": f"{g1['safety']['steps']:,}", "clipped": f"{g1['safety']['clipped_steps']:,}", "seconds": round(g1["wall_s"]), "line": line}

    # ---- voice
    t, total, vo, srt = 0.0, sum(s["dur"] for s in film["shots"]), [], []
    for shot in film["shots"]:
        lines = shot.get("vo_real", shot["vo"]) if shot.get("swappable") and kind == "real" else shot.get("vo", [])
        for ln in lines:
            text = spoken(film, ln["text"])
            if re.search(r"glove", text, re.I) and not re.search(r"emulated|Nobody is wearing", text):
                stop(f"voice line says gloves without 'emulated': {text!r}")
            key = line_key(text, film["voice"])
            meta = json.loads((work / "vo" / f"{key}.json").read_text())
            if ln["at"] + meta["dur"] > shot["dur"] - 0.15:
                stop(f"{shot['id']}: the line is {meta['dur']:.1f} s and does not fit in the shot: {text!r}")
            vo.append({"file": f"{key}.mp3", "at": round(t + ln["at"], 3), "dur": meta["dur"], "shot": shot["id"], "text": text, "words": meta["words"]})
        facts["shots"].setdefault(shot["id"], {})["start"] = t
        t += shot["dur"]
    if total > 120:
        stop(f"the film is {total} s, over two minutes")
    facts.update(vo=vo, total=total)
    # shot 8 cuts from the runner's numbers to the dataset where the voice gets to "The clean ones"
    v8 = next(v for v in vo if v["shot"] == "8-checked")
    i = max(k for k, w in enumerate(v8["words"]) if w["w"][0].isupper())
    facts["shots"]["8-checked"]["cut"] = round(v8["at"] - facts["shots"]["8-checked"]["start"] + v8["words"][i]["s"] - 0.2, 3)
    # the planned silence: three seconds from the moment the page shows the count
    q = facts["shots"]["4-without"]["start"] + facts["shots"]["4-without"]["result_at"]
    facts["quiet"] = [round(q, 3), round(q + 3.0, 3)]
    for v in vo:                                             # sidecar captions, one per sentence
        start = None
        for k, w in enumerate(v["words"]):
            start = w["s"] if start is None else start
            if re.search(r"[.!?:]$", w["w"]) or k == len(v["words"]) - 1:
                srt.append((v["at"] + start, v["at"] + w["e"], " ".join(x["w"] for x in v["words"] if start <= x["s"] <= w["s"])))
                start = None

    # ---- stage everything the render needs
    edit = work / "edit"
    if edit.exists():
        shutil.rmtree(edit)
    (edit / "in").mkdir(parents=True)
    (edit / "out").mkdir()
    cards(film, facts, edit / "in")
    need_files = [f"{x}.mp4" for x in used if x != "6-page"] + ["6-page.mp4", "shot2.mp4", "6-panel.png", "8-panel.png"]
    for name in need_files:
        os.link(clips / name, edit / "in" / name)
    for v in vo:
        if not (edit / "in" / v["file"]).exists():
            os.link(work / "vo" / v["file"], edit / "in" / v["file"])
    os.link(work / "vo" / "music.mp3", edit / "in" / "music.mp3")
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
    print(f"film {total:g} s, shot 2 is the {kind} clip, silence {facts['quiet'][0]:g} to {facts['quiet'][1]:g} s; ready in {edit}")


# ---------------------------------------------------------------- render (in the sandbox)

def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", *args], check=True)


def render() -> None:
    film, facts = json.loads(Path("film.json").read_text()), json.loads(Path("facts.json").read_text())
    fps, sf = film["fps"], facts["shots"]
    x264 = ["-an", "-r", str(fps), "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "15360"]
    end = "fps=30,format=yuv420p,setsar=1"
    hold = "tpad=stop_mode=clone:stop_duration=4"
    parts = []
    for i, shot in enumerate(film["shots"]):
        sid, dur = shot["id"], shot["dur"]
        out, f = f"out/s{i + 1}.mp4", sf[shot["id"]]
        I = lambda *names: [x for name in names for x in ("-i", f"in/{name}")]
        if sid == "1-all":
            ins, g = I("1-all.mp4"), f"[0:v]{hold},fade=t=in:st=0:d=0.5,{end}[v]"
        elif sid == "2-hand":
            ins, g = I("shot2.mp4"), f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,{hold},{end}[v]"
        elif sid == "3-tables":
            ins = I(*shot["clip"])
            g = (f"[0:v][1:v][2:v]hstack=inputs=3,drawbox=x=638:y=0:w=4:h={H}:c=black:t=fill,"
                 f"drawbox=x=1278:y=0:w=4:h={H}:c=black:t=fill,{hold},{end}[v]")
        elif sid in ("4-without", "5-with"):
            a, b = ("4a.mp4", "4b.mp4") if sid == "4-without" else ("5a.mp4", "5b.mp4")
            ins = I(a, b, "card-speed.png")
            g = f"[0:v][1:v]concat=n=2:v=1,{hold}[c];[c][2:v]overlay=0:0:enable='lt(t,{f['a']:.3f})',{end}[v]"
        elif sid == "6-policy-tab":
            lead = 1.8                                       # the whole page first, then its panel beside the stage
            ins = I("6-page.mp4", "6-left.mp4", "6-panel.png")
            g = (f"[1:v]{hold}[l];[2:v]scale={PANEL}:{H}[p];[l][p]hstack=inputs=2,trim=0:{dur - lead},setpts=PTS-STARTPTS,fps=30[b];"
                 f"[0:v]trim=0:{lead},setpts=PTS-STARTPTS,fps=30[a];[a][b]concat=n=2:v=1,{end}[v]")
        elif sid == "7-nobody":
            ins, g = I("7-nobody.mp4"), f"[0:v]{hold},{end}[v]"
        elif sid == "8-checked":
            cut = f["cut"]
            ins = ["-loop", "1", "-framerate", "30", "-t", f"{cut:.3f}", "-i", "in/card-log.png"] + I("8-left.mp4", "8-panel.png")
            g = (f"[1:v]{hold}[l];[2:v]scale={PANEL}:{H}[p];[l][p]hstack=inputs=2,trim=0:{dur - cut:.3f},setpts=PTS-STARTPTS,fps=30[b];"
                 f"[0:v]fps=30,format=yuv420p[a];[a][b]concat=n=2:v=1,{end}[v]")
        elif sid == "9-address":
            ins = ["-loop", "1", "-framerate", "30", "-t", str(dur + 1), "-i", "in/card-address.png"]
            g = f"[0:v]fade=t=out:st={dur - 0.8}:d=0.8,{end}[v]"
        else:
            raise SystemExit(f"no recipe for shot {sid}")
        ff(*ins, "-filter_complex", g, "-map", "[v]", "-frames:v", str(int(round(dur * fps))), *x264, out)
        parts.append(out)
        print("shot", sid, flush=True)
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
              "planned_silence": [q0, q1], "silences_over_2.5s": quiet}
    Path("out/report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "plan":
        plan(Path(sys.argv[2]).resolve())
    elif len(sys.argv) >= 2 and sys.argv[1] == "render":
        render()
    else:
        raise SystemExit(__doc__)
