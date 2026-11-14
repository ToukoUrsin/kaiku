"""The pitch deck: deck/kaiku.html and deck/kaiku.pdf, built from the study's own numbers.

    python deck/build.py                # the deck in force: no real gloves were worn
    python deck/build.py --gloves       # deck/kaiku-gloves.*: only after a person drove in real Smartgloves
    python deck/build.py --one-glove    # deck/kaiku-one-glove.*: the same, when it was one glove

Needs playwright with chromium for the PDF.

The chart is drawn from kaiku/policy/pretrained/study.json, so the deck cannot disagree with the study.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "kaiku" / "policy" / "pretrained" / "study.json"
DEMO = "kaiku-echo.fly.dev"
SOURCE = "github.com/ToukoUrsin/kaiku"
W, H = 1920, 1080

# What the deck says about who drove. Nothing says gloves were worn until a real session exists.
WORDS = {
    "none": {
        "title": "One hand,<br>48 robots.",
        "each": "Every one of them is a person at one robot, at one table.",
        "quarter": "The operator works a quarter of the time.",
        "after": "Then nobody drives.",
        "rokoko": "<b>A person's hand drives the fingers,</b> seen by a camera, down the path Rokoko's gloves use. "
                  "Rokoko's driver and hand solver feed that path too, tested with its emulated Smartgloves II. No real gloves were worn.",
    },
    "gloves": {
        "title": "One pair of<br>Rokoko gloves,<br>48 robots.",
        "each": "Every one of them is a person in gloves, at one robot, at one table.",
        "quarter": "The operator wears the gloves a quarter of the time.",
        "after": "Then the gloves come off.",
        "rokoko": "<b>Real Smartgloves on a real person.</b> Rokoko's driver and hand solver read them. Kaiku takes their stream as it is.",
    },
}
WORDS["one-glove"] = {**WORDS["gloves"], "title": "One Rokoko glove,<br>48 robots.",
                      "quarter": "The operator wears the glove a quarter of the time.", "after": "Then the glove comes off.",
                      "rokoko": "<b>One real Smartglove on a real person.</b> Rokoko's driver and hand solver read it. "
                                "Its fingers drive the robot's fingers."}


def pct(share: float) -> str:
    return f"{round(share * 100)}%"


def chart(curve: list[dict]) -> str:
    """Share of unseen scenes sorted against the operator's demonstrations, alone and with echoes."""
    left, right, top, bottom = 110, 1380, 50, 520
    xs = [left + (right - left) * i / (len(curve) - 1) for i in range(len(curve))]

    def y(share: float) -> float:
        return bottom - (bottom - top) * share

    out = [f'<svg class="chart" viewBox="0 0 1620 620" width="1620" height="620" role="img" '
           f'aria-label="Share of new scenes sorted, by number of demonstrations, alone and with echoes">']
    for share in (0.0, 0.5, 1.0):
        cls = "axis" if share == 0 else "grid"
        out.append(f'<line class="{cls}" x1="{left}" x2="{right}" y1="{y(share)}" y2="{y(share)}"/>')
        out.append(f'<text class="tick" x="{left - 24}" y="{y(share) + 9}" text-anchor="end">{pct(share)}</text>')
    for x, point in zip(xs, curve):
        out.append(f'<text class="tick" x="{x}" y="{bottom + 48}" text-anchor="middle">{point["demos"]}</text>')
    out.append(f'<text class="axis-title" x="{right + 36}" y="{bottom + 48}">demonstrations</text>')
    marked = next(i for i, p in enumerate(curve) if p["demos"] == 16)
    out.append(f'<line class="gap" x1="{xs[marked]}" x2="{xs[marked]}" y1="{y(curve[marked]["plain"]) - 14}" '
               f'y2="{y(curve[marked]["kaiku"]) + 14}"/>')
    for series in ("plain", "kaiku"):
        path = " ".join(f'{"M" if i == 0 else "L"}{x:.1f} {y(p[series]):.1f}' for i, (x, p) in enumerate(zip(xs, curve)))
        out.append(f'<path class="series {series}" d="{path}"/>')
        for i, (x, p) in enumerate(zip(xs, curve)):
            out.append(f'<circle class="dot {series}" cx="{x:.1f}" cy="{y(p[series]):.1f}" r="{11 if i == marked else 7}"/>')
    k, p = curve[marked]["kaiku"], curve[marked]["plain"]
    out.append(f'<text class="value" x="{xs[marked] - 26}" y="{y(k) - 8}" text-anchor="end">{pct(k)}</text>')
    out.append(f'<text class="value quiet" x="{xs[marked] - 26}" y="{y(p) - 26}" text-anchor="end">{pct(p)}</text>')
    last = curve[-1]
    out.append(f'<text class="name" x="{right + 36}" y="{y(last["kaiku"]) - 4}">with echoes</text>')
    out.append(f'<text class="name quiet" x="{right + 36}" y="{y(last["plain"]) + 30}">operator alone</text>')
    out.append("</svg>")
    return "\n".join(out)


def slides(study: dict, words: dict) -> list[str]:
    curve = [p for p in study["curve"] if p["demos"] >= 2]
    by = {p["demos"]: p for p in study["curve"]}
    about = study["about"]
    return [
        f"""<section class="slide cover" style="background-image:linear-gradient(90deg,#0c0d10 0%,#0c0d10f2 26%,#0c0d1000 56%),url(img/lead.jpg)">
  <div class="mark">Kaiku</div>
  <h1>{words["title"]}</h1>
  <p class="foot">Junction 2026 · The Robot Teleoperation Challenge by Rokoko</p>
</section>""",
        f"""<section class="slide text">
  <p class="kicker">The problem</p>
  <h2>A robot learns a job by watching a person do it.</h2>
  <p class="lede">Sorting parts took our robot 128 demonstrations to get to {pct(by[128]["plain"])}.<br>
  {words["each"]}</p>
</section>""",
        """<section class="slide picture" style="background-image:linear-gradient(180deg,#0c0d10e6 0%,#0c0d1099 16%,#0c0d1000 30%),url(img/overview.jpg)">
  <div class="caption">
    <h2>Sort the parts once.</h2>
    <p class="lede">47 more robots sort theirs at the same moment, each at a table of its own.</p>
  </div>
</section>""",
        """<section class="slide text">
  <p class="kicker">Kaiku is Finnish for echo</p>
  <h2>It is not a copy.</h2>
  <ol class="points">
    <li><b>Every table is different.</b> The parts lie somewhere else, turned another way, sometimes another shape.</li>
    <li><b>Each robot redoes your motion</b> around its own parts, while you are still moving.</li>
    <li><b>Physics checks every robot,</b> and so does the challenge's safety filter. Only the clean runs are saved, in the challenge's dataset format.</li>
  </ol>
</section>""",
        f"""<section class="slide figure">
  <h2>16 demonstrations: {pct(by[16]["plain"])} alone, {pct(by[16]["kaiku"])} with echoes.</h2>
  <p class="lede">32 with echoes beat 128 without. {words["quarter"]}</p>
  {chart(curve)}
  <p class="note">Share of {about["scenes"]} scenes nobody had seen in which every part ended in its own bin. Mean of {about["runs"]} training runs.</p>
</section>""",
        f"""<section class="slide picture" style="background-image:linear-gradient(180deg,#0c0d10e6 0%,#0c0d1099 16%,#0c0d1000 30%),url(img/policy.jpg)">
  <div class="caption">
    <h2>{words["after"]}</h2>
    <p class="lede">The policy trained on the echoes sorts tables it has never seen. 48 robots, nobody driving.</p>
  </div>
</section>""",
        f"""<section class="slide text">
  <p class="kicker">What is real</p>
  <h2>Built on the challenge's own parts.</h2>
  <ol class="points">
    <li>{words["rokoko"]}</li>
    <li><b>The G1 gets its real commands.</b> The challenge's own runner took 1,800 in a row through its safety filter, none clipped.</li>
    <li><b>The data is the challenge's format,</b> LeRobot v3.0 and GR00T, ready to train on.</li>
    <li class="quiet"><b>Not yet: a real G1.</b> Everything runs in a physics twin of the sorting station.</li>
  </ol>
</section>""",
        f"""<section class="slide text last">
  <p class="kicker">Try it now</p>
  <h2 class="link">{DEMO}</h2>
  <p class="lede">Press Train and the page runs the comparison again, on demonstrations it just recorded.<br>Or download the scene that just finished as a dataset.</p>
  <p class="foot">{SOURCE} · MIT</p>
</section>""",
    ]


CSS = """
@page { size: 1920px 1080px; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; background: #0c0d10; }
body { font-family: "Inter", system-ui, sans-serif; color: #f4f5f7; -webkit-font-smoothing: antialiased;
       font-feature-settings: "cv11", "ss03"; font-variant-numeric: tabular-nums; }
.slide { width: 1920px; height: 1080px; padding: 120px 150px; position: relative; overflow: hidden;
         page-break-after: always; break-after: page; background: #0c0d10 center / cover no-repeat;
         display: flex; flex-direction: column; justify-content: center; }
h1 { font-size: 96px; line-height: 1.04; letter-spacing: -0.035em; font-weight: 600; margin: 0; }
h2 { font-size: 76px; line-height: 1.08; letter-spacing: -0.03em; font-weight: 600; margin: 0; max-width: 1500px; }
.lede { font-size: 38px; line-height: 1.4; color: #b3b7c0; margin: 40px 0 0; max-width: 1300px; letter-spacing: -0.01em; }
.kicker { font-size: 26px; color: #9a9da6; margin: 0 0 36px; letter-spacing: 0; }
.mark { font-size: 34px; font-weight: 600; letter-spacing: -0.01em; position: absolute; top: 96px; left: 150px; }
.foot { position: absolute; left: 150px; bottom: 96px; font-size: 24px; color: #9a9da6; margin: 0; }
.picture { justify-content: flex-start; padding-top: 56px; }
.picture h2 { font-size: 64px; }
.picture .lede { margin-top: 14px; font-size: 32px; color: #d3d6dc; }
.points { list-style: none; padding: 0; margin: 64px 0 0; max-width: 1380px; }
.points li { font-size: 36px; line-height: 1.4; color: #b3b7c0; padding: 30px 0; border-top: 1px solid #ffffff1f; }
.points li b { color: #f4f5f7; font-weight: 600; }
.points li.quiet, .points li.quiet b { color: #9a9da6; }
.figure { justify-content: flex-start; padding-top: 96px; }
.figure h2 { font-size: 66px; max-width: none; white-space: nowrap; }
.figure .lede { margin-top: 22px; font-size: 34px; max-width: none; }
.chart { margin-top: 56px; overflow: visible; }
.chart .grid { stroke: #ffffff14; stroke-width: 1; }
.chart .axis { stroke: #ffffff3d; stroke-width: 1; }
.chart .gap { stroke: #ffffff3d; stroke-width: 1; }
.chart .tick, .chart .axis-title { font-size: 24px; fill: #9a9da6; }
.chart .series { fill: none; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }
.chart .series.kaiku { stroke: oklch(0.82 0.17 152); }
.chart .series.plain { stroke: #9a9da6; }
.chart .dot { stroke: #0c0d10; stroke-width: 4; }
.chart .dot.kaiku { fill: oklch(0.82 0.17 152); }
.chart .dot.plain { fill: #9a9da6; }
.chart .value { font-size: 44px; font-weight: 600; fill: #f4f5f7; letter-spacing: -0.02em; }
.chart .name { font-size: 28px; font-weight: 500; fill: #f4f5f7; }
.chart .quiet { fill: #9a9da6; }
.note { position: absolute; left: 150px; bottom: 60px; font-size: 22px; color: #9a9da6; margin: 0; }
.link { font-size: 64px; letter-spacing: -0.025em; }
"""


def build(variant: str = "none") -> Path:
    study = json.loads(STUDY.read_text())
    html = ("<!doctype html><html lang=en><meta charset=utf-8><title>Kaiku</title>"
            f"<style>{CSS}</style><body>\n" + "\n".join(slides(study, WORDS[variant])) + "\n</body></html>\n")
    page = HERE / ("kaiku.html" if variant == "none" else f"kaiku-{variant}.html")
    page.write_text(html)
    return page


def pdf(page: Path) -> Path:
    from playwright.sync_api import sync_playwright
    out = page.with_suffix(".pdf")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        tab = browser.new_page(viewport={"width": W, "height": H})
        tab.goto(page.as_uri())
        tab.wait_for_load_state("networkidle")
        tab.pdf(path=str(out), width=f"{W}px", height=f"{H}px", print_background=True)
        browser.close()
    return out


if __name__ == "__main__":
    import sys
    print(pdf(build(next((name for name in ("gloves", "one-glove") if f"--{name}" in sys.argv), "none"))))
