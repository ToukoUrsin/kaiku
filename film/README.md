# The submission film

The no-gloves cut: nobody has worn real Rokoko gloves with Kaiku, and nothing in it says so.
Films and everything they are made from live in `/bulk3/runs/robotteleop/film/` (not in git).
v4's work files are under `/bulk3/runs/robotteleop/film/v4/`; its recordings are v3's
(`v4/rec` and `v4/hand` point at `v3/rec` and `v3/hand`).

## Build it

    film/make.sh v4

Writes `/bulk3/runs/robotteleop/film/kaiku-nogloves-v4.mp4`, a contact sheet (one frame every 2 s),
captions (`.srt`) and a report (lengths, loudness, the planned silence, what shot 1 is and where its
hand was put) next to it. Only what changed is made again. The cut renders in a `drun` sandbox.

With the final open (11 s) the film is 1:49.1: open 0:11, body 1:36.1, end card 0:02.

## The cold open and the end card

The film starts with a cold open and ends on an end card, both made elsewhere. `open` in
`film.json` names the folder (`/bulk3/runs/opens/final/kaiku/`) and the files (`open.mp4`, `end.mp4`).
Every build reads them from that folder as they are then:

- The open goes first, untouched: its own picture and its own sound at its own level. No voice, no
  music and no type of ours is over it. (It is 24 frames a second and the film is 30, so each of its
  frames is shown for one or two film frames; nothing else is done to it.) The end card goes last.
- It takes whatever length the files have. The body's music, voice and captions start where the
  open ends.
- **The build stops, loudly, if the whole film would pass 2:00** (`open.max`), and says by how much.
  It prints a note when the film is over 1:50 (`open.aim`).
- Shot 1 starts on the open's last frame (a robot hand held upright on black). The build finds the
  robot's hand in that frame (the bright shape: its top, its middle, its width) and puts the hand
  film full frame with its hand there, on a cut (the open's maker asks for a cut, no dissolve). After 1.1 s the picture moves to its
  place beside the robot's hand. If the last frame is not a bright hand on black, `open.hand` in
  `film.json` is used and the build says so.

**When the final open is in the folder, this rebuilds the film with it. Nothing is edited:**

    film/make.sh v4

To build the body alone, without the open and the end card (`kaiku-nogloves-v4-noopen.mp4`):

    film/make.sh v4 --no-open

To drop only the end card, set `open.end` to `""` in `film.json`.

## Shot 1: a real hand

Shot 1 (7.1 s) is film of a real hand, played into the hosted page as its camera. The page reads it
as it reads a visitor's camera and the lead robot's fingers follow. The film shows the page's own
camera picture and the robot's hand side by side, from one run, lined up on the clock they were
recorded with (the build stops if any picture is more than 0.12 s from its robot frame, if the page
lost the hand, or if the runner did not say a camera was driving).

The hand is "Finger-counting in Dutch" by S. Perquin (Wikimedia Commons, CC BY-SA 4.0). The film
says so on screen in shot 1 and under the address at the end. The take:

    ffmpeg -t 7.4 -i Finger-counting_in_Dutch.webm -an -vf "crop=1067:800:0:190,scale=640:480,fps=30" -pix_fmt yuv420p cam640.y4m
    nice -n 10 /usr/bin/python film/hand.py 'https://kaiku-echo.fly.dev/?intro=0&quality=low' cam640.y4m /bulk3/runs/robotteleop/film/v3/hand/take4 --seconds 36 --cpu

`hand.py` presses what a visitor presses (Drive it yourself, Use my hand, Grasp it myself) and hands
the turn back at the end. `takes.hand.take` in `film.json` names the take the film uses, and
`open.film_hand` says where the hand is in its picture when the shot starts.

## The slot: a real gloved hand as the first shot

One command, no JSON to edit:

    film/make.sh v5 --shot2 /path/to/glove.mp4

`glove.mp4` is a phone clip of a real hand in a Rokoko glove driving Kaiku: 10 seconds or longer,
any size, upright or sideways. Three options, all optional:

    film/make.sh v5 --shot2 /path/to/glove.mp4 --from 3 --turn 90 --hand 0.5,0.4,0.55

`--from 3` starts 3 s into the clip. `--turn 90` (or 180, 270) turns it clockwise, for a clip that
plays on its side. `--hand` says where the hand is in the (turned) clip when it starts: its centre
from the left, its centre from the top, and its height, as shares of the picture (default
`0.5,0.45,0.5`: a hand in the middle, half the picture tall).

What it changes, and nothing else:

- The film is written to `kaiku-gloves-v5.mp4` (with `.srt`, contact sheet, report). The no-gloves
  film is not overwritten.
- The first shot after the open is the clip instead of the hand film, for 10 s instead of 7.1 s
  (`dur_real` in `film.json`), so the film is 2.9 s longer (1:52.0 with the 11 s draft open; the
  2:00 stop still applies).
- It lines up with the open's last frame the same way: the clip starts with its hand where the
  robot's hand was and about as tall, then after 1.1 s settles to show the whole picture. Where the
  picture does not fill 16:9 it sits over a blurred copy of itself. Sound is dropped; HDR is brought
  down to ordinary video.
- Head: "A person's hand in a Rokoko Smartglove drives the robot". Label: "A real hand, a real glove".
- Voice: "This is a person's hand in a Rokoko glove. The glove drives the first robot's fingers."
- The hand film's credit leaves shot 1 and the end.

Everything after shot 1 is the same film, word for word. The gloves words are true only of a clip
that shows what they say: a real glove on a real hand, driving Kaiku. `film/make.sh v4` alone goes
back to the hand film.

Checked on 14 November 2026 with two 15 s test clips cut from the hand film: one on its side, as a
whole film (`--from 2 --turn 90 --hand 0.45,0.6,0.55`, 1:52.0, the hand on the robot's hand at the
cut), and one upright with no options, through the checks and shot 1, with and without the open.
The test film and the clips were deleted.

## Change a word or a number

Everything spoken, every number and all timing is in `film/film.json`. Edit, then `film/make.sh vN`.

The build stops unless every number in `numbers` is what the checked-in files say
(`kaiku/policy/pretrained/study.json`, `baselines.json`, the runner's `summary.json`) and what the
page said in the take that shows it. Nothing is typed in the edit only. It also stops if the number
slide is longer than 12 s, the runner's log is on screen for more than 5 s, the hand film runs
outside 6 to 8 s, the body is not between 1:25 and 1:47, or the whole film passes 2:00. A take may be
longer than its shot (the shot uses its start). When the hosted page changes, record it again (both
only listen, apart from what `--at` names; never Train both):

    nice -n 10 /usr/bin/python film/rec.py https://kaiku-echo.fly.dev /bulk3/runs/robotteleop/film/v3/rec/operator --seconds 175 --wait copyfree --at '22:copy_scene:null'
    nice -n 10 /usr/bin/python film/rec.py https://kaiku-echo.fly.dev /bulk3/runs/robotteleop/film/v3/rec/policy --seconds 75 --wait standin --at '6:mode:"policy"'
    film/g1log.sh /bulk3/runs/robotteleop/film/v4/work

then set the `start`/`end` of the clips in `film.json`, move the changed `work/clips/N-*` and
`work/jobs/N-*` away, and run `film/make.sh vN`.

To look at one shot without cutting the film: `film/edit.py plan WORK`, then in `WORK/edit`
`python3 edit.py render 1-hand` writes `out/s1.mp4`.

## Type sizes

Nothing the film's own type says is under 36 px at 1080p, and every number is 44 px or larger; the
page's own lines are enlarged to 38 px (its source line) and 46 px (its banner) by film-only CSS in
`film.json`. The exceptions are credits and labels beside the picture: the hand film's credit (30 px
in shot 1, 28 px at the end), the label on the runner's robot (34 px) and the runner's scrolling log
(22 px), whose one line that matters is shown at 44 px.

## The voice

`voice.provider` in `film.json` picks who speaks. v3 and v4 are Gemini's speech model (voice
Charon), because ElevenLabs had no credit left (it resets on 5 December 2026). The word timings for
the captions are Deepgram's transcription of each line, and the build stops if what was heard is not
the line's own words. The music is v2's score, kept for the same reason. `film.json` says how to go back.

## What is what

| File | Does |
|---|---|
| `film.json` | words, numbers, timing, the cold open's folder, which seconds of which recording each shot uses |
| `rec.py` | records a page's stream; presses only what `--at` names, never Train or Drive |
| `hand.py` | shot 1: plays a film of a hand into the hosted page as its camera, records the stream, the page's camera picture and what the page read, all on one clock |
| `g1log.sh` | shot 8: the challenge's own runner (mock robot) follows a local Kaiku for 60 s; keeps its log, its summary and the local Kaiku's stream |
| `shoot.py` | replays a recording in headless Chrome frame by frame and writes what the page said (`.says.jsonl`) |
| `capture.py` | films every take in `film.json`, in drun sandboxes |
| `vo.py` | voice (Gemini or ElevenLabs) and music (ElevenLabs), keys from hsec |
| `edit.py` | checks the film against the files and what the page said, draws the film's own type, then cuts it with the open in front (ffmpeg, in a drun sandbox) |
| `make.sh` | all of it, in order |

Recordings used, all under `/bulk3/runs/robotteleop/film/v3/`: `rec/operator` (the hosted page, 48
robots: the fleet, the comparison with and without Kaiku), `rec/policy` (Policy mode), `rec/g1` (the
local Kaiku the runner followed) and `hand/take4` (shot 1). v4 films them again from new cameras
(three neighbouring tables, close) and with larger type; nothing new was pressed on the live page.
