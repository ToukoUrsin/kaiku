# The submission film

The no-gloves cut: nobody has worn real Rokoko gloves with Kaiku, and nothing in it says so.
Films and everything they are made from live in `/bulk3/runs/robotteleop/film/` (not in git);
v3's recordings, takes and work files are under `/bulk3/runs/robotteleop/film/v3/`.

## Build it

    film/make.sh v3

Writes `/bulk3/runs/robotteleop/film/kaiku-nogloves-v3.mp4` (1:50.5), a contact sheet (one frame
every 2 s), captions (`.srt`) and a report (loudness, the planned silence, what shot 1 is) next to it.
Only what changed is made again. The cut renders in a `drun` sandbox.

## Shot 1: a real hand

Shot 1 (0:00 to 0:14) is film of a real hand, played into the hosted page as its camera. The page
reads it as it reads a visitor's camera and the lead robot's fingers follow. The film shows the
page's own camera picture and the robot's hand side by side, from one run, lined up on the clock
they were recorded with (the build stops if any picture is more than 0.12 s from its robot frame,
if the page lost the hand, or if the runner did not say a camera was driving).

The hand is "Finger-counting in Dutch" by S. Perquin (Wikimedia Commons, CC BY-SA 4.0). The film
says so on screen in shot 1 and in the end card. The take:

    ffmpeg -t 7.4 -i Finger-counting_in_Dutch.webm -an -vf "crop=1067:800:0:190,scale=640:480,fps=30" -pix_fmt yuv420p cam640.y4m
    nice -n 10 /usr/bin/python film/hand.py 'https://kaiku-echo.fly.dev/?intro=0&quality=low' cam640.y4m /bulk3/runs/robotteleop/film/v3/hand/take4 --seconds 36 --cpu

`hand.py` presses what a visitor presses (Drive it yourself, Use my hand, Grasp it myself) and hands
the turn back at the end. `takes.hand.take` in `film.json` names the take the film uses.

## Swap in a person's own hand

    film/make.sh v4 --shot2 /path/to/hand.mp4

With `--shot2` the clip fills shot 1, the head reads "A person's hand moves the robot's fingers",
the voice says `vo_real`, and the hand film's credit leaves the end card. `film/make.sh v4` alone
goes back to the hand film. The clip must be:

- 16:9, 1920x1080 or larger, 14 s or longer (the first 14 s are used; sound is dropped).
- A screen recording of `https://kaiku-echo.fly.dev/?intro=0&shot=lead` in a real browser: press
  Drive it yourself, Use my hand, Grasp it myself.
- A person's bare hand in the camera picture beside the lead robot's hand: open, close, pinch, slowly,
  so the robot's fingers are seen closing with theirs.
- The page's own source line readable ("A visitor's hand, seen by a camera, ..."). No gloves in it.

## Change a word or a number

Everything spoken, every number and all timing is in `film/film.json`. Edit, then `film/make.sh vN`.

The build stops unless every number in `numbers` is what the checked-in files say
(`kaiku/policy/pretrained/study.json`, `baselines.json`, the runner's `summary.json`) and what the
page said in the take that shows it. Nothing is typed in the edit only. When the hosted page changes,
record it again (both only listen, apart from what `--at` names; never Train both):

    nice -n 10 /usr/bin/python film/rec.py https://kaiku-echo.fly.dev /bulk3/runs/robotteleop/film/v3/rec/operator --seconds 175 --wait copyfree --at '22:copy_scene:null'
    nice -n 10 /usr/bin/python film/rec.py https://kaiku-echo.fly.dev /bulk3/runs/robotteleop/film/v3/rec/policy --seconds 75 --wait standin --at '6:mode:"policy"'
    film/g1log.sh /bulk3/runs/robotteleop/film/v3/work

then set the `start`/`end` of the clips in `film.json`, move the changed `work/clips/N-*` and
`work/jobs/N-*` away, and run `film/make.sh vN`.

## The voice

`voice.provider` in `film.json` picks who speaks. v3 is Gemini's speech model (voice Charon), because
ElevenLabs had no credit left (it resets on 5 December 2026). The word timings for the captions are
Deepgram's transcription of each line, and the build stops if what was heard is not the line's own
words. The music is v2's score, kept for the same reason. `film.json` says how to go back.

## What is what

| File | Does |
|---|---|
| `film.json` | words, numbers, timing, which seconds of which recording each shot uses |
| `rec.py` | records a page's stream; presses only what `--at` names, never Train or Drive |
| `hand.py` | shot 1: plays a film of a hand into the hosted page as its camera, records the stream, the page's camera picture and what the page read, all on one clock |
| `g1log.sh` | shot 8: the challenge's own runner (mock robot) follows a local Kaiku for 60 s; keeps its log, its summary and the local Kaiku's stream |
| `shoot.py` | replays a recording in headless Chrome frame by frame and writes what the page said (`.says.jsonl`) |
| `capture.py` | films every take in `film.json`, in drun sandboxes |
| `vo.py` | voice (Gemini or ElevenLabs) and music (ElevenLabs), keys from hsec |
| `edit.py` | checks the film against the files and what the page said, draws the film's own type, then cuts it (ffmpeg, in a drun sandbox) |
| `make.sh` | all of it, in order |

Recordings used, all under `/bulk3/runs/robotteleop/film/v3/`: `rec/operator` (the hosted page, 48
robots: the fleet, the comparison with and without Kaiku, the panels, the end), `rec/policy` (Policy
mode), `rec/g1` (the local Kaiku the runner followed) and `hand/take4` (shot 1).
