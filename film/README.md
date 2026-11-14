# The submission film

The no-gloves cut: nobody has worn real Rokoko gloves with Kaiku, and nothing in it says so.
Films and everything they are made from live in `/bulk3/runs/robotteleop/film/` (not in git).

## Build it

    film/make.sh v2

Writes `/bulk3/runs/robotteleop/film/kaiku-nogloves-v2.mp4`, a contact sheet (one frame every 2 s),
captions (`.srt`) and a loudness report next to it. Only what changed is made again.

## Swap in a real shot 2

    film/make.sh v3 --shot2 /path/to/hand.mp4

Shot 2 (0:09 to 0:24) is one file, `work/clips/shot2.mp4`, and one voice line. Until a person films
their hand it is the stand-in: the lead robot close up while the scripted hand drives, the page's own
source line on screen, and a voice line that says so. With `--shot2` the clip replaces it and the
voice says the real line (`vo_real` in `film.json`). `film/make.sh v4` alone goes back to the stand-in.

The clip must be:

- 16:9, 1920x1080 or larger, 15 s or longer (the first 15 s are used; sound is dropped).
- A screen recording of `https://kaiku-echo.fly.dev/?intro=0&shot=lead` in a real browser: press
  Drive it yourself, Use my hand, Grasp it myself.
- A person's bare hand in the camera picture beside the lead robot's hand: open, close, pinch, slowly,
  so the robot's fingers are seen closing with theirs.
- The page's own source line readable ("A person's hand, seen by a camera, ..."). No gloves in it.

## Change a word or a number

Everything spoken, every number and all timing is in `film/film.json`. Edit, then `film/make.sh vN`.

The numbers on screen are the page's own pixels from the take. The numbers the voice says are in
`numbers`, and the build stops if the page did not say the same in that take. So when the study
numbers change on the hosted page:

    .venv/bin/python film/rec.py https://kaiku-echo.fly.dev /bulk3/runs/robotteleop/film/rec/study --seconds 20 --wait standin

then in `film.json` set `"rec": "study"` and `"at": 10` on the still `6-panel`, start/end 1 to 3 on
`6-page`, 3 to 20 on `6-left` (drop `6-leftb`), put the new numbers in `numbers`, delete
`work/clips/6-*`, and run `film/make.sh vN`. `rec.py` only listens.

## What is what

| File | Does |
|---|---|
| `film.json` | words, numbers, timing, which seconds of which recording each shot uses |
| `rec.py` | records the hosted page's stream; presses only what `--at` names, never Train or Drive |
| `shoot.py` | replays a recording in headless Chrome frame by frame and writes what the page said (`.says.jsonl`) |
| `capture.py` | films every take in `film.json`, in drun sandboxes |
| `vo.py` | voice and music (ElevenLabs, key from hsec) |
| `g1log.sh` | runs the challenge's own runner against a local Kaiku for the numbers in shot 8 |
| `edit.py` | checks the film against what the page said, then cuts it (ffmpeg, in a drun sandbox) |
| `make.sh` | all of it, in order |

Recordings used: `rec/operator2` (the stand-in driving, the comparison with and without Kaiku) and
`rec/policy` (Policy mode). Both are the hosted page with 48 robots.
