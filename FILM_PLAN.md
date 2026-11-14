# Kaiku film plan

Two minutes at most, public. Two cuts share every shot but shot 2: the one in force says no gloves
were worn; the other is used only after a real glove session exists.

## The story

1. A robot learns a job from demonstrations, and a person gives them one at a time.
2. With Kaiku one person's hand drives one robot and 47 more redo the job live, each on its own table.
3. A policy trained on 16 demonstrations sorts almost nothing. With the echoes, 81%. Then nobody drives.

**The moment a judge retells:** "They switched it off and 47 robots grabbed at parts that weren't
there. Switched it on, same scene, all 47 sorted. Then I held my hand up to my own laptop and the
robot's fingers followed mine."

## Shots (1:54)

| Time | Screen | Voiceover, word for word | Number on screen |
|---|---|---|---|
| 0:00–0:09 | All 48 robots sorting, slow push in | "One person is sorting parts with this robot. Forty-seven more are doing the same job, right now." | 48 robots |
| 0:09–0:24 | **No gloves:** the camera picture of a bare hand beside the lead robot's hand; the fingers close together. **Gloves:** phone footage of the gloved hand beside the lead robot's hand | **No gloves:** "Nobody is wearing anything. A camera sees the hand, and the robot's fingers follow the fingers, down the same path Rokoko's gloves use." **Gloves:** "The gloves are Rokoko Smartgloves. Rokoko's driver reads them, and the robot's fingers follow the fingers." | The page's own source line |
| 0:24–0:34 | Three tables side by side: the same colours, parts in other places | "Every table is different. The parts lie somewhere else, turned another way. So the motion can't just be copied." | none |
| 0:34–0:50 | See it without Kaiku: the robots reach for parts that are not there | "Copy it anyway and this happens." (three seconds of silence on the misses) | the banner's count without Kaiku |
| 0:50–1:04 | The same scene again, Kaiku on | "Kaiku redoes the motion on every table, around the parts that are really there, while the person is still moving. Same scene." | the banner's count with Kaiku |
| 1:04–1:22 | Policy tab: the two curves | "Why it matters: from sixteen demonstrations a policy sorted almost nothing. From the same sixteen and their echoes, eighty-one percent. Thirty-two with echoes beat a hundred and twenty-eight without." | 81%, 32 against 128 |
| 1:22–1:38 | Policy mode, all robots, nobody driving | "Then nobody drives. Forty-eight robots, scenes none of them has seen." | the live share of scenes sorted |
| 1:38–1:48 | Dataset tab, then the challenge runner's log | "Every run goes through the humanoid challenge's safety filter. Physics decides what counts. The clean ones download in the challenge's own format." | 1,800 commands, 0 clipped |
| 1:48–1:54 | The address, a hand held up to a laptop | "Hold your hand up. Drive it yourself." | kaiku-echo.fly.dev |

## Where each shot comes from

All screens are the hosted page at 1920×1080, recorded with `tools/film/record.py` and rendered with
`tools/film/capture.py` (run through `drun`, output to `/bulk3/runs/robotteleop/film/`). No account.

| Shot | URL and state | Reset |
|---|---|---|
| 1 | `/?clean=1&shot=overview&quality=high`, Operator mode | none |
| 2, no gloves | `/?intro=0&shot=lead`, Drive it yourself, Use my hand, Grasp it myself; screen recording of a real browser with a person's hand | hand the turn back |
| 2, gloves | the glove session's replay, `/?clean=1&shot=lead`; phone footage from the sitting | none |
| 3 | `/?clean=1&env=1`, `env=7`, `env=19`, tiled in the edit | none |
| 4, 5 | `/?shot=overview&quality=high`, See it without Kaiku; one take holds both halves | wait half a minute between takes |
| 6 | `/?tab=policy&result=1` | none |
| 7 | `/?clean=1&shot=overview`, Policy mode | switch back to Operator |
| 8 | `/?tab=dataset`; `tests/test_g1.py` output | none |
| 9 | title card | none |

## Must be true before filming

- The source line on the page says what is really driving in every shot. Nothing says gloves in the no-gloves cut.
- Shot 2 without gloves needs a person's hand in front of a real camera. Until Touko films it, the cut uses the page's own camera picture from a screen recording by whoever is at the laptop. It is the one shot an agent cannot make.
- The numbers in shots 4 to 6 are read off the page in that take, not typed in the edit.
- The 81% and the 32 against 128 match the README on the day. If the baseline study changes them, the voiceover changes with them.
- A judge alone on the address gets to shot 4 in two presses and to shot 2 in three, under 30 seconds.
- Proof from outside our own simulator: Rokoko's driver and hand solver in shot 2, the humanoid challenge's own runner and safety filter in shot 8.
