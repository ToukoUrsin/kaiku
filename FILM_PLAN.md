# Kaiku film plan

Two minutes at most, public. This is the plan of film v3 (1:50.5), the no-gloves cut: nobody has worn
real gloves with Kaiku and nothing in it says so. How to build it: `film/README.md`.

## The story

1. A real hand moves a robot's fingers. Nothing is worn on it.
2. One robot is driven and 47 echoes redo the job live, each on its own table. Copying the motion
   fails; Kaiku's echoes sort every table.
3. The echoes are worth demonstrations: a policy trained on 32 sorts 58% of new scenes alone, 95%
   with their echoes. Then nobody drives.

**The moment a judge retells:** "They switched it off and 40 robots grabbed at parts that weren't
there. Switched it on, same tables, all 47 sorted. And 32 demonstrations with echoes beat 128 without."

## Shots (1:50.5)

Everything is at real speed. The words and numbers below are filled in from `film/film.json`; the
build stops if a number is not what the checked-in files and the page say.

| Time | Screen | Voiceover, word for word | In the film's own type |
|---|---|---|---|
| 0:00 to 0:14 | The page's camera picture of a real hand beside the robot's hand, same run, same moment | "This is film of a real hand. We play it into the page as its camera, and the robot's fingers follow it. Nothing is worn on the hand." | "A real hand moves the robot's fingers"; "Film of a real hand (it loops), played into the page as its camera"; "Hand: S. Perquin, CC BY-SA 4.0" |
| 0:14 to 0:24.5 | All 48 robots sorting, slow push in | "On the live page, a scripted hand drives one robot. Forty-seven echoes redo the job as it happens, each on its own table." | "One robot is driven. 47 echoes redo the job, each on its own table."; "On the live page a scripted hand drives." |
| 0:24.5 to 0:36.5 | See it without Kaiku: the robots reach for parts that are not there. The page's banner counts | "Every table is different. So if the others just copy the motion, this happens." (2.8 s without music on the result) | "Kaiku off: every robot gets the same motion"; "Real time" |
| 0:36.5 to 0:48.5 | The same tables, Kaiku on. The page's own line: "Without Kaiku: 7 of 47 tables sorted. With Kaiku: 47 of 47." | "Same tables, Kaiku on. Each echo redoes the motion around the parts that are really there." | "Kaiku on, the same tables"; "Real time" |
| 0:48.5 to 1:04.5 | The page's study panel beside the numbers | "Why it matters. A policy trained on thirty-two demonstrations sorts fifty-eight percent of new scenes. With their echoes, ninety-five. Thirty-two with echoes beat a hundred and twenty-eight without." | 58% alone, 95% with their echoes; "32 with echoes beat 128 without"; "128 demonstrations alone: 92%"; "Share of 96 unseen scenes sorted, mean of 3 training runs. The study's demonstrations came from a scripted hand." |
| 1:04.5 to 1:16 | The page's matched comparison beside the numbers | "Against a MimicGen-style replay, from the same sixteen demonstrations: eight percent with the replay, seventy-three with Kaiku's echoes. The replay is our own implementation." | 8% MimicGen-style replay, 73% Kaiku's echoes; "The replay is our own implementation, not MimicGen's code." |
| 1:16 to 1:25 | Policy mode, all robots, the page's banner counts | "Then nobody drives. The policy sorts scenes it has never seen." | "Then nobody drives"; "Real time" |
| 1:25 to 1:36 | The runner's log line by line as it was printed, beside the local Kaiku it followed | "This is the humanoid challenge's own runner, with a mock robot. Eighteen hundred commands went through its safety filter. None was clipped." | "The humanoid challenge's own runner and safety filter, with a mock robot"; "55 seconds later"; "1,800 commands through the safety filter. 0 clipped." |
| 1:36 to 1:42.5 | The page's dataset panel | "Every clean run is saved as a demonstration, in the format the challenge asks for." | "Every clean run is a demonstration, in the challenge's format" |
| 1:42.5 to 1:50.5 | The live page, whole | "It is live at kaiku echo dot fly dot dev. Open it and drive it yourself." | "Live now. Drive it yourself."; kaiku-echo.fly.dev; the hand film's full credit |

## Where each shot comes from

All page shots are the hosted page (`https://kaiku-echo.fly.dev`, master f632b46) at 1920x1080:
its stream recorded with `film/rec.py` (it only listens) and drawn again frame by frame by
`film/shoot.py` in `drun` sandboxes. Recordings and takes: `/bulk3/runs/robotteleop/film/v3/`.

| Shot | Source | Pressed |
|---|---|---|
| 1 | `hand/take4`: `film/hand.py` plays "Finger-counting in Dutch" (S. Perquin, Wikimedia Commons, CC BY-SA 4.0) into the page as its camera | Drive it yourself, Use my hand, Grasp it myself; the turn handed back |
| 2 to 6, 9, 10 | `rec/operator`, Operator mode, a scripted hand driving | See it without Kaiku, once |
| 5, 6 numbers | `kaiku/policy/pretrained/study.json`, `baselines.json`, and the page's own panels | none |
| 7 | `rec/policy`, Policy mode | Policy |
| 8 | `film/g1log.sh`: the challenge's runner (`--backend mock`) follows a local Kaiku for 60 s; its log, its summary, the local Kaiku's stream | none on the live page |

Train both is never pressed.

## What is true in it, and what is not claimed

- Shot 1 is film of a real hand, not a person at the laptop. The film says so in type and in voice.
  The hand and the robot are one run on one clock; the sync is not edited.
- Shots 2 to 7 and the study's demonstrations are driven by a scripted hand. The voice, the film's
  type or the page's own source line says so in each.
- Shot 8 is the challenge's own runner and safety filter with a mock robot, not a real G1.
- No gloves are worn or claimed.
- The voice is synthetic (Gemini speech); the music is v2's score.

## When a person's hand is filmed

`film/make.sh vN --shot2 FILE` puts a screen recording of a person's own hand in shot 1, changes the
head and the voice line to say so, and drops the hand film's credit. `film/README.md` says what the
clip must be. It is the one shot an agent cannot make.
