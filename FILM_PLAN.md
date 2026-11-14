# Kaiku film plan

Two minutes at most, public. This is the plan of film v4, the no-gloves cut: nobody has worn real
gloves with Kaiku and nothing in it says so. With the draft cold open (11 s) it is 1:49.1: open
0:11, body 1:36.1, end card 0:02. How to build it: `film/README.md`.

## The story

1. Cold open, made elsewhere: "One hand. 48 simulated robots." It ends on a robot hand held upright
   on black. The film cuts from it to a real hand in the same place.
2. A real hand moves a robot's fingers. Then, once: everything here is a simulator, the policy is
   told where the parts are, and on the live page a scripted hand drives when no visitor does.
3. All 48 robots at real speed. Copying the motion fails on the same tables that Kaiku's echoes sort.
4. One number slide: 32 demonstrations, 58% alone, 95% with their echoes. Then nobody drives, and
   the live page's address comes up over the policy sorting alone.

**The moment a judge retells:** "They switched it off and 40 robots grabbed at parts that weren't
there. Switched it on, same tables, all 47 sorted. And 32 demonstrations with echoes beat 128 without."

## Shots

Everything is at real speed. The words and numbers below are filled in from `film/film.json`; the
build stops if a number is not what the checked-in files and the page say. Times are with the 11 s
draft open; the build takes whatever length the open has.

| Time | Screen | Voiceover, word for word | In the film's own type |
|---|---|---|---|
| 0:00 to 0:11 | The cold open, as it is: its picture and its sound, nothing of ours over it | none | none (the open's own: "One hand. 48 simulated robots.") |
| 0:11 to 0:18.1 | The hand film full frame, its hand where the robot's hand was in the open's last frame; after 1.1 s it moves beside the robot's hand, same run, same moment | "This is film of a real hand, played into the page as its camera. The robot's fingers follow it." | "A real hand moves the robot's fingers"; "Film of a hand, played into the page as its camera"; "Hand: S. Perquin, CC BY-SA 4.0"; "The robot's hand, same run, same moment" |
| 0:18.1 to 0:30.1 | The driven robot close, the page's own line under it | "Everything here runs in a simulator, a twin of Junction's G1 sorting station. The policy is told where the parts are. On the live page, a scripted hand drives when no visitor does." | "Everything here runs in a simulator"; "A twin of Junction's G1 sorting station"; "The policy is told where the parts are"; "On the live page, a scripted hand drives when no visitor does" |
| 0:30.1 to 0:36.6 | All 48 robots sorting, slow push in | "Forty-seven echoes redo the driven robot's job as it happens, each on its own table." | "47 echoes redo the driven robot's job, each on its own table" |
| 0:36.6 to 0:48.6 | Kaiku off. Three neighbouring tables close, then all 48: the robots reach for parts that are not there. The page's banner counts: 7 of 47 | "Every table is different. So if the others just copy the motion, they reach for parts that are not there." (2.8 s without music on the result) | "Real time"; the page's banner, enlarged |
| 0:48.6 to 1:03.1 | The same tables, Kaiku on. The same three tables close, then all 48. The page's banner: 47 of 47, without: 7. Its count of saved demonstrations rises | "Same tables, Kaiku on. Each echo redoes the motion around the parts that are really there." then "Every clean run is saved as a demonstration." | "Real time"; "Every clean run is saved as a demonstration" |
| 1:03.1 to 1:13.9 | The one number slide, over the robots dimmed | "A policy trained on thirty-two demonstrations sorts fifty-eight percent of new scenes. With their echoes, ninety-five. Thirty-two with echoes beat a hundred and twenty-eight without." | "A policy trained on 32 demonstrations"; 58% alone; 95% with their echoes; "32 with echoes beat 128 without"; "128 demonstrations alone: 92%"; "Share of 96 unseen scenes sorted, mean of 3 training runs. The study's demonstrations came from a scripted hand." |
| 1:13.9 to 1:22.2 | Three tables at the other end of the rank, sorting | "From the same sixteen demonstrations, our own MimicGen-style replay reaches eight percent. Kaiku's echoes reach seventy-three." | "The same 16 demonstrations, multiplied two ways"; 8% MimicGen-style replay; 73% Kaiku's echoes; "Share of 96 unseen tables sorted. The replay is our own implementation, not MimicGen's code." |
| 1:22.2 to 1:30.5 | The local Kaiku the runner followed; the runner's log for 5 s, its summary line large | "The humanoid challenge's own runner, with a mock robot: eighteen hundred commands through its safety filter, none clipped." | "The humanoid challenge's own runner and safety filter, with a mock robot"; the runner's own line, 44 px; "1,800 commands through the safety filter. 0 clipped." |
| 1:30.5 to 1:40.5 | Policy mode: three tables close, then all 48 | "Then nobody drives. The policy sorts scenes it has never seen." | "Then nobody drives"; "Real time" |
| 1:40.5 to 1:47.1 | The policy still sorting alone, the page's banner counting what it sorted | "It is live at kaiku echo dot fly dot dev. Open it and drive it yourself." | kaiku-echo.fly.dev; "Open it and drive it yourself"; the hand film's full credit |
| 1:47.1 to 1:49.1 | The end card, as it is | none | none (the card's own) |

## Where each shot comes from

The open and the end card are read from `/bulk3/runs/opens/final/kaiku/` when the film is built
(`open` in `film/film.json`).

All page shots are the hosted page (`https://kaiku-echo.fly.dev`) at 1920x1080: its stream recorded
with `film/rec.py` (it only listens) and drawn again frame by frame by `film/shoot.py` in `drun`
sandboxes. v4 uses v3's recordings (`/bulk3/runs/robotteleop/film/v3/`), filmed again from new
cameras and with larger type; nothing new was pressed on the live page for v4.

| Shot | Source | Pressed |
|---|---|---|
| 1 | `hand/take4`: `film/hand.py` plays "Finger-counting in Dutch" (S. Perquin, Wikimedia Commons, CC BY-SA 4.0) into the page as its camera | Drive it yourself, Use my hand, Grasp it myself; the turn handed back |
| 2 to 7 | `rec/operator`, Operator mode, a scripted hand driving | See it without Kaiku, once |
| 6, 7 numbers | `kaiku/policy/pretrained/study.json`, `baselines.json`, and the page's own panels (read, not shown) | none |
| 8 | `film/g1log.sh`: the challenge's runner (`--backend mock`) follows a local Kaiku for 60 s; its log, its summary, the local Kaiku's stream | none on the live page |
| 9, 10 | `rec/policy`, Policy mode | Policy |

Train both is never pressed.

## What is true in it, and what is not claimed

- Everything is a simulator, and the film says so once, early, in type and in voice, with the two
  things a judge would otherwise have to ask: the policy is told where the parts are, and a scripted
  hand drives the live page when no visitor does.
- Shot 1 is film of a real hand, not a person at the laptop. The film says so in type and in voice.
  The hand and the robot are one run on one clock; the sync is not edited.
- The comparison with a MimicGen-style replay is our own implementation of the replay. The voice
  and the type say so.
- The runner shot is the challenge's own runner and safety filter with a mock robot, not a real G1.
- No gloves are worn or claimed.
- The voice is synthetic (Gemini speech); the music is v2's score. Neither plays over the open.

## When a person's gloved hand is filmed

`film/make.sh vN --shot2 FILE` puts a phone clip of a real gloved hand in the first shot after the
open, lined up with the open's last frame the same way, changes the head and the voice line to say
so, and drops the hand film's credit. `film/README.md` has the exact command. It is the one shot an
agent cannot make.
