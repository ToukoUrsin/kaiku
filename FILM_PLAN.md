# Kaiku film plan

Two minutes at most, public. This is the plan of film v5, the no-gloves cut: nobody has worn real
gloves with Kaiku and nothing in it says so. With the final cold open (11 s) it is 1:51.9: open
0:11, body 1:38.9, end card 0:02. How to build it: `film/README.md`.

What v5 changed from v4: a plain cut from the open's last frame to the hand, the hand as tall as the robot's; the
replay comparison says its counts (5 and 214 clean runs of the same 752 tables) before its shares; the number slide
says the 32 demonstrations are the scripted hand's, counts their 1,498 echoes and says the echoes cost no one's time;
the comparison's wide shots are framed so the driven robot in front and its table are clear of the page's line; the
film's own type no longer repeats the voice word for word.

## The story

1. Cold open, made elsewhere: "One hand drives / 48 simulated robots." It ends on a robot hand held upright
   on black. The film cuts from it, with no dissolve, to a real hand in the same place and at the same size.
2. A real hand moves a robot's fingers. Then, once: everything here is a simulator, the policy is
   told where the parts are, and on the live page a scripted hand drives when no visitor does.
3. All 48 robots at real speed. Copying the motion fails on the same tables that Kaiku's echoes sort.
4. One number slide: the scripted hand's 32 demonstrations, 58% alone, 95% with their 1,498 echoes, more than
   128 demonstrations alone; the echoes cost no one's time. Then the same 16 demonstrations turned into more two
   ways: 5 clean runs by our replay, 214 by Kaiku's echoes, and what a policy trained with each sorts (8%, 73%).
   Then nobody drives, and the live page's address comes up over the policy sorting alone.

**The moment a judge retells:** "They switched it off and 40 robots grabbed at parts that weren't
there. Switched it on, same tables, all 47 sorted. And 32 demonstrations and their echoes beat 128 alone, and the echoes cost no one any time."

## Shots

Everything is at real speed. The words and numbers below are filled in from `film/film.json`; the
build stops if a number is not what the checked-in files and the page say. Times are with the 11 s
final open; the build takes whatever length the open has.

| Time | Screen | Voiceover, word for word | In the film's own type |
|---|---|---|---|
| 0:00 to 0:11 | The cold open, as it is: its picture and its sound, nothing of ours over it | none | none (the open's own: "One hand drives / 48 simulated robots.") |
| 0:11 to 0:17.8 | A plain cut from the open's last frame to the hand film, its hand where the robot's hand was and as tall; after 1.1 s it moves beside the robot's hand, same run, same moment | "This is film of a real hand, played into the page as its camera. The robot's fingers follow it." | "A real hand moves the robot's fingers"; "Film of a hand, as the page's camera"; "Hand: S. Perquin, CC BY-SA 4.0"; "The robot's hand, same moment" |
| 0:17.8 to 0:29.8 | The driven robot close, the page's own line under it | "Everything here runs in a simulator, a twin of Junction's G1 sorting station. The policy is told where the parts are. On the live page, a scripted hand drives when no visitor does." | "Everything here runs in a simulator"; "A twin of Junction's G1 sorting station"; "The policy is told where the parts are"; "No visitor driving? A scripted hand drives" |
| 0:29.8 to 0:36.3 | All 48 robots sorting, slow push in | "Forty-seven echoes redo the driven robot's job as it happens, each on its own table." | "1 robot driven, 47 echoes, each on its own table" |
| 0:36.3 to 0:48.3 | Kaiku off. Three neighbouring tables close, then all 48, framed so the driven robot in front and its table are whole and the page's line sits beside them: the robots reach for parts that are not there. The page's banner counts: 7 of 47 | "Every table is different. So if the others just copy the motion, they reach for parts that are not there." (2.8 s without music on the result) | "Real time"; the page's banner, enlarged |
| 0:48.3 to 1:02.8 | The same tables, Kaiku on. The same three tables close, then all 48, framed the same way. The page's banner: 47 of 47, without: 7. Its count of saved demonstrations rises | "Same tables, Kaiku on. Each echo redoes the motion around the parts that are really there." then "Every clean run is saved as a demonstration." | "Real time"; "Clean runs are saved as demonstrations" |
| 1:02.8 to 1:15.8 | The one number slide, over the robots dimmed | "A policy trained on the scripted hand's thirty-two demonstrations sorts fifty-eight percent of new scenes. With their echoes, ninety-five, more than with a hundred and twenty-eight demonstrations. The echoes cost no one's time." | "A policy trained on the scripted hand's 32 demonstrations"; 58% alone; 95% with their 1,498 echoes; "32 + 1,498 echoes beat 128 alone"; "128 demonstrations alone: 92%. The echoes cost no one's time."; "Share of 96 unseen scenes sorted, mean of 3 training runs." |
| 1:15.8 to 1:28.0 | Three tables at the other end of the rank, sorting | "From the same sixteen demonstrations, our MimicGen-style replay kept five runs, Kaiku's echoes two hundred and fourteen. Policies trained on them sort eight and seventy-three percent." | "The same 16 demonstrations, the same 752 tables"; 5 "clean runs kept by our MimicGen-style replay"; 214 "clean runs kept by Kaiku's echoes"; 8% "of new tables sorted by a policy trained with the 5"; 73% "trained with the 214"; "Each policy: the 16 demonstrations plus the runs kept, tested on 96 unseen tables, mean of 3 runs. The replay is our own, not MimicGen's code." |
| 1:28.0 to 1:36.3 | The local Kaiku the runner followed; the runner's log for 5 s, its summary line large | "The humanoid challenge's own runner, with a mock robot: eighteen hundred commands through its safety filter, none clipped." | "The challenge's own runner and safety filter, with a mock robot"; the runner's own line, 44 px; "1,800 commands through the safety filter. 0 clipped." |
| 1:36.3 to 1:43.3 | Policy mode: three tables close, then all 48 | "Then nobody drives. The policy sorts scenes it has never seen." | "Then nobody drives"; "Real time" |
| 1:43.3 to 1:49.9 | The policy still sorting alone, the page's banner counting what it sorted | "It is live at kaiku echo dot fly dot dev. Open it and drive it yourself." | kaiku-echo.fly.dev; "Open it and drive it yourself"; the hand film's full credit |
| 1:49.9 to 1:51.9 | The end card, as it is | none | none (the card's own) |

## Where each shot comes from

The open and the end card are read from `/bulk3/runs/opens/final/kaiku/` when the film is built
(`open` in `film/film.json`).

All page shots are the hosted page (`https://kaiku-echo.fly.dev`) at 1920x1080: its stream recorded
with `film/rec.py` (it only listens) and drawn again frame by frame by `film/shoot.py` in `drun`
sandboxes. v5 uses v3's recordings (`/bulk3/runs/robotteleop/film/v3/`) and v4's takes of them; it films again
only the comparison's two wide shots (framed a little further back), the robots under the replay comparison and
the number slide's background (each a second longer). Nothing new was pressed on the live page for v4 or v5.

| Shot | Source | Pressed |
|---|---|---|
| 1 | `hand/take4`: `film/hand.py` plays "Finger-counting in Dutch" (S. Perquin, Wikimedia Commons, CC BY-SA 4.0) into the page as its camera | Drive it yourself, Use my hand, Grasp it myself; the turn handed back |
| 2 to 7 | `rec/operator`, Operator mode, a scripted hand driving | See it without Kaiku, once |
| 6, 7 numbers | `kaiku/policy/pretrained/study.json` (the row with 32: 58%, 95%, 1,498 echoes; 128 alone: 92%), `baselines.json` (generation: 5 and 214 clean of 752; policies: 8% and 73%), and the page's own panels (read, not shown) | none |
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
  and the type say so, and both give the counts behind the shares: the replay kept 5 clean runs, Kaiku's echoes 214,
  of the same 752 tables; each policy was trained on the 16 demonstrations plus what was kept.
- The study's 32 demonstrations came from the scripted hand, not a person. The slide says so on screen and in the
  voice, with the 58% and the 95%, and shows that the 32 bring 1,498 echoes.
- The runner shot is the challenge's own runner and safety filter with a mock robot, not a real G1.
- No gloves are worn or claimed.
- The voice is synthetic (Gemini speech); the music is v2's score. Neither plays over the open.

## When a person's gloved hand is filmed

`film/make.sh vN --shot2 FILE` puts a phone clip of a real gloved hand in the first shot after the
open, lined up with the open's last frame the same way, changes the head and the voice line to say
so, and drops the hand film's credit. `film/README.md` has the exact command. It is the one shot an
agent cannot make.
