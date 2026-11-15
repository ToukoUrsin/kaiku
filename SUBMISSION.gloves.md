# Kaiku, after a real glove session

Use this instead of [SUBMISSION.md](SUBMISSION.md) once a person has driven Kaiku in real Smartgloves, and not before. `python -m kaiku.app.sitting` says so in `SITTING.md` when that session exists and is valid.

**Punchline:** One pair of Rokoko gloves, 48 robots.

**Challenge:** The Robot Teleoperation Challenge by Rokoko

**Demo:** https://kaiku-echo.fly.dev

## Description

Kaiku is Finnish for echo. One person in Rokoko Smartgloves sorts parts with a simulated Unitree G1. 47 more G1s do the same job at the same moment, each on its own table where the parts lie somewhere else. Each robot aims the person's motion at its own parts while they are still moving. We call those 47 runs echoes.

Robots learn by watching a job being done, and there are never enough recordings. A policy trained on 32 demonstrations got every part in the right bin on 58% of new tables. With the 1,498 echoes that worked added, 95%. It took 128 demonstrations without echoes to get 92%.

The session in the film is a person wearing real Smartgloves at Junction, read by Rokoko's driver and hand solver. The hosted demo replays it with its echoes when nobody drives. A camera plugs into the same code, so you can drive it without gloves.

MimicGen also turns one demonstration into many, but afterwards, from recordings. Kaiku does it live. On 752 harder tables, with the parts moved much further than in the demo, the echoes sorted 214. Replaying the recording the way MimicGen does sorted 5. We wrote that replay ourselves.

What is not real: it all runs in a simulator, a MuJoCo copy of Junction's G1 sorting station, and the simulator tells the policy where the parts are. The demonstrations behind the numbers came from a scripted hand, not a person. No real G1 has moved.

In the demo, press Drive it yourself, then Use my hand. "See it without Kaiku" shows 47 robots copying the same motion and missing. Runs that get every part in the right bin are saved in LeRobot and GR00T format, and the page offers them as a download.

## How we used Modal

Every policy above was trained on Modal, 63 runs, each on its own GPU. On the live page, "Train both" trains two policies on Modal while you watch, one with the echoes and one without.

## Deck line

One operator in Rokoko gloves drives one robot. 47 more redo the job on their own tables, and a policy learns from all 48.

## Booth pitch

One person wore Rokoko gloves and sorted parts with one robot. Behind it 47 more robots did the same job, each on a table where the parts lie somewhere else. A policy trained on 32 demonstrations sorted 58% of new tables. With the echoes, 95%. That is the recording playing now. Hold your hand up to the camera and you drive it yourself.

## If it was one glove

Rokoko's booth had one glove. If that is the session, change three lines and nothing else:

- Punchline: One Rokoko glove, 48 robots.
- First sentence of the description: One person wearing one Rokoko Smartglove sorts parts with a Unitree G1, and 47 more G1s do the same job at the same moment.
- What is real: one real Smartglove on a real person. Its fingers drove the robot's fingers; the hand's position came from the Coil Pro, or from the laptop's pointer when there was none.

## Technologies

Rokoko Smartgloves, Rokoko Device SDK, Unitree G1 with Dex3 hands, MuJoCo, MediaPipe, PyTorch, Modal, LeRobot, GR00T, three.js, Fly.io
