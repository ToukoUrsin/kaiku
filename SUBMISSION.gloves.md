# Kaiku, after a real glove session

Use this instead of [SUBMISSION.md](SUBMISSION.md) once a person has driven Kaiku in real Smartgloves, and not before. `python -m kaiku.app.sitting` says so in `SITTING.md` when that session exists and is valid.

**Punchline:** One pair of Rokoko gloves, 48 robots.

**Challenge:** The Robot Teleoperation Challenge by Rokoko

**Demo:** https://kaiku-echo.fly.dev

## Description

Kaiku is Finnish for echo. One person in Rokoko Smartgloves sorts parts with a simulated Unitree G1. 47 more G1s do the same job at the same moment, each on its own table where the parts lie somewhere else, turned differently, sometimes a different shape. So the motion is not copied. It is redone on every table, relative to the parts that are actually there, while the person is still moving. We call those 47 runs echoes.

Robots learn by watching the job being done, and there are never enough recordings. We trained a policy on 32 demonstrations from a scripted hand, not a person. It got every part in the right bin on 58% of new tables. With the 1,498 echoes of those 32 it got 95%. Four times the demonstrations, 128 without echoes, got 92%. Each number is the average of 3 training runs, tested on 96 tables the policy never saw.

The session in the film is a person wearing real Smartgloves at Junction, read by Rokoko's driver and hand solver. The hosted demo replays it with its echoes when nobody drives. A camera plugs into the same code as the gloves, so you can drive it without them.

What is not real: everything runs in a simulator, a MuJoCo twin of Junction's G1 sorting station, and the simulator tells the policy where the parts are. No real G1 has moved.

In the demo, press Drive it yourself, then Use my hand, and hold your hand up to the camera. Press See it without Kaiku to watch 47 robots copy the same motion and miss, then the same tables again with Kaiku on.

MimicGen and DexMimicGen also turn one demonstration into many, but afterwards, from recordings. Kaiku does it live. We also compared the two on 752 tables with the parts moved much further than in the demo. Kaiku's echoes sorted every part on 214 of them, a bit over a quarter. Replaying the recording the way MimicGen does managed 5. We wrote that replay ourselves. A policy trained with those 5 sorted 8% of new tables. With 5 echoes it sorted 19%, with all 214, 73%.

Every command is checked against the speed and workspace limits Junction's humanoid challenge sets for its real G1. Runs that pass, with every part in the right bin, are saved in LeRobot and GR00T format, and the page offers them as a download.

## How we used Modal

Every number above was trained on Modal, 63 runs in all, 48 for the curve and 15 for the comparison, each on its own GPU. The Train both button on the live page trains two policies on Modal while you watch, one with the echoes and one without, four to six minutes from the click to the result.

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

Rokoko Smartgloves II, Rokoko Device SDK (RGMP, hand solver), Unitree G1 with Dex3 hands, MuJoCo, MediaPipe hand landmarks, Python, PyTorch, Modal, LeRobot v3.0, GR00T, three.js, Fly.io
