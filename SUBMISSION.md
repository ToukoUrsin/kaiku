# Kaiku

This is the version in force: no real gloves have been worn. The version for after a real glove session is [SUBMISSION.gloves.md](SUBMISSION.gloves.md).

**Punchline:** One hand, 48 robots.

**Challenge:** The Robot Teleoperation Challenge by Rokoko

**Demo:** https://kaiku-echo.fly.dev

## Description

Kaiku is Finnish for echo. You hold your hand up to a webcam and sort parts with a simulated Unitree G1. 47 more G1s do the same job at the same moment, each on its own table where the parts lie somewhere else, turned differently, sometimes a different shape. So the motion is not copied. It is redone on every table, relative to the parts that are actually there, while you are still moving. We call those 47 runs echoes.

Robots learn by watching the job being done, and there are never enough recordings. We trained a policy on 32 demonstrations from a scripted hand, not a person. It got every part in the right bin on 58% of new tables. With the echoes of those 32 it got 95%. Four times the demonstrations, 128 without echoes, got 92%. Each number is the average of 3 training runs, tested on 96 tables the policy never saw.

Rokoko gloves plug into the same code as the camera. The gloves give the fingers. The camera finds where the hand is, so you need no Coil Pro. We had no real gloves, so we have only tested this with the emulated Smartgloves II in Rokoko's Device SDK. With it, a pinch picks a part up.

What is not real: everything runs in a simulator, a MuJoCo twin of Junction's G1 sorting station, and the simulator tells the policy where the parts are. When no visitor drives the demo, the scripted hand does.

In the demo, press Drive it yourself, then Use my hand. Press See it without Kaiku to watch 47 robots copy the same motion and miss, then the same tables again with Kaiku on.

MimicGen and DexMimicGen also turn one demonstration into many, but afterwards, from recordings. Kaiku does it live. We also compared the two on 752 tables with the parts moved much further than in the demo. Kaiku's echoes sorted every part on 214 of them, a bit over a quarter. Replaying the recording the way MimicGen does managed 5. We wrote that replay ourselves.

Every command is checked against the speed and workspace limits Junction's humanoid challenge sets for its real G1. Runs that pass, with every part in the right bin, are saved in LeRobot and GR00T format, and the page offers them as a download.

## How we used Modal

Every number above was trained on Modal, 63 runs in all, each on its own GPU. The Train both button on the live page trains two policies on Modal while you watch, one with the echoes and one without, about four minutes from the click to the result.

## Deck line

A person's hand drives one robot. 47 more redo the job on their own tables, and a policy learns from all 48.

## Booth pitch

Hold your hand up to the laptop camera. The robot's fingers follow yours and you sort a part. Behind it 47 more robots do the same job, each on a table where the parts lie somewhere else. A policy trained on 16 demonstrations sorted almost no new scenes. With the echoes, 81%. Rokoko gloves plug into the same stream. We tested that with Rokoko's driver and its emulated gloves, because we had no real ones.

## Technologies

Rokoko Device SDK (RGMP, hand solver, emulated Smartgloves II), Unitree G1 with Dex3 hands, MuJoCo, MediaPipe hand landmarks, Python, PyTorch, Modal, LeRobot v3.0, GR00T, three.js, Fly.io
