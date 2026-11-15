# Kaiku

This is the version in force: no real gloves have been worn. The version for after a real glove session is [SUBMISSION.gloves.md](SUBMISSION.gloves.md).

**Punchline:** One hand, 48 robots.

**Challenge:** The Robot Teleoperation Challenge by Rokoko

**Demo:** https://kaiku-echo.fly.dev

## Description

Kaiku is Finnish for echo. Hold your hand up to a webcam and a simulated Unitree G1's hand goes where yours goes. Its fingers close when yours do, and you sort parts with it. 47 more G1s do the same job at the same moment, each on its own table where the parts lie somewhere else. Each robot aims your motion at its own parts while you are still moving. We call those 47 runs echoes.

Robots learn by watching a job being done, and there are never enough recordings. A policy trained on 32 demonstrations got every part in the right bin on 58% of new tables. With the 1,498 echoes that worked added, 95%. It took 128 demonstrations without echoes to get 92%.

MimicGen also turns one demonstration into many, but afterwards, from recordings. Kaiku does it live. On 752 harder tables, with the parts moved much further than in the demo, the echoes sorted 214. Replaying the recording the way MimicGen does sorted 5. We wrote that replay ourselves.

What is not real: it all runs in a simulator, a MuJoCo copy of Junction's G1 sorting station, and the simulator tells the policy where the parts are. The demonstrations behind the numbers came from a scripted hand, not a person. We had no real Rokoko gloves. Gloves plug into the same code as the camera, and we tested that with the emulated Smartgloves II in Rokoko's Device SDK.

In the demo, press Drive it yourself, then Use my hand. "See it without Kaiku" shows 47 robots copying your motion and missing. Runs that get every part in the right bin are saved in LeRobot and GR00T format, and the page offers them as a download.

## How we used Modal

Every policy above was trained on Modal, 63 runs, each on its own GPU. On the live page, "Train both" trains two policies on Modal while you watch, one with the echoes and one without.

## Deck line

A person's hand drives one robot. 47 more redo the job on their own tables, and a policy learns from all 48.

## Booth pitch

Hold your hand up to the laptop camera. The robot's fingers follow yours and you sort a part. Behind it 47 more robots do the same job, each on a table where the parts lie somewhere else. A policy trained on 32 demonstrations sorted 58% of new tables. With the echoes, 95%. Rokoko gloves plug into the same stream. We tested that with Rokoko's driver and its emulated gloves, because we had no real ones.

## Technologies

Rokoko Device SDK, Unitree G1 with Dex3 hands, MuJoCo, MediaPipe, PyTorch, Modal, LeRobot, GR00T, three.js, Fly.io
